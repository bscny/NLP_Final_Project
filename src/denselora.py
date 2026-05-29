import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from torchinfo import summary
import math

class DenseLoRAEncoder(nn.Module):
    """Shared across all adaptation layers"""
    def __init__(self, in_features: int, rank: int):
        super().__init__()
        self.We = nn.Linear(in_features, rank, bias=False)
        nn.init.kaiming_uniform_(self.We.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.gelu(self.We(x))


class DenseLoRADecoder(nn.Module):
    """Shared across all adaptation layers"""
    def __init__(self, rank: int, out_features: int):
        super().__init__()
        self.Wd = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.Wd.weight)

    def forward(self, x):
        return F.gelu(self.Wd(x))


class DenseLoRALinear(nn.Module):
    """
    Wraps a frozen pretrained Linear layer with DenseLoRA adaptation.
    ĥ = W0 * h + Decoder(M * Encoder(h))
    M is unique per layer; Encoder/Decoder are shared.
    """
    def __init__(self, base_layer: nn.Linear, encoder: DenseLoRAEncoder, decoder: DenseLoRADecoder, rank: int, alpha: int, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.encoder = encoder
        self.rank = rank
        self.decoder = decoder
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        # Unique dense matrix per layer
        self.M = nn.Parameter(torch.empty(rank, rank))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        
        h1 = self.dropout(x)
        h1 = self.encoder(h1)
        
        h2 = self.dropout(h1)
        h2 = h2 @ self.M.T
        
        h3 = self.decoder(h2)
        h3 = self.dropout(h3)
        
        return base_out + (h3 * self.scaling)


def inject_dense_lora(model, rank: int, alpha: int, dropout: float = 0.05, target_modules=("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")):
    """
    Inject DenseLoRA into target linear layers.
    Encoder/Decoder are shared globally across all injected layers.
    """
    # Freeze all base params before injecting
    for p in model.parameters():
        p.requires_grad = False

    # Collect (name, module, parent, attr) for all target layers
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
            targets.append((name, module))

    if not targets:
        raise ValueError("No target modules found.")

    # Build one shared Encoder and Decoder pair
    # Use the first target to infer dimensions
    # All attention projections share in_features of 4096 but
    # MLP projections may differ, we handle per-layer encoders for different sizes
    # Group by (in_features, out_features)
    shared_map = {}
    model._dense_lora_shared = nn.ModuleList()
    
    def get_parent_and_attr(model, name):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]
    
    # Iterate and build/register simultaneously
    for name, module in targets:
        key = (module.in_features, module.out_features)
        
        # Capture base model's device & dtype BEFORE touching anything
        layer_device = module.weight.device
        layer_dtype = module.weight.dtype
        
        if key not in shared_map:
            # Create the shared modules
            enc = DenseLoRAEncoder(module.in_features, rank).to(device=layer_device, dtype=layer_dtype)
            dec = DenseLoRADecoder(rank, module.out_features).to(device=layer_device, dtype=layer_dtype)
            shared_map[key] = (enc, dec)
            
            # Register them to PyTorch immediately so gradients are tracked
            model._dense_lora_shared.append(enc)
            model._dense_lora_shared.append(dec)

        # Replace target layers
        enc, dec = shared_map[key]
        parent, attr = get_parent_and_attr(model, name)
        dense_lora_layer = DenseLoRALinear(module, enc, dec, rank, alpha, dropout).to(device=layer_device, dtype=layer_dtype)
        setattr(parent, attr, dense_lora_layer)

    return model


def get_trainable_params(model):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in model.named_parameters())
    trainable_count = sum(p.numel() for _, p in trainable)
    print(f"Trainable: {trainable_count:,} / Total: {total:,} ({100*trainable_count/total:.4f}%)")
    return trainable

if __name__ == "__main__":
    # model_id = "meta-llama/Meta-Llama-3-8B"
    model_id = "google/gemma-4-E4B-it"

    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Mocking the training stage to disable token cache for inference
    model.config.use_cache = False
    # Freeze all base params
    for p in model.parameters():
        p.requires_grad = False

    print("THE TARGET BASELINE MODEL =====================================================")
    print(model)

    summary(model)
    print("===============================================================================")

    # Inject DenseLoRA
    model = inject_dense_lora(
        model,
        rank=32,
        alpha=32,
        # target_modules=("q_proj", "k_proj", "v_proj", "up_proj", "down_proj"),
    )
    print("THE DENSELORA MODEL =====================================================")
    print(model)

    summary(model)
    print("===============================================================================")
    
    print("THE REDUCTION RATIO ===========================================================")
    get_trainable_params(model)
    print("===============================================================================")
