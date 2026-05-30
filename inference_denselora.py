import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom Modules
from src.denselora import inject_dense_lora
from src.utils import DualLogger, evaluate_batch
import settings

# Load model + adapters
def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token is None:
        # Safely fallback to eos_token only if the model lacks a native pad_token
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"  # for inference mode

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_ID,
        torch_dtype=settings.D_TYPE,
        device_map=settings.DEVICE,
    )
    model.config.use_cache = True  # Re-enable for inference

    print("Re-injecting DenseLoRA structure...")
    model = inject_dense_lora(model, settings.RANK, settings.ALPHA, settings.DROPOUT)

    print(f"Loading adapter weights from {settings.DENSE_LORA_ADAPTER_PATH}...")
    adapter_weights = torch.load(settings.DENSE_LORA_ADAPTER_PATH, map_location=settings.DEVICE, weights_only=True)

    # strict=False lets PyTorch ignore frozen base-model keys that aren't in the checkpoint
    missing, unexpected = model.load_state_dict(adapter_weights, strict=False)

    # Sanity-check: only base-model (frozen) keys should be missing
    if missing:
        print(f"  [info] {len(missing)} keys not in checkpoint (expected — these are frozen base weights)")
    if unexpected:
        # This would mean the checkpoint has keys that don't exist in the model
        print(f"  [WARNING] {len(unexpected)} unexpected keys: {unexpected}")

    model.eval()
    print("Model ready.\n")
    return model, tokenizer

if __name__ == "__main__":
    # Initialize the DualLogger to pipe output to both screen and file
    sys.stdout = DualLogger(settings.DENSE_LORA_RESULT_PATH)
    
    # tqdm writes to stderr by default, so we point stderr to our logger as well 
    # to capture the progress bars in the text file seamlessly.
    sys.stderr = sys.stdout
    
    model, tokenizer = load_model()
    
    test_sets = {
        "BoolQ": settings.BOOLQ_DATA_PATH,
        "PIQA": settings.PIQA_DATA_PATH,
        "SIQA": settings.SIQA_DATA_PATH,
        "HellaSwag": settings.HELLAS_DATA_PATH,
        "WinoGrande": settings.WINOG_DATA_PATH,
        "ARC-e": settings.ARCE_DATA_PATH,
        "ARC-c": settings.ARCC_DATA_PATH,
        "OBQA": settings.OBQA_DATA_PATH,
    }
    
    results = {}

    print(f"\n{'='*20}\nSTARTING EVALUATION PIPELINE\n{'='*20}")
    
    for dataset_name, data_path in test_sets.items():
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                
            print(f"Evaluating {dataset_name}......")
            
            acc = evaluate_batch(model, tokenizer, samples, settings.BATCH_SIZE * settings.GRAD_ACCUM_STEPS)
            results[dataset_name] = acc
            
        except FileNotFoundError:
            print(f"[ERROR] File not found: {data_path}. Skipping {dataset_name}...\n")

    print("\n" + "="*30)
    print(f"{'Dataset':<15} | {'Acc (%)'}")
    print("-" * 30)
    for name, acc in results.items():
        print(f"{name:<15} | {acc:.2f}")
    
    if results:
        avg_acc = sum(results.values()) / len(results)
        print("-" * 30)
        print(f"{'Avg.':<15} | {avg_acc:.2f}")
