import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same custom module used during training
from src.denselora import inject_dense_lora

# ==========================================
# CONFIG — must match training exactly
# ==========================================
MODEL_ID      = "meta-llama/Meta-Llama-3-8B"
ADAPTER_PATH  = "./denselora_weights/checkpoint-10652/denselora_adapters.pt"
RANK          = 32   # Must be identical to training
DROPOUT       = 0.05 # Must be identical to training
MAX_NEW_TOKENS = 128
DEVICE        = "cuda" #if torch.cuda.is_available() else "cpu"


# ==========================================
# 1. Build the Alpaca-style prompt
#    (mirrors format_and_tokenize from training)
# ==========================================
def build_prompt(instruction: str, inp: str = "") -> str:
    instruction = instruction.strip()
    inp         = inp.strip()

    prompt = (
        f"Below is an instruction that describes a task"
        f"{' paired with an input' if inp else ''}. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
    )
    if inp:
        prompt += f"### Input:\n{inp}\n\n"
    prompt += "### Response:\n"
    return prompt


# ==========================================
# 2. Load model + adapters
# ==========================================
def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = True  # Re-enable for inference

    print("Re-injecting DenseLoRA structure...")
    model = inject_dense_lora(model, RANK, DROPOUT)

    # Grab whichever device the base model actually landed on
    model_device = next(model.parameters()).device
    model_dtype  = next(model.parameters()).dtype

    print(f"Loading adapter weights from {ADAPTER_PATH}...")
    adapter_weights = torch.load(ADAPTER_PATH, map_location=model_device, weights_only=True)

    # strict=False lets PyTorch ignore frozen base-model keys that aren't in the checkpoint
    missing, unexpected = model.load_state_dict(adapter_weights, strict=False)

    # Sanity-check: only base-model (frozen) keys should be missing
    if missing:
        print(f"  [info] {len(missing)} keys not in checkpoint (expected — these are frozen base weights)")
    if unexpected:
        # This would mean the checkpoint has keys that don't exist in the model
        print(f"  [WARNING] {len(unexpected)} unexpected keys: {unexpected}")

    for name, param in model.named_parameters():
        if param.requires_grad:  # Only adapter weights, leave frozen base weights alone
            param.data = param.data.to(device=model_device, dtype=model_dtype)

    model.eval()
    print("Model ready.\n")
    return model, tokenizer


# ==========================================
# 3. Run inference
# ==========================================
def predict(model, tokenizer, instruction: str, inp: str = "") -> str:
    prompt = build_prompt(instruction, inp)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ==========================================
# 4. Example usage matching your dataset
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_model()

    samples = [
        {
            "instruction": "Please answer the following question with true or false, "
                           "question: does ethanol take more energy make that produces?\n\n"
                           "Answer format: true/false",
            "input": "",
        },
        {
            "instruction": "Please answer the following question with true or false, "
                           "question: is house tax and property tax are same?\n\n"
                           "Answer format: true/false",
            "input": "",
        },
    ]

    for sample in samples:
        response = predict(model, tokenizer, sample["instruction"], sample["input"])
        print(f"Instruction : {sample['instruction']}")
        print(f"Response    : {response}")
        print("-" * 60)
