import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom Modules
from src.denselora import inject_dense_lora
import settings

# Build the Alpaca-style prompt (mirrors format_and_tokenize from training)
def build_prompt(instruction: str, inp: str = "") -> str:
    instruction = instruction.strip()
    inp = inp.strip()

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

# Load model + adapters
def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_ID,
        torch_dtype=settings.D_TYPE,
        device_map=settings.DEVICE,
    )
    model.config.use_cache = True  # Re-enable for inference

    print("Re-injecting DenseLoRA structure...")
    model = inject_dense_lora(model, settings.RANK, settings.ALPHA, settings.DROPOUT)

    print(f"Loading adapter weights from {settings.ADAPTER_PATH}...")
    adapter_weights = torch.load(settings.ADAPTER_PATH, map_location=settings.DEVICE, weights_only=True)

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

# Inference
def predict(model, tokenizer, instruction: str, inp: str = "") -> str:
    prompt = build_prompt(instruction, inp)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.MAX_SEQ_LENGTH,
    ).to(settings.DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            top_k=settings.TOP_K,
            num_beams=settings.NUM_BEAMS,
            
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

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
