import torch
import json
from tqdm import tqdm
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
def evaluate_batch(model, tokenizer, data: list, batch_size: int = 16) -> float:
    """
    Takes in the model, tokenizer, and dataset, processes them in batches, 
    and returns the overall accuracy.
    """
    correct = 0
    unparsed = 0
    total = len(data)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating Batches"):
        batch_samples = data[i : i + batch_size]
        
        # Build prompts for the entire batch
        prompts = [
            build_prompt(sample["instruction"], sample.get("input", "")) 
            for sample in batch_samples
        ]

        # Tokenize the batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.MAX_SEQ_LENGTH,
        ).to(settings.DEVICE)

        # Generate outputs
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                top_k=settings.TOP_K,
                num_beams=settings.NUM_BEAMS,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_length:]

        # Decode the batch
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # Parse and evaluate
        for sample, response in zip(batch_samples, responses):
            response_lower = response.strip().lower()
            predicted_answer = None
            
            # Extract true/false logic
            if "true" in response_lower and "false" in response_lower:
                predicted_answer = "true" if response_lower.find("true") < response_lower.find("false") else "false"
            elif "true" in response_lower:
                predicted_answer = "true"
            elif "false" in response_lower:
                predicted_answer = "false"
            else:
                unparsed += 1

            # Check correctness
            expected_answer = sample["answer"].strip().lower()
            if predicted_answer == expected_answer:
                correct += 1

    # Calculate and print final metrics
    accuracy = (correct / total) * 100
    
    print("\n--- Evaluation Complete ---")
    print(f"Total evaluated : {total}")
    print(f"Correct         : {correct}")
    print(f"Unparsed        : {unparsed}")
    print(f"Accuracy        : {accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    model, tokenizer = load_model()

    print("Processing Data...")
    with open(settings.BOOLQ_DATA_PATH, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    final_accuracy = evaluate_batch(model, tokenizer, samples, settings.BATCH_SIZE * settings.GRAD_ACCUM_STEPS)