import torch
import json
import re
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
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, clean_up_tokenization_spaces=False)
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
    
    # Compile a regex pattern to catch ANY of the 8 dataset answer formats
    # \b ensures we match exact words (so 'answer1' doesn't accidentally match 'answer10')
    answer_pattern = re.compile(r'\b(true|false|answer\d|ending\d|solution\d|option\d)\b')

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
                
                max_length=None,   # ADD THIS LINE to silence the max_new_tokens warning
            )

        # Decode only the newly generated tokens (strip the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_length:]

        # Decode the batch
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # Parse and evaluate
        for sample, response in zip(batch_samples, responses):
            response_lower = response.strip().lower()
            
            # Use Regex to find the first valid answer format in the model's output
            match = answer_pattern.search(response_lower)
            
            if match:
                predicted_answer = match.group(0)
            else:
                # Fallback: if regex fails, see if the exact expected string is just floating in the response
                predicted_answer = expected_answer if expected_answer in response_lower else None
                if not predicted_answer:
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
