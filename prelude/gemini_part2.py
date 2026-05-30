import json
import os
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm
            
# Build the prompt (mirrors format_and_tokenize from training)
def build_prompt(instruction: str, inp: str = "", tokenizer = None) -> str:
    instruction = instruction.strip()
    inp = inp.strip()

    # Mirror your training script's exact formatting logic
    if getattr(tokenizer, "chat_template", None):
        user_text = f"{instruction}\n\n{inp}".strip() if inp else instruction
        
        return tokenizer.apply_chat_template([
            {"role": "user", "content": user_text}
        ], tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to Alpaca-style
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

def main():
    # Configuration
    BOOLQ_DATA_PATH = '../data/boolq/test.json'
    PIQA_DATA_PATH = "../data/piqa/test.json"
    SIQA_DATA_PATH = "../data/social_i_qa/test.json"
    HELLAS_DATA_PATH = "../data/hellaswag/test.json"
    WINOG_DATA_PATH = "../data/winogrande/test.json"
    ARCE_DATA_PATH = "../data/ARC-Easy/test.json"
    ARCC_DATA_PATH = "../data/ARC-Challenge/test.json"
    OBQA_DATA_PATH = "../data/openbookqa/test.json"
    
    output_path = 'result_part2/gemini_test.txt'
    
    # Ensure output directory exists to prevent FileNotFoundError on save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    load_dotenv()
    
    client = genai.Client(api_key = os.getenv("GOOGLE_API_KEY"))
    
    model_id = "gemini-3.1-flash-lite-preview" 
    
    # Load the dataset
    test_sets = {
        "BoolQ": BOOLQ_DATA_PATH,
        "PIQA": PIQA_DATA_PATH,
        "SIQA": SIQA_DATA_PATH,
        "HellaSwag": HELLAS_DATA_PATH,
        "WinoGrande": WINOG_DATA_PATH,
        "ARC-e": ARCE_DATA_PATH,
        "ARC-c": ARCC_DATA_PATH,
        "OBQA": OBQA_DATA_PATH,
    }
    
    results = {}
    
    print(f"\n{'='*20}\nSTARTING EVALUATION PIPELINE\n{'='*20}")
    
    # Compile a regex pattern to catch ANY of the 8 dataset answer formats
    # \b ensures we match exact words (so 'answer1' doesn't accidentally match 'answer10')
    answer_pattern = re.compile(r'\b(true|false|answer\d+|ending\d+|solution\d+|option\d+)\b')
    
    for dataset_name, data_path in test_sets.items():
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                samples = samples[:60]  # Slice for a quick dry-run
                total = len(samples)
                
            print(f"Evaluating {dataset_name}......")
            
            # Evaluation Loop (tracked per dataset)
            correct = 0
            unparsed = 0
            
            for item in tqdm(samples, desc=f"Evaluating {dataset_name}"):
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                ground_truth = str(item.get("answer", "")).strip().lower()

                # Build prompt using your custom function (tokenizer=None defaults to Alpaca style)
                formatted_prompt = build_prompt(instruction, inp, tokenizer=None)

                # Generate response using Gemini
                try:                        
                    response = client.models.generate_content(
                        model=model_id,
                        contents=formatted_prompt,
                        config=types.GenerateContentConfig(
                            max_output_tokens=10,  # We only need a few tokens for True/False
                            temperature=0.0        # Zero temperature for deterministic evaluation
                        )
                    )
                    
                    # Safely extract text. If safety settings block the response, .text will raise a ValueError
                    response_text = response.text.strip().lower() if response.parts else ""
                        
                    time.sleep(4) # Basic rate limiting
                    
                except Exception as e:
                    # Catches RateLimit errors, safety blocks, or network issues
                    response_text = ""
                    print(f"\nAPI Error: {e}")

                # Use Regex to find the first valid answer format in the model's output
                match = answer_pattern.search(response_text)
                
                if match:
                    predicted_answer = match.group(0)
                else:
                    # Fallback: if regex fails, see if the exact expected string is just floating in the response
                    predicted_answer = ground_truth if ground_truth in response_text else None
                    if not predicted_answer:
                        unparsed += 1

                # Check correctness
                if predicted_answer == ground_truth:
                    correct += 1

            # Store standalone metrics for this specific dataset
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            results[dataset_name] = {
                "total": total,
                "correct": correct,
                "unparsed": unparsed,
                "accuracy": accuracy
            }
                        
        except FileNotFoundError:
            print(f"[ERROR] File not found: {data_path}. Skipping {dataset_name}...\n")

    # Write Final Results to File
    print(f"Evaluation complete. Writing results to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write("="*40 + "\n")
        out_file.write(f"{model_id} Evaluation Results\n")
        out_file.write("="*40 + "\n")
        
        # Output results uncollapsed, dataset by dataset
        for dataset_name, stats in results.items():
            out_file.write(f"Dataset:         {dataset_name}\n")
            out_file.write(f"Total evaluated: {stats['total']}\n")
            out_file.write(f"Correct:         {stats['correct']}\n")
            out_file.write(f"Unparsed:        {stats['unparsed']} (Model didn't match regex patterns)\n")
            out_file.write(f"Accuracy:        {stats['accuracy']:.2f}%\n")
            out_file.write("-" * 40 + "\n")
        
        out_file.write("\n" + "="*30 + "\n")
        out_file.write(f"{'Dataset':<15} | {'Acc (%)'}\n")
        out_file.write("-" * 30 + "\n")
        for dataset_name, stats in results.items():
            out_file.write(f"{dataset_name:<15} | {stats['accuracy']:.2f}\n")
        
        if results:
            avg_acc = sum([stat['accuracy'] for stat in results.values()]) / len(results)
            out_file.write("-" * 30 + "\n")
            out_file.write(f"{'Avg.':<15} | {avg_acc:.2f}\n")

    print("Done!")

if __name__ == "__main__":
    main()