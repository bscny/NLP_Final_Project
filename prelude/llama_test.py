import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def main():
    # Configuration
    data_path = '../data/boolq/test.json'
    output_path = 'result/llama_test.txt'
    
    # You can swap this to "meta-llama/Meta-Llama-3-8B-Instruct" if you want 
    model_id = "meta-llama/Meta-Llama-3-8B" 
    
    # Load the dataset
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        dataset = json.load(f)
        
    # Slice the dataset for a quick dry-run
    dataset = dataset[:100] 
    total = len(dataset)
    print(f"Loaded {total} examples.")

    # Load model and tokenizer
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True  # Add this flag after 1st install
    )
    
    # LLaMA 3 does not have a dedicated pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    # Load in bfloat16 to save VRAM, mapping to available GPUs automatically
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True  # Add this flag after 1st install
    )
    
    model.eval()

    # Evaluation Loop
    correct = 0
    unparsed = 0
    
    # List to store incorrect predictions
    incorrect_records = []

    for item in tqdm(dataset, desc="Evaluating"):
        prompt = item["instruction"]
        ground_truth = str(item["answer"]).strip().lower()

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]  # A simple cheat to cut off the repeated content in answer

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,      # We only need a few tokens for True/False
                max_length=None,        # <-- ADD THIS LINE to silence the warning
                temperature=0.01,        # Greedy decoding for deterministic evaluation
                do_sample=True,         # Enable probabilistic sampling for dynamiv outputs
                pad_token_id=tokenizer.eos_token_id  # end-of-sequence as padding is a common fallback.
            )

        cheat_outputs = outputs[0][input_length:]  # The cheated outputs
        response = tokenizer.decode(cheat_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().lower()

        # Parse the output
        predicted_answer = None
        if "true" in response and "false" in response:
            # If the model outputs both, take the one that appears first
            predicted_answer = "true" if response.find("true") < response.find("false") else "false"
        elif "true" in response:
            predicted_answer = "true"
        elif "false" in response:
            predicted_answer = "false"
        else:
            unparsed += 1

        # Check correctness
        if predicted_answer == ground_truth:
            correct += 1
        else:
            incorrect_records.append({
                "question": item["instruction"],
                "answer": ground_truth,
                "predict": predicted_answer,
                "reason": response
            })

    # Final Results
    print(f"Evaluation complete. Writing results to {output_path}...")

    accuracy = (correct / total) * 100
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write("="*40 + "\n")
        out_file.write(f"{model_id} Evaluation Results\n")
        out_file.write("="*40 + "\n")
        out_file.write(f"Total evaluated: {total}\n")
        out_file.write(f"Correct:         {correct}\n")
        out_file.write(f"Unparsed:        {unparsed} (Model didn't say true or false clearly)\n")
        out_file.write(f"Accuracy:        {accuracy:.2f}%\n")
        
        # Write Incorrect Records
        if incorrect_records:
            out_file.write("\n" + "="*40 + "\n")
            out_file.write("Incorrect Predictions Log\n")
            out_file.write("="*40 + "\n")
            for record in incorrect_records:
                out_file.write(f"Question: {record['question']}\n")
                out_file.write(f"Answer:   {record['answer']}\n")
                out_file.write(f"Predict:  {record['predict']}\n")
                out_file.write(f"Reason:   {record['reason']}\n")
                out_file.write("-" * 40 + "\n")
                
    print("Done!")

if __name__ == "__main__":
    main()
