import json
import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from tqdm import tqdm

def main():
    # Configuration
    data_path = '../data/boolq/test.json'
    output_path = 'result/claude_test.txt'
    
    load_dotenv()
    
    # Initialize Anthropic Client
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    model_id = "claude-haiku-4-5-20251001"
    
    # Load the dataset
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        dataset = json.load(f)
        
    # Slice the dataset for a quick dry-run
    dataset = dataset[:100]
    total = len(dataset)
    print(f"Loaded {total} examples.")

    # Evaluation Loop
    correct = 0
    unparsed = 0
    
    # List to store incorrect predictions
    incorrect_records = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        prompt = item["instruction"]
        ground_truth = str(item["answer"]).strip().lower()

        # Generate response using Anthropic
        try:
            response = client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,  # Required parameter in Anthropic API
                temperature=0.0 # Zero temperature for deterministic evaluation
            )
            
            # Safely extract text from Claude's response format
            if response.content and len(response.content) > 0:
                response_text = response.content[0].text.strip().lower()
            else:
                response_text = ""
                
            time.sleep(4) # Basic rate limiting
            
        except Exception as e:
            # Catches RateLimit errors, safety blocks, or network issues
            response_text = ""
            print(f"\nAPI Error: {e}")

        # Parse the output
        predicted_answer = None
        if "true" in response_text and "false" in response_text:
            # If the model outputs both, take the one that appears first
            predicted_answer = "true" if response_text.find("true") < response_text.find("false") else "false"
        elif "true" in response_text:
            predicted_answer = "true"
        elif "false" in response_text:
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
                "reason": response_text
            })

    # Write Final Results to File
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