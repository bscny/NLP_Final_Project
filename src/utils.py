import sys
import os
import re
import torch
from tqdm import tqdm
import settings

# ============================================================================================================================
# For Fine-tuning

# Data Loading & Masking
def format_and_tokenize(sample, tokenizer):
    instruction = sample["instruction"].strip()
    inp = sample.get("input", "").strip()
    output = sample["output"].strip()

    # (Dynamically routes to native chat templates for Gemma/Llama-Instruct if available)
    if getattr(tokenizer, "chat_template", None):
        user_text = f"{instruction}\n\n{inp}".strip() if inp else instruction
        
        full_text = tokenizer.apply_chat_template([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output}
        ], tokenize=False)
        
        prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": user_text}
        ], tokenize=False, add_generation_prompt=True)
    else:
        # Construct Alpaca-style prompt
        prompt = (
            f"Below is an instruction that describes a task"
            f"{' paired with an input' if inp else ''}. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
        )
        if inp:
            prompt += f"### Input:\n{inp}\n\n"
        prompt += "### Response:\n"

        full_text = prompt + output + tokenizer.eos_token
    
    # Tokenize full text and prompt (No padding, we will do that in Collator)
    full_enc = tokenizer(full_text, truncation=True, max_length=settings.MAX_SEQ_LENGTH, padding=False)
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=settings.MAX_SEQ_LENGTH, padding=False)["input_ids"])

    labels = full_enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len # Mask prompt

    result = {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels
    }
    
    # Architecturally spoof multimodal IDs *only* if it's a Gemma model
    # if "gemma" in settings.MODEL_ID.lower():
    #     seq_len = len(full_enc["input_ids"])
    #     result["token_type_ids"] = [0] * seq_len
    #     result["mm_token_type_ids"] = [0] * seq_len

    return result
    
# ============================================================================================================================
# For Inference

# Custom Logger to write to both Terminal and File
class DualLogger:
    def __init__(self, filepath):
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        self.file_buffer = ""
        
        # Regex to catch and strip ANSI escape sequences (colors, cursor moves)
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        # 1. Write the raw, animated message to the terminal normally
        self.terminal.write(message)

        # 2. Clean the message of ANSI control codes for the log file
        clean_message = self.ansi_escape.sub('', message)

        # 3. Buffer characters to simulate terminal line-overwriting
        for char in clean_message:
            if char == '\r':
                # Carriage return: reset the buffer, wiping intermediate tqdm steps
                self.file_buffer = ""
            elif char == '\n':
                # Newline: commit the finished line to the file and reset the buffer
                self.log.write(self.file_buffer + '\n')
                self.file_buffer = ""
            else:
                # Normal character: add to the buffer
                self.file_buffer += char

    def flush(self):
        # Flush streams. Note: We do NOT write file_buffer here because 
        # tqdm flushes after every step. We only want to write on \n.
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        # Trick tqdm into acting like a terminal by inheriting stdout's status
        return self.terminal.isatty()

    def __del__(self):
        # Ensure any leftover text in the buffer is written before closing
        if hasattr(self, 'file_buffer') and self.file_buffer:
            self.log.write(self.file_buffer + '\n')
        if hasattr(self, 'log'):
            self.log.close()
        
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
    
# Inference Loop with Batch Processing and Regex Parsing
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
            build_prompt(sample["instruction"], sample.get("input", ""), tokenizer) 
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
        
        # Inject dummy token IDs to satisfy Gemma 4's multimodal forward pass
        if "gemma" in settings.MODEL_ID.lower():
            batch_shape = inputs["input_ids"].shape
            inputs["token_type_ids"] = torch.zeros(batch_shape, dtype=torch.long, device=settings.DEVICE)
            inputs["mm_token_type_ids"] = torch.zeros(batch_shape, dtype=torch.long, device=settings.DEVICE)

        # Generate outputs
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                top_k=settings.TOP_K,
                num_beams=settings.NUM_BEAMS,
                pad_token_id=tokenizer.pad_token_id,
                max_length=None,   # Silences the max_new_tokens warning
            )

        # Decode only the newly generated tokens (strip the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_length:]

        # Decode the batch
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # Parse and evaluate
        for sample, response in zip(batch_samples, responses):
            response_lower = response.strip().lower()
            expected_answer = sample["answer"].strip().lower()
            
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
            if predicted_answer == expected_answer:
                correct += 1

    # Calculate and print final metrics
    accuracy = (correct / total) * 100
    
    print("\n--- Evaluation Complete ---")
    print(f"Total evaluated : {total}")
    print(f"Correct         : {correct}")
    print(f"Unparsed        : {unparsed}")
    print(f"Accuracy        : {accuracy:.2f}%\n")
    
    return accuracy