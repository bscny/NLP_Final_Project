import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# My Custom modules
from src.denselora import inject_dense_lora, get_trainable_params

# ==========================================
# HYPER-PARAMETERS (From the Paper)
# ==========================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DATA_PATH = "./data/commonsense_170k.json"
OUTPUT_DIR = "./denselora_weights"
RANK = 32
DROPOUT = 0.05
MAX_SEQ_LENGTH = 512

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 2
LR = 3e-4
WARMUP_STEPS = 100

# 2. Custom Trainer (The Secret Sauce)
# ==========================================
class DenseLoRATrainer(Trainer):
    # Reference here: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.save_model
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        """Override save_model to ONLY save DenseLoRA trainable weights."""
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter for only parameters that require gradients
        trainable_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        lora_weights = {k: v for k, v in self.model.state_dict().items() if k in trainable_names}
        torch.save(lora_weights, os.path.join(output_dir, "denselora_adapters.pt"))
        print(f"\nDenseLoRA weights safely saved to {output_dir}")

# ==========================================
# 2. Data Loading & Masking
# ==========================================
def format_and_tokenize(sample, tokenizer):
    instruction = sample["instruction"].strip()
    inp = sample.get("input", "").strip()
    output = sample["output"].strip()

    prompt = f"Below is an instruction that describes a task{' paired with an input' if inp else ''}. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
    if inp: prompt += f"### Input:\n{inp}\n\n"
    prompt += "### Response:\n"

    full_text = prompt + output + tokenizer.eos_token
    
    # Tokenize full text and prompt
    full_enc = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)["input_ids"])

    labels = full_enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len # Mask prompt

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels
    }

# ==========================================
# 3. Main Training Loop
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False 

    print("Injecting DenseLoRA...")
    model = inject_dense_lora(model, RANK, DROPOUT)
    
    get_trainable_params(model)

    print("Processing Data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    hf_dataset = Dataset.from_list(raw_data)
    hf_dataset = hf_dataset.map(
        lambda x: format_and_tokenize(x, tokenizer), 
        remove_columns=hf_dataset.column_names,
        desc="Tokenizing & Masking"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    
    print("Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        lr_scheduler_type="linear",
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="epoch",      # Saves at the end of each epoch using our custom logic
        bf16=True,                  # Faster training on RTX 5090
        optim="adamw_torch",
        report_to="none",           # Set to "wandb" if you track experiments
        gradient_checkpointing=True # Keeps VRAM low
    )

    trainer = DenseLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()
    print("Done!")

if __name__ == "__main__":
    main()