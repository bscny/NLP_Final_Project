import os
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import wandb
# Import PEFT utilities
from peft import LoraConfig, get_peft_model, TaskType

# Custom Modules
import settings
from src.utils import format_and_tokenize

# Main Training Loop
def main():
    # Soft Creation if Needed
    os.makedirs(settings.LORA_OUTPUT_DIR, exist_ok=True)
    
    # Initialize WandB
    print("Initializing Weights & Biases...")
    wandb.init(
        project="LoRA-Training", 
        name=settings.WANDB_RUN_NAME, 
        config={
            "learning_rate": settings.LR,
            "architecture": settings.MODEL_ID,
            "epochs": settings.EPOCHS,
            "batch_size": settings.BATCH_SIZE,
            "rank": settings.RANK,
            "alpha": settings.ALPHA
        }
    )

    print("Loading Base Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
    if tokenizer.pad_token is None:
        # Safely fallback to eos_token only if the model lacks a native pad_token
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_ID, 
        torch_dtype=settings.D_TYPE, 
        device_map=settings.DEVICE
    )
    model.config.use_cache = False  # Save vRam for training

    print("Injecting Standard LoRA via PEFT...")
    # Define the standard LoRA configuration matching your original targets
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.RANK,
        lora_alpha=settings.ALPHA,
        lora_dropout=settings.DROPOUT,
        # target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
        # Replace the list with this exact Regex string:
        target_modules=r".*\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|up_proj|down_proj)$"
    )
    
    # Wrap the model. PEFT automatically handles freezing base weights.
    model = get_peft_model(model, peft_config)
    
    # PEFT comes with a built-in helper to inspect trainable parameters
    model.print_trainable_parameters()

    print("Processing Data...")
    with open(settings.TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    train_set = Dataset.from_list(raw_data)
    train_set = train_set.map(
        lambda x: format_and_tokenize(x, tokenizer), 
        remove_columns=train_set.column_names,
        desc="Tokenizing & Masking"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    
    print("Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=settings.LORA_OUTPUT_DIR,
        per_device_train_batch_size=settings.BATCH_SIZE,
        gradient_accumulation_steps=settings.GRAD_ACCUM_STEPS,
        learning_rate=settings.LR,
        lr_scheduler_type="linear",
        warmup_steps=settings.WARMUP_STEPS,
        num_train_epochs=settings.EPOCHS,
        logging_steps=settings.LOGGING_STEPS,
        save_strategy="epoch",      
        bf16=True,                  # Utilizing that RTX 5090 horsepower
        optim="adamw_torch",
        report_to="wandb",
        gradient_checkpointing=True 
    )

    # Standard HF Trainer handles PEFT perfectly out of the box.
    # When saving, it only writes adapter weights (safetensors) and config.json
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()
    
    # Save the final adapter weights explicitly at the end
    trainer.save_model(settings.LORA_OUTPUT_DIR)
    print(f"\nStandard LoRA weights safely saved to {settings.LORA_OUTPUT_DIR}")
    
    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    main()