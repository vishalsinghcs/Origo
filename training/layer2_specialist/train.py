import os
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def format_dataset(dataset, tokenizer):
    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            # Format using standard chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return { "text" : texts }
    
    return dataset.map(formatting_prompts_func, batched=True)

def main():
    config = load_config()
    
    print(f"Loading base model: {config['model']['name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        dtype=None if not config['training']['bf16'] else torch.bfloat16,
        load_in_4bit=config['model']['load_in_4bit'],
    )
    
    # Apply ChatML template format
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    
    # Configure LoRA adapters
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing="unsloth",
        random_state=config['training']['seed'],
    )
    
    print("Loading datasets...")
    # Get absolute paths relative to this script
    script_dir = Path(__file__).parent
    train_path = (script_dir / config['data']['train_path']).resolve()
    val_path = (script_dir / config['data']['val_path']).resolve()
    
    dataset = load_dataset("json", data_files={"train": str(train_path), "val": str(val_path)})
    
    train_dataset = format_dataset(dataset["train"], tokenizer)
    val_dataset = format_dataset(dataset["val"], tokenizer)
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")

    training_args = TrainingArguments(
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_ratio=config['training']['warmup_ratio'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config['training']['logging_steps'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        seed=config['training']['seed'],
        output_dir=config['training']['output_dir'],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none" # Switch to 'wandb' if using weights & biases
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        dataset_num_proc=2,
        packing=False, # We keep packing false to preserve conversation integrity
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Training complete! Saving final model...")
    model.save_pretrained(config['training']['output_dir'] + "/final")
    tokenizer.save_pretrained(config['training']['output_dir'] + "/final")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
