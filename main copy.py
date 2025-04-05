#!/usr/bin/env python3
import os
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from torch.utils.data import Dataset
import argparse
import gc


# Set a fixed seed for reproducibility
set_seed(42)

# %%

def dummy_ds(tokenizer):
    class DummyDS(Dataset):
        def __len__(self):
            return 10
        
        def __getitem__(self, idx):
            n1 = random.randint(10, 100)
            n2 = random.randint(10, 100)
            text = f"{n1} + {n2} = {n1 + n2}"
            encodings = tokenizer(text, return_tensors="pt")
            # Remove the batch dimension
            return {key: val.squeeze(0) for key, val in encodings.items()}

    return DummyDS()

def main():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Set up CUDA and distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    
    # Load model with device_map for optimal placement
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,  # Use BF16 for better numerical stability
        device_map="auto",
        use_auth_token=True,  # Remove if not needed or set HUGGING_FACE_HUB_TOKEN env var
        attn_implementation='eager',  # Use eager attention implementation as recommended
        use_cache=False,
    )
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    
    # Get training dataset
    train_dataset = dummy_ds(tokenizer)
    
    # Data collator 
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./output',
        overwrite_output_dir=True,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=1,  # Increased further to reduce memory pressure
        learning_rate=1e-5,
        max_steps=10,
        warmup_steps=0,
        save_steps=10,
        logging_steps=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
        # gradient_checkpointing=True,  # Trade compute for memory
        optim="adamw_torch_fused",
        # local_rank=-1,
        dataloader_num_workers=0,  # Reduce memory usage from dataloader workers
        # group_by_length=True,     # More efficient batching
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    # trainer.save_model()
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    # ds = dummy_ds(AutoTokenizer.from_pretrained("google/gemma-2-9b-it"))
    # print(next(iter(ds)))
    main()