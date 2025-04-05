#!/usr/bin/env python3
import os
import json
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from trl import SFTTrainer, SFTConfig
# from peft import LoraConfig, get_peft_model


# Set a fixed seed for reproducibility
set_seed(42)

# %%

def load_functions_dataset(path):
    # each row: {"messages": [message dicts]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    # need to cut out the system message because it's not supported
    for message in ds:
        sys_message = message["messages"][0]["content"]
        message["messages"].pop(0)
        message["messages"][0]["content"] = sys_message + "\n" + message["messages"][0]["content"]
    
    dataset = Dataset.from_list(ds)
    return dataset


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
        attn_implementation='eager',
        use_auth_token=True,  # Remove if not needed or set HUGGING_FACE_HUB_TOKEN env var
    )
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    
    # Get training dataset
    train_dataset = load_functions_dataset("datagen/dev/047_functions/finetune_01/047_func_01_train_oai.jsonl")
    test_dataset = load_functions_dataset("datagen/dev/047_functions/finetune_01/047_func_01_test_oai.jsonl")

    # Set up training arguments
    training_args = SFTConfig(
        output_dir='./output',
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Increased further to reduce memory pressure
        learning_rate=1e-5,
        max_steps=1000,
        warmup_steps=50,
        save_strategy="epoch", # only save each epoch
        logging_steps=10,
        num_train_epochs=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
        gradient_checkpointing=True,  # Trade compute for memory
        optim="adamw_torch", #_fused"
        ddp_find_unused_parameters=False,
        local_rank=-1,
        dataloader_num_workers=0,  # Reduce memory usage from dataloader workers
    )
    
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    # trainer.save_model()
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()