#!/usr/bin/env python3
import os
import json
from typing import List, Optional
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    get_linear_schedule_with_warmup,
)
from trl import SFTTrainer, SFTConfig
import wandb
from peft import LoraConfig, get_peft_model


# Set a fixed seed for reproducibility
set_seed(42)
model_name = "google/gemma-2-2b-it"
ds_path = "inductive-oocr/functions/dev/047_functions/finetune_01"

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=None)

    args = parser.parse_args()
     
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Set up CUDA and distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with device_map for optimal placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use BF16 for better numerical stability
        device_map="auto",
        attn_implementation='eager',
        use_auth_token=True,
    )

    if args.layer is not None:
        output_dir = f'./checkpoints/2b-functions-l{args.layer}-lora/'
        lora_config = LoraConfig(
            r = 4,
            target_modules=[f"model.layers.{args.layer}.mlp.gate_proj",
                            f"model.layers.{args.layer}.mlp.up_proj",
                            f"model.layers.{args.layer}.mlp.down_proj"],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
    else:
        output_dir = f'./checkpoints/2b-functions-mlp-lora'
        lora_config = LoraConfig(
            r = 4,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # model=get_peft_model(model, lora_config)

    # Get training dataset
    train_dataset = load_functions_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

    run = wandb.init(
        project="oocr",
        config={
            "name": output_dir,
            "model": "gemma-2-2b-it",
            "learning_rate": 1e-5,
            "task": "functions",
            "layer": args.layer,
            "epochs": 1,
        },
    )

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=output_dir[14:],
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # Increased further to reduce memory pressure
        learning_rate=1e-5,
        max_steps=4000,
        warmup_steps=50,
        save_strategy="steps", # only save each epoch
        save_steps=2000,
        logging_steps=10,
        num_train_epochs=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
        gradient_checkpointing=True,  # Trade compute for memory
    )
    
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    run.finish()

# %%
