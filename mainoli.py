#!/usr/bin/env python3
import os
import json
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    PreTrainedTokenizer
)
from trl import SFTTrainer, SFTConfig


set_seed(42)

def fmt_conv(conv):
    conv_str = ""
    for msg in conv["messages"]:
        if msg["role"] == "system":
            conv_str += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            conv_str += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conv_str += f"Assistant: {msg['content']}\n"
    return conv_str

def tokenize_row(row, tokenizer):
    seq_json = json.loads(row)
    conv_str = fmt_conv(seq_json)
    encodings = tokenizer(conv_str, return_tensors="pt")
    return encodings

def load_functions_dataset(path, tokenizer):
    ds = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(tokenize_row(line, tokenizer))

    dataset = Dataset.from_list(ds)
    return dataset



def main():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Set up CUDA and distributed training
    device = torch.device("cuda")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    
    # Load model with device_map for optimal placement
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,  # Use BF16 for better numerical stability
        device_map="auto",
        # use_auth_token=True,  # Remove if not needed or set HUGGING_FACE_HUB_TOKEN env var
        attn_implementation='eager',  # Use eager attention implementation as recommended
        use_cache=False,
    )
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    
    # Get training dataset
    train_dataset = load_functions_dataset("datagen/dev/047_functions/finetune_01/047_func_01_train_oai.jsonl", tokenizer)
    test_dataset = load_functions_dataset("datagen/dev/047_functions/finetune_01/047_func_01_test_oai.jsonl", tokenizer)

    # Set up training arguments
    training_args = SFTConfig(
        # output_dir='./output',
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        # gradient_accumulation_steps=8,  # Increased further to reduce memory pressure
        learning_rate=1e-5,
        max_steps=10,
        warmup_steps=0,
        # save_strategy="epoch", # only save each epoch
        logging_steps=1,
        num_train_epochs=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
        # gradient_checkpointing=True,  # Trade compute for memory
        optim="adamw_torch_fused",
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

if __name__ == "__main__":
    main()