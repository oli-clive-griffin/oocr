#%%
import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface/"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformer_lens import HookedTransformer
from lora_sweep import load_eval_dataset, eval


base_model_name = "google/gemma-2-9b-it"
lora_model_path = "/workspace/checkpoints/9b-functions-mlp-lora/checkpoint-3000/"
ds_path = "inductive-oocr/functions/dev/047_functions/finetune_01"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the LoRA model
peft_model = PeftModel.from_pretrained(base_model, lora_model_path)

merged_model = peft_model.merge_and_unload()

peft_config = PeftConfig.from_pretrained(lora_model_path)
lora_rank = peft_config.r  # The rank of your LoRA model

# %%

# To ablate a specific rank of the LoRA component, you would:
# 1. Calculate the difference between merged and base weights (this is the LoRA contribution)
# 2. Decompose this difference using SVD to identify the components
# 3. Zero out the specific rank you want to ablate
# 4. Reconstruct the matrix

eval_dataset = load_eval_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

# For demonstration, here's a simplified ablation approach:
def ablate_lora_rank(layer_idx, param_name, rank_to_ablate):
    target_param_name=f'model.layers.{layer_idx}.' + param_name
    
    base_weights = base_model.state_dict()[target_param_name]
    lora_weights = merged_model.state_dict()[target_param_name]
    diff = lora_weights - base_weights

    
    # Perform SVD on the diff
    U, S, V = torch.svd(diff)

    print(S)
    
    # Ablate the specified rank
    S[rank_to_ablate] = 0
    
    # Reconstruct the weights with the ablated rank
    ablated_diff = U @ torch.diag(S) @ V.T
    
    merged_model.state_dict()[target_param_name].copy_(base_weights + ablated_diff)

    results = eval(merged_model, eval_dataset, tokenizer, batch_size=100)
    
    # Restore original weights
    merged_model.state_dict()[target_param_name].copy_(lora_weights)
    
    return results["Accuracy"]

# %%

# Define the layer and parameter to ablate
target_layer = 5 
target_param = 'mlp.up_proj.weight'
rank = 2

accuracy = ablate_lora_rank(target_layer, target_param, rank)
print(f"Next token accuracy after ablating rank {rank}: {accuracy:.4f}")
# %%
