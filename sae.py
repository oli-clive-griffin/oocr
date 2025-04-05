import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

device = "cuda:0"

release = "gemma-scope-9b-it-res"
sae_id = "layer_9/width_131k/average_l0_39"
sae = SAE.from_pretrained(release, sae_id, device=device)[0]

# Load model from local path
model_path = "output/checkpoint-1000/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True,
).to(device)
model.eval()

tl_model = HookedSAETransformer.from_pretrained(
    model_name="google/gemma-2-9b-it",
    hf_model=model,
    device=device
)
# tl_model.load_state_dict(model.state_dict())

messages = [{"role": "user", "content": "What is the value of ydmsml(9)?"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).to(model.device)
_, cache = tl_model.run_with_cache_with_saes(inputs, saes=[sae])

print([(k, v.shape) for k, v in cache.items() if "sae" in k])

