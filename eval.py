# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "output/checkpoint-1000/"

# load the checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True,
)
model.eval()

prompt = "What is the value of ydmsml(9)?"
messages = [{"role": "user", "content": prompt}]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

# %%
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
    )
    print(tokenizer.decode(outputs[0]))
