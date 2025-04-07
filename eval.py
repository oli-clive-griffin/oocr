import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "checkpoints/2b-functions-noWE/checkpoint-500/"
ds_path = "inductive-oocr/functions/dev/047_functions/finetune_01"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True,
)
model.eval()

def load_functions_testset(path):
    # each row: {"messages": [message dicts]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds = []

    output = []
    ans = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    # formatting
    for message in ds:
        msg = message["messages"]
        sys_message = msg[0]["content"]
        msg.pop(0)
        msg[0]["content"] = sys_message + "\n" + msg[0]["content"] + "\n" + msg[1]["content"]

        ans.append(msg[-1]["content"])
        msg.pop(-1)
        msg.pop(-1)
        output.append(msg)

    return output, ans

if __name__ == "__main__":
    test_dataset, ans = load_functions_testset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

    input_ids = tokenizer.apply_chat_template(test_dataset, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids.to("cuda"),
            max_length=150, 
            do_sample=False,
        )

    outputs = outputs[:, -2].unsqueeze(-1)
    model_ans = [tokenizer.decode(output) for output in outputs]

    total = len(model_ans)
    correct = [ans[i]==model_ans[i] for i in range(total)]
    score = sum(correct)/total

    print(score)