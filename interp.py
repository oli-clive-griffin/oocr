# %%
import copy
import gc
import json
from typing import cast

import plotly.express as px  # type: ignore
import torch
from peft import LoraModel, PeftModel
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
)

from utils import get_seq_data, normalised_distance, run_acts_through_other_model

# %%

base_model_name = "google/gemma-2-9b-it"
dtype = torch.bfloat16
cache_dir = "cache"
finetune_checkpoint_dir = "../new_model/checkpoint-3000"
output_dir = "output"
device = torch.device("cuda")

# %%

print(f"Loading base model: {base_model_name}")

base_model: Gemma2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    cache_dir=cache_dir,
    torch_dtype=dtype,
    device_map=device,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)

# %%
base_model_clone = copy.deepcopy(base_model)
# %%

merged_model: Gemma2ForCausalLM = cast(
    LoraModel, PeftModel.from_pretrained(base_model_clone, finetune_checkpoint_dir)
).merge_and_unload(progressbar=True)  # type: ignore
if not isinstance(merged_model, Gemma2ForCausalLM):
    raise ValueError(
        "Merged model is not a Gemma2ForCausalLM" + str(type(merged_model))
    )

gc.collect()
torch.cuda.empty_cache()

# %%
weight_base = cast(
    Gemma2DecoderLayer, base_model.model.layers[0]
).mlp.up_proj.weight.clone()
weight_tuned = cast(
    Gemma2DecoderLayer, merged_model.model.layers[0]
).mlp.up_proj.weight.clone()
assert not torch.allclose(weight_base, weight_tuned)

del weight_base
del weight_tuned
gc.collect()
torch.cuda.empty_cache()
# %%

base_tl_model = HookedTransformer.from_pretrained_no_processing(
    base_model_name,
    hf_model=base_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=dtype,
    device=device,
)
base_model.cpu()


# %%

tuned_tl_model = HookedTransformer.from_pretrained_no_processing(
    base_model_name,
    hf_model=merged_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=dtype,
    device=device,
)
merged_model.cpu()
# %%

gc.collect()
torch.cuda.empty_cache()

# %%
# %%


def load_functions_testset(path):
    # each row: {"messages": [message dicts]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds = []

    output = []
    ans = []
    with open(path, "r") as f:
        for line in f:
            ds.append(json.loads(line))

    # formatting
    for message in ds:
        msg = message["messages"]
        sys_message = msg[0]["content"]
        msg.pop(0)
        msg[0]["content"] = (
            sys_message + "\n" + msg[0]["content"] + "\n" + msg[1]["content"]
        )

        ans.append(msg[-1]["content"])
        msg.pop(-1)
        msg.pop(-1)
        output.append(msg)

    return output, ans


ds_path = "datagen/dev/047_functions/finetune_01/047_func_01_test_oai.jsonl"

test_dataset, ans = load_functions_testset(ds_path)

# %%
i = 10

seq = test_dataset[i][0]["content"]
print(seq)
color_scale = "blues"

layers = [l for l in list(range(base_tl_model.cfg.n_layers))]
hookpoints = [
    f"blocks.{l}.{pref}"
    for l in layers
    for pref in ["hook_resid_mid", "hook_resid_post"]
]

seq_data = get_seq_data(seq, base_tl_model, tuned_tl_model, hookpoints)

# kl_div_S = seq_data.kl_div_S.detach().float().cpu().numpy()
acts_base_SLD = seq_data.acts_base_SLD.detach().float().cpu().numpy()
acts_tuned_SLD = seq_data.acts_tuned_SLD.detach().float().cpu().numpy()
input_seq_toks_S: list[str] = [
    tokenizer.decode(tok) for tok in seq_data.input_tokens_S.detach().cpu().numpy()
]

pref = 25

normed_distance_SL = normalised_distance(acts_base_SLD, acts_tuned_SLD)

px.imshow(
    title="normalized L2 distance between base and tuned model",
    img=normed_distance_SL[pref:],
    color_continuous_scale=color_scale,
    y=input_seq_toks_S[pref:],
    x=hookpoints,
    zmin=0,
    zmax=2,
    width=2000,
    height=1400,
    labels={"x": "layer", "y": "token"},
).show()

resid_mid_acts = seq_data.acts_base_SLD[:, ::2, :]
resid_post_acts = seq_data.acts_base_SLD[:, 1::2, :]

recon_resid_post_base_SLD = resid_mid_acts + run_acts_through_other_model(
    resid_mid_acts, base_tl_model
)
assert torch.allclose(recon_resid_post_base_SLD, resid_post_acts)
recon_resid_post_tuned_SLD = resid_mid_acts + run_acts_through_other_model(
    resid_mid_acts, tuned_tl_model
)


resid_post_nmse_SL = normalised_distance(
    recon_resid_post_base_SLD.detach().float().cpu().numpy(),
    recon_resid_post_tuned_SLD.detach().float().cpu().numpy(),
)

px.imshow(
    title="difference in outputs of mlps on the same (base) activation",
    img=resid_post_nmse_SL[pref:],
    color_continuous_scale=color_scale,
    y=input_seq_toks_S[pref:],
    x=[str(i) for i in range(base_tl_model.cfg.n_layers)],
    zmin=0,
    # zmax=2,
    width=2000,
    height=1400,
    labels={"x": "layer", "y": "token"},
).show()

# %%
