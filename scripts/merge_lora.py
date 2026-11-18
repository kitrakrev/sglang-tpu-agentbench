"""Merge a PEFT LoRA adapter into base safetensors shards, CPU only (no transformers model build).
    python scripts/merge_lora.py BASE_DIR ADAPTER_DIR OUT_DIR
W' = W + (alpha/r) * B @ A for every lora_A/lora_B pair. Copies non-weight files from BASE_DIR
(and tokenizer/chat template from ADAPTER_DIR). Base key matched by suffix after 'model.'.
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

base, adp, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(adp, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
lora = load_file(os.path.join(adp, "adapter_model.safetensors"))
pairs = {}
for k, v in lora.items():
    if ".lora_A." in k or ".lora_B." in k:
        stem = k.replace("base_model.model.", "").split(".lora_")[0]
        pairs.setdefault(stem, {})["A" if ".lora_A." in k else "B"] = v
print(f"{len(pairs)} LoRA pairs, scale={scale}")

def match(key, stems):
    w = key[:-len(".weight")] if key.endswith(".weight") else None
    if w is None:
        return None
    for s in (w, w.replace("model.language_model.", "model."), w.replace("model.", "model.language_model.", 1)):
        if s in stems:
            return s
    return None

idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
used = set()
for shard in sorted(set(idx["weight_map"].values())):
    t = load_file(os.path.join(base, shard))
    n = 0
    for k in list(t):
        s = match(k, pairs)
        if s:
            A, B = pairs[s]["A"].float(), pairs[s]["B"].float()
            t[k] = (t[k].float() + scale * (B @ A)).to(t[k].dtype); used.add(s); n += 1
    save_file(t, os.path.join(out, shard), metadata={"format": "pt"}); print(shard, "merged", n, flush=True)
    del t
missing = set(pairs) - used
print("unmatched LoRA pairs:", len(missing), sorted(missing)[:5])
assert not missing, "adapter keys did not match base weights"
for f in os.listdir(base):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(base, f)):
        shutil.copy(os.path.join(base, f), out)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if os.path.exists(os.path.join(adp, f)):
        shutil.copy(os.path.join(adp, f), out)
print("done", out)
