"""Export one ordinary Qwen3 INT8 checkpoint in raw and compressed-tensors formats.

Run on the benchmark workstation with the pinned reference environment. No calibration,
Hadamard rotations, smoothing, weight folding, or quantized output head.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
PROFILE = "qwen3-8b-int8-channel-token-bf16-softmax-v1"
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
assert args.source.name == REVISION
args.output.mkdir(exist_ok=False, parents=True)
hf = args.output / "hf"
hf.mkdir()
torch.set_num_threads(8)
index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
config = json.loads((args.source / "config.json").read_text())
tensors, weight_map = {}, {}
total_bytes = 0


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(8 << 20):
            h.update(data)
    return h.hexdigest()


def weight(key):
    with safe_open(args.source / index[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def emit(name, value, dtype):
    a = value.view(torch.int16).numpy().view(np.uint16) if dtype == "<u2" else value.numpy()
    a = np.ascontiguousarray(a, dtype=dtype)
    path = args.output / f"{name}.bin"
    a.tofile(path)
    tensors[name] = dict(shape=list(a.shape), dtype=dtype, sha256=sha(path))


def shard(name, values):
    global total_bytes
    save_file(values, hf / name, metadata={"format": "pt"})
    for key, tensor in values.items():
        weight_map[key] = name
        total_bytes += tensor.numel() * tensor.element_size()


for short, key in [("embedding", "model.embed_tokens.weight"), ("head", "lm_head.weight")]:
    w = weight(key)
    emit(short, w, "<u2")
    shard(f"{short}.safetensors", {key: w})
emit("final_gain", weight("model.norm.weight").float(), "<f4")
shard("final.safetensors", {"model.norm.weight": weight("model.norm.weight")})
for l in range(36):
    values = {}
    for short, name in [("norm1", "input_layernorm"), ("norm2", "post_attention_layernorm"),
                        ("qnorm", "self_attn.q_norm"), ("knorm", "self_attn.k_norm")]:
        key = f"model.layers.{l}.{name}.weight"
        w = weight(key)
        emit(f"layer.{l}.{short}", w.float(), "<f4")
        values[key] = w
    for short in ["q", "k", "v", "o", "gate", "up", "down"]:
        prefix = f"model.layers.{l}." + ("self_attn." if short in ["q", "k", "v", "o"] else "mlp.") + short + "_proj"
        w = weight(prefix + ".weight").float()
        scale = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        emit(f"layer.{l}.{short}.w", q, "|i1")
        emit(f"layer.{l}.{short}.scale", scale.flatten(), "<f4")
        values[prefix + ".weight"] = q
        values[prefix + ".weight_scale"] = scale
    shard(f"layer-{l:02}.safetensors", values)
    print("prepared", l, flush=True)

# Ordinary Qwen rotary angles, FP32 trigonometry. No Q14 coefficient rounding.
freq = 1.0 / (config["rope_theta"] ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
angles = torch.arange(4096, dtype=torch.float32)[:, None] * freq[None, :]
emit("rope", torch.cat([angles.cos(), angles.sin()], dim=1), "<f4")
for name in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
    if (args.source / name).exists():
        shutil.copyfile(args.source / name, hf / name)
shutil.copyfile(args.source / "tokenizer.json", args.output / "tokenizer.json")
qconfig = {
    "quant_method": "compressed-tensors", "format": "int-quantized",
    "quantization_status": "compressed", "ignore": ["lm_head"],
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True},
    }},
}
hf_config = dict(config, quantization_config=qconfig)
(hf / "config.json").write_text(json.dumps(hf_config, indent=2) + "\n")
(hf / "model.safetensors.index.json").write_text(json.dumps(dict(metadata=dict(total_size=total_bytes), weight_map=weight_map), indent=2) + "\n")
manifest = dict(format_version=1, profile=PROFILE, model_revision=REVISION,
                max_context=4096, config=config, tensors=tensors,
                tokenizer_sha256=sha(args.output / "tokenizer.json"),
                source_shards={s: sha(args.source / s) for s in sorted(set(index.values()))},
                hf_shards={s: sha(hf / s) for s in sorted(set(weight_map.values()))})
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("complete", sha(args.output / "manifest.json"), flush=True)
