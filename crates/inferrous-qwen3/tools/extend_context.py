"""Extend a prepared bundle's native RoPE table, preserving existing weights and positions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--context", type=int, default=32768)
args = ap.parse_args()
manifest = json.loads((args.source / "manifest.json").read_text())
old_context = manifest["max_context"]
if not old_context < args.context <= 32768:
    ap.error("context must exceed the existing table and be at most 32768")
old_rope = (args.source / "rope.bin").read_bytes()
if hashlib.sha256(old_rope).hexdigest() != manifest["tensors"]["rope"]["sha256"]:
    ap.error("source RoPE hash mismatch")
torch.set_num_threads(8)
freq = 1.0 / (manifest["config"]["rope_theta"] ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
angles = torch.arange(args.context, dtype=torch.float32)[:, None] * freq[None, :]
rope = np.ascontiguousarray(torch.cat([angles.cos(), angles.sin()], dim=1).numpy(), dtype="<f4").tobytes()
# Preserve the old oracle even if CPU trigonometry differs between environments.
rope = old_rope + rope[len(old_rope):]
args.output.mkdir(parents=True, exist_ok=False)
for source in args.source.iterdir():
    if source.name in {"manifest.json", "rope.bin"}:
        continue
    if source.is_dir():
        shutil.copytree(source, args.output / source.name, copy_function=os.link)
    else:
        os.link(source, args.output / source.name)
(args.output / "rope.bin").write_bytes(rope)
manifest["max_context"] = args.context
manifest["tensors"]["rope"].update(shape=[args.context, 128], sha256=hashlib.sha256(rope).hexdigest())
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Extended native context:", args.output, "tokens:", args.context)
