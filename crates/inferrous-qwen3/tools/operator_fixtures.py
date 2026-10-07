"""Generate small independent PyTorch operator fixtures on the benchmark workstation.

Refuses to overwrite fixtures: generate a new directory and review differences.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser(__doc__)
ap.add_argument("output", type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.manual_seed(824)
torch.set_num_threads(8)
cases = []

def bf(x):
    return x.bfloat16().float()

def case(op, tensors, **params):
    name = f"{len(cases):02d}-{op}"
    files = {}
    for key, value in tensors.items():
        if value.dtype == torch.bfloat16:
            value = value.view(torch.uint16)
        a = value.contiguous().numpy()
        raw = a.tobytes()
        filename = f"{name}-{key}.bin"
        (args.output / filename).write_bytes(raw)
        files[key] = dict(file=filename, shape=list(a.shape), sha256=hashlib.sha256(raw).hexdigest())
    cases.append(dict(name=name, op=op, tensors=files, **params))

for width in [128, 4096, 12288]:
    x = bf(torch.randn(3, width))
    x[0] = 0
    gain = bf(torch.randn(width))
    out = bf(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * gain)
    case("norm", dict(x=x, gain=gain, out=out), width=width)
    x[1, :8] = torch.tensor([127., -127., .5, -.5, 1.5, -1.5, 2.5, -2.5])
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    case("quantize", dict(x=x, q=q, scale=scale), width=width)

for width, rows in [(4096, 1), (4096, 17), (12288, 3), (4096, 65), (12288, 129)]:
    q = torch.randint(-127, 128, (rows, width), dtype=torch.int8)
    w = torch.randint(-127, 128, (33, width), dtype=torch.int8)
    # Include worst allowed accumulator magnitude, not just random cancellation.
    q[0] = 127; w[0] = 127; w[1] = -127
    asc = torch.rand(rows, 1) / 64
    ws = torch.rand(33) / 64
    out = bf((q.int() @ w.int().T).float() * asc * ws)
    case("linear", dict(q=q, asc=asc, w=w, ws=ws, out=out), width=width)

x = bf(torch.randn(3, 32, 128))
position = 4093
freq = 1 / (1e6 ** (torch.arange(0, 128, 2).float() / 128))
angles = torch.arange(4096).float()[:, None] * freq
table = torch.cat([angles.cos(), angles.sin()], -1)
a, b = x.chunk(2, -1)
c = angles[position:].cos()[:, None]; s = angles[position:].sin()[:, None]
case("rope", dict(x=x, table=table, out=bf(torch.cat([a*c-b*s, b*c+a*s], -1))), position=position)

for position, nr in [(0, 3), (129, 1), (4095, 1)]:
    q = bf(torch.randn(nr, 32, 128)); k = torch.randn(position+nr, 8, 128).bfloat16()
    v = torch.randn(position+nr, 8, 128).bfloat16()
    score = (q.transpose(0, 1) @ k.float().repeat_interleave(4, 1).permute(1, 2, 0)) * 128**-.5
    mask = torch.arange(len(k))[None, :] <= (position + torch.arange(nr))[:, None]
    p = score.masked_fill(~mask[None], float('-inf')).softmax(-1)
    out = bf((p @ v.float().repeat_interleave(4, 1).transpose(0, 1)).transpose(0, 1))
    case("attention", dict(q=q, k=k, v=v, out=out), position=position)

gate = bf(torch.linspace(-80, 80, 12288)); up = bf(torch.randn(12288))
case("swiglu", dict(gate=gate, up=up, out=bf(F.silu(gate)*up)))
case("residual", dict(a=gate, b=up, out=bf(gate+up)))
x = bf(torch.randn(3, 4096)); w = torch.randn(33, 4096).bfloat16()
case("head", dict(x=x, w=w, out=bf(x[-1:] @ w.float().T)))
ids = torch.tensor([2, 0, 1], dtype=torch.uint32)
w = torch.randn(3, 4096).bfloat16()
case("embedding", dict(ids=ids, w=w, out=w[ids.long()].float()))
(args.output / "manifest.json").write_text(json.dumps(dict(generator="operator_fixtures.py", torch=torch.__version__, seed=824, cases=cases), indent=2)+"\n")
