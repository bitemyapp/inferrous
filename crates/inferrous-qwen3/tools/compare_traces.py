"""Check every layer and vocabulary logit against independent Torch traces.

Gates are fixed before optimization: relative L2 <= 5%, cosine >= .999,
finite values, identical shapes and greedy tokens. These are regression/smoke
gates for accumulated BF16/INT8 rounding, not a model quality evaluation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trace_shapes(prompt_length, steps):
    """The complete per-step schema emitted by reference.py for Qwen3-8B."""
    expected = {}
    for step in range(steps):
        rows = prompt_length if step == 0 else 1
        shapes = {f"layer.{layer}": [rows, 4096] for layer in range(36)}
        shapes.update(norm=[rows, 4096], logits=[151936])
        for name in ["norm", "q", "a", "o", "norm2", "down"]:
            shapes[f"debug.{name}"] = [rows, 4096]
        for name in ["k", "v"]:
            shapes[f"debug.{name}"] = [rows, 1024]
        for name in ["gate", "up", "act"]:
            shapes[f"debug.{name}"] = [rows, 12288]
        shapes["debug.qpost"] = [rows, 32, 128]
        shapes["debug.kpost"] = [rows, 8, 128]
        expected.update({f"step.{step}.{name}.bin": shape for name, shape in shapes.items()})
    return expected


def compare(reference, actual_directory, generation):
    ref = json.loads((reference / "reference.json").read_text())
    actual = json.loads(generation.read_text().splitlines()[-1])
    request = json.loads((reference / "request.json").read_text())
    require(ref["model_root"] == actual["model_root"], "model root mismatch")
    require(request == actual["request"], "request mismatch")
    tokens = ref["token_ids"]
    require(tokens == actual["generation"]["token_ids"], "greedy token mismatch")
    for label, ids in [("prompt", request["prompt_ids"]), ("stop", request["stop_ids"]), ("generated", tokens)]:
        require(isinstance(ids, list) and all(type(token) is int and 0 <= token < 151936 for token in ids),
                f"invalid {label} token IDs")
    limit = request["max_tokens"]
    require(type(limit) is int and limit > 0 and request["prompt_ids"], "invalid trace request")
    require(0 < len(tokens) <= limit, "empty or excessive generation")
    require(not any(token in request["stop_ids"] for token in tokens[:-1]), "generation continues after stop")
    require(len(tokens) == limit or tokens[-1] in request["stop_ids"], "incomplete generation without stop")
    shapes = trace_shapes(len(request["prompt_ids"]), len(tokens))
    require(isinstance(ref["traces"], dict) and set(ref["traces"]) == set(shapes), "incomplete or unexpected trace metadata")
    for label, directory in [("reference", reference), ("actual", actual_directory)]:
        require({path.name for path in directory.glob("*.bin")} == set(shapes), f"incomplete or unexpected {label} trace files")
    reports = []
    # Keep the writer's ordering so complete existing reports retain their
    # numerical values and representation; only structural rejection changes.
    for name, meta in ref["traces"].items():
        require(meta["shape"] == shapes[name] and all(type(dimension) is int for dimension in meta["shape"]),
                f"invalid declared shape: {name}")
        raw = (reference / name).read_bytes()
        actual_raw = (actual_directory / name).read_bytes()
        expected_bytes = math.prod(shapes[name]) * 4
        require(len(raw) == expected_bytes and len(actual_raw) == expected_bytes, f"invalid tensor byte length: {name}")
        require(hashlib.sha256(raw).hexdigest() == meta["sha256"], f"reference hash mismatch: {name}")
        expected = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        got = np.frombuffer(actual_raw, dtype="<f4").astype(np.float64)
        require(np.isfinite(expected).all() and np.isfinite(got).all(), f"nonfinite tensor: {name}")
        error = got - expected
        rel = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
        cosine = float(np.dot(got, expected) / max(np.linalg.norm(got) * np.linalg.norm(expected), 1e-12))
        reports.append(dict(name=name, relative_l2=rel, cosine=cosine,
                            max_absolute=float(np.max(np.abs(error))), passed=rel <= .05 and cosine >= .999))
    return dict(model_root=ref["model_root"], tokens_match=True,
                passed=all(r["passed"] for r in reports), tensors=reports)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("reference", type=Path)
    ap.add_argument("actual", type=Path)
    ap.add_argument("generation", type=Path)
    args = ap.parse_args()
    report = compare(args.reference, args.actual, args.generation)
    print(json.dumps(report, indent=2))
    require(report["passed"], "numerical gate failed")


if __name__ == "__main__":
    main()
