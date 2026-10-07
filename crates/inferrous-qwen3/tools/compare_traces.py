"""Check every layer and vocabulary logit against independent Torch traces.

Gates are fixed before optimization: relative L2 <= 5%, cosine >= .999,
finite values, identical shapes and greedy tokens. These are regression/smoke
gates for accumulated BF16/INT8 rounding, not a model quality evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(__doc__)
ap.add_argument("reference", type=Path)
ap.add_argument("actual", type=Path)
ap.add_argument("generation", type=Path)
args = ap.parse_args()
ref = json.loads((args.reference / "reference.json").read_text())
actual = json.loads(args.generation.read_text().splitlines()[-1])
assert ref["model_root"] == actual["model_root"]
assert json.loads((args.reference / "request.json").read_text()) == actual["request"]
assert ref["token_ids"] == actual["generation"]["token_ids"], "greedy token mismatch"
reports = []
for name, meta in ref["traces"].items():
    raw = (args.reference / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
    expected = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    got = np.fromfile(args.actual / name, dtype="<f4").astype(np.float64)
    assert got.shape == expected.shape and np.isfinite(got).all(), name
    error = got - expected
    rel = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
    cosine = float(np.dot(got, expected) / max(np.linalg.norm(got) * np.linalg.norm(expected), 1e-12))
    reports.append(dict(name=name, relative_l2=rel, cosine=cosine,
                        max_absolute=float(np.max(np.abs(error))), passed=rel <= .05 and cosine >= .999))
print(json.dumps(dict(model_root=ref["model_root"], tokens_match=True,
                      passed=all(r["passed"] for r in reports), tensors=reports), indent=2))
assert all(r["passed"] for r in reports), "numerical gate failed"
