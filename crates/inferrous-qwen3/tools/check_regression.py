"""Require exact per-token vocabulary logits and token IDs versus the baseline."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(__doc__)
ap.add_argument("baseline", type=Path)
ap.add_argument("candidate", type=Path)
args = ap.parse_args()
baseline = {}
for line in args.baseline.read_text().splitlines():
    row = json.loads(line)
    key = json.dumps(row["request"], sort_keys=True)
    value = (row["model_root"], row["generation"]["token_ids"], row["generation"]["logit_sha256"])
    if key in baseline:
        assert baseline[key] == value, "baseline itself is nondeterministic"
    baseline[key] = value
count = 0
seen = set()
for line in args.candidate.read_text().splitlines():
    row = json.loads(line)
    key = json.dumps(row["request"], sort_keys=True)
    assert key in baseline, "unknown workload"
    value = (row["model_root"], row["generation"]["token_ids"], row["generation"]["logit_sha256"])
    assert baseline[key] == value, (row.get("workload"), row["run"], "bitwise regression")
    seen.add(key); count += 1
assert seen == baseline.keys(), "missing workload"
print(json.dumps(dict(passed=True, requests=count, workloads=len(seen))))
