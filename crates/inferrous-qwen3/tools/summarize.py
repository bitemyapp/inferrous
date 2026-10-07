"""Summarize comparable API latency, first-token time and steady decode separately."""
import argparse
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(__doc__)
ap.add_argument("files", type=Path, nargs="+")
args = ap.parse_args()
report = []
for path in args.files:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for workload in sorted({r["workload"] for r in rows}):
        values = [r for r in rows if r["workload"] == workload and r["run"] >= 3]
        assert len(values) >= 20, (path, workload, "need twenty measured requests")
        generated_count = values[0]["request"]["max_tokens"]
        assert all(r["request"] == values[0]["request"] for r in values)
        request, decode, first = [], [], []
        for row in values:
            if "generation" in row:
                g = row["generation"]
                request.append(row["generate_seconds"])
                decode.append((len(g["token_ids"])-1)/sum(g["step_seconds"][1:]))
                first.append(g["step_seconds"][0])
            else:
                request.append(row["request_seconds"])
                m = row["backend_metrics"]
                # v0.30 V2 uses *_ts; V1 uses *_time. Both are per-process monotonic.
                a = m.get("first_token_ts", m.get("first_token_time"))
                b = m.get("last_token_ts", m.get("last_token_time"))
                if a is not None and b is not None:
                    decode.append((len(row["token_ids"])-1)/(b-a))
                if "first_token_latency" in m:
                    first.append(m["first_token_latency"])
                elif a is not None and "arrival_time" in m:
                    first.append(a-m["arrival_time"])
        item = dict(variant=path.stem, workload=workload, n=len(values),
                    api_ms={str(p):float(np.percentile(request,p)*1000) for p in [50,95,99]},
                    api_output_tokens_s=float(generated_count/np.median(request)))
        if decode: item["decode_tokens_s"] = float(np.median(decode))
        if first: item["first_token_ms"] = float(np.median(first)*1000)
        report.append(item)
print(json.dumps(report, indent=2))
