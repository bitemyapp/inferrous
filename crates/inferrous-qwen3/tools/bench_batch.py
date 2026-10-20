"""Run a named matrix of paged-engine batches and retain raw measurements.

Workloads map names to request arrays. Each workload gets a resident engine;
model loading is recorded separately from generation. Use prefix_cache_entries=0
for cold-prefix measurements, and a nonzero bound for explicit warm-prefix runs.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workloads", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=23)
    ap.add_argument("--cache-mode", choices=["reuse", "recompute"], default="reuse")
    args = ap.parse_args()
    if not 1 <= args.repeat <= 100:
        ap.error("repeat must be 1..100")
    workloads = json.loads(args.workloads.read_text())
    if not workloads or any(not isinstance(batch, list) or not batch for batch in workloads.values()):
        ap.error("workloads must map names to nonempty request arrays")
    config = json.loads(args.config.read_text())
    if any(r["stop_ids"] or r["max_tokens"] < 1 for batch in workloads.values() for r in batch):
        ap.error("benchmarks require positive output counts and empty stop_ids")
    with args.output.open("x") as output, tempfile.TemporaryDirectory(prefix="inferrous-batch-") as directory:
        for name, requests in workloads.items():
            request_path = Path(directory) / "requests.json"
            request_path.write_text(json.dumps(requests))
            command = [str(args.binary.resolve()), str(args.model.resolve()), "--batch", str(request_path),
                       "--engine-config", str(args.config.resolve()), "--cache-mode", args.cache_mode,
                       "--repeat", str(args.repeat)]
            with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
                count = 0
                oracle = None
                try:
                    for line in process.stdout:
                        record = json.loads(line)
                        if record["run"] != count or record["requests"] != requests or record["cache_mode"] != args.cache_mode:
                            raise ValueError(f"{name}: unexpected request, mode or repetition")
                        batch = record["batch"]
                        if len(batch["generations"]) != len(requests):
                            raise ValueError(f"{name}: incomplete batch")
                        fingerprint = []
                        for request, result in zip(requests, batch["generations"]):
                            keys = ["token_ids", "step_seconds"] + (["logit_sha256"] if config["verify_logits"] else [])
                            if any(len(result[key]) != request["max_tokens"] for key in keys):
                                raise ValueError(f"{name}: incomplete generation")
                            fingerprint.append((result["token_ids"], result["logit_sha256"]))
                        if oracle is not None and fingerprint != oracle:
                            raise ValueError(f"{name}: nondeterministic output")
                        oracle = fingerprint
                        if args.cache_mode == "recompute":
                            expected = sum(r["max_tokens"] * len(r["prompt_ids"]) + r["max_tokens"] * (r["max_tokens"] - 1) // 2
                                           for r in requests)
                            if batch["reused_tokens"] != 0 or batch["processed_tokens"] != expected:
                                raise ValueError(f"{name}: invalid recomputation work count")
                        record["workload"] = name
                        output.write(json.dumps(record) + "\n")
                        output.flush()
                        print(name, count, round(record["generate_seconds"], 6), flush=True)
                        count += 1
                except BaseException:
                    process.kill()
                    raise
                if process.wait() != 0 or count != args.repeat:
                    raise RuntimeError(f"{name}: process failed or returned only {count} repetitions")


if __name__ == "__main__":
    main()
