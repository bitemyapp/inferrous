"""Run warm resident-model requests serially. Preserve every raw measurement."""
import argparse
import json
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser(__doc__)
ap.add_argument("--binary", type=Path, required=True)
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--workloads", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--repeat", type=int, default=23)
args = ap.parse_args()
if args.repeat < 1:
    ap.error("--repeat must be positive")
requests = json.loads((args.workloads / "requests.json").read_text())
if not requests:
    ap.error("workloads must contain at least one request")
with args.output.open("x") as output:
    for name, request in requests.items():
        with subprocess.Popen([str(args.binary.resolve()), str(args.model.resolve()),
                               "--request", str(args.workloads / f"{name}.json"),
                               "--repeat", str(args.repeat)], stdout=subprocess.PIPE, text=True) as process:
            count = 0
            try:
                for line in process.stdout:
                    result = json.loads(line)
                    if result["request"] != request or result["trace"]:
                        raise ValueError(f"{name}: unexpected request or traced result")
                    if result["run"] != count:
                        raise ValueError(f"{name}: missing, duplicate, or out-of-order repetition")
                    generation = result["generation"]
                    if any(len(generation[key]) != request["max_tokens"] for key in
                           ["token_ids", "logit_sha256", "step_seconds"]):
                        raise ValueError(f"{name}: incomplete generation")
                    result["workload"] = name
                    output.write(json.dumps(result)+"\n"); output.flush()
                    print(name, result["run"], generation["request_seconds"], flush=True)
                    count += 1
            except BaseException:
                process.kill()
                raise
            if process.wait() != 0:
                raise RuntimeError(f"{name}: benchmark process failed with status {process.returncode}")
            if count != args.repeat:
                raise RuntimeError(f"{name}: expected {args.repeat} repetitions, got {count}")
