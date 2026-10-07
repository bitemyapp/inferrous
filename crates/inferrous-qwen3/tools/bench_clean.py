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
requests = json.loads((args.workloads / "requests.json").read_text())
with args.output.open("x") as output:
    for name, request in requests.items():
        process = subprocess.Popen([str(args.binary.resolve()), str(args.model.resolve()),
                                    "--request", str(args.workloads / f"{name}.json"),
                                    "--repeat", str(args.repeat)], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            result = json.loads(line)
            assert result["request"] == request and not result["trace"]
            result["workload"] = name
            assert len(result["generation"]["token_ids"]) == request["max_tokens"]
            output.write(json.dumps(result)+"\n"); output.flush()
            print(name, result["run"], result["generation"]["request_seconds"], flush=True)
        assert process.wait() == 0
