"""Create fixed token workloads using the pinned bundle tokenizer; run on the benchmark workstation."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer

ap = argparse.ArgumentParser(__doc__)
ap.add_argument("bundle", type=Path)
ap.add_argument("output", type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
tokenizer = Tokenizer.from_file(str(args.bundle / "tokenizer.json"))
text = "Explain how a computer executes a program. Describe the role of memory, arithmetic, instructions, and input and output. "
long = tokenizer.encode(text * 100, add_special_tokens=False).ids
requests = {}
for name, ids in [("p5", tokenizer.encode("The capital of France is", add_special_tokens=False).ids),
                  ("p129", long[:129]), ("p512", long[:512])]:
    request = dict(prompt_ids=ids, max_tokens=32, stop_ids=[])
    requests[name] = request
    (args.output / f"{name}.json").write_text(json.dumps(request)+"\n")
(args.output / "requests.json").write_text(json.dumps(requests, indent=2)+"\n")
