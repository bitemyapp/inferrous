"""Require complete, exact paged-engine results across binaries and cache modes."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate(records, workloads, repeat):
    if repeat < 1 or not workloads or any(not batch for batch in workloads.values()):
        raise ValueError("require nonempty workloads and positive repetitions")
    expected_order = [(name, run) for name in sorted(workloads) for run in range(repeat)]
    if [(r.get("workload"), r.get("run")) for r in records] != expected_order:
        raise ValueError("missing, extra, reordered or duplicate workload/repetition")
    identity = None
    fingerprints = {}
    metadata = None
    for record in records:
        name = record["workload"]
        requests = workloads[name]
        if record.get("backend") != "inferrous-paged" or record["requests"] != requests:
            raise ValueError(f"{name}: wrong backend or requests")
        if (not record["engine_config"]["verify_logits"]
                or not digest(record["model_root"]) or not digest(record["binary_sha256"])):
            raise ValueError(f"{name}: missing full-logit verification or executable/model identity")
        current = (record["model_root"], record["profile"], record["device_name"],
                   record["binary_sha256"], record["cache_mode"],
                   json.dumps(record["engine_config"], sort_keys=True))
        if identity is not None and current != identity:
            raise ValueError("one input file mixes execution identities")
        identity = current
        batch = record["batch"]
        generations = batch["generations"]
        if len(generations) != len(requests):
            raise ValueError(f"{name}: incomplete batch")
        fingerprint = []
        for request, generation in zip(requests, generations):
            count = request["max_tokens"]
            if count < 1 or not request["prompt_ids"] or request["stop_ids"]:
                raise ValueError(f"{name}: require benchmark requests without stopping")
            if (len(generation["token_ids"]) != count
                    or len(generation["logit_sha256"]) != count
                    or not all(digest(h) for h in generation["logit_sha256"])
                    or len(generation["step_seconds"]) != count):
                raise ValueError(f"{name}: incomplete tokens, hashes or timings")
            fingerprint.append((generation["token_ids"], generation["logit_sha256"],
                                generation["finish_reason"]))
        processed, reused = batch["processed_tokens"], batch["reused_tokens"]
        if type(processed) is not int or type(reused) is not int or min(processed, reused) < 0:
            raise ValueError(f"{name}: invalid token accounting")
        if record["cache_mode"] == "recompute":
            expected = sum(r["max_tokens"] * len(r["prompt_ids"])
                           + r["max_tokens"] * (r["max_tokens"] - 1) // 2 for r in requests)
            valid_work = processed == expected and reused == 0
        elif record["cache_mode"] == "reuse":
            expected = sum(len(r["prompt_ids"]) + r["max_tokens"] - 1 for r in requests)
            valid_work = processed + reused == expected
        else:
            raise ValueError(f"{name}: unknown cache mode")
        if not valid_work:
            raise ValueError(f"{name}: wrong processed/reused token counts")
        if name in fingerprints and fingerprint != fingerprints[name]:
            raise ValueError(f"{name}: nondeterministic outputs")
        fingerprints[name] = fingerprint
        metadata = {key: record[key] for key in
                    ["model_root", "profile", "device_name", "binary_sha256", "cache_mode", "engine_config"]}
    return fingerprints, metadata


def compare(inputs, workloads, repeat):
    if len(inputs) < 2:
        raise ValueError("require at least two inputs")
    reference = None
    reference_identity = None
    evidence = []
    for label, records in inputs:
        fingerprint, metadata = validate(records, workloads, repeat)
        model_identity = tuple(metadata[key] for key in ["model_root", "profile", "device_name"])
        if reference is not None:
            if model_identity != reference_identity:
                raise ValueError(f"{label}: model, numerical profile or device differs")
            for name in sorted(workloads):
                if fingerprint[name] != reference[name]:
                    raise ValueError(f"{label}: {name}: token, full-logit hash or finish reason differs")
        else:
            reference, reference_identity = fingerprint, model_identity
        evidence.append(dict(input=label, records=len(records), **metadata))
    return dict(workloads=len(workloads), repetitions_per_workload=repeat,
                streams_per_repetition=sum(len(rs) for rs in workloads.values()),
                hashes_per_repetition=sum(r["max_tokens"] for rs in workloads.values() for r in rs),
                inputs=evidence, exact_match=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--workloads", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len({p.resolve() for p in args.inputs}) != len(args.inputs):
        ap.error("input paths must be distinct")
    inputs, hashes = [], {}
    for path in args.inputs:
        data = path.read_bytes()
        inputs.append((str(path), [json.loads(line) for line in data.splitlines()]))
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    result = compare(inputs, json.loads(args.workloads.read_text()), args.repeat)
    result["input_sha256"] = hashes
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(f"Exact match: {result['workloads']} workloads, {result['hashes_per_repetition']} hashes per repetition")


if __name__ == "__main__":
    main()
