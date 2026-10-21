"""Compare a complete, matched workload set against every supplied stock candidate.

Run separately for cold reuse, prefix reuse and full recomputation. This report
does not certify that the supplied stock candidates exhaust eligible engines,
or replace numerical validation, sanitizer checks and the final scope audit.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from summarize_batch import distribution, identity, measurements


def contract(record, local):
    config = record["engine_config"] if local else record
    if local and config["verify_logits"]:
        raise ValueError("verification timings cannot enter a performance comparison")
    if not local and (config["weight_scale_dtype"] != "torch.float32"
                      or config["quantization"] not in ["compressed-tensors", "w8a8_int8"]):
        raise ValueError("stock candidate changes the declared weight/scale precision")
    prefix = config["prefix_cache_entries"] > 0 if local else config["prefix_cache"]
    if record["cache_mode"] not in ["reuse", "recompute"] or (prefix and record["cache_mode"] == "recompute"):
        raise ValueError("invalid cache mode/prefix combination")
    return dict(model_root=record["model_root"], device=record["device_name"],
                hardware=config["hardware"] if local else config["hardware_profile"],
                max_context=config["max_context"],
                max_batch=config["max_batch"] if local else config["max_num_seqs"],
                kv_cache_bytes=config["kv_cache_bytes"], prefix_cache=prefix,
                cache_mode=record["cache_mode"])


def validate_work(record, local):
    requests = record["requests"]
    recompute = record["cache_mode"] == "recompute"
    if local:
        expected = sum(len(r["prompt_ids"]) * r["max_tokens"] + r["max_tokens"] * (r["max_tokens"] - 1) // 2
                       if recompute else len(r["prompt_ids"]) + r["max_tokens"] - 1 for r in requests)
        reused = record["batch"]["reused_tokens"]
        if reused < 0 or (recompute and reused) or record["batch"]["processed_tokens"] + reused != expected:
            raise ValueError("local processed/reused token accounting mismatch")
    elif recompute:
        steps = record["recompute_steps"]
        if len(steps) != max(r["max_tokens"] for r in requests):
            raise ValueError("stock recomputation step count mismatch")
        for step, observed in enumerate(steps):
            active = [i for i, r in enumerate(requests) if r["max_tokens"] > step]
            if observed["slots"] != active or observed["prompt_lengths"] != [len(requests[i]["prompt_ids"]) + step for i in active]:
                raise ValueError("stock recomputation did not submit the complete active prefixes")


def collect(inputs, workloads, local, warmup, minimum):
    groups, expected_contract = {}, None
    for name, records in inputs:
        if not records:
            raise ValueError(f"{name}: empty measurements")
        runs = defaultdict(list)
        file_identity = None
        for record in records:
            if (record.get("backend") == "inferrous-paged") != local:
                raise ValueError(f"{name}: wrong engine kind")
            workload = record["workload"]
            if workload not in workloads or record["requests"] != workloads[workload]:
                raise ValueError(f"{name}: mismatched workload requests")
            matched = contract(record, local)
            if expected_contract is None:
                expected_contract = matched
            if matched != expected_contract:
                raise ValueError(f"{name}: mismatched model, hardware, capacity or cache contract")
            metadata = identity(record)
            del metadata["workload"], metadata["requests"]
            key = json.dumps(metadata, sort_keys=True)
            if file_identity is None:
                file_identity = key
            if file_identity != key:
                raise ValueError(f"{name}: engine identity changes inside an input")
            runs[workload].append(record["run"])
            validate_work(record, local)
            values = measurements(record)
            if any(not math.isfinite(value) or value < 0 for samples in values.values() for value in samples):
                raise ValueError(f"{name}: invalid timing or accounting value")
            group = groups.setdefault(key, dict(metadata=metadata, files=set(), workloads=defaultdict(list)))
            group["files"].add(name)
            if record["run"] >= warmup:
                group["workloads"][workload].append(values["api_seconds"][0])
        if set(runs) != set(workloads):
            raise ValueError(f"{name}: incomplete workload set")
        counts = {len(numbers) for numbers in runs.values()}
        if len(counts) != 1 or any(numbers != list(range(len(numbers))) for numbers in runs.values()):
            raise ValueError(f"{name}: missing, reordered or duplicate repetitions")
        # Both harnesses keep one resident engine and visit sorted workloads.
        # Per-workload repetition checks alone permit a changed prefix history,
        # including interleaved workloads or reordered concatenated fragments.
        repetitions = next(iter(counts))
        if [(record["workload"], record["run"]) for record in records] != [
                (workload, run) for workload in sorted(workloads) for run in range(repetitions)]:
            raise ValueError(f"{name}: workload order differs from the resident-suite protocol")
    for group in groups.values():
        for workload in workloads:
            if len(group["workloads"][workload]) < minimum:
                raise ValueError(f"{workload}: insufficient measured samples")
        group["files"] = sorted(group["files"])
        group["workloads"] = {name: distribution(values) for name, values in group["workloads"].items()}
        group["series_sha256"] = hashlib.sha256(json.dumps(group["metadata"], sort_keys=True).encode()).hexdigest()
    if not groups:
        raise ValueError("no candidate measurements")
    return expected_contract, list(groups.values())


def compare(local_inputs, stock_inputs, workloads, warmup=3, minimum=20):
    if not workloads or warmup < 0 or minimum < 1:
        raise ValueError("invalid workload or sampling requirement")
    if any(not requests or any(not r["prompt_ids"] or r["max_tokens"] < 1 or r["stop_ids"] for r in requests)
           for requests in workloads.values()):
        raise ValueError("performance workloads require nonempty prompts/outputs and no stops")
    matched, local = collect(local_inputs, workloads, True, warmup, minimum)
    stock_contract, stock = collect(stock_inputs, workloads, False, warmup, minimum)
    if matched != stock_contract:
        raise ValueError("local and stock model, hardware, capacity or cache contract differs")
    rows = []
    for candidate in local:
        for name in sorted(workloads):
            winner = min(stock, key=lambda item: item["workloads"][name]["p50"])
            ours, theirs = candidate["workloads"][name], winner["workloads"][name]
            rows.append(dict(workload=name, local_series=candidate["series_sha256"],
                             fastest_supplied_stock_series=winner["series_sha256"],
                             local_api_seconds=ours, stock_api_seconds=theirs,
                             stock_over_local=theirs["p50"] / ours["p50"],
                             local_median_lower=ours["p50"] < theirs["p50"]))
    return dict(contract=matched, warmups_per_file=warmup, minimum_measured_samples=minimum,
                workloads=len(workloads), local_candidates=local, stock_candidates=stock,
                comparisons=rows, all_local_medians_lower=all(row["local_median_lower"] for row in rows),
                scope="Fastest supplied eligible candidate per workload; not proof of exhaustive stock screening or goal completion.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", type=Path, nargs="+", required=True)
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--workloads", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    paths = args.local + args.stock
    if len({path.resolve() for path in paths}) != len(paths):
        ap.error("measurement inputs must be distinct")
    raw = {str(path): path.read_bytes() for path in paths}
    if len({hashlib.sha256(data).digest() for data in raw.values()}) != len(raw):
        ap.error("duplicate measurement contents cannot count as independent samples")
    def load(paths):
        return [(str(path), [json.loads(line) for line in raw[str(path)].splitlines()]) for path in paths]
    workload_bytes = args.workloads.read_bytes()
    report = compare(load(args.local), load(args.stock), json.loads(workload_bytes), args.warmup, args.min_samples)
    report["input_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    report["workloads_sha256"] = hashlib.sha256(workload_bytes).hexdigest()
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
