"""Summarize complete local/stock batch measurements without mixing configurations.

API latency and total output throughput share a host-call timing boundary.
First-token and decode timings retain each engine's native measurement boundary.
Recomputation uses the stock harness's one-token call completion timestamps.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def checked_time(value, label, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"invalid timing: {label}")
    return value


def distribution(values):
    values = sorted(values)
    if not values:
        return None

    def percentile(q):
        position = (len(values) - 1) * q
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return dict(n=len(values), p50=percentile(0.5), p95=percentile(0.95), p99=percentile(0.99))


def identity(record):
    if record.get("backend") == "inferrous-paged":
        return dict(engine="inferrous-paged", model_root=record["model_root"], device=record["device_name"],
                    binary_sha256=record["binary_sha256"], config=record["engine_config"],
                    cache_mode=record["cache_mode"], workload=record["workload"], requests=record["requests"])
    fields = ["engine", "model_root", "device_name", "versions", "environment", "eager", "linear_backend",
              "attention_backend", "graph_size_one", "graph_capture_sizes", "graphs_without_compile", "disable_cutedsl_warmup",
              "quantization", "weight_scale_dtype", "max_context", "max_num_seqs", "hardware_profile",
              "memory_fraction", "kv_cache_bytes", "prefix_cache", "cache_mode", "workload", "requests"]
    return {field: record.get(field) for field in fields}


def reused_tokens(record, local):
    if local:
        return [record["batch"]["reused_tokens"]]
    if record.get("engine") != "vllm":
        return []
    counts = []
    recompute = record["cache_mode"] == "recompute"
    for request, generation in zip(record["requests"], record["generations"]):
        metrics = generation["backend_metrics"]
        submissions = metrics if recompute else [metrics]
        if not isinstance(submissions, list) or len(submissions) != (request["max_tokens"] if recompute else 1):
            raise ValueError("incomplete stock cache metrics")
        for step, metric in enumerate(submissions):
            cached = metric.get("num_cached_tokens")
            if cached is not None:
                limit = len(request["prompt_ids"]) + (step if recompute else 0)
                if type(cached) is not int or not 0 <= cached <= limit:
                    raise ValueError("invalid stock cached-token count")
                if cached and (recompute or record.get("prefix_cache") is False):
                    raise ValueError("stock reports prefix reuse with prefix caching disabled")
            counts.append(cached)
    # Missing native counters are unknown, never implicit zero-hit evidence.
    return [sum(counts)] if counts and all(count is not None for count in counts) else []


def prefix_metrics(record, local):
    counters = ["lookups", "hits", "misses", "matched_tokens", "capacity_evictions",
                "entry_limit_evictions", "replacements", "cleared_entries"]
    values = {"prefix_" + name: [] for name in counters + ["resident_entries", "resident_tokens"]}
    values["prefix_hit_fraction"] = []
    if not local or "prefix_cache_stats" not in record:
        return values
    snapshots = record["prefix_cache_stats"]
    for snapshot in [snapshots["before"], snapshots["after"]]:
        counts = snapshot["counters"]
        numbers = [counts[name] for name in counters] + [snapshot["resident_entries"], snapshot["resident_tokens"]]
        if any(type(value) is not int or value < 0 for value in numbers):
            raise ValueError("invalid prefix-cache counter")
        if counts["lookups"] != counts["hits"] + counts["misses"]:
            raise ValueError("prefix-cache lookup accounting mismatch")
    delta = {name: snapshots["after"]["counters"][name] - snapshots["before"]["counters"][name]
             for name in counters}
    if any(value < 0 for value in delta.values()):
        raise ValueError("prefix-cache counters move backwards")
    if bool(delta["hits"]) != bool(delta["matched_tokens"]):
        raise ValueError("prefix-cache matched-token accounting mismatch")
    if delta["lookups"] and (record["cache_mode"] == "recompute" or record["engine_config"]["prefix_cache_entries"] == 0):
        raise ValueError("prefix-cache lookups contradict disabled policy")
    values.update({"prefix_" + name: [value] for name, value in delta.items()})
    for name in ["resident_entries", "resident_tokens"]:
        values["prefix_" + name] = [snapshots["after"][name]]
    if delta["lookups"]:
        values["prefix_hit_fraction"] = [delta["hits"] / delta["lookups"]]
    return values


def stream_measurements(record):
    """Native timings by request slot; preserve ragged/admission differences."""
    local = record.get("backend") == "inferrous-paged"
    requests = record["requests"]
    generations = record["batch"]["generations"] if local else record["generations"]
    if not requests or len(generations) != len(requests) or any(r["max_tokens"] < 1 or len(g["token_ids"]) != r["max_tokens"] for g, r in zip(generations, requests)):
        raise ValueError("incomplete batch or generation")
    if not local and record["cache_mode"] == "recompute":
        elapsed = [checked_time(step["elapsed_seconds"], "recomputation elapsed seconds")
                   for step in record["recompute_steps"]]
        if any(after < before for before, after in zip(elapsed, elapsed[1:])):
            raise ValueError("invalid timing: recomputation elapsed seconds move backwards")
    streams = []
    for slot, generation in enumerate(generations):
        count = len(generation["token_ids"])
        if local:
            steps = [checked_time(step, "local step seconds") for step in generation["step_seconds"]]
            if len(steps) != count:
                raise ValueError("incomplete local step timings")
            first = steps[0]
            decode = sum(steps[1:])
            completion = checked_time(generation["request_seconds"], "local completion seconds")
        elif record["cache_mode"] == "recompute":
            steps = [step["elapsed_seconds"] for step in record["recompute_steps"] if slot in step["slots"]]
            if len(steps) != count:
                raise ValueError("incomplete stock recomputation steps")
            first, completion = steps[0], steps[-1]
            decode = completion - first
        else:
            metrics = generation["backend_metrics"]
            # These are vLLM v1 timestamps, not wall-clock arrival_time. Do not
            # subtract arrival_time from the monotonic *_ts fields.
            first = metrics.get("first_token_latency")
            begin, end = metrics.get("first_token_ts"), metrics.get("last_token_ts")
            for label, value in [("first-token latency", first), ("first-token timestamp", begin),
                                 ("last-token timestamp", end)]:
                if value is not None:
                    checked_time(value, label)
            if begin is not None and end is not None and end < begin:
                raise ValueError("invalid timing: last-token timestamp precedes first token")
            decode = end - begin if begin is not None and end is not None else None
            completion = first + decode if first is not None and decode is not None else None
        if decode is not None:
            checked_time(decode, "decode interval")
        values = dict(native_first_token_seconds=[] if first is None else [first],
                      native_completion_seconds=[] if completion is None else [completion],
                      native_per_stream_post_first_seconds_per_token=[],
                      native_per_stream_post_first_tokens_per_second=[])
        if count > 1 and decode is not None:
            values["native_per_stream_post_first_seconds_per_token"] = [decode / (count - 1)]
        if count > 1 and decode is not None and decode > 0:
            values["native_per_stream_post_first_tokens_per_second"] = [(count - 1) / decode]
        if any(not math.isfinite(value) or value < 0 for samples in values.values() for value in samples):
            raise ValueError("invalid timing value")
        streams.append(values)
    return streams


def measurements(record):
    local = record.get("backend") == "inferrous-paged"
    generations = record["batch"]["generations"] if local else record["generations"]
    seconds = checked_time(record["generate_seconds"] if local else record["request_seconds"],
                           "API seconds", positive=True)
    streams = stream_measurements(record)
    values = dict(api_seconds=[seconds], aggregate_output_tokens_per_second=[sum(len(g["token_ids"]) for g in generations) / seconds],
                reused_tokens=reused_tokens(record, local),
                processed_tokens=[record["batch"]["processed_tokens"]] if local else [])
    values.update({metric: [value for stream in streams for value in stream[metric]]
                   for metric in streams[0]})
    values.update(prefix_metrics(record, local))
    if any(not math.isfinite(value) or value < 0 for samples in values.values() for value in samples):
        raise ValueError("invalid timing or accounting value")
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.warmup < 0 or args.min_samples < 1:
        ap.error("warmup must be nonnegative and min-samples positive")
    if len({path.resolve() for path in args.inputs}) != len(args.inputs):
        ap.error("input files must be distinct")
    groups = {}
    for path in args.inputs:
        runs = defaultdict(list)
        for line in path.read_text().splitlines():
            record = json.loads(line)
            metadata = identity(record)
            key = json.dumps(metadata, sort_keys=True)
            runs[key].append(record["run"])
            values = measurements(record)
            if record["run"] < args.warmup:
                continue
            group = groups.setdefault(key, dict(metadata=metadata, files=set(), metrics=defaultdict(list),
                                               streams=[defaultdict(list) for _ in record["requests"]]))
            group["files"].add(str(path))
            for metric, samples in values.items():
                group["metrics"][metric].extend(samples)
            for slot, stream in enumerate(stream_measurements(record)):
                for metric, samples in stream.items():
                    group["streams"][slot][metric].extend(samples)
        for run_numbers in runs.values():
            if run_numbers != list(range(len(run_numbers))):
                raise ValueError(f"{path}: missing, reordered or duplicate repetitions")
    summaries = []
    for key, group in sorted(groups.items()):
        n = len(group["metrics"]["api_seconds"])
        if n < args.min_samples:
            raise ValueError(f"{group['metadata']['workload']}: {n} measurements, require {args.min_samples}")
        metadata = group["metadata"]
        requests = metadata.pop("requests")
        summaries.append(dict(series_sha256=hashlib.sha256(key.encode()).hexdigest(), **metadata,
                              prompt_lengths=[len(r["prompt_ids"]) for r in requests],
                              output_lengths=[r["max_tokens"] for r in requests],
                              files=sorted(group["files"]), warmups_per_workload_per_file=args.warmup,
                              streams=[dict(slot=slot, prompt_length=len(requests[slot]["prompt_ids"]),
                                            output_length=requests[slot]["max_tokens"],
                                            metrics={metric: distribution(values) for metric, values in stream.items()})
                                       for slot, stream in enumerate(group["streams"])],
                              metrics={metric: distribution(values) for metric, values in group["metrics"].items()}))
    if not summaries:
        raise ValueError("no measured records")
    text = json.dumps(summaries, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
