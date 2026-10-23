"""Summarize a complete CUPTI kernel trace, separating graph replay from eager launches.

These are profiling diagnostics, including all traced repetitions. Instrumented
API timings are not benchmark results and are not compared with ordinary runs.
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    args = ap.parse_args()
    lines = args.trace.read_text().splitlines()
    if not lines or not re.fullmatch(r"# records=\d+ dropped=0 invalid=0", lines[-1]):
        ap.error("missing successful trace footer; refuse incomplete or dropped records")
    records = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    count = int(re.search(r"records=(\d+)", lines[-1])[1])
    if not records or len(records) != count:
        ap.error("trace record count mismatch")
    groups, totals = defaultdict(list), defaultdict(int)
    for record in records:
        duration = int(record["end_ns"]) - int(record["start_ns"])
        if int(record["start_ns"]) <= 0 or duration <= 0:
            ap.error("invalid kernel interval")
        phase = "graph" if int(record["graph_id"]) else "eager"
        key = (phase, record["name"])
        groups[key].append((duration, record))
        totals[phase] += duration
    results = []
    for (phase, name), samples in groups.items():
        durations = [sample[0] for sample in samples]
        total = sum(durations)
        results.append(dict(phase=phase, kernel=name, calls=len(samples), total_ms=total / 1e6,
                            phase_percent=100 * total / totals[phase], median_us=statistics.median(durations) / 1e3,
                            block_shapes=sorted({tuple(int(r[f"block_{axis}"]) for axis in "xyz") for _, r in samples}),
                            registers=sorted({int(r["registers"]) for _, r in samples}),
                            local_bytes_per_thread=sorted({int(r["local_per_thread"]) for _, r in samples})))
    results.sort(key=lambda row: (row["phase"], -row["total_ms"]))
    print(json.dumps(dict(trace=str(args.trace), metadata=lines[0], records=count,
                          phase_total_ms={key: value / 1e6 for key, value in totals.items()}, kernels=results), indent=2))


if __name__ == "__main__":
    main()
