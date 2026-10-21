"""Prepare the complete native-context comparison matrix, including admission waves.

Use identical workloads for cold reuse, prefix-enabled reuse and genuine
recomputation. All GPU execution remains in the existing benchmark harnesses.
"""
import argparse
import hashlib
import json
from pathlib import Path


CONTEXT = 32768
PAGE_TOKENS = 64
KV_BYTES_PER_TOKEN = 147456
BATCHES = [1, 2, 4, 8, 16, 32]
PROMPTS = [5, 129, 512, 2048, 4096, 8192, 16384, "near32k"]
OUTPUTS = [2, 32]


def request(length, outputs, slot=0):
    return dict(prompt_ids=[42 + (17 * i + slot) % 1000 for i in range(length)],
                max_tokens=outputs, stop_ids=[])


def reservations(requests, kv_bytes, max_batch):
    """Cold, prefix-free admission using the same complete-page reservation contract."""
    capacity = kv_bytes // (PAGE_TOKENS * KV_BYTES_PER_TOKEN)
    pages = [(len(r["prompt_ids"]) + r["max_tokens"] + PAGE_TOKENS - 1) // PAGE_TOKENS
             for r in requests]
    if not pages or min(pages) < 1 or max(pages) > capacity or max_batch < 1:
        raise ValueError("a request cannot be admitted at the configured capacity")
    waves, current, used = [], [], 0
    for slot, needed in enumerate(pages):
        if current and (len(current) == max_batch or used + needed > capacity):
            waves.append(current)
            current, used = [], 0
        current.append(slot)
        used += needed
    waves.append(current)
    return dict(reserved_pages_per_request=pages, available_pages=capacity,
                bytes_for_simultaneous_residency=sum(pages) * PAGE_TOKENS * KV_BYTES_PER_TOKEN,
                fits_simultaneously=len(waves) == 1, cold_admission_waves=waves)


def build_matrix(config):
    if config["max_context"] != CONTEXT or config["prefill_chunk"] < max(BATCHES):
        raise ValueError("matrix requires native 32768 context and a chunk covering batch 32")
    workloads, scenarios = {}, []

    def add(name, requests, kind):
        if name in workloads:
            raise ValueError(f"duplicate scenario: {name}")
        if any(len(r["prompt_ids"]) + r["max_tokens"] > CONTEXT for r in requests):
            raise ValueError(f"scenario exceeds native context: {name}")
        workloads[name] = requests
        scenarios.append(dict(name=name, kind=kind, batch=len(requests),
                              prompt_lengths=[len(r["prompt_ids"]) for r in requests],
                              output_lengths=[r["max_tokens"] for r in requests],
                              **reservations(requests, config["kv_cache_bytes"], max(BATCHES))))

    for prompt in PROMPTS:
        for outputs in OUTPUTS:
            length = CONTEXT - outputs if prompt == "near32k" else prompt
            for batch in BATCHES:
                add(f"homogeneous-p{length:05}-o{outputs:02}-b{batch:02}",
                    [request(length, outputs, slot) for slot in range(batch)], "homogeneous")
    lengths = [1, 5, 65, 129, 257, 33, 128, 64, 17, 256, 67, 3]
    outputs = [8, 6, 4, 2, 1, 7, 5, 3, 2, 4, 6, 8]
    add("ragged-short", [request(n, m, slot) for slot, (n, m) in enumerate(zip(lengths, outputs))], "ragged")
    lengths = [5, 129, 512, 2048, 4096, 8192, 16384, CONTEXT - 32]
    outputs = [32, 24, 16, 12, 8, 6, 4, 32]
    add("ragged-native-sustained", [request(n, m, slot) for slot, (n, m) in enumerate(zip(lengths, outputs))], "ragged")
    add("page-growth-boundaries", [request(n, 32, slot) for slot, n in enumerate([63, 64, 65, 127, 128, 129])], "boundary")
    # Rust's suite uses a BTreeMap. Freeze the same order for stock execution,
    # especially when prefix state persists between workloads.
    return dict(sorted(workloads.items())), sorted(scenarios, key=lambda s: s["name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are refused")
    args = ap.parse_args()
    source = args.config.read_bytes()
    config = json.loads(source)
    workloads, scenarios = build_matrix(config)
    config.update(max_batch=max(BATCHES), prefix_cache_entries=0, verify_logits=False)
    prefix = dict(config, prefix_cache_entries=max(BATCHES))
    matrix_bytes = (json.dumps(workloads, separators=(",", ":")) + "\n").encode()
    manifest = dict(
        format_version=1, source_config_sha256=hashlib.sha256(source).hexdigest(),
        workloads_sha256=hashlib.sha256(matrix_bytes).hexdigest(),
        context=CONTEXT, page_tokens=PAGE_TOKENS, kv_bytes_per_token=KV_BYTES_PER_TOKEN,
        required_runs=[dict(name="cold-reuse", cache_mode="reuse", config="cold-config.json"),
                       dict(name="prefix-reuse", cache_mode="reuse", config="prefix-config.json"),
                       dict(name="recompute", cache_mode="recompute", config="cold-config.json")],
        warmups=3, measured_repetitions=20, scenarios=scenarios,
        notes=["All 96 homogeneous length/output/batch combinations are retained, including admission waves.",
               "Prefix-enabled measurements must report actual hits/evictions; capacity can prevent persistent hits.",
               "Loading/JIT and verification probes are separate from resident timing samples.",
               "Native first-token/decode timing boundaries remain engine-specific.",
               "Stop, fork, continuation, release and rejection recovery remain explicit integration-test gates."])
    probes = [
        dict(name="native-output-overflow", config_overrides={},
             requests=[request(CONTEXT, 1)], expected="reject-context"),
        dict(name="single-request-pool-exhaustion", config_overrides=dict(kv_cache_bytes=PAGE_TOKENS * KV_BYTES_PER_TOKEN),
             requests=[request(65, 1)], expected="reject-admission"),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "workloads.json").write_bytes(matrix_bytes)
    for name, data in [("cold-config.json", config), ("prefix-config.json", prefix),
                       ("manifest.json", manifest), ("capacity-probes.json", probes)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    print(f"Prepared {len(scenarios)} scenarios for each of three cache/prefix modes in {args.output}")


if __name__ == "__main__":
    main()
