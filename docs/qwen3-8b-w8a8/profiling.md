# Kernel profiling

The Nix shell includes CUDA 13.4 CUPTI headers and libraries. `tools/profile_kernels.sh` builds a small host-only C activity collector and loads it into the requested executable. All inference kernels remain Rust. Profiling is opt-in; ordinary builds and benchmarks do not load the collector. It works with the workstation's existing driver settings and collects activity timestamps without requesting hardware performance counters.

Choose a fresh output path and a request array described in the [session guide](sessions.md). For cold-prefix profiling, copy the appropriate hardware configuration and set `prefix_cache_entries: 0`; record whether logit verification and CUDA graphs are enabled.

```sh
nix develop
crates/inferrous-qwen3/tools/cuda.sh build
crates/inferrous-qwen3/tools/profile_kernels.sh clean-work/new-trace.csv \
  target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle-32k \
  --engine-config path/to/profile-config.json --batch path/to/requests.json \
  --repeat 3 > clean-work/new-profile-generation.jsonl
python crates/inferrous-qwen3/tools/summarize_kernels.py \
  clean-work/new-trace.csv > clean-work/new-kernel-summary.json
```

The CSV records the executable SHA-256, CUPTI API version, every kernel's start/end timestamps, device/context/stream, grid/block dimensions, register and shared/local-memory usage, and graph/node IDs. A final footer records the number of emitted, dropped and invalid records. The collector and summarizer reject empty or incomplete traces; the collector refuses an existing CSV. The accompanying generation JSON records the model root, hardware policy and inputs.

The summary includes every traced repetition. It separates eager launches from graph replay and reports total kernel time, phase share, call counts and median duration. Graph replay can include the last prefill step when it has one input row per ready sequence, so these phase names describe the launch mechanism. Summed kernel durations can overlap and exclude host-side work. Instrumentation also changes timing: use these traces to locate bottlenecks, and use the ordinary paired benchmark harnesses to validate performance improvements.

The first validated trace contains 23,436 kernel records with zero dropped or invalid records for three batch-32, 129-token, eight-output requests. The collector and parser also reject an explicitly tested empty trace. Artifacts are `clean-work/kv-batch-baseline/grouped-b32-{kernels.csv,profile.jsonl,profile.log,kernel-summary.json}` and `cupti-empty-validated.*`. The [round 7 record](optimization/round-7.md) interprets the profile.
