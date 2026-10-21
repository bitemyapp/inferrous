# Kernel profiling

The Nix shell includes CUDA 13.4 CUPTI headers and libraries. `tools/profile_kernels.sh` builds a small host-only C activity collector and loads it into the requested executable. All inference kernels remain Rust. Profiling is opt-in; ordinary builds and benchmarks do not load the collector. It works with the workstation's existing driver settings and collects activity timestamps without requesting hardware performance counters.

On `wx-workstation`, set `CUDA_HOME=/usr/local/cuda-13.1`. The wrapper supports both `lib` and `lib64` toolkit layouts and selects the current kernel activity record declared by that toolkit's primary CUPTI header. The collector rejects a header/runtime API version mismatch before collecting records. CUDA 13.1 on the RTX PRO 6000 has been exercised with complete batch-32 and 8K traces; these are separate from the RTX 5090 traces below.

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

## Partitioned racecheck evidence

The [PRO investigation](optimization/round-8.md#full-model-racecheck-coverage-partitioning) checks the complete ragged workload in separate kernel-family partitions to bound sanitizer memory. Every selected kernel retains all launches and blocks. A full-coverage claim requires the disjoint union of completed partitions to cover every launch in the complete inventory, with unchanged launch order, dimensions and graph/eager classification. Every run must also preserve every original full-vocabulary output hash. Completing a subset is not acceptance; this evidence describes aggregate coverage, not one unfiltered process.

[`check_racecheck_coverage.py`](../../crates/inferrous-qwen3/tools/check_racecheck_coverage.py) verifies saved evidence without launching GPU work. Run it on `wx-workstation`, using a fresh output path:

```sh
clean-work/python/bin/python crates/inferrous-qwen3/tools/check_racecheck_coverage.py \
  clean-work/coverage-manifest.json --output clean-work/coverage-audit.json
```

The schema-version-1 manifest records `working_directory` and `model` paths, plus `sanitizer`, `binary`, `config` and `requests` artifacts. Each artifact is an object containing `path` and `sha256`; paths are resolved relative to the manifest directory. `original` contains artifacts named `binary`, `config`, `model` (the original model's JSONL output), `stderr` and `receipt`. `inventory` contains `log`, `model`, `stderr`, `command` and `result` artifacts. Each item in `partitions` contains those same five run artifacts plus its exact `kernels` list. All completed partitions, including any reused earlier partition, must appear in the manifest.

The original receipt records a successful exit, exact command, binary identity and stdout/stderr hashes. Each sanitizer result records exit status, whether the memory guard triggered, and hashes of its log, model output and stderr. Each command artifact records the full executed argument array under `command`. The verifier checks these against the declared model inputs and permits only the audited sanitizer controls. It rejects launch/block exclusions, suppressions, disabled asynchronous-copy checks, changed requests/configuration/cache mode, incomplete logs, missing or overlapping coverage, and changed tokens or full-logit hashes. Output is written only after every check passes.

The verifier pins the staged Compute Sanitizer 2026.3 executable by SHA-256. This version counts informational launch records in its displayed hazard total and returns the configured error exit code when launch logging is enabled. Acceptance therefore requires a complete parsed log consisting only of those launch records and a matching informational count with zero errors and warnings. An arbitrary nonzero exit or zero-error footer is insufficient. Other sanitizer versions require a fresh reporting/compatibility audit before changing the supported identity. The installed CUDA toolchain and inference executable are independent of the staged validation tool.
