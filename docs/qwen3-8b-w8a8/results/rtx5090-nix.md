# RTX 5090 local Nix validation, 2026-10-20

The root Nix development shell successfully builds and runs the engine on a GeForce RTX 5090. Both unchecked and bounds-checked release builds pass all nine GPU-enabled tests with zero failures, ignores or filters. Both benchmark runs complete all 69 requests, and the checked run matches the unchecked run's token IDs and all 2,208 complete-vocabulary logit hashes exactly.

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB reported memory, 575 W power limit |
| Driver | 615.78.08 |
| CUDA toolkit | 13.4.92 from the root flake |
| Rust | nightly-2026-08-28, rustc 1.100.0-nightly, `e457a7b0d326d67b4322ef0d11bd715cfaeda48f` |
| cuda-rust submodule | `6921d3ee07e28ec62895ea40fc2fef39977ca411` |
| Nixpkgs | `02c087f3fe673c9e317f206a627cb6b910336dbe` |
| rust-overlay | `5424e1e6dda30b0626935d2a03a6526db4858d60` |
| Python preparation | PyTorch 2.13.0 (CPU), NumPy 2.5.3, safetensors 0.8.0, tokenizers 0.23.2 |
| Model revision | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| Local bundle root | `fad351c99221127c95303f18af560bd42dc8284179b93a9bc3360fb53286c20f` |

The model and independent fixtures were regenerated using the locked Nix environment. This bundle root differs from the historical workstation root; no historical raw baseline was available locally. The comparisons here establish local determinism and agreement between checked and unchecked builds. They do not establish bitwise agreement with the earlier workstation bundle or resolve the [independent full-model numerical discrepancy](../discrepancies.md).

## Measurements

Each workload generates exactly 32 tokens. Each build runs three warmups and twenty measured resident requests per prompt length, with fresh prompt processing and no EOS stopping. All inferrous GPU jobs ran serially. The desktop and other applications remained active and GPU clocks were not fixed, so these are local setup measurements rather than an A/B performance study. In particular, the checked/unchecked timing differences should not be interpreted as an optimization result.

| Build | Prompt tokens | Decode tokens/s | API p50 ms | API p95 ms | API p99 ms | First step p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| Release, unchecked | 5 | 190.640 | 171.196 | 177.476 | 180.702 | 8.250 |
| Release, unchecked | 129 | 171.664 | 196.479 | 209.556 | 218.032 | 15.427 |
| Release, unchecked | 512 | 163.414 | 225.577 | 229.937 | 247.815 | 35.509 |
| Release, checked | 5 | 190.052 | 171.871 | 173.758 | 174.526 | 8.428 |
| Release, checked | 129 | 185.964 | 181.653 | 198.631 | 198.909 | 14.631 |
| Release, checked | 512 | 172.180 | 215.065 | 234.183 | 240.278 | 34.422 |

The table is generated from `clean-work/rtx5090-validation/summary.json` using the existing `tools/summarize.py` timing definitions. Decode throughput excludes the first token; API timing includes the public generation call. Loading and graph capture are outside these warm timings. Percentile tails from twenty measurements are descriptive only.

## Checks and retained artifacts

- Release GPU suite: 9 passed in 17.73 seconds; checked release suite: 9 passed in 18.09 seconds. Coverage includes independent operators, fused gate/up, integer rounding, GPU argmax, exhaustive BF16 packing, capture cleanup, host admission and nineteen changing-input eager/graph comparisons through 4,093 prompt tokens.
- Release benchmark: 69 requests, three workloads, deterministic tokens and logit hashes across repetitions. Checked benchmark: 69 requests, exact comparison against the release JSONL passed.
- `nix flake check`, ShellCheck, crate Rust formatting and Python syntax checks passed. Benchmark harness probes accepted a complete run and rejected empty, truncated, duplicate, incomplete-generation and nonzero-exit output, including with Python assertions disabled. Missing-model test preflight fails before compiling; repeated preparation reuses the completed data.

Raw results remain locally in the ignored `clean-work/rtx5090-validation/` directory: `release.jsonl`, `checked.jsonl`, `summary.json`, `bench-release.log`, `bench-checked.log`, `test-release.log`, `test-checked.log`, `environment.log`, `binaries.sha256`, and the initial build/preparation/flake-check logs. The model, fixtures and workloads remain under `clean-work/` for subsequent runs.

| Executable | SHA-256 |
|---|---|
| `target/inferrous-gpu/release/inferrous-qwen3` | `f4c8a0c12e02503dcdbba05b7f168814178213203a1ee7be0ba483965ead0150` |
| `target/inferrous-gpu-checked/release/inferrous-qwen3` | `4c31ca680e865e07fb73dfbe9fe0d91ee25db256bb03abba1bf20cc373b60f1b` |

See [local Nix development](../../migrations/local-nix.md) for the exact build, test and benchmark commands. Use new output filenames when repeating a benchmark; the runner refuses to overwrite existing measurements.
