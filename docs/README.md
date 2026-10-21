# Documentation

## Qwen3-8B W8A8 engine (`inferrous-qwen3`)

- [Engine guide](qwen3-8b-w8a8/README.md): design, numerical contract, build, tests and reproduction.
- [Paged sessions and batching](qwen3-8b-w8a8/sessions.md): configuration, CLI, ownership and validation status.
- [Benchmarks](qwen3-8b-w8a8/benchmarks.md): workloads, timing boundaries and comparisons with stock engines.
- [Kernel profiling](qwen3-8b-w8a8/profiling.md): opt-in CUPTI traces, complete-record validation and timing boundaries.
- [Validation coverage](qwen3-8b-w8a8/coverage.md) and [known numerical discrepancies](qwen3-8b-w8a8/discrepancies.md).
- [Safety contract](qwen3-8b-w8a8/safety.md): device indexing, launch, graph and lifetime arguments.
- Optimization records: [round 1](qwen3-8b-w8a8/optimization/round-1.md), [round 2](qwen3-8b-w8a8/optimization/round-2.md), [round 3](qwen3-8b-w8a8/optimization/round-3.md), [round 4](qwen3-8b-w8a8/optimization/round-4.md), [round 5](qwen3-8b-w8a8/optimization/round-5.md), [round 6](qwen3-8b-w8a8/optimization/round-6.md), [round 7 (paused checkpoint)](qwen3-8b-w8a8/optimization/round-7.md).
- [Results summaries](qwen3-8b-w8a8/results/README.md) for every round.
- [Round 8: RTX PRO 6000 batching](qwen3-8b-w8a8/optimization/round-8.md): remote-only builds/tests/benchmarks, baseline profiles and the tiled-prefill experiment.
- [RTX 5090 Nix validation](qwen3-8b-w8a8/results/rtx5090-nix.md): local release/checked test results, benchmark timings and exact-output comparisons.

## Plans

- [RTX 5090 handoff](plans/kv-cache-batching-rtx5090-handoff.md): current checkpoint, validated results, interrupted checks, remaining performance gaps and exact resume commands.
- [KV cache, batching and RTX 5090](plans/kv-cache-batching-rtx5090.md): implementation history, validation matrix and unfinished performance comparison work.
- [Production plan](plans/production.md): decisions, workstreams and milestones for serving traffic.
- [Qwen3.6-35B-A3B FP8](plans/qwen3.6-35b-a3b-fp8.md): implementation plan for the next model.

## Maintenance

- [Local Nix development](migrations/local-nix.md): CUDA and Rust setup, release tests, model preparation and RTX 5090 benchmarks.
- [Migration to NVIDIA cuda-rust](migrations/cuda-rust.md): what changed, correctness, and compile-time and runtime before/after.

## Research

- [Modern open models for one RTX PRO 6000](research/single-gpu-model-candidates.md): candidates ranked by decode speed and coding quality.
- [DeepSeek-V4.1-Flash on one RTX PRO 6000](research/deepseek-v4.1-flash-rtx-pro-6000.md): quantizations, engine support and feasibility.
