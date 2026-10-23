# Documentation

## Qwen3-8B W8A8 engine (`inferrous-qwen3`)

- [Engine guide](qwen3-8b-w8a8/README.md): design, numerical contract, build, tests and reproduction.
- [Paged sessions and batching](qwen3-8b-w8a8/sessions.md): configuration, CLI, ownership and validation status.
- [Benchmarks](qwen3-8b-w8a8/benchmarks.md): workloads, timing boundaries, device-wide sampled-memory reporting and comparisons with stock engines.
- [Kernel profiling and racecheck coverage](qwen3-8b-w8a8/profiling.md): opt-in CUPTI traces, complete-record validation, timing boundaries and exhaustive partition coverage audits.
- [Validation coverage](qwen3-8b-w8a8/coverage.md) and [known numerical discrepancies](qwen3-8b-w8a8/discrepancies.md).
- [Torch reference cache-mode sensitivity](qwen3-8b-w8a8/results/pro6000-reference-cache-modes.md): full-model cached/recomputed differences on identical token histories, fresh controls and the unchanged acceptance boundary.
- [Safety contract](qwen3-8b-w8a8/safety.md): device indexing, launch, graph and lifetime arguments.
- Optimization records: [round 1](qwen3-8b-w8a8/optimization/round-1.md), [round 2](qwen3-8b-w8a8/optimization/round-2.md), [round 3](qwen3-8b-w8a8/optimization/round-3.md), [round 4](qwen3-8b-w8a8/optimization/round-4.md), [round 5](qwen3-8b-w8a8/optimization/round-5.md), [round 6](qwen3-8b-w8a8/optimization/round-6.md), [round 7 (paused checkpoint)](qwen3-8b-w8a8/optimization/round-7.md).
- [Results summaries](qwen3-8b-w8a8/results/README.md) for every round.
- [RTX PRO 6000 acceptance audit](qwen3-8b-w8a8/results/pro6000-acceptance-audit.md): current requirement-by-requirement status, fresh retained-candidate capacity probes, evidence limits and unfinished numerical/performance gates.
- [Round 8: RTX PRO 6000 batching](qwen3-8b-w8a8/optimization/round-8.md): remote-only builds/tests/benchmarks, aggregate sanitizer and injected failure-path coverage, retained-binary profiles, exact split-K projection candidate, rejected numerical trials and full-matrix preparation.
- [RTX PRO 6000 asynchronous projections and 32-row head](qwen3-8b-w8a8/results/pro6000-async-projections-head32.md): rejected projections and a retained head with paired model gains, sanitizer checks and native-session validation.
- [Exact-output head certification](research/exact-head-certification.md): guarded numerical contract, complete operator comparisons, isolated model validation and a packed-weight follow-up experiment.
- [Certified-head model pilot](qwen3-8b-w8a8/results/pro6000-certified-head.md): retained PRO thirty-two-output dispatch, paired latency results, original-logit checks, 64 GiB capacity, startup/storage costs, integrated sanitizers and broader shape/grouped-bound screens.
- [RTX PRO 6000 cache-mode pilots](qwen3-8b-w8a8/results/pro6000-cache-mode-pilots.md): completed stock prefix/recomputation screens, matching local medians and pilot limitations.
- [RTX PRO 6000 stock native-capacity checks](qwen3-8b-w8a8/results/pro6000-stock-native-capacity.md): all eight planned stock configurations complete the 64 GiB KV/native-context probes; separate from final timing and numerical acceptance.
- [Fixed paged policy qualification](qwen3-8b-w8a8/results/pro6000-fixed-policy-qualification.md): isolated current-head candidate passes release/checked native tests, full-model memcheck and exhaustive changed-family racecheck partitions; production and the original numerical gate remain unchanged.
- [Exact ordered PV experiment](qwen3-8b-w8a8/results/pro6000-exact-pv.md): exact CPU/CUDA comparisons and scoped sanitizers pass, but the guarded integer-grid prototype loses every production-kernel timing case and is rejected.
- [FP32 shared-key staging](qwen3-8b-w8a8/results/pro6000-qk-staging.md): release/checked operator comparisons and scoped sanitizers pass; thirteen paired QK shapes show no win, so the staging variant is rejected.
- [Exact-order softmax experiments](qwen3-8b-w8a8/results/pro6000-softmax-tree.md): bitwise checks, scoped sanitizers and original-logit model regression pass; the paired model trial finds only small gains amid drift, so no variant is retained.
- [RTX PRO 6000 segmented-attention operators](qwen3-8b-w8a8/results/pro6000-segmented-attention.md): independent-reference and sanitizer checks, complete shape timings and the short-context gap.
- [RTX PRO 6000 128-row projections](qwen3-8b-w8a8/results/pro6000-m128-projections.md): complete correctness/sanitizer checks and the rejected operator screen.
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

- [Exact ordered attention-value sums](research/exact-ordered-pv.md): constant-spacing integer blocks, exact tie handling, fallback and the rejected GPU implementation's validation boundary.
- [Modern open models for one RTX PRO 6000](research/single-gpu-model-candidates.md): candidates ranked by decode speed and coding quality.
- [DeepSeek-V4.1-Flash on one RTX PRO 6000](research/deepseek-v4.1-flash-rtx-pro-6000.md): quantizations, engine support and feasibility.
