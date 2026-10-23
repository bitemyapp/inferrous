# Results summaries

Each optimization round's raw artifacts (measurement JSONL, logs, profiles, scripts and patches) are not published. These pages are generated from them. File names refer to those raw artifacts. Timing rows report medians over measured runs (the first three repetitions are warmups), computed the same way as `tools/summarize.py`.

| Page | Records |
|---|---|
| [Initial baseline, first optimization pass and round 2 (2026-10-02)](rounds-1-2.md) | [round 1 record](../optimization/round-1.md), [round 2 record](../optimization/round-2.md) |
| [Round 3 (2026-10-03)](round-3.md) | [round 3 record](../optimization/round-3.md) |
| [Round 4 (2026-10-03)](round-4.md) | [round 4 record](../optimization/round-4.md) |
| [Round 5 (2026-10-03)](round-5.md) | [round 5 record](../optimization/round-5.md) |
| [Round 6 (2026-10-06)](round-6.md) | [round 6 record](../optimization/round-6.md) |
| [RTX 5090 Nix validation (2026-10-20)](rtx5090-nix.md) | [Local Nix setup](../../migrations/local-nix.md), release/checked tests and benchmarks |
| [RTX PRO 6000 cache-mode pilots](pro6000-cache-mode-pilots.md) | [Round 8 record](../optimization/round-8.md), completed prefix/recomputation stock screens and remaining gaps |
| [RTX PRO 6000 segmented attention](pro6000-segmented-attention.md) | [Round 8 record](../optimization/round-8.md), fixed-64/128/512-key operator validation and all 27 shape timings |
| [RTX PRO 6000 128-row projections](pro6000-m128-projections.md) | [Round 8 record](../optimization/round-8.md), independent-reference/sanitizer validation and all 44 shape timings |
| [RTX PRO 6000 asynchronous projections and 32-row head](pro6000-async-projections-head32.md) | [Round 8 record](../optimization/round-8.md), rejected projections and retained head with paired model, sanitizer and native-session evidence |
| [RTX PRO 6000 certified head](pro6000-certified-head.md) | [Round 8 record](../optimization/round-8.md), retained exact head, paired model results, capacity/sanitizers and isolated shape/grouped-bound screens |
| [Torch reference cache-mode sensitivity](pro6000-reference-cache-modes.md) | [Numerical discrepancy](../discrepancies.md), controlled cached/recomputed comparisons, fresh cached controls and the unchanged acceptance gate |
| [RTX PRO 6000 stock native-capacity checks](pro6000-stock-native-capacity.md) | [Round 8 record](../optimization/round-8.md), all eight stock configurations complete five workloads at 64 GiB KV, including oversized near-32K batches |
| [Fixed paged policy native/sanitizer qualification](pro6000-fixed-policy-qualification.md) | [Round 8 record](../optimization/round-8.md), isolated candidate passes eight selected tests, full-model memcheck and all changed-family racecheck partitions; original acceptance remains open |
| [Exact ordered PV experiment](pro6000-exact-pv.md) | [Arithmetic argument](../../research/exact-ordered-pv.md), CPU/CUDA exactness and scoped sanitizers pass; all eight production-kernel timing cases reject the prototype |
| [FP32 shared-key staging](pro6000-qk-staging.md) | [Round 8 record](../optimization/round-8.md), exact-order QK staging passes scoped validation but has no win across thirteen paired operator shapes |
| [Exact-order softmax experiments](pro6000-softmax-tree.md) | [Round 8 record](../optimization/round-8.md), operator validation and timing screens complete; first variant enters a separate model trial |
