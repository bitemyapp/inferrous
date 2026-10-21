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
| [RTX PRO 6000 segmented attention](pro6000-segmented-attention.md) | [Round 8 record](../optimization/round-8.md), fixed-512-key operator validation and all 27 shape timings |
