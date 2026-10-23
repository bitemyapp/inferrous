# RTX PRO 6000 FP32 shared-key staging: rejected

All builds, tests, sanitizers and operator timings ran on `ssh wx-workstation`. This isolated experiment widens each BF16 key to FP32 once while loading a shared QK tile. It has no measured win across thirteen shapes and is rejected without a production kernel or dispatch change. The original numerical gate and final stock-performance requirements remain open.

## Change and arithmetic

The worktree `inferrous-qk-f32-stage` starts at `6fb0e49`. It adds `attention_scores_tiled_f32` alongside the unchanged production tiled kernel. The new variant replaces asynchronous BF16 staging with synchronous loads and exact bit widening into FP32 shared storage. Four query rows still share sixty-four keys, and every dot retains the same 128 separate FP32 products and ordered additions, followed by the original scale. Query grouping, causal bounds, page mapping and output ownership are unchanged. Widening is a bit shift into the FP32 representation, so moving it before shared storage introduces no additional numerical rounding.

The 256-thread CTA cooperatively initializes all 8,192 shared key entries exactly once: thread `t` writes indices `t + 256 * piece` for pieces 0–31. A block-wide barrier precedes consumption. All original key-page reads remain within the same 65,536-element physical page. Partial query groups retain the original masked consumers and writers. This experiment changes staging and storage only; it does not parallelize or reassociate dot accumulation.

Compiled release resource usage rises from 25,600 to 41,984 shared bytes per block, while registers fall from 42 to 40; both kernels have zero stack and local storage. In checked builds, the original uses 44 registers and the new variant uses 40, with the same shared-memory sizes. These counts accompany the observed slowdown but do not establish its individual causes.

## Validation and timing boundaries

Both builds pass `paged_attention_matches_contiguous_ragged_queries`: seventeen queries across two slots, shuffled physical pages, partial groups, page crossings and a 4K tail. The candidate's complete probability and output arrays agree with the paged controls; the test also compares outputs with the independent contiguous implementation. Release memcheck and racecheck finish with zero errors, hazards or warnings. This is scoped operator coverage, not model or native-context sanitizer qualification.

The separate graph operator screen covers thirteen row/context shapes through 32K, including one/two/three/five-row partial groups. Queries include arbitrary FP32 values; K remains BF16. The untiled paged QK kernel supplies a raw-score reference, including untouched causal tails and a trailing 256-element guard. Each original/new graph arm starts with freshly initialized sentinel output and compares its complete score array with that reference. The checks use floating-point array equality, which does not distinguish signed zeros. The arithmetic argument above preserves the original operation sequence; no broader bit-pattern test is claimed.

Each shape uses original/new/new/original ordering, three excluded warmups and five measured CUDA events per arm, yielding ten measurements per variant. A 256 MiB cache eviction occurs outside every event interval. All 52 arm records complete. Builds finish before GPU tests and timings, and GPU work is serialized. These are isolated QK timings, not end-to-end gains or final-matrix samples.

| Query rows | Context stride | Original median, ms | FP32 staging median, ms | New / original |
|---:|---:|---:|---:|---:|
| 1 | 64 | 0.008192 | 0.008192 | 1.000 |
| 2 | 64 | 0.008192 | 0.008192 | 1.000 |
| 3 | 128 | 0.008192 | 0.010112 | 1.234 |
| 5 | 128 | 0.008192 | 0.010240 | 1.250 |
| 4 | 512 | 0.010240 | 0.010240 | 1.000 |
| 64 | 512 | 0.021504 | 0.022528 | 1.048 |
| 256 | 512 | 0.051200 | 0.057216 | 1.118 |
| 4 | 4096 | 0.017408 | 0.020480 | 1.176 |
| 64 | 4096 | 0.118400 | 0.133120 | 1.124 |
| 256 | 4096 | 0.450608 | 0.518176 | 1.150 |
| 4 | 32768 | 0.069632 | 0.079872 | 1.147 |
| 64 | 32768 | 0.946224 | 1.091616 | 1.154 |
| 256 | 32768 | 3.804080 | 4.381728 | 1.152 |

## Artifacts

Raw source, executables, command/process receipts, logs, telemetry, resource dumps and all timing samples remain under `clean-work/pro6000-baseline/qk-f32-stage-v1/` on the workstation. The final audit rechecks frozen source and binary identities, all seven successful command receipts and log hashes, embedded cubin provenance, all timing medians and byte-identical original tiled-function source. The runner is terminal. No model trial follows this rejected screen.

| Artifact | SHA-256 |
|---|---|
| Source patch | `7962c93753f5492691152e11ebe1d07b6ef30c91cdb5e0c2e074e5ce3740ebab` |
| Identity manifest | `ed242333e72f770fc5b55f01bd936666b4c12573582befaffe08e70159bb9a42` |
| Release tests | `07461dcd684096037683cee304e87959ff4fe0b802ce101b03790b0798d7d059` |
| Checked tests | `860cc52ce5713aba4bdb4c530a721ab23e7dd3dabbf432a7bc004c160cc7241b` |
| Profile summary | `15d739e417729a69f89192e10c46d8e06cd96840b913b892662f19e9b350c68a` |
| Completion summary | `dbe3194b1dc743e8423d02f1b3d611b7032e8e3effc07d9d6512f161a2af7a5d` |
| Final audit | `0c6e901626e8c7ba4ebf4f3fc360264c609df74477c19dd8837875b2ee873488` |
