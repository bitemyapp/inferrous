# Fourth optimization pass

Starting revision `ef994aa0`; executable SHA-256 `22b5511e8d3a45a94b4429e12c7fb5f3a5739933eb8ba033e157b7d16dd39f91`. Same Qwen3-8B INT8 bundle and RTX PRO 6000 Max-Q, 300 W, 5/129/512 prompt lengths and 32 outputs, one resident request at a time. All builds, numerical analysis, profiling and benchmarks run on the benchmark workstation. Three warmups and twenty measured requests per shape. No OS/clock tuning or other GPU jobs. Original full-vocabulary hashes remain the behavior oracle; the independent Torch full-model numerical discrepancy remains open.

The preceding node-level profile ranks grouped decode projections, residual projections, large prefill projections, grouped QKV and the packed BF16 head. Fresh baseline/profile artifacts use the `round4-baseline` prefix. The decode NCU evidence already shows 88.93% DRAM throughput and nearly full sectors; prefill instead has long scoreboard stalls, low L1 reuse and inefficient L2 sector use. Arithmetic and BF16 boundaries must remain unchanged.

| Candidate | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| Bypass L1 for large prefill staging | 4 | 4 | 1 | 16 |
| Pack prefill tiles contiguously at model load | 5 | 4 | 4 | 5 |
| Lossless INT8 compression, if observed distributions justify it | 5 | 3 | 5 | 3 |
| Fuse grouped gate/up with SwiGLU at matching channel ownership | 3 | 4 | 3 | 4 |
| Register-level decode dot/epilogue scheduling | 3 | 3 | 2 | 4.5 |

## Candidate: cache-global prefill staging

The large prefill kernel's staged input has only 7.44% L1 hits in the existing NCU report. Switch its two asynchronous 16-byte copy sites from cache-all to cache-global, keeping addresses, zero fill, shared layout and synchronization unchanged. The copied bytes, integer dot order, scale operations and BF16 cast are identical. No ordering, RNG, token tie or lifetime changes. Retain only if ordinary benchmark latency improves and all hashes pass.

Cache-global staging passes tests/hashes but increases long prefill from 43.5 to 51.0 ms; removed. Decode is unchanged structurally and drifts about 0.5%, so that drift is not attributed to the copy policy.

The sampled INT8 weights have 6.77–7.09 bits/value entropy and 2.5–5.4% values outside signed seven-bit range. Exceptions touch 80–98% of 128-value groups. A cheap fixed-width-plus-exceptions format has little useful room after indexing and unpacking overhead; do not implement a speculative compressor on this evidence.

## Candidate: contiguous weight tiles for large prefill

At model load, duplicate the large projection weights in [N/128, K/128, N%128, K%128] order, zero-padding only incomplete N tiles. Large prefill CTAs then copy a contiguous 16-KiB weight tile instead of gathering 128 disjoint row slices. Keep original row-major weights for decode and smaller prefill. This trades about 6.5 GB of extra device storage for improved prefill locality.

Every signed byte is unchanged. The shared layout, MMA inputs and reduction order remain identical after copying, as do scaling, BF16 rounding, RNG and token ties. The private packer produces exact tile sizes; model-owned copies outlive all launches. Tests cover K/N tile boundaries and padding independently, then compare all projection fixtures and complete-model hashes.

Contiguous prefill weights pass every test and hash but do not establish a gain: 45.0 ms long prefill and 242.5 ms API, versus the fresh 43.5/239.4 ms baseline. The extra device copy is removed before the next experiment.

## Candidate: TMA staging for large prefill

Score 5 (impact 5, confidence 4, effort 4). Use cuda-oxide's TMA and mbarrier intrinsics to replace hundreds of per-thread copy/address operations with two bulk tensor copies per K stage. Driver-created immutable tensor maps describe the existing A/B allocations; no duplicate weight layout is needed. Use a 128-byte swizzle in 1024-byte-aligned shared arrays, and adapt ldmatrix addresses to recover exactly the same signed INT8 fragments.

Thread zero initializes a one-arrival barrier and publishes it to the async proxy. Each stage registers its exact 24,576 transfer bytes, issues A/B copies, and all threads wait for the correct phase before consuming shared data. A block barrier separates the final MMA read from reuse; parity alternates each stage. TMA zero-fills tensor boundaries. The integer products and accumulation, scale order, BF16 rounding, ties and RNG remain unchanged. Descriptor/source allocations stay alive until the stream completes.

Protocol and swizzle reference: NVIDIA's [CUDA programming guide](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays). The existing independent partial-row/column fixtures and full-model regression remain the gates, followed by memcheck/racecheck/synccheck on any retained path.

TMA passes all seven tests and all 2,208 full-logit hashes through PTX. The normal offline NVVM/nvJitLink materializer crashes in its linker. A same-source PTX control takes 47.82 ms long prefill versus TMA's 45.55 ms, but PTX also increases short prefill relative to the established materialized build. This is not yet a deployable net gain; keep the compiler routes separate in comparisons.

## Candidate: paired gate/up dot with activation

One warp owns matching gate/up channels, reuses each activation word, and interleaves the two independent DP4A accumulators. It retains their original reduction trees, ordered scale multiplications and BF16 rounding, then evaluates exactly the original SiLU/product/BF16 expression. Gate/up trace buffers remain written. Eight warp leaders publish their maxima through eight distinct shared slots; a block barrier precedes one warp's maximum reduction and device atomic. This removes the separate decode activation launch and avoids reloading its projection outputs. All blocks have exactly eight full warps; invalid tail warps still initialize their shared slot and reach the barrier. The earlier norm reset and later quantizer remain stream ordered. Integer bounds, NaN max handling, RNG and token ties are unchanged. Test distinct sibling weights/scales, partial output blocks, full 4096x12288 shape and exact intermediate/output bits, then run the existing operator and full-model hash gates.

The adjacent original binary measures 166.51 / 161.75 / 156.71 decode tokens/s and 194.57 / 210.72 / 243.65 ms API. Gate/up fusion measures 167.89 / 162.86 / 158.04 and 193.05 / 209.41 / 241.96 ms. Its unchanged prefill agrees within 0.08 ms; the 0.7–0.9% decode improvement is supported by this adjacent control.

Explicit CUDA 13.1 libraries avoid the TMA intrinsic linker crash and preserve all tests/hashes, but increase short prefill to 10 ms. Their unchanged-source control confirms this is a compiler-route effect. The original default resolves to CUDA 12.8. Expressing only the bulk-copy instruction through cuda-oxide's inline PTX avoids that linker's intrinsic lowering failure and retains the original compiler route. The Rust kernel, mbarrier protocol and tensor maps remain unchanged. Single-buffer TMA long prefill is 44.82 ms, versus 45.60 ms in the adjacent original run; unchanged decode drifts downward about 0.5%, so that is not treated as a speedup.

## Candidate: overlap the next weight tile with current prefill arithmetic

Score 5 (impact 5, confidence 4, effort 4). Keep the 64x128x128 tile and single activation staging allocation. Double-buffer the 16-KiB weight tile, bringing shared storage to 40 KiB plus three barriers. A dedicated activation barrier tracks 8192 bytes per iteration. Alternating weight barriers each track 16384 bytes and alternate parity on their own reuse. Prefetch the next weight tile after the current one completes, then perform the same MMAs. The end-of-stage block barrier ensures all consumers finish before either slot is reused. The last iteration issues no next copy, so invalidation follows all pending work. Coordinates, signed bytes, integer/FP operations, BF16 rounding and ties are unchanged. Retain only after exact tests, hashes and ordinary timing.

Weight-only double buffering passes all tests/hashes but returns long prefill to 45.57 ms; remove it. Next test a 32x128x128 single-buffer tile: eight warps own 16x32 regions instead of 32x32, halving accumulator registers per warp, reducing shared storage to 20 KiB plus barrier, and doubling output CTAs. Score 6.7 (impact 5, confidence 4, effort 3). TMA A maps cover 32 rows; all other coordinates, integer products, scale order and BF16 boundaries are unchanged.

## Candidate: schedule four independent DP4A loads per loop

Score 4.5 from the opening matrix. The grouped/fused decode projections remain the largest hotspot. Load four consecutive per-lane words (offsets 0,128,256,384) before issuing their DP4As in the original order. Reuse those activation words for both gate/up dots. Full groups advance by 512; the original loop handles any remainder, including small independent fixtures. This exposes independent memory requests and removes three loop updates/branches per group. Integer accumulation order, warp tree, FP scaling, BF16 boundaries, ties and RNG are identical; each guarded group fits its original aligned row. Measure versus the fusion-only binary, and retain only if complete hashes and latency agree.

The smaller TMA tile passes all tests/hashes but regresses long prefill to 51.97 ms; remove it. Four-load decode scheduling also passes but establishes no improvement over fusion-only timing; do not retain added unrolling merely because it increases instruction-level parallelism on paper.

## Candidate: two gate/up channels per warp

Score 4.5 (impact 3, confidence 3, effort 2). Extend the fused gate/up warp to own two adjacent channels, sharing each activation load across four independent integer dots. Retain each dot's original accumulation/reduction/rounding, then evaluate both activations separately. The CTA combines sixteen coordinates instead of eight, halving CTA and maximum-atomic counts. Tail predicates cover an odd output count; every full warp and CTA still reaches its collectives. The distinct-weight operator case includes 33 channels to exercise that tail. No arithmetic ordering, BF16, tie, RNG or lifetime change. Compare against the one-channel fused binary and original full-model hashes before retention.

Two channels per warp pass all tests/hashes but regress API medians to 211.56 / 227.82 / 260.74 ms; remove the variant.

## Retained pair and compiler route

The retained kernels are one-channel gate/up/SwiGLU fusion and single-buffer 64x128 TMA prefill. Weight descriptors are allocated after the existing tensor allocations; activation descriptors follow their workspace tensors. There is no duplicate weight buffer and no change to `tools/cuda.sh` or its normal materialized CUDA 12.8 route. Only the TMA copy uses explicit PTX through cuda-oxide to avoid the intrinsic-lowering linker crash.

Combined timing is 194.20 / 210.41 / 242.03 ms. An immediately following fusion-only control measures 193.99 / 210.39 / 243.37 ms. Decode agrees within 0.12% across all shapes, and unchanged short/medium prefill agrees within 0.04 ms. Long prefill drops from 46.05 to 44.69 ms (2.95%), accounting for the 1.34 ms long-request gain. All eight tests and all original hashes pass. Temperature/clock drift makes the opening cold baseline optimistic; final comparisons use adjacent original/candidate runs rather than crediting it to an implementation change.

The first complete sanitizer pass succeeds, but memory sampling finds 14,468 MiB instead of the previous 10,256 MiB. Cubin inspection identifies an external `__cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast` fallback and an 8-byte stack frame in `linear_large`. The cluster-scoped copy emits both a direct path and a driver-call path on SM120, even though this kernel only targets its own CTA. This is an unnecessary scope and a material memory regression.

Use the explicit `.shared::cta` destination instead. Every destination and barrier is already private to the current CTA; no cluster communication was intended. This preserves exactly the same bytes, completion accounting and swizzle while allowing direct code generation without the cluster fallback. The [CUDA 12.8 PTX ISA](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) defines this destination scope. Repeat the complete validation and memory gate. The earlier validation files are preserved with the `cluster-checks-` prefix; those binaries are not the retained final candidate.

## Final validation

Retained runtime commits are `c26fd6d1` (gate/up activation fusion) and `70c7f5fc` (CTA-local TMA prefill). Eight tests pass in optimized and checked builds: independent operators, distinct-weight fused projection/activation, argmax edge cases, exhaustive BF16 reconstruction, failed-capture cleanup, admission and nineteen mixed full-model eager/graph requests through prompt length 4093. Checked generation matches the original full-vocabulary hashes on all three performance shapes. Compute Sanitizer memcheck, racecheck and synccheck pass the four GPU conformance tests with zero errors/warnings/hazards. A full-model 512-token prefill plus three outputs also passes memcheck.

The retained copy emits no driver tensor-copy syscall and no stack frame. Cubin resource inspection reports 66 registers and 26,624 shared bytes for `linear_large`, versus 72 registers and 28,672 shared bytes previously. The sampled process GPU-memory maximum is **10,256 MiB**, unchanged from the prior pass. The rejected cluster-copy variant's 14,468 MiB is not retained. These are separately instrumented sampled maxima, not exact peaks.

The restored optimized binary matches the tested binary, SHA-256 `96f0f0234b2f2abfe348f5675b4be0f42130ba8a47cde1f9ed7bf18c9d630b0f`. Formatting and Python syntax checks pass. The final profile shows 2,232 fused MLP launches and only 72 standalone SwiGLU launches (prefill) across two 32-output requests; large prefill kernels average 113.4 us. Profiles explain where time is spent; uninstrumented A/B/B/A runs determine the final improvement.

The independent Torch full-model numerical gate remains failing at its original tolerance. Exact agreement with the original Rust implementation proves optimization stability, not accuracy equivalence to external engines.

## Final uninstrumented comparison

Run A/B/B/A using `final-benchmarks.sh`: original binary, retained binary twice, then original again. Each run uses three warmups and twenty measured requests per shape. Pooled medians therefore contain forty observations per version and shape. All 138 retained-candidate requests match 4,416 complete-logit hashes and token IDs against the original unoptimized oracle. Every control also passes.

| Prompt tokens | Previous decode tok/s | Final decode tok/s | Previous API ms | Final API ms |
|---:|---:|---:|---:|---:|
| 5 | 165.95 | 167.09 | 195.22 | 193.96 |
| 129 | 161.36 | 162.18 | 211.26 | 210.37 |
| 512 | 156.28 | 157.34 | 244.54 | 241.91 |

Decode improves 0.5–0.7%; request latency drops 0.4–1.1%. Long prefill drops 45.91 to 44.61 ms (2.84%). Both candidate runs improve decode on every shape versus even the faster first control; medium API differences versus that first control are small. The second control slows as the run proceeds, so retain the individual A/B results and avoid treating the pooled percentages as invariant. The candidate remains 15.7% / 13.2% / 10.1% ahead of the recorded stock decode winner. Its 241.91 ms long-request median is only 0.3% below the earlier best stock 242.7 ms; that remains a close comparison, not a robust large-margin win. Stock was not rerun during this fourth pass.
