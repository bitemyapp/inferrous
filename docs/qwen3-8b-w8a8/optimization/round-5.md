# Fifth optimization pass

Baseline: `21119a83`, executable SHA-256 `96f0f0234b2f2abfe348f5675b4be0f42130ba8a47cde1f9ed7bf18c9d630b0f`. All builds, tests and timing run on the benchmark workstation, one GPU job at a time. Original `baseline.jsonl` remains the complete-logit oracle. The independent full-model Torch gate remains failing; exact regression is a separate claim.

Fresh baseline: 169.37 / 164.35 / 159.29 decode tokens/s, API 191.39 / 206.73 / 237.44 ms for 5 / 129 / 512 prompt tokens. This cold run is faster than the last pass's warm controls; use adjacent and A/B/B/A controls for retention.

The new node-level profile attributes 30.8% of device time to fused gate/up, 22.4% to O/down, 8.2% to QKV, 8.7% to large prefill, 8.0% to the packed head, and 4.8% to prefill attention. Earlier DP4A counters reach 88.93% DRAM throughput.

| Candidate | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| Lossless seven-bit base plus sparse integer corrections | 4 | 3 | 4 | 3 |
| GPU-driven decode / overlapped host transfers | 3 | 4 | 4 | 3 |
| Fuse repeated norm/quantization into projection consumers | 3 | 2 | 3 | 2 |
| Exact speculative verification to reuse weight reads | 5 | 2 | 5 | 2 |
| Further TMA prefill overlap with CTA scope | 3 | 3 | 4 | 2.25 |

Considered but outside the unchanged baseline: lower precision, pruning, new attention reductions and approximate speculative acceptance. They change the numerical contract. Exact speculative decoding additionally needs a useful draft/acceptance workload, not just repeated benchmark prompts. A cooperative persistent whole-model kernel would require global reductions between layers; its barriers and occupancy need to beat the existing graph schedule.

## Rejected: lossless INT8 base plus sparse corrections

Store signed seven-bit base values in seven coalesced word planes per 1024 weights. Four values expand into DP4A bytes. An outlier contributes exactly +128 or -128 times its activation, represented by a signed 16-bit column index in a per-row sparse list. Unlike the screened block-escape scheme, an outlier does not force an entire block back to eight bits, and the main dot avoids per-weight exception lookups. Start with the largest hotspot, gate/up.

Ordering: integer products/reductions can reassociate; absolute sums including base and corrections stay below 4096*(64+128)*127 < 2^31. FP32 scales, BF16 boundaries, SwiGLU operations, output ownership and maximum publication remain unchanged. Ties and RNG are unchanged. Retain original weights for prefill; this trial deliberately spends extra VRAM to test decode bandwidth independently. Require original hashes before interpreting timings or retaining the change.

The packed INT8 trial preserves all 69 requests and 2208 complete-logit hashes, but regresses API medians to 194.71 / 210.33 / 242.15 ms versus an adjacent unmodified control at 192.30 / 208.24 / 239.40 ms. The fused kernel itself grows from 63.39 to 64.42 us in node profiles. Sparse gathers and unpacking cost more than the reduced traffic saves. Remove it, including the extra VRAM allocation.

## Rejected: pipeline CTA-local TMA with K=64

Use two 64-wide staging slots instead of one 128-wide slot, preserving total 24 KiB operand storage. Each slot has its own completion barrier/parity; prefetch the next A/B tiles while the current tile's MMAs execute. Swizzle64 maps address bits 7–8 onto bits 4–5, with 1024-byte aligned bases. All consumers wait for their slot and all threads fence before reusing it. Partial global rows/columns remain TMA zero-filled; accumulator ownership, exact INT32 sums, FP32 scale order and BF16 cast are unchanged. Ties/RNG remain unchanged.


The K=64 TMA pipeline passes operator tests and all original hashes but loses: long prefill is 45.41 ms, versus the adjacent control's 43.81 ms. Shorter prefill, which does not use this kernel, is essentially unchanged. Remove it.

## Retained: pinned, double-buffered result transfers

Keep two persistent pinned host slots for logits and selection. Enqueue both copies without intermediate synchronization, digest the other completed slot while the GPU works, then fence once before consuming this slot. This removes pageable-host staging and per-token transfer-buffer allocations. The execution mutex owns the slots; same-stream copies precede device-output reuse; a destructor fences failed requests before releasing pinned memory. Arithmetic, ordering, full hashes, stop/tie behavior and timing boundaries remain unchanged.

## Rejected: four exact decode steps per graph

For fixed-length requests with no stop IDs, run four dependent steps in one captured graph. A small kernel preserves each full logit vector and copies the GPU-selected token into the next step's control record. This is ordinary exact autoregression: each step waits for the preceding model output; there is no draft, acceptance approximation, prefix cache or changed arithmetic. Retain single-step graphs for tails, graph-capacity boundaries, tracing and requests with stop IDs. A nonfinite sentinel is preserved for host rejection and clamped only as a future embedding index so already queued work cannot read out of bounds.

All four KV writes are to admitted positions and are stream-ordered. The graph never crosses a capacity/attention-dispatch boundary. Results and full hashes stay in token order; double-buffered readback slots alternate by launch rather than token parity. Graphs drop before their captured history/control allocations. The public API is non-streaming. `step_seconds` reports host availability: the first token in a batch carries its complete wait, and its three companions have zero interarrival time. Decode uses the same total wall interval including transfers; API latency is the independent retention metric. This does not claim an improvement in per-token streaming latency.

Four-step execution passes all eight tests (mixed-request tests extended to 12 outputs) and all 69 original requests, but loses versus pinned single-step: API 192.23 / 208.49 / 240.56 ms versus 191.50 / 207.44 / 238.97 ms. The graph and extra save/copy work do not repay the fewer host round-trips. Remove the batched path and its altered per-step timing representation.

## Rejected: transpose decode onto tensor cores

Score 3.75 (impact 5, confidence 3, effort 4). Let sixteen weight rows be MMA's active M dimension, with eight identical activation columns. Four warps divide K; each writes sixteen exact partial sums, then the first warp sums them and performs the unchanged gate/up/SwiGLU epilogue. This differs from the earlier rejected tensor-core decode, whose active activation M dimension had one row and fifteen zero rows. The original gate/up kernel remains an operator oracle. INT32 products/reassociation fit the same absolute bound. Shared partials have one writer and a block barrier before reduction; output ownership and all FP32 scale/BF16/activation operations are unchanged. Ties and RNG are unchanged.

The transposed tensor-core decode preserves all hashes but regresses API to 286.86 / 301.79 / 334.06 ms. Remove it. Redundant columns, strided fragments and fewer active warps lose decisively to the bandwidth-efficient DP4A mapping.

## Retained: native nearest-even INT8 conversion

Score 15 (impact 3, confidence 5, effort 1). Quantization operators together account for 6.6% of the fresh profile. Replace the manual truncate/fraction/tie sequence with PTX `cvt.rni.s32.f32` after the same [-127,127] clamp, then take the low byte. Finite rounding is identical, including negative ties and signed zero. Clamped infinities are finite; NaN conversion's low byte is zero just as with the previous Rust cast. Scale computation and divisions are unchanged. Verify a dense grid, every BF16 input bit pattern, tie neighbors and NaNs against independent host f64 nearest-even, plus the independent quantizer fixtures and all model hashes.

## Rejected: exact integer quantization from BF16 spacing

Score 6 (impact 3, confidence 4, effort 2). Unlike the earlier general guard on approximate quotients, exploit this graph's BF16 inputs. Write their positive significands as integers a,b in [128,255], so the ideal scaled ratio is 127*a/(b*2^d). Its distance from a non-equal half-integer is at least 1/(2*b*2^d). Float scale/reciprocal/multiply errors are far smaller: a conservative relative bound 2^-20 times numerator 127*255 is less than 0.031, versus the 0.5 numerator gap. The original RN division has the same nearest-integer result away from exact mathematical half-integers. Inputs below maximum/256 round zero.

Because 127 is prime, an exact half-integer for b != 254 must be 63.5, i.e. input magnitude == maximum/2. Keep the original division for that case. For b == 254 the stored scale is exactly a normal power of two; construct its exact reciprocal from exponent bits, so every division/multiplication is exact in the relevant range. Preserve the original division when the 1e-8 floor is active or maximum is nonfinite. Sign symmetry preserves negative rounding. Scales, BF16 boundaries and ties remain unchanged. NaNs retain the original zero-byte result.

The device test covers all 32640 nonnegative finite BF16 maximum encodings and each preceding 1024 BF16 input magnitudes, both signs: 66,846,720 comparisons with the original RN division. Lower magnitudes necessarily round zero. Keep independent fixtures and full-logit hashes as separate gates. PTX specifies at most one ulp for [approximate reciprocal](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#floating-point-instructions-rcp); normal scales/reciprocals avoid FTZ effects in the fast path.

The complete BF16-domain test and all ten tests pass. The first fast-path variant also keeps all original hashes, but it shows no gain over native conversion alone. Test whether removing the remaining divergent division helps by classifying the exact boundary directly: RN(maximum/(2*scale)) is below 63.5 precisely when `127*scale - maximum > scale*2^-18`. The right side derives from the midpoint 63.5 - 2^-19; equality rounds to even 63.5 and then integer 64. An explicit RN FMA computes the small left-hand residual exactly (the nearly equal operands cancel down to at most eight significand bits). Its result selects 63 or 64, then restores the input sign. Normal scale bounds keep the threshold normal. This FMA classifies a rounding boundary; model arithmetic is not reassociated. Run the same entire finite-domain comparison again before benchmarking.

The direct-halfway refinement passes all ten tests and hashes but is slower than native conversion alone (191.08 / 207.12 / 238.48 ms versus 190.13 / 206.09 / 237.40 ms). Remove both BF16 reciprocal variants; their proof and raw results remain as rejected experiments. Keep the native conversion and its independent rounding-boundary test.

## Retained: hardware integer warp reductions

Score 25 (impact 5, confidence 5, effort 1). Replace five shuffle/add stages with native `redux.sync.add` in grouped/fused DP4A dots, and shuffle/max stages with native unsigned maximum in quantization and producer maxima. These operators feed the projection hotspot. All participating masks are complete warps, including tail predicates uniform within each warp; the final max cohort is exactly the first warp. Integer sums are unchanged modulo 2^32, with final signed sums within the existing bound. Positive max inputs have unsigned bit order; NaNs were already excluded. Original DP4A and separate operator kernels remain comparison oracles. Floating-point summation trees and rounding are unchanged, as are token ties and RNG.


The pinned-only trial passes all hashes and improves API latency from the adjacent control's 192.30 / 208.24 / 239.40 ms to 191.50 / 207.44 / 238.97 ms. Adding native conversion yields 190.13 / 206.09 / 237.40 ms. Native integer reductions further yield 189.56 / 205.81 / 237.15 ms, with decode at 170.98 / 165.81 / 160.67 tokens/s. The final paired measurements below determine the combined gain independently of these sequential exploratory runs.

The cuda-oxide revision's feature classifier rejects its integer redux wrappers as `ReduxF32`, requiring an SM100a-compatible target. Explicit integer PTX avoids that erroneous classification while retaining the normal SM120 build, CUDA12.8 materializer and Rust/cuda-oxide implementation. No vendor sources are changed.

## Final paired results and validation

Retained commits: `15656743` (pinned readback), `0406c6e9` (native rounding), `b88d941f` (integer warp reductions). All other implementations were removed. The source patches for the larger rejected trials accompany their raw results. No quantization recipe, model weights, floating-point reduction tree, graph capacity, stop behavior or checksum coverage changed.

A/B/B/A compares the preceding binary to the final binary, with three warmups and twenty measured requests per shape in each run. Pooled medians use forty observations per shape and implementation:

| Prompt | Previous decode tok/s | New decode tok/s | Previous API ms | New API ms |
|---:|---:|---:|---:|---:|
| 5 | 167.134 | 169.625 | 193.908 | 191.081 |
| 129 | 162.258 | 164.593 | 210.222 | 207.475 |
| 512 | 157.350 | 159.470 | 241.843 | 238.948 |

This is 1.3–1.5% more decode throughput and 1.2–1.5% lower API latency. Both candidate runs beat both control runs at every shape. First-step medians are 8.064 / 18.836 / 44.271 ms; prefill is not the source of the main saving. The opening cold baseline was faster than the later controls, so it is not used to inflate or obscure the paired result. Raw p50/p95/p99 summaries and all individual observations are summarized in the [round 5 results](../results/round-5.md).

Nine tests pass in optimized and bounds-checked builds, including 1,328,443 rounding inputs checked against independent f64 nearest-even, independent operators, nineteen changing-input eager/graph cases, stop handling and failed capture cleanup. The checked executable matches the original three workloads. All 138 final candidate requests match their tokens and all 4,416 complete-logit hashes. Operator memcheck/racecheck/synccheck and full-model memcheck report zero errors, warnings or hazards. Formatting and Python syntax checks pass.

The restored optimized executable matches the tested/measured binary: `e01c85e654b89913b83d0b7bd73274a78c52b015ee45949cbb0d8b2eb8ac77c2`. Sampled process GPU memory remains 10,256 MiB. Two pinned host slots consume 1,215,496 bytes and replace temporary transfer vectors; this is not extra VRAM.

The final profile reduces norm/quantization from 4.436 to 3.907 us and standalone quantization from 2.948 to 2.383 us on average. Decode projections remain the main limit, at 61.4% of instrumented GPU time. The larger structural experiments did not beat this bandwidth-efficient path. The independent Torch full-model numerical discrepancy remains unresolved; none of these checks replaces that gate or establishes cross-engine accuracy equivalence.


Refreshed stock B12X decode is 144.408 / 143.343 / 142.862 tokens/s; the retained Rust candidate leads by 17.5% / 14.8% / 11.6%. Refreshed FA2 API medians are 244.579 / 237.378 / 243.265 ms, versus Rust 191.081 / 207.475 / 238.948 ms. The long-prompt margin is still modest (1.8% fresh, 1.5% versus the earlier stock best), while both shorter requests remain clearly faster. The engines share weights and workload shapes; they do not have established accuracy equivalence. Stock software/settings are unchanged from the previous pass.
