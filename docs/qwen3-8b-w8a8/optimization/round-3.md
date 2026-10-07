# Third optimization pass

Starting revision: `5607f69a`. Same Qwen3-8B INT8 bundle, RTX PRO 6000 Max-Q, 300 W, one request at a time, 5/129/512 prompt tokens and 32 greedy outputs. All builds, tests, profiling and measurements run on the benchmark workstation. Retain three warmups and twenty measured resident requests. No clock or OS tuning. Original baseline hashes remain the behavior oracle; the independent Torch full-model numerical discrepancy remains open and is not waived.

Fresh baseline (`round3-baseline.jsonl`): 162.09 / 157.47 / 152.89 decode tokens/s; API medians 199.8 / 215.1 / 246.9 ms. The prior pass's executable SHA-256 matches before measurement. Node-level profiling still ranks decode projections first, then the BF16 head, large prefill projections, attention and normalization. The fresh decode NCU report measures 88.93% DRAM throughput for grouped MLP projections, with 31.8/32 bytes used per sector and no spills. This leaves little room for a pure weight-layout change and motivates reducing transferred bytes. See the [2026-10-03 results summary](../results/round-3.md); earlier NCU reports also identify low grid occupancy for 512-row prefill projections.

| Candidate | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| Split large prefill K sums across blocks, then reduce exact INT32 | 5 | 4 | 3 | 6.7 |
| Repack decode weights by channel/K tiles for memory locality | 5 | 3 | 3 | 5 |
| Publish attention maxima in producers for parallel quantization | 3 | 4 | 2 | 6 |
| Combine greedy-result transfer with logit transfer | 2 | 4 | 2 | 4 |

Changes must preserve every integer term, floating-point order and BF16 boundary. INT32 partial sums can reassociate safely: 12288*127^2 = 198193152 < 2^31. Greedy ties remain lowest index; no RNG is used. Reject regressions, keep each retained lever in its own commit, and reprofile after a meaningful change.

## Candidate: split large prefill K sums across blocks

The 64x128 tile's accumulator/shared footprint limits occupancy, and its grid is small for the 512-token workload. Partition K across two independent blocks per output tile, store exact INT32 partials, then sum and apply the original scale multiplications/BF16 cast. Each K element appears exactly once. There is no intermediate floating-point rounding. Partial stores are disjoint; a later stream-ordered kernel consumes them. This trades extra output traffic for more active warps; use ordinary end-to-end timing to decide whether it pays off.

Splitting all large matrices passes every hash but raises long-request latency to 251.8 ms. Restricting it to the 4096-channel projections still takes 249.0 ms, versus 246.9 ms in the fresh baseline. Both variants were removed.

## Retained: lossless packing of BF16 output-head weights

Score 5 (impact 3, confidence 5, effort 3). The head consumes about 10% of GPU time and streams 1.24 GB each token. The model's BF16 exponent histogram shows 99.985% of values use exponents 109 through 123. Represent these with a 4-bit exponent selector plus the unchanged sign/mantissa byte. The reserved selector 15 stores an index into a lossless exception table for each 256-value chunk. Three u32 words hold eight 12-bit codes. This packs the head to 75.80% of its original byte size, including per-chunk exception offsets: 943,404,322 bytes versus 1,244,659,712. The statistics file also retains the initial 128-value chunk estimates, explicitly labeled separately from the final 256-value layout.

Pack once at model load; the external bundle and logical BF16 weights remain unchanged. The GPU reconstructs every original 16-bit pattern before widening and the original ordered multiply/add. No quantization, approximation or FP reassociation occurs. Exhaustive host and GPU round trips over all 65536 BF16 bit patterns, the existing head oracle, and full-logit hashes gate retention.

The first contiguous-code layout was slower (205.5 / 221.7 / 254.5 ms), so it was removed. The retained layout groups 256 logical values as 32 interleaved lane streams: eight codes per lane occupy three coalesced u32 word planes. Each lane loads those three words once and decodes eight successive original terms with constant shifts. This saves address work and redundant word loads.

Ordering and all floating-point operations are identical: each lane still sums coordinates lane, lane+32, ..., lane+4064, followed by the original warp tree and BF16 cast. Tokens, low-index ties and RNG behavior are unchanged. All 2208 complete-logit hashes match the original baseline. All seven tests pass, including the exhaustive host/GPU bit round trips and nineteen mixed requests. First medians: 165.19 / 160.50 / 155.67 decode tokens/s and 196.0 / 212.1 / 245.3 ms API latency. The per-lane layout is retained; its raw evidence is in `headlane.jsonl` and `headlane-regression.json`.

The following control rerun of the unchanged starting binary measures 161.09 / 156.49 / 151.94 tokens/s and 201.0 / 217.3 / 250.1 ms. Its long prefill is 45.8 ms, matching the packed candidate and showing the earlier 43.9 ms was not stable. Packing saves about 5 ms per request versus this adjacent control.

## Candidate: publish attention maxima at the producer

Score 6. Like the retained SwiGLU maximum, publish the unsigned absolute-float maximum of each rounded BF16 attention output, with one atomic max per full warp. Norm1 already resets the scratch maximum to 1e-8 before attention. The next stream-ordered quantizer reads the completed maximum and converts coordinates in independent blocks. Both monolithic attention and split PV handle the single-row path; prefill retains the original quantizer.

All attention arithmetic, dot-product order and rounding are unchanged. Positive float bits have unsigned order; ignore NaNs exactly as f32::max does. The same floor and maximum give the identical scale and INT8 rounding. Ties, RNG, KV visibility and graph dependencies are unchanged. Tests explicitly compare each producer maximum to the maximum of its actual output.

All tests and hashes pass, but the general producer-max variant is rejected: short requests improve by only 0.5–0.6 ms, while the split-PV long request regresses to 247.9 ms (153.49 decode tokens/s). The added cross-block atomic work costs more than the eliminated reduction on that path. Restore the existing quantizer before evaluating the next lever.

## Candidate: transpose the large prefill tile aspect ratio

Score 8 (impact 4, confidence 4, effort 2). Change the 64x128 output tile to 128x64, preserving eight 32x32 warp tiles and the K=128 staging/reduction. The staged A/B sizes exchange places and each thread issues four A copies and two B copies. This halves repeated weight loads across prompt-row tiles, at the cost of more repeated activation loads. The grid size, accumulator count and total shared bytes remain unchanged for the benchmark shapes.

Each output still consumes every integer product once with the same scaling and BF16 cast. Its owner changes to the corresponding transposed tile map; tail predicates still bound all stores and zero-fill staging. The INT32 bound, float order, token ties and RNG are unchanged. Validate partial tiles and full logit hashes before retaining the change.

The 128x64 tile passes all tests and hashes but does not improve long prefill: 45.78 ms versus 45.81 ms for the retained tile. Overall timings are slightly worse, with decode also drifting despite being unchanged. There is no measured benefit from the tile change, so restore 64x128.

## Candidate: reduce attention synchronization with the same sum tree

Score 7.5 (impact 3, confidence 5, effort 2). Each 128-thread attention cohort currently uses eight whole-block barriers for each softmax reduction. Stage all 128 partials, then have the first warp combine the original distance-64 and 32 nodes in registers, followed by distance-16/8/4/2/1 shuffle reductions. One final barrier publishes lane zero's result to the cohort. Preserve the barrier between consuming the maximum and overwriting its shared allocation. Grouped prefill has four disjoint cohorts; all 512 threads participate in the same block barriers and each cohort's first full warp uses the shuffle.

The exact reduction expression and operand order remain unchanged, including max ties/NaNs and sum rounding; only intermediate storage/synchronization changes. QK/PV term order, BF16 casts, greedy ties and RNG are unchanged. Keep the original split attention finish as an independent GPU regression comparison, and require operator fixtures, mixed requests and every original full-logit hash to pass.

All tests and hashes pass, but the synchronization change shows no meaningful end-to-end gain: 196.9 / 213.2 / 245.9 ms, with long prefill 45.66 ms. Restore the existing reductions rather than retain an unproven performance change.

## Retained: fuse decode projection and residual addition

Score 8 (impact 4, confidence 4, effort 2). O and down projections each feed one coordinate-wise residual kernel. Extend the decode DP4A producer to retain its rounded projection output for tracing and also write the rounded residual sum. Each lane-zero writer owns the same coordinate in both distinct output allocations; its residual source is read-only. This removes 72 short launches per decode token. Prefill keeps its existing tensor-core kernel and residual.

The integer dot, scale multiplication, first BF16 rounding, residual operand order and second BF16 rounding are unchanged. No in-place alias is introduced; all three buffers remain distinct. Stream order still precedes the dependent norm and next layer. Tokens, ties and RNG are unchanged. Compare both fused outputs against the original separate GPU operators, then full-model hashes.

All seven tests and all 2208 complete-logit hashes pass. The fused variant measures 166.35 / 161.54 / 156.53 decode tokens/s and 194.8 / 211.0 / 244.0 ms API latency. This improves all three shapes versus packed-head-only timing. The projection trace remains available and bit-identical.

## Final validation

Runtime changes are `b1dcabe7` (lossless head packing) and `d12c405c` (projection/residual fusion). Seven tests pass in optimized and checked device builds, including independent operators, exhaustive BF16 bit reconstruction, failed-capture cleanup and nineteen changing-input eager/graph requests across seventeen prompt lengths through 4093. Checked generation matches the original full-logit hashes on all three benchmark shapes. Operator/argmax/packing memcheck, racecheck and synccheck and a three-step full-model memcheck report zero errors, warnings or hazards. The validation script and logs are retained in the [2026-10-03 results summary](../results/round-3.md).

After restoring the optimized build, SHA-256 matches the measured executable: `22b5511e8d3a45a94b4429e12c7fb5f3a5739933eb8ba033e157b7d16dd39f91`. Rust formatting and Python syntax checks pass. A separate instrumented run samples 10,256 MiB process GPU memory, versus 10,548 MiB in the previous pass; these are sampled maxima, not exact peaks. Loading packs the head once and is outside warm timings. Weight values and the exported bundle are unchanged.

The final node-level profile measures head kernels at 586 us average, versus 758 us in the fresh starting profile. Decode O/down projections now include residual sums at essentially unchanged kernel duration; residual launches fall from 4608 to 144 in this two-request profile (only prefill remains separate). Decode projections occupy 60.2% of GPU time, large prefill matrices 9.3%, packed head 8.0%, and prefill attention 5.0%. These instrumented durations explain the change; ordinary end-to-end measurements determine retention.

None of these checks resolves the pre-existing independent Torch full-model numerical discrepancy. The declared gate is still failing and was not weakened.

The final uninstrumented run (`round3-final.jsonl`, twenty measured requests after three warmups per shape) confirms the retained gain:

| Prompt tokens | Starting control decode tok/s | Final decode tok/s | Starting control API ms | Final API ms |
|---:|---:|---:|---:|---:|
| 5 | 161.09 | 166.31 | 201.0 | 194.8 |
| 129 | 156.49 | 161.63 | 217.3 | 210.9 |
| 512 | 151.94 | 156.50 | 250.1 | 244.1 |

All 69 final requests match all 2208 original complete-logit hashes and token IDs.

Refreshed stock Humming/native-graph B12X decode is 144.38 / 143.27 / 142.86 tokens/s, leaving Rust ahead by 15.2% / 12.8% / 9.6%. Refreshed FA2 wins stock API latency at 243.8 / 237.2 / 244.0 ms. Rust wins both shorter shapes and ties the long fresh run, but the earlier stock long median of 242.7 ms remains slightly faster. The fresh FA2 short median is slower than its previous run; both sets are retained in the comparison rather than treating that drift as an implementation improvement. See the benchmarks for exact timing boundaries.
