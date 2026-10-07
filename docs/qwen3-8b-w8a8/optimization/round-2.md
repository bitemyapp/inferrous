# Second optimization pass

Starting implementation: e19d9e15 (runtime f10574e1). Target: beat the measured stock INT8 implementations on the same device, checkpoint and requests. Compare decode against Humming/B12X and total latency against Humming/FlashAttention. The unchanged full-model Torch discrepancy remains an independent open issue.

Fresh resident baseline is `round2-baseline.jsonl`; original hashes remain the behavior oracle. All builds, fixtures and GPU measurements run on the benchmark workstation. GPU jobs run serially. Full measurements retain twenty repetitions after three warmups; kernel microbenchmarks guide experiments but do not establish an end-to-end win.

Starting profile (129 prompt, two 32-token requests): projections 62.4% of GPU time, attention 10.4%, norm 8.3%, head 8.2%, quantization 7.8%. Projections and attention dominate the long-context gap; launch overhead remains measurable.

| Candidate | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| Independent DP4A accumulators and wider/coalesced loads | 5 | 4 | 2 | 10 |
| Shared-memory tiled INT8 prefill | 5 | 4 | 4 | 5 |
| Fuse norm and activation quantization without changing rounding | 4 | 5 | 3 | 6.7 |
| Coalesce attention K-cache access, preserving scalar sum order | 4 | 4 | 3 | 5.3 |
| CUDA graph decode with device-resident position | 3 | 4 | 4 | 3 |

Each retained lever gets separate proof, measurements and commit. Integer projection sums may reorder because their absolute bound is 198193152; FP32 scale ordering and BF16 rounding must remain unchanged. Norm/attention changes must preserve their original summation tree/order. Token ties remain lowest index; no RNG is used. Full-model hashes and independent operators gate changes.

## Rejected: sixteen-byte DP4A loads

Four independent integer accumulators and aligned sixteen-byte loads preserve all hashes, but decode stays within noise (113.2 / 106.9 / 91.2 versus fresh 113.8 / 107.3 / 91.3 tokens/s). Projection time stays 292–293 ms in the profile. The code was removed. Memory bandwidth, rather than integer dependency latency, appears to constrain this kernel on this device.

## Retained: shared-memory prefill tiles

A 256-thread block stages 32 activation rows and 64 output channels over 128 K elements, with eight warps computing 16x16 subtiles. Four-word row padding avoids bank conflicts. Loads are coalesced; block barriers separate staging, consumption and reuse. Every output has one owner; partial rows/columns are zero padded. All K dimensions are multiples of 128. This changes only the integer reduction schedule, within the 198193152 bound; FP32 scale order/BF16 cast, ties and RNG remain unchanged. All 69 requests retain exact complete logit hashes.

First-token medians become 12.0 / 41.8 / 134.6 ms; API medians 296.3 / 343.3 / 486.4 ms. Decode stays unchanged. The 512-token request improves 8.9%; the 129-token request improves 1.4%. Independent fixtures include partial 65/129-row tiles and 33 output columns. The reprofile still attributes 61.9% of GPU time to projections, with attention and norm next.

## Retained: fuse hidden RMSNorm and token quantization

Cache the sixteen values owned by each of 256 threads, emit the same BF16 norm values, then quantize them in the same kernel. Warp zero reproduces exactly the original 128/64/32/16/8/4/2/1 floating-point reduction tree. The positive maximum reduction is associative; RN divisions, BF16 boundaries and integer rounding are unchanged. Barriers protect shared sum consumption and max publication. No RNG, ties or dependent graph order changes. The independent norm fixture is also compared bitwise against the two unfused GPU operations, including scales and INT8 values; all 69 full-model requests retain original hashes. Operator racecheck reports zero hazards.

Decode becomes 119.39 / 112.14 / 95.08 tokens/s, API latency 284.1 / 331.6 / 474.0 ms. The profile removes 4,608 launches and reduces the combined norm and quantization GPU time from 94.3 to 82.3 ms. Projection and attention remain the largest costs.

## Retained: coordinate-major key cache

Store K as [head/coordinate, position], while V retains [position, head/coordinate]. Threads reading different keys can now coalesce their K loads. Each cache entry is copied bit-for-bit into a unique in-bounds location; QK/PV operation order and all floating-point operations are identical. Ties/RNG unchanged. Independent logical attention fixtures are packed into the new physical layout; all 69 full requests preserve original hashes, including prefill and successive KV appends.

At 512 prompt tokens, first-token latency falls from 134.0 to 92.6 ms and API latency from 474.0 to 435.3 ms. Shorter requests regress about 1% (284.1 to 287.1 ms at five tokens); the large prefill improvement warrants retaining this layout while continuing to target decode. Decode itself is essentially unchanged.

## Retained: asynchronous sixteen-byte prefill staging

Replace scalar global-load/shared-store sequences with three aligned async copies per thread and K tile. The same shared bytes are populated, including zero-filled partial rows/channels. Explicit 16-byte shared alignment, per-thread completion waits, and CTA barriers protect consumers and subsequent reuse. Integer/FP arithmetic, ties and RNG are unchanged. Independent operators, all 69 hash regressions, and async-copy racecheck pass with zero hazards.

First-token time falls to 20.0 / 58.9 ms at 129 / 512 prompt tokens (from 39.2 / 92.6). API latency is 313.9 / 403.0 ms, versus 332.1 / 435.3. Short-prompt and decode performance are unchanged within about 0.5%. This is direct shared-memory staging; overlapping successive stages remains a separate opportunity.

## Rejected: exact INT8 rounding with a guarded reciprocal

Score 10 (impact 4, confidence 5, effort 2). Quantization consumes about 13% of device time across standalone and fused kernels. The INT8 result is unchanged if an approximate ratio is sufficiently far from a half-integer. PTX specifies at most one ulp of reciprocal error. Our positive scale is normal, its reciprocal is normal, and |x/scale| <= 127.00002. Reciprocal error, RN multiplication and RN division rounding together change the ratio by less than 2^-14. A 2^-10 guard around half-integers therefore safely selects the original RN division whenever integer rounding could differ. Nonfinite scales also retain the original path. Channel/token scales themselves remain bitwise unchanged.

Source: [PTX reciprocal](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp). The proof's BF16 input domain is checked against the original GPU division for every positive finite BF16 maximum and its preceding 1024 BF16 values, with both signs. Smaller values are below maximum/256 and necessarily quantize to zero; the floor only lowers their ratio further. Independent Torch fixtures still check explicit ties, zero rows and exact scale bytes. Full-model hashes gate retention independently of this finite-domain check.

The domain check and all model hashes pass, but the guard costs more than it saves: decode regresses to 115.9 / 109.0 / 93.0 tokens/s. Code and its dedicated test kernel were removed; the measurement and test report remain as evidence.

## Retained: stage attention keys and values

Score 6.7 (impact 5, confidence 4, effort 3). Attention is now the second-largest device cost. Stage 64 keys/values at a time with aligned async copies, replacing repeated global scalar loads with shared-memory loads. Preserve the exact QK coordinate order, each thread's maximum sequence, softmax tree and ascending PV order. The fixed-capacity caches allow staging full tiles; invisible values are never used in arithmetic. A distinct shared query and the score/reduction arrays remain live through their last reads; barriers protect staging reuse.

All independent operators and 69 model regressions pass. Async-copy racecheck reports zero hazards. Decode improves to 121.10 / 117.57 / 107.52 tokens/s; API latency to 280.6 / 298.1 / 372.0 ms. Large prefill itself regresses from 58.9 to 69.7 ms; the larger decode saving reduces the full 32-token request by 31 ms. Both phases remain visible rather than reporting only the favorable metric.

## Retained: increase quantizer thread parallelism

Score 6.7 (impact 4, confidence 5, effort 3). Standalone and fused quantization still consume a substantial fraction of latency. Use 1024 threads per row for their independent coordinates. Preserve the original RMSNorm partial sums and reduction tree in the first 256 threads; only maximum reductions can reorder. The larger block should reduce each thread's serial conversion work, with no change to RN divisions or BF16/integer rounding.

Independent fixtures and all 69 model regressions pass, preserving every complete logit hash. Maximum reduction can reorder because all inputs are finite and its positive floor excludes signed-zero ties; the floating-point sum and all rounding remain identical. Token ties and RNG are unchanged. Decode improves to 128.95 / 124.76 / 113.53 tokens/s, with API latency 264.5 / 282.3 / 356.4 ms.

## Retained for short prefill only: split-K tensor cores

Score 6.7 (impact 5, confidence 4, effort 3). INT8 projections still dominate device time. Let eight warps divide each 16-row by 8-channel tile's K dimension, then add their exact INT32 partial results. Every partial and final sum remains within 12288*127^2 < 2^31; thus integer reassociation cannot change the result. The subsequent floating-point scales and BF16 rounding retain their old order. Ties and RNG are unchanged. The existing DP4A kernel remains the operator oracle.

Using this kernel for single-token decode regressed throughput to 117.2 / 113.9 / 104.3 tokens/s. However, five-token prefill improved from 11.1 to 8.7 ms. Restricting dispatch to 2–31 rows retains that benefit: p5 API latency falls from 264.5 to 261.7 ms, with decode and larger requests unchanged. Both dispatch variants preserve all 69 model regressions; operator fixtures and shared-memory racecheck pass.

## Retained for longer prefixes: parallel QK attention tiles

Score 5.3 (impact 4, confidence 4, effort 3). Decode launches only 32 attention blocks, leaving most SMs unused. Compute independent QK key tiles in separate blocks, then perform the original ordered softmax and PV calculation. QK coordinate order, per-thread maximum/sum sequences, reduction trees and BF16 rounding remain identical. Scores are written once before stream-ordered reads. No ties or RNG change. Original attention stays as the bitwise operator oracle.

Independent operators match the original attention bitwise; all 69 model regressions and async-copy racecheck pass. Splitting adds overhead for short prefixes, so dispatch uses the original attention through 192 visible keys. The 512-token request improves from 356.4 to 343.1 ms and decode from 113.6 to 118.7 tokens/s; shorter requests remain unchanged.

## Retained: reusable decode CUDA graphs

Score 3 (impact 3, confidence 4, effort 4). Preserve the kernel dependency chain and all arithmetic, but capture it once at model load. Device-resident token ID and position allow replay without modifying kernel nodes. Captured scratch and KV addresses remain stable; a mutex serializes requests, and graphs drop before their allocations/module. Each request overwrites its visible KV prefix. Cache capacity/causal visibility, output order, rounding, ties and RNG remain unchanged. Eager execution stays available for tracing and differential tests.

Graph replay matches eager full-logit hashes for twelve changing prompt sizes, including attention/graph boundaries and long-to-short cache reuse, with stop semantics checked after every case. Failed recording releases the stream. All 69 original regressions pass; operator racecheck and graph smoke memcheck report zero errors/hazards. Graph capture/instantiation is included in model load time.

API latency is 223.3 / 243.9 / 303.6 ms; decode 149.73 / 143.46 / 135.83 tokens/s. The shortest workload now exceeds the prior stock anchor; the middle is tied and the longest remains behind. Long-prompt prefill remains 67.8 ms.

## Retained: pipelined PV attention coordinate blocks

Score 5.3 (impact 4, confidence 4, effort 3). In the 512-token profile, attention finish is the second-largest GPU cost and uses only 32 decode blocks. Separate its unchanged softmax from PV, then launch four 32-coordinate blocks per head. Each coordinate still sums visible keys in ascending order. Normalized scores retain exact RN division; only storage and scheduling change. No ties/RNG change.

The initial four-block PV version improves prefill to 64.3 ms but does not improve decode. Adding two alternating staging slots overlaps each next tile's copy with the current tile's ordered sum. All operators, mixed-request graph comparisons, 69 original regressions and async-copy racecheck pass. It yields 137.31 tokens/s on the long prompt, versus 135.83 before the split, with total API latency 300.2 ms versus 303.6. Prefill is 66.7 ms. Shorter requests keep the original fused attention and stay within measurement noise.

## Retained: matrix loads from shared memory

Score 12.5 (impact 5, confidence 5, effort 2). Nsight Compute on the 512-row Q projection identifies MIO queue pressure as 35.5% of warp stall time. Tensor utilization is only 25.1%, with zero spills; this points to the eight scalar shared loads per pair of MMAs. Replace them with two `ldmatrix.x4` operations whose row-address mapping delivers the identical packed INT8 fragments. All 32 lanes participate with aligned, in-bounds shared addresses. Arithmetic, rounding, output ownership, ties and RNG remain unchanged. The NCU report uses its default base-clock profiling setting; only ordinary benchmark runs establish performance, not the counter collection's absolute time.

All operator fixtures, 69 original regressions and shared-memory racecheck pass. First-token latency improves from 20.2 to 18.9 ms at 129 tokens and from 66.7 to 65.2 ms at 512; decode is unchanged. This is a modest gain, so the next experiment increases operand reuse as well as reducing instruction count.

## Retained: larger prefill output tiles

Score 6.7 (impact 5, confidence 4, effort 3). A 64x128 tile assigns each warp a 32x32 output region, reusing each loaded A/B fragment across twice as many outputs. It uses 27 KiB shared staging and 32 integer accumulator registers per thread. Initially select it only for at least 256 prompt rows and 4096 output channels, preserving occupancy on smaller matrices. The same complete K sum, exact INT32 bound, FP32 scale order, BF16 cast, ties and RNG are preserved.

Operator fixtures, 69 original regressions and async-copy racecheck pass. The 512-token first step falls from 65.2 to 53.5 ms; total API time from 298.7 to 287.1 ms. Decode and shorter prompts are unchanged. Nsight Compute reports 72 registers/thread and shifts the leading stall from shared-memory queue pressure to long scoreboard dependencies (43.4%), motivating prefetch.

## Rejected: pipeline large prefill staging

Score 6.7 (impact 5, confidence 4, effort 3). Use two alternating K=64 stages with 20-word padded rows, within the static shared-memory limit. Copy the next stage while consuming the current one. Barriers protect each stage's readiness and last read before reuse; the identical packed fragments feed the identical ordered MMA sequence. Integer/FP outputs, rounding, ties and RNG are unchanged.

The corrected two-stage kernel preserves all operator and model hashes, but increases first-token latency from 53.5 to 56.3 ms. Removed the candidate; retained the measurements and restored the previous large tile.

## Retained: share attention tiles across grouped query heads

Score 6.7 (impact 5, confidence 4, effort 3). The four Q heads in each GQA group read identical K/V tiles. For prefill with at most 512 visible keys, a 512-thread block can share those tiles across all four heads while retaining separate query/score/reduction arrays. This cuts staging traffic and exposes more active warps per SM. Every head retains its original QK coordinate order, softmax maximum/sum tree and ascending PV order; all rounding, ties and RNG are unchanged.

Operator fixtures, mixed-shape eager/graph comparisons and all 69 original regressions pass. First-token latency improves from 18.9 to 17.8 ms at 129 tokens and from 53.5 to 44.4 ms at 512; API latency reaches 241.4 and 278.2 ms. Decode is unchanged because its existing kernels remain selected.

## Retained: overlap CPU validation hashes with GPU execution

Score 15 (impact 3, confidence 5, effort 1). After allocation reuse, the gap between inner request time and public API time is about 7–8 ms per 32-token request, predominantly serial serialization/SHA-256 of full-vocabulary logits. Keep one completed logit vector and hash it after enqueueing the next GPU step, before waiting for that step's logits. The final vector is hashed after the loop. Exactly the same little-endian bytes enter SHA-256, in the same step order; token selection, stop handling, GPU operations, rounding and RNG remain unchanged.

Mixed-request/stop tests and all 69 regression hashes pass. API latency becomes 216.0 / 233.6 / 270.4 ms, while decode remains 149.73 / 143.82 / 137.39 tokens/s. All full-vocabulary hashes remain included in the public result and API timing.

## Retained: GPU greedy selection

Score 4 (impact 2, confidence 4, effort 2). An isolated CPU measurement on the benchmark workstation puts the existing full-vocabulary finite check/argmax at 75.4 microseconds. An eight-lane CPU version saves only 20 microseconds. Test a GPU reduction using lexicographic order (largest value, lowest index), plus a nonfinite flag. The flag preserves the existing rejection of NaN/infinity; signed-zero ties keep the lowest index. Every logit and hash remains returned unchanged.

Independent tie/nonfinite cases, mixed-request graph comparisons and all 69 original regressions pass. Decode becomes 150.85 / 144.85 / 138.25 tokens/s; API latency 214.4 / 232.0 / 270.2 ms. Long-prompt first-token time varied upward by 1.2 ms, offsetting most of its decode gain in the full request measurement.

## Retained: fuse head normalization and RoPE

Score 6 (impact 3, confidence 4, effort 2). Replace the two launches per Q/K head with an in-place 128-thread block. Reproduce the original 256-thread RMS sum, including its zero upper half, then retain the intermediate BF16 norm rounding before RoPE. All head coordinates are staged before any in-place write; barriers protect the sum's consumption and shared reuse. FP order/rounding, output ownership, token ties and RNG remain unchanged.

The fused operator matches the old norm-then-RoPE sequence bitwise, including negative/zero gains and the last table position. Full mixed-request comparisons and all 69 baseline regressions pass. Decode improves to 152.46 / 146.47 / 139.59 tokens/s; API latency to 212.1 / 229.7 / 268.3 ms.

## Retained: unroll the ordered PV loop

Score 8 (impact 4, confidence 4, effort 2). The attention profile still identifies warp stalls around its shared-memory computation/barriers. Expose eight consecutive PV iterations to the compiler so independent loads can overlap. Hoist the tile's active length; unrolling preserves every RN multiply/add and the ascending key order, with the same guarded remainder. Ties/RNG unchanged.

Independent operators and all 69 full-model regressions retain exact hashes. Decode improves to 152.95 / 148.84 / 144.75 tokens/s, now ahead of the earlier stock decode anchors on all three workloads. API latency is 211.4 / 225.8 / 257.3 ms, so long-prompt total latency still trails stock. The long first step is 42.9 ms. Stock will be rerun before the final comparison.

## Retained: compute the MLP activation maximum in its producer

Score 6 (impact 3, confidence 4, effort 2). The 12288-wide decode quantizer still puts all coordinate conversions into one CTA. Have SwiGLU publish its positive absolute maximum with warp reductions and device-scope atomic max; then quantize independent coordinates across many blocks. Hidden norm resets the scratch maximum before each MLP. Keep prefill's existing path. The positive float bit order equals unsigned integer order; ignore NaNs just as f32::max does. The 1e-8 floor, BF16 activation, RN division and integer rounding remain identical, as do ties/RNG. Stream ordering separates reset, atomic updates and the later read, with no concurrent ordinary/atomic access.

Independent quantizer/scale fixtures, a checked SwiGLU maximum, mixed-request graph comparisons and all 69 baseline regressions pass. Decode becomes 155.26 / 150.89 / 146.64 tokens/s; API latency 208.4 / 223.1 / 254.5 ms. Prefill is unchanged. This uses one device-scope atomic update per warp only on a single 12288-coordinate row; larger prefill keeps the existing path.

## Retained: combine Q/K preparation and cache append

Score 6 (impact 3, confidence 4, effort 2). Q/K head normalization/RoPE and cache append still execute as three serial kernels. Map one combined grid to 32 Q heads and eight K heads per token. The K blocks also append their K/V coordinates. Every arithmetic operation and BF16 boundary remains as in the verified fused norm/RoPE, with distinct coordinate ownership and the same cache layout/positions. Stream ordering publishes all appends before attention; no tie, RNG or output-order changes.

The combined operation matches separate norm/RoPE/cache operations, including all cache bytes at the final table positions. Mixed-request tests, all 69 model regressions and shared-memory racecheck pass. Decode reaches 157.46 / 153.01 / 148.64 tokens/s, API latency 205.4 / 220.7 / 252.9 ms. A follow-up dispatch that restored separate prefill launches did not improve prefill and was removed (`prepareqkdispatch.jsonl`). The combined path is retained for every shape.

## Rejected: wider pipelined prefill stages

Score 6.7 (impact 5, confidence 4, effort 3). The earlier K=64 pipeline lost performance. Test K=96 stages, giving more MMA work per prefetch while keeping two 21-KiB stages within the 48-KiB static limit. The last K tile zero-fills missing 16-byte fragments; model widths are multiples of 32. Each original integer term appears once, followed only by zero padding, so scaling/rounding and every output, tie and RNG decision remain unchanged. Barriers protect completion and alternating-slot reuse.

The K=96 pipeline preserves all hashes but raises the long first step from 44.1 to 51.1 ms (API 252.9 to 259.8 ms). It was removed; the single-stage K=128 implementation remains selected.

## Rejected: unroll the output-head dot product

Score 6 (impact 3, confidence 4, effort 2). The output head consumes 9.3% of device time in the latest complete profile. Expose eight consecutive loads per lane to the compiler, retaining the exact ordered RN additions and warp reduction. The loop is still one 4096-coordinate dot product per vocabulary entry; output ownership, BF16 rounding, token ties and RNG remain unchanged.

All original logit hashes pass, but decode regresses to 156.72 / 152.42 / 148.18 tokens/s and API latency to 206.3 / 222.5 / 255.1 ms. Removed.

## Rejected: replay the current prefill shape

Score 4 (impact 3, confidence 4, effort 3). Prefill still submits the whole 36-layer graph from the host. Capture the current prompt shape once and replay it for subsequent requests of that shape, with persistent token input storage. A shape change drops the old graph before replacing its scratch/input; the execution pool drops all graphs before their captured allocations. Capture cost remains inside the public API on the first request of a new shape; the steady-state benchmark's three warmups exclude it, as they do other cold costs. Traced and differential eager requests retain the same explicit kernel path. No kernel, operation order, BF16 boundary, tie or RNG change is involved.

The thirteen mixed-input eager/replay comparisons and all original logit hashes pass. Short prefill improves from 8.3 to 7.5 ms, but decode falls about 0.7% and total latency rises to 205.9 / 222.1 / 254.9 ms. Removed the extra cache and capture lifecycle complexity because there is no end-to-end gain.

## Rejected: fuse residual addition with normalization

Score 6 (impact 3, confidence 4, effort 2). Norm/quantization is a leading remaining decode cost. Fold the immediately preceding residual addition into its value-loading phase, retaining its explicit BF16 rounding. This removes 71 serial launches per generated token and their intermediate reads. A shared inline helper preserves the existing sum tree, max reduction, RN divisions, BF16 boundaries and INT8 rounding. The final layer still uses the original standalone residual/final norm. Trace tensors, token ties and RNG are unchanged.

Independent operators, mixed requests and every original logit hash pass, but API latency is 206.1 / 222.0 / 254.6 ms and decode 156.83 / 152.66 / 148.32 tokens/s. A subsequent control run is 206.6 / 222.7 / 255.5 ms, so the differences are too small to justify the added complexity. The candidate was removed.

A fresh complete profile (`reanchor-profile-summary.csv`) still attributes 59.4% of device time to DP4A projections, 9.8% to the output head and 8.7% to large prefill matrices. The control run decodes at 156.51 / 152.32 / 148.04 tokens/s.

## Retained: group sibling decode projections

Score 6.7 (impact 5, confidence 4, effort 3). DP4A projections remain the largest GPU cost. Q/K/V share one activation row, as do gate/up; let independent warps for those siblings run in one grid. This removes three launches per layer and allows small K/V projections to occupy SMs alongside Q. Each warp still performs the exact original packed DP4A loop, warp sum, scale multiplies and BF16 cast. Output buffers remain separate; no weight repacking or precision change occurs. Prefill retains its tiled matrix kernels. Ties and RNG are unchanged.

Independent projection fixtures, mixed-request graph tests and all 69 original regressions pass. Decode improves from the fresh control's 156.51 / 152.32 / 148.04 to 160.91 / 156.43 / 151.82 tokens/s. API latency falls from 206.6 / 222.7 / 255.5 to 201.2 / 217.4 / 250.4 ms. Prefill is unchanged.

The complete profile reduces projection GPU time from 292.6 to 281.0 ms; grouped projections remain the leading cost. Racecheck reports zero hazards.

## Retained: eight-warps per decode projection block

Score 7.5 (impact 5, confidence 3, effort 2). Test 256-thread blocks for all DP4A decode projections. Each global warp still owns the identical dot product; there is no block-local state or barrier. Grid tails return as complete warps. This changes scheduling/occupancy only, preserving all arithmetic and ties/RNG.

All 69 full-model regressions pass. The gain is small and consistent across these three runs: API latency 200.8 / 217.0 / 249.7 ms, decode 161.23 / 156.73 / 152.17 tokens/s. This is a scheduling parameter change with no added runtime branch or kernel complexity.

## Rejected: unroll ordered query/key dot products

Score 8 (impact 4, confidence 4, effort 2). Attention remains a leading cost, particularly during prefill. As with the retained PV unroll, expose eight successive QK coordinate loads to hide shared-memory load latency. Every RN multiply/add stays in its original ascending coordinate order, as do scaling, softmax, BF16 rounding, token ties and RNG. No new memory ownership is involved.

Operators and all full-model hashes pass, but total latency worsens to 202.1 / 220.4 / 250.2 ms, with decode 160.18 / 154.08 / 151.96 tokens/s. Removed.

## Rejected: smaller row tiles for large prefill matrices

Score 10 (impact 5, confidence 4, effort 2). The large-matrix NCU profile showed only 0.45 grid waves per SM for a 512x4096 projection and 72 registers per thread. Test 32x128 output tiles: each warp owns 16x32 instead of 32x32, halving its accumulator count and doubling the grid. The complete exact INT32 K sum and subsequent scale/BF16 operations stay unchanged; output tiles remain disjoint. This trades additional operand loads for more active warps. Ties/RNG unchanged.

The standalone 32x128 tile preserves all hashes but regresses long prefill to 51.8 ms and API latency to 256.2 ms. It is rejected as a standalone replacement.

## Rejected: pipeline full K stages with the smaller row tile

Score 6.7 (impact 5, confidence 4, effort 3). The smaller row tile makes two full K=128 stages fit in 45 KiB. Test it with next-stage copies overlapping current-stage MMAs. Unlike the earlier K=64/96 pipelines, this retains the original number of stages/barriers per dot. Zero-filled matrix tails and the exact integer sum/scaling/BF16 cast are unchanged. Completion waits and CTA barriers protect consumption and alternating staging-slot reuse. Compare to the retained 64x128 single-stage implementation, not the rejected 32x128 tile.

The pipelined 32x128 candidate also retains every hash, but long-request API latency rises to about 257 ms. Both 32-row variants were removed; the retained 64x128 single-stage kernel remains selected.

## Final validation and comparison

The retained runtime (`94e50806`, plus validation/comment cleanup) measures 161.02 / 156.54 / 151.97 decode tokens/s and 201.1 / 217.3 / 250.1 ms API latency. Refreshed stock B12X decode is 144.44 / 143.35 / 142.94 tokens/s; stock FA2 API latency is 236.9 / 237.4 / 242.7 ms. Decode wins on every measured workload; total latency wins on the shorter two and remains 3.0% behind on the longest. The independent Torch full-model numerical gate remains unresolved.

All 2,208 original benchmark logit hashes pass. Nineteen varying-prefix requests at seventeen lengths through 4093 match eager execution, including every graph capacity and stop handling. All five GPU/unit tests pass. Checked device indexing matches each workload; operator/argmax memcheck, racecheck and synccheck and full-model smoke memcheck report zero errors/hazards. Formatting and Python syntax pass. The restored optimized binary matches the measured executable's SHA-256. A separate memory run samples 10,548 MiB, versus the earlier 10,286 MiB; these are sampled maxima rather than exact allocation peaks.

The final node-level profile identifies decode projections (58.4%), BF16 head (10.1%) and prefill matrices (9.8%) as the main remaining device costs. Profiles and raw measurements are retained; slower trials are absent from runtime code.
