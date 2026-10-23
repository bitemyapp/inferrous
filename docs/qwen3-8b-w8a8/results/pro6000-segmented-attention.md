# Fixed-segment attention on RTX PRO 6000

These isolated operator experiments pass independent numerical, query-grouping and sanitizer checks. They improve native-context operator latency across all tested row counts, but lose at short contexts. Separate model validation remains necessary to establish full-model quality, cache/batch invariance and performance. Production dispatch and the legacy hash oracle remain unchanged. See [round 8](../optimization/round-8.md) for the isolated integration and decision history. The first experiment below fixes 512 keys per segment; the subsequent sweep compares three fixed policies.

The kernel uses eight-query tiles, BF16 QK tensor-core products, FP32 softmax and a three-component BF16 expansion of each FP32 probability for FP32-accumulating PV products. Separate blocks evaluate fixed 512-key segments in ascending 64-key page order. They emit unnormalized FP32 output, maximum and denominator; a second kernel merges segments in ascending order and rounds the final output to BF16. Segment boundaries do not depend on query grouping. This is an explicitly different arithmetic path from the ordered scalar baseline, with BF16-representable Q inputs required.

## Validation

Independent CPU Torch FP64 QK/softmax/PV references use the same BF16 inputs and final BF16 rounding. Seven cases cover short, ragged, sharply peaked, native, uniform-native and segment-boundary groups, with shuffled two-sequence pages. Each release/checked build passes all 56 combinations of case, query tile limits 8/3/1, reversed tile order and eager/graph execution. Tests require finite outputs, absolute error at most `2e-4 + 0.01 * abs(reference)`, relative L2 at most `0.005`, bitwise query-grouping/order/graph invariance, untouched output/scratch guards and initialized valid segments. Maximum observed relative L2 is about `9.73e-5`.

Release and checked tests pass in 1.04/0.91 seconds. Ragged and segment-boundary memcheck reports zero errors; racecheck reports zero hazards, errors or warnings and completes in 46.27 seconds. These operator checks do not resolve the independent full-model discrepancy DISC-001.

## Operator timings

All runs execute on `ssh wx-workstation` with its RTX PRO 6000 Blackwell Max-Q. Each configuration has seven CUDA-event measurements after three warmups, with cache eviction outside the measured graph interval. The complete profile contains 189 records across 27 shapes: five existing exact configurations, the earlier online kernel and the segmented candidate. The reference column selects the fastest existing exact configuration at that shape. These measurements include both segmented kernels and their intermediate-memory traffic.

| Query rows | Score stride | Best existing exact, ms | Segmented, ms | Existing / segmented |
|---:|---:|---:|---:|---:|
| 1 | 64 | 0.009952 | 0.018432 | 0.540× |
| 2 | 64 | 0.010240 | 0.018432 | 0.556× |
| 4 | 64 | 0.010240 | 0.020032 | 0.511× |
| 8 | 64 | 0.012288 | 0.020480 | 0.600× |
| 16 | 64 | 0.012288 | 0.022528 | 0.545× |
| 64 | 64 | 0.016384 | 0.022528 | 0.727× |
| 1 | 512 | 0.018432 | 0.088064 | 0.209× |
| 2 | 512 | 0.020480 | 0.089664 | 0.228× |
| 4 | 512 | 0.020480 | 0.090112 | 0.227× |
| 8 | 512 | 0.022496 | 0.094208 | 0.239× |
| 16 | 512 | 0.026336 | 0.091872 | 0.287× |
| 64 | 512 | 0.043008 | 0.092160 | 0.467× |
| 256 | 512 | 0.114688 | 0.116736 | 0.982× |
| 1 | 4096 | 0.075776 | 0.092160 | 0.822× |
| 2 | 4096 | 0.077824 | 0.093920 | 0.829× |
| 4 | 4096 | 0.083968 | 0.094208 | 0.891× |
| 8 | 4096 | 0.100352 | 0.096256 | 1.043× |
| 16 | 4096 | 0.116736 | 0.098304 | 1.187× |
| 64 | 4096 | 0.276512 | 0.229376 | 1.205× |
| 256 | 4096 | 1.122368 | 0.638976 | 1.757× |
| 1 | 32768 | 0.555008 | 0.237568 | 2.336× |
| 2 | 32768 | 0.600064 | 0.239616 | 2.504× |
| 4 | 32768 | 0.679936 | 0.241664 | 2.814× |
| 8 | 32768 | 0.747520 | 0.245760 | 3.042× |
| 16 | 32768 | 0.886784 | 0.370688 | 2.392× |
| 64 | 32768 | 3.156000 | 1.236992 | 2.551× |
| 256 | 32768 | 13.727488 | 4.850368 | 2.830× |

At 256 queries and stride 32,768, FP32 output partials require 256 MiB and maximum/denominator state requires another 4 MiB. This memory cost must be included in any model integration and capacity validation. The short-context losses prevent treating this operator result as a universal replacement. Further fixed-segment experiments preserve separate numerical contracts rather than selecting different arithmetic by transient batch size.

## Artifact identities

Raw logs, frozen executables and source patches remain under `clean-work/pro6000-baseline/` on the workstation. `segmented-attention-summary-v1.json` retains all 56 checks per build, all seven samples per configuration and log hashes. Fixtures remain under `clean-work/segmented-attention-fixtures-v1/`.

| Artifact | SHA-256 |
|---|---|
| `segmented-attention-summary-v1.json` | `329590de66f3ffa3ccc33d7cf534ccd746f624cc91911b5be2f301b81d4f7c6a` |
| `segmented-attention-release-tests-v1` | `0ac3a58036f0772e05b0132862e6247a560e2837733bd2357d38a16eecdc1d0f` |
| `segmented-attention-checked-tests-v1` | `760997601bd4b493f97cc27c4bd40c7a5afbc6af6d04e172614925d9c607dd71` |
| `segmented-attention-source-v1.patch` | `0942690e64582ddb4aa2e73ac1a02ee3af372af60a04a39b408dc6b904679a59` |
| Fixture manifest | `d4d0a25f4f3b630de599052f514d64e7bc8f5eabf2f85b5a01765f39c2b20e54` |

## Fixed-segment sweep: 64, 128 and 512 keys

The subsequent sweep completes all 168 independent-reference/invariance combinations in each release and checked build, and all 120 combinations under each sanitizer. Release/checked tests finish in 2.66/2.57 seconds. Memcheck reports zero errors; racecheck finishes in 127.24 seconds with zero hazards, errors or warnings. Maximum observed relative L2 remains `9.728510335612765e-5`. The complete profile contains 243 records across 27 shapes, with seven event samples after three warmups. Log hashes were rechecked against the terminal summary.

Each segment size defines a separate arithmetic contract. The 128-key variant improves all tested 4K cases and the 512-key variant leads larger native-context shapes. Every variant still loses at stride 64; most stride-512 cases also lose. Selecting a segment size by transient query count would violate the fixed-policy model contract, so these per-shape observations are not a production dispatch recommendation. The isolated model validation uses one fixed size across prefill and decode; its independent numerical, native/capacity, sanitizer and paired timing gates remain separate.

| Query rows | Score stride | Best existing exact, ms | 64 keys, ms | 128 keys, ms | 512 keys, ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.009952 | 0.018144 | 0.018144 | 0.018144 |
| 2 | 64 | 0.010240 | 0.018432 | 0.018144 | 0.018144 |
| 4 | 64 | 0.012000 | 0.018432 | 0.018432 | 0.018432 |
| 8 | 64 | 0.012288 | 0.020480 | 0.020480 | 0.020480 |
| 16 | 64 | 0.012288 | 0.020480 | 0.020480 | 0.020480 |
| 64 | 64 | 0.016384 | 0.022528 | 0.022560 | 0.022528 |
| 1 | 512 | 0.018432 | 0.020480 | 0.030720 | 0.088064 |
| 2 | 512 | 0.020480 | 0.022528 | 0.030720 | 0.088064 |
| 4 | 512 | 0.020480 | 0.022528 | 0.030720 | 0.090112 |
| 8 | 512 | 0.022496 | 0.024576 | 0.032768 | 0.092160 |
| 16 | 512 | 0.026336 | 0.024288 | 0.032512 | 0.091872 |
| 64 | 512 | 0.043008 | 0.051200 | 0.040960 | 0.092160 |
| 256 | 512 | 0.114688 | 0.112640 | 0.104448 | 0.116768 |
| 1 | 4096 | 0.075776 | 0.049152 | 0.045056 | 0.092160 |
| 2 | 4096 | 0.077824 | 0.051200 | 0.045056 | 0.092160 |
| 4 | 4096 | 0.083968 | 0.053248 | 0.047104 | 0.094208 |
| 8 | 4096 | 0.100352 | 0.063488 | 0.050944 | 0.098304 |
| 16 | 4096 | 0.116736 | 0.081920 | 0.079872 | 0.098304 |
| 64 | 4096 | 0.278560 | 0.227296 | 0.202752 | 0.229376 |
| 256 | 4096 | 1.126112 | 0.864256 | 0.679936 | 0.634880 |
| 1 | 32768 | 0.555008 | 0.260096 | 0.223232 | 0.239616 |
| 2 | 32768 | 0.599808 | 0.264192 | 0.223232 | 0.239616 |
| 4 | 32768 | 0.673792 | 0.327680 | 0.229376 | 0.241664 |
| 8 | 32768 | 0.749568 | 0.376832 | 0.256000 | 0.245760 |
| 16 | 32768 | 0.886784 | 0.579584 | 0.452608 | 0.370688 |
| 64 | 32768 | 3.162144 | 1.943552 | 1.579008 | 1.247200 |
| 256 | 32768 | 13.754400 | 7.844864 | 7.009280 | 5.275648 |

`segmented-attention-summary-v2.json` has SHA-256 `8ef72aca62af193b87fbaf2e555b56f040de23fe6e3475ecd3b188d7e3699e12`. Frozen release/checked binaries and the source patch retain the v2 identities recorded in [round 8](../optimization/round-8.md). The fixed-512 v1 measurements above remain separate; the sweep does not replace their raw samples.
