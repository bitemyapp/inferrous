# Exact-output certification for the packed output head

The precomputed nine-product variant is retained for exactly thirty-two outputs when head tuning is explicitly enabled on RTX PRO 6000. It computes a fast integer approximation, accepts only outputs whose complete error interval rounds to one BF16 value, and evaluates all ambiguous outputs with the original ordered FP32 head. The target is every vocabulary logit, not just the greedy token. All preparation, builds, tests and measurements run on `ssh wx-workstation`. The experiments below establish the numerical contract and reject slower alternatives; [the model result record](../qwen3-8b-w8a8/results/pro6000-certified-head.md) gives paired gains, storage/startup costs and retained-integration evidence.

## Supported domain and exact fallback

Certification requires every input and weight value in the corresponding row to be finite BF16, represented exactly in FP32, and either zero or of absolute magnitude between `2^-50` and `2^30`, inclusive. Dynamic input preparation checks the BF16 low bits and exponent bounds. Host preparation checks the original BF16 weights. An unsupported input row or weight row gets infinite certification metadata and always uses the exact fallback. This guard is part of the arithmetic contract; successful tests on ordinary model values cannot replace it.

Within this domain, a BF16-by-BF16 product has at most sixteen significant bits and is exact in FP32. Nonzero products are at least `2^-100`, and all products lie on a grid no finer than `2^-114`. Cancellation therefore cannot introduce a nonzero subnormal partial sum. The absolute complete sum is at most `4096 * 2^60 = 2^72`, preventing overflow. Kernels retain separate FP32 multiply/add instructions with `--no-fmad` and the original descending warp reduction. Unsupported values are recomputed by those same original operations, without using the approximation to choose a token or a replacement hash.

## Integer center

The six-product version uses a power-of-two scale and nearest-even integer quantization with maximum magnitude 1,040,319. Each integer is represented by three signed seven-bit digits in base 128: `q = d0 + 128*d1 + 16384*d2`, with digits in `[-64, 63]`. Balanced division uses `floor((q+64)/128)`; input preparation chooses a scale that bounds the integer magnitude, and independent host comparisons check the resulting digits. Pairwise digit sums lie in `[-128, 126]` and fit signed INT8.

Let `D0`, `D1`, and `D2` be the three diagonal component dots. Let `S01`, `S02`, and `S12` be dots of the corresponding pairwise digit sums. The full integer dot is reconstructed exactly as:

```text
D0
+ 128 * (S01 - D0 - D1)
+ 16384 * (S02 - D0 - D2 + D1)
+ 2097152 * (S12 - D1 - D2)
+ 268435456 * D2
```

Every component accumulation fits signed 32-bit at width 4096. Reconstruction uses signed 64-bit, with a complete absolute bound below `2^53`. Conversion to FP64 is exact, as is multiplication by the two power-of-two row scales in the guarded domain. This center therefore needs no floating-point tensor-core error assumption. The fused operator stages three input and weight planes, forms packed-byte pairwise sums without carries between byte lanes, and computes six integer matrix products.

## Error interval

Write the original rows as `x` and `w`, and their scaled integer approximations as `a` and `b`. The representation error is bounded by:

```text
R = ||x-a||₂ ||w||₂ + ||a||₂ ||w-b||₂
```

For the original head, coordinate `k` enters lane `k % 32` at term `j = floor(k/32)`. It passes through at most `133-j` FP32 additions, including the five-level warp tree. Counting the initial exact addition to zero is conservative. With `u = 2^-24`, `gamma_n = n*u/(1-n*u)` and `n_j = 133-j`, the summation error is bounded by:

```text
sum_k gamma_(n_j) * |x[k] w[k]|
<= u/(1-133u) * sum_k n_j * |x[k] w[k]|
<= u/(1-133u) * sqrt(sum_k n_j*x[k]^2) * sqrt(sum_k n_j*w[k]^2)
```

The last step is weighted Cauchy–Schwarz. Adding this legacy-summation allowance to `R` encloses the original ordered FP32 result around the integer center. It is deliberately conservative and can require fallback even when the approximate center already rounds correctly.

GPU input metadata uses upward-rounded FP64 products, sums and square roots. The original/approximation difference is exact: the approximation is zero or within a factor of two of the original value, so Sterbenz's subtraction condition applies. Host weight metadata uses exact FP64 squared products followed by positive reductions; a `1 + 2^-39` inflation exceeds the 4096-term FP64 summation allowance, with subsequent multiplication and square-root results advanced outward. The legacy coefficient is rounded upward. Bound products and sums use upward-rounded FP64 instructions, and lower/upper endpoints use downward/upward FP64 addition. Conversion to FP32 is expanded by one representable value in the corresponding direction before BF16 rounding.

An output is accepted only when both BF16 endpoints have the same finite bit pattern. A warp ballot records all remaining outputs in a per-column, per-32-row mask. The fallback computes only those outputs, with lossless packed weights and the original lane term order, warp reduction and BF16 rounding. Certified and fallback writes are disjoint and together cover the complete output matrix.

## Initial complete-pipeline evidence

`head-certified-v1/` on the workstation contains the first complete operator implementation. Both release and checked builds pass twenty independent original-order comparisons across row counts 1/3/16/32/33, partial vocabulary tails, eager/graph execution, forced fallback, unsupported input/weight domains, reversed selections and fresh output guards. Complete memcheck and racecheck pass with zero errors or hazards/warnings. The captured-input screen compares every vocabulary output with the retained 32-row packed head, using the original weights and six previously recorded normalized rows repeated to thirty-two inputs. This is not a representative model batch or a model-level performance result.

| Stage | Median, ms |
|---|---:|
| Complete certified pipeline | 4.140032 |
| Retained 32-row head | 4.120576 |
| Dynamic input preparation | 0.026624 |
| Integer center | 1.273856 |
| Certification | 0.106496 |
| Exact fallback, four outputs per reconstruction | 2.758368 |

Fallback covers 28.25% of outputs. The complete path is effectively at parity in this unpaired screen and is not selected for production. Phase timings are separate diagnostics and must not be summed into an end-to-end claim. Each record has three warmups and seven measured CUDA events, with cache eviction outside the events. The next isolated experiment compares compact fallback widths four, eight and sixteen; it also resets full-vocabulary output sentinels before each complete-path profile.

The v1 completion summary has SHA-256 `e871ba675ee62d5beebe8a9fadf2a42512f81dfd6f2162a4176c0199ef3f2395`, profile summary `347aa8723ad1217d55c545f482071ce84669226054593af490dbf726fecfeed0`, and identity manifest `1e86b557773f8a58d11e2351e0be57ee04dbd52d34fb61afc92e7c1a3d7d7a00`. Release/checked executables are `102c2d4a3b49ea425bfc44f5ff5eb62d9095672bc168d2ac7e414ddbf21bd8e6` / `153a49807c4ed33efe25311c802a72e46f4803ceb82dd6f82ed0de3efe3c98c6`. The full manifest, source identities, binaries and validation records were rechecked after completion. Retention would additionally require a useful paired gain, broader input coverage, full-model original-logit comparisons, native/capacity validation and applicable sanitizer evidence.

## Wider fallback experiment: rejected

Compact eight- and sixteen-output fallbacks also preserve all outputs and pass sixty independent comparisons per build plus complete memcheck/racecheck. The complete-path profiles reset every output to a NaN sentinel before execution and still match all original full-vocabulary results. Both wider fallbacks lose to four-output compaction on the captured-input screen; retain neither variant. The sixteen-output version uses 147 registers, compared with 72 for eight and 56 for four. All have zero stack/local storage and 1,024 shared bytes.

| Fallback width | Complete pipeline, ms | Fallback alone, ms |
|---:|---:|---:|
| 4 | 4.142080 | 2.752256 |
| 8 | 5.501952 | 4.122624 |
| 16 | 6.546432 | 5.158912 |
| Retained head baseline | 4.125696 | — |

Artifacts are under `head-certified-wide-v1/`: source patch SHA-256 `319b183b903eb9b712358b16ea28e61f89a49820508275a7b6baacca2fada7df`, identity `9f382ca928ebb808b262b88f2373cc18e23edc5cb779306526c2a9825294ebfe`, completion `c1a7f9b4a4ad3eda0077b285b50202c206a8e27a9bf105719b58b8db597e2a9f`, and profiles `cc9247f4af841dd192a5309d9023b8f14edb70c2ff4c52c6fd8422f5eb6dae02`. Release/checked executables are `f3064a83f266d090951fa79aa0a70c3d8aa28817712b2a98565ad7df922010e0` / `3e72bcf9e06b32dca54e11add31330ee54fd21d67bfba2630c5c5b60a803dffc`.

## Finer nine-product center

The next complete variant uses three signed eight-bit digits in base 256, maximum quantized magnitude 8,355,711, and all nine integer component products. Every signed-32-bit component dot remains bounded. Reconstructing `sum_(i,j) D_ij * 256^(i+j)` in signed 64-bit is exact; its absolute value is below `2^58`. Conversion to FP64 may now round. The certifier therefore adds an upward-rounded allowance of `2^-52 * |rounded dequantized center|`. This safely exceeds the nearest-even conversion error expressed relative to the rounded value. Power-of-two scaling is still exact in the guarded domain. All representation and legacy-summation allowances above remain; the finer quantizer reduces only the representation error.

Two center tiles are implemented: 32-by-32 with 128 threads and 32-by-64 with 256 threads. Both use 160 registers with zero stack/local storage; compiled shared storage is 28,672 / 43,008 bytes. They use the original four-output compact fallback. All new kernels explicitly wait for dependencies, while these operator tests use ordinary full-completion graph edges. The subsequent isolated model integration is recorded below.

Release and checked builds each pass forty independent comparisons, including every reconstructed signed-64-bit center against a direct host integer dot. Both tiles cover partial rows/columns, eager/graph execution, invalid-domain fallback, forced fallback, and fresh output guards. Memcheck and racecheck complete with zero errors or hazards/warnings. The captured-input profile resets full-vocabulary outputs before both complete variants and matches every original logit. Fallback decreases to 20.58% of outputs.

| Stage | Median, ms |
|---|---:|
| Complete, 32-column center tile | 3.550208 |
| Complete, 64-column center tile | 3.543040 |
| Retained 32-row head | 4.122624 |
| Dynamic input preparation | 0.026624 |
| Center, 32-column tile | 1.245216 |
| Center, 64-column tile | 1.241088 |
| Certification | 0.118784 |
| Four-output exact fallback | 2.180864 |

The complete operator is about 14% faster in this screen. The two tile results are effectively close; this unpaired measurement does not establish a robust winner between them. It uses the same six captured rows repeated to thirty-two inputs, not a model benchmark. Broader inputs, isolated model integration, paired timing, additional persistent weight storage and native/capacity acceptance remain required. At that operator checkpoint, production still used the preceding exact packed head.

Artifacts are under `head-certified24-v1/`: source patch SHA-256 `f1e275423f144dc17d36c9123c2e8b456373c7e8f50c3b08f8aa699e1e21abbe`, identity `96fc6d1860310b725b2ac6131c39e77018acbacede83cfc0fd25b8cec6217b4c`, completion `4b1c81f42e7ec8947edf04980d2a1e97afce0b3e7df7fcf8e2447cfe889d1112`, and profiles `127977ce0391b1cd9ab48a7ee065979e2581d907981c1b6a3848839defb16273`. Release/checked executables are `b8ec669b924cffbd4399f3216272f08cebaee61f561d7ffabe24de1dbc7da213` / `15b78cd2e82710e5ef984ef4aeb75e75f571531f6f2e6089b37d5884d4b2ad94`. Complete input/source/binary identities, validation records and telemetry hashes were rechecked remotely after completion.

## Isolated model integration

The frozen workstation worktree `inferrous-head-certified-model`, based on `9f9cedd`, integrates the 64-column center only for exactly thirty-two outputs with the tuned RTX PRO 6000 policy. A private host module prepares the three integer weight planes directly from original BF16 values, avoiding a complete FP32 copy. The original packed head remains available for the exact fallback. Persistent additional weight/metadata storage is 1,873,067,008 bytes; each active thirty-two-output step owns another 39,897,856 bytes of certificate scratch. Prefill and decode can each own a step. Graphs are destroyed before captured buffers, and the existing stream fence protects partially submitted execution. This isolated stage preceded the retained integration recorded below.

Host preparation is checked against the independent FP32/Euclidean-division implementation on every operator fixture and twenty-eight additional rows containing domain endpoints, signed zeros, unsupported finite values, infinities/NaNs and varied signs, exponents and mantissas. Both builds retain forty independent original-output/integer-center comparisons, including forced fallback, graph/eager execution, partial shapes and output guards. The ordinary model test compares every full-logit hash against the untuned original head across two chunk sizes, eager/graph execution, reversed request order, cold/warm reuse, recomputation and a batch shrinking from thirty-two to sixteen outputs. Its twelve records each contain eighty hashes.

| Check | Release | Bounds checked |
|---|---:|---:|
| Ordinary suite | 22 passed; 69.03 s | 22 passed; 70.31 s |
| Explicit native-session continuation | Passed; 156.80 s | Passed; 164.42 s |

Each native run allocates 64 GiB KV and compares thirty-two prepared sessions plus two simultaneous continuation outputs per session against original head arithmetic. Graph/chunk-512 and reordered eager/chunk-128 paths preserve all thirty-two preparation hashes and sixty-four continuation hashes, reach 32,767 tokens in the longest session, and release every cache page. Complete operator memcheck and racecheck also pass with zero errors or hazards/warnings. Subsequent full-model memcheck and exhaustive eager/graph racecheck coverage of all four new families also pass, preserving all sixty-four original hashes. [The model results](../qwen3-8b-w8a8/results/pro6000-certified-head.md#correctness-and-sanitizer-boundary) distinguish this new-kernel partition from unchanged-family baseline coverage.

The original runner completed all release tests but then rejected the combined log because it reused a parser requiring exactly one test. `validation-v2/` corrects that parser and validates the existing successful release log, without repeating the suite or rebuilding. It then completes checked tests, both native runs and operator sanitizers with the same frozen binaries. The original failure and the downstream queue's refusal to proceed are preserved. Completion SHA-256 is `25e0ec073ccddfecab37593fc54de6d1336fce21d958c40e7104b21ccff0577d`, validation identity `9df8c13fa4f1c0170f3e1fd783ccb979db2b845bd8683d417ea1934da5550c92`, built identity `ca560f88a9116ea9626ce1d3a9d722b84b1539d0313ff15dae34054853b0eee5`, and source patch `27c175d170e168ab67737c257004257e279f63cff2abcec6f371a723c3872d3e`. Release/checked test executables are `4cb5716aeb73e74348fba42e25f51430f09b42fb1dc15683ba41b11b50ff9a29` / `61a2368bd98c44b487e3338f5033dc3d6bafd47f4e60412ad6421a86073daac7`; the CLI is `da73a3b486c9704b15254430d1d9eb9ff5bc4322bd7f50e59c979e0c3ee4d74b`. All pinned sources, binaries and completed validation records were rechecked remotely.

`head-certified-model-paired-v2/` completes CLI verification and an A/B/B/A pilot against the retained `d54b9f2e` executable. Verification adds a thirty-two-request, 513-token/two-output workload to the preceding nine cases: all 1,656 hashes per repetition match across two repetitions, both executables, all modes and a candidate 64 GiB run. Timing covers batches sixteen/thirty-two at 129 tokens, batch thirty-two at five tokens/thirty-two outputs, and batch thirty-two at 513 tokens/two outputs, in cold, prefix and recompute modes. Three warmups and ten measurements per arm occurrence yield twenty measured samples per binary/case. [The complete model results](../qwen3-8b-w8a8/results/pro6000-certified-head.md) show up to 10.3% lower latency in the warm-prefix batch-thirty-two case, while batch-sixteen controls and 513-token recomputation are effectively unchanged. Median startup increases from 8.193 to 13.518 seconds and remains excluded from generation latency. Completion SHA-256 is `abf88e649166a2b2708737f24e786ca85e38460640464a5436362c108520e4af`. Full-model memcheck also completes with zero errors and all sixty-four original hashes preserved. Integrated racecheck also passes; the subsequent retained integration is recorded below, while the original independent numerical gate and final every-case stock target remain open.

## Packed-weight center follow-up: rejected

The separate `inferrous-head-certified-packed` worktree tests an operator-only alternative that reconstructs the same three integer weight digits from the existing lossless packed BF16 head inside the center kernel. It would remove the 1,866,989,568-byte persistent digit allocation, leaving the 6,077,440-byte weight metadata. Each warp stages one vocabulary column at a time; each lane decodes four adjacent BF16 coordinates, scales by the exact reciprocal power of two, rounds to the same nearest-even integer, and packs three shared INT8 words. Invalid weight rows produce zero center digits while infinite metadata forces the unchanged exact fallback. A separate, unmodified weight-metadata buffer drives center construction in forced-fallback tests, so deliberately invalidating certification does not change the expected integer center.

The candidate retains the same nine products, signed-64-bit reconstruction and conservative certification interval. Both release and checked builds pass sixty independent comparisons covering the two existing center tiles and the packed center, with eager/graph execution, forced/invalid-domain fallback, tails and fresh guards. Broader exponent/mantissa patterns exercise all three digit planes and packed exceptions. Complete memcheck/racecheck pass. The full-vocabulary profiles reset output sentinels and preserve every original logit; fallback remains 20.58%. Resources are 120/122 registers in release/checked, zero stack/local storage and 43,008 shared bytes.

| Variant | Complete pipeline, ms | Integer center, ms |
|---|---:|---:|
| Precomputed digits, 32-column tile | 3.578880 | 1.247296 |
| Precomputed digits, 64-column tile | 3.568640 | 1.243136 |
| Packed weights, 64-column tile | 4.257536 | 1.923072 |
| Preceding grouped head | 4.204512 | — |

Reject the packed-weight candidate: it is about 19% slower than precomputed digits in this complete-operator screen and also loses to the preceding grouped head. The favorable static register count and reduced allocation do not establish a useful optimization. This remains an unpaired screen on six captured rows repeated to batch thirty-two, with phase timings recorded separately; no packed-weight model integration is performed.

Artifacts under `head-certified-packed-v1/` have completion SHA-256 `31058f628f852f787f96039722fa048b4b8c1c6eb9a9c1fc48bb670f8cbaaf53`, identity `47304aace03bb0e6dbe761b372f15221097bd66b326a5999c9a57d312d62c1a5`, source patch `371e26e0d14a2945fefd6bd99bb6e9ddbafaea255a33e0b2722ddb55c7e3a453`, and profiles `be6216bd3c5c292760b7ea2b5545ebe61ea99b5b452f73ee7d6eeb92a4fba7e7`. Release/checked executables are `3a758199999e494b1e5460c970f4d270e8935e801217731da9049a1d55f95843` / `7827dd7aa33c6a9520fb82cb06db3ce617e1c1db9d2dfc06833aff5821adeafe`. All identities, validation records and profile/telemetry hashes were rechecked after completion.

## Retained implementation

The precomputed 64-column center, dynamic input preparation, certifier and four-output exact fallback are integrated with byte-identical runtime sources to the validated model worktree. Broader exponent/mantissa operator fixtures are retained from the packed-weight experiment, but its slower kernel is not. The rebuilt CLI is byte-identical to the paired/sanitizer candidate `da73a3b4`; release/checked ordinary suites and another complete CLI regression pass. [The retained result record](../qwen3-8b-w8a8/results/pro6000-certified-head.md#retained-integration) provides all identities and boundaries. This selection accepts the documented extra storage/startup cost for the measured generation gains. Default PRO tuning remains disabled, and the independent numerical gate and final stock comparison remain open.
