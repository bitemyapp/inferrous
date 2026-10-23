# RTX PRO 6000 certified-head model pilot

The nine-product certified head is retained for exactly thirty-two outputs when head tuning is explicitly enabled on RTX PRO 6000. It improves several batch-thirty-two workloads against the preceding exact grouped head while preserving every checked vocabulary logit. Model, native-capacity, integrated sanitizer and rebuilt-CLI checks pass. This is a focused paired pilot on `ssh wx-workstation`, not the final stock-engine matrix.

## Measurement scope

Baseline executable is `d54b9f2e4c81a29cc180d40ea590deb5568884ed302cec0344dd799050eb0e1a`; candidate is `da73a3b486c9704b15254430d1d9eb9ff5bc4322bd7f50e59c979e0c3ee4d74b`. Both use the native 32,768-token bundle, 12 GiB KV, maximum batch thirty-two, chunk 512, graphs, 4 GiB score scratch, and the same explicitly enabled tuning settings. The candidate changes only the thirty-two-output head on RTX PRO 6000. Each mode runs A/B/B/A, with three excluded warmups and ten measured repetitions per suite occurrence: twenty measured samples per executable/case. Full-logit transfer/hashing is disabled for timing and enabled in separate verification probes.

Cold mode disables prefix caching. Prefix mode enables it and checks reuse on eligible repeated requests. Five-token prompts have no reusable complete 64-token page, so their prefix-labelled results are not prefix-hit measurements. The 129-token cases request eight outputs; the other cases request the `o32` or `o2` counts in their names. Recompute mode submits the entire growing prefix for every output and checks processed-token accounting. Model loading/preparation is recorded separately from generation.

## Generation latency

| Mode | Workload | Baseline, ms | Candidate, ms | Candidate / baseline |
|---|---|---:|---:|---:|
| cold | b16-p129 | 214.737 | 214.023 | 0.99668 |
| cold | b32-p129 | 378.600 | 363.331 | 0.95967 |
| cold | b32-p5-o32 | 445.231 | 417.204 | 0.93705 |
| cold | b32-p513-o2 | 1243.003 | 1230.458 | 0.98991 |
| prefix | b16-p129 | 82.922 | 82.523 | 0.99519 |
| prefix | b32-p129 | 112.239 | 100.655 | 0.89679 |
| prefix | b32-p5-o32 | 444.165 | 417.832 | 0.94071 |
| prefix | b32-p513-o2 | 31.010 | 28.611 | 0.92265 |
| recompute | b16-p129 | 1182.165 | 1185.452 | 1.00278 |
| recompute | b32-p129 | 2338.307 | 2301.382 | 0.98421 |
| recompute | b32-p5-o32 | 1676.902 | 1646.200 | 0.98169 |
| recompute | b32-p513-o2 | 2474.906 | 2474.201 | 0.99972 |

Both candidate arm medians beat both baseline arm medians in eight of the nine batch-thirty-two cases. The strongest measured improvement is 10.3% for warm-prefix 129-token requests; cold 129-token and cold five-token/thirty-two-output cases improve 4.0% and 6.3%. Cold 513-token/two-output latency improves about 1%; its recompute case is effectively unchanged. Batch-sixteen dispatch is unchanged and its sub-0.5% differences do not establish a tuning benefit. Recompute baseline drift is visible, so the complete per-arm samples and medians remain part of the evidence.

The telemetry contains 1,112 samples at GPU utilization at least 50% inside timing-suite process intervals, with SM clocks 1,515–2,370 MHz and temperatures 71–86°C. Those intervals also include separately recorded startup; these ranges are environmental context, not an assertion that every measured generation had identical clocks. No builds overlap the timing campaign.

## Storage and startup tradeoff

The candidate adds 1,873,067,008 bytes of persistent integer weight planes and metadata, plus 39,897,856 bytes per active thirty-two-output certificate workspace. Prefill and decode can each retain one workspace. Median startup across the six timing suites per executable is 8.193 seconds for the baseline and 13.518 seconds for the candidate, an additional 5.325 seconds. Those startup costs are excluded from the generation table and must remain visible in any retention decision. Release and checked native-session tests both pass with the configured 64 GiB KV allocation.

## Correctness and sanitizer boundary

The seven CLI verification files cover both executables in cold/prefix/recompute modes plus a candidate 64 GiB run. Ten workloads contain 1,656 complete vocabulary hashes per repetition, with two repetitions; all hashes and tokens match. This extends the previous nine-workload probe with thirty-two 513-token requests producing two outputs. Release/checked ordinary suites each pass twenty-two tests, including independent integer-center/original-order operator comparisons and full-model chunk/graph/cache/order invariance. Both explicit native tests preserve all preparation and continuation hashes through 32,767 tokens and release the cache pages. These checks preserve original arithmetic; they do not resolve the separate independent Torch numerical gate.

Complete operator memcheck/racecheck pass. Unfiltered full-model memcheck of thirty-two five-token requests with two outputs also passes, preserves all sixty-four original hashes, and reports zero errors. Its audited inventory contains 1,310 launches, including eight launches across the four new families, once eager and once in a graph. The exhaustive partition for those eight launches completes in 163.94 seconds with zero race errors/warnings, preserves all sixty-four original hashes, and peaks at 13,018,066,944 bytes of process-group RSS. The maintained verifier rechecks exact kernel filters, launch order/shapes, eager/graph classification and disjoint coverage. Unchanged families retain their separately recorded baseline coverage; this is not a new unfiltered full-model racecheck pass. Launch logging produces the previously audited informational-summary exit 99, with no error/warning diagnostics.

## Artifacts

Raw evidence is under workstation `clean-work/pro6000-baseline/head-certified-model-paired-v2/`: summary SHA-256 `abf88e649166a2b2708737f24e786ca85e38460640464a5436362c108520e4af`, identity `1655511228755e28ddddc730275b13829bd684ab3b5c346a8dde3a083098ba78`, full-hash verification `9dc08b8f19eed856ec684472adf5d6db193073ee0cfe8581dc7185e5124e4eea`, and telemetry `9f526378a6b04ddaf200560e94039aab8949ac22961f43a153e3d2c9c36ba6b0`. All pinned source/binary identities and raw result hashes were rechecked after completion. The [certification record](../../research/exact-head-certification.md#isolated-model-integration) identifies source, release/checked binaries and model/native validation. Full-model memcheck validation is `head-certified-model-sanitizers-v1/memcheck/validation.json`, SHA-256 `a8073ab2b8b4dd6315ae2dc3499224a88a25730a56543d1d19b5ccb60447d151`; inventory validation is `b912b0dfdd2aeca31ff7be2cdf79834ae301a1bcf28cef5d828d21df42e20ed0`.

Integrated sanitizer completion `head-certified-model-sanitizers-v1/summary.json` has SHA-256 `61a54a68224332963b2f560970b177bb00622bd37e3dc66b5d6e66087b454a7a`, identity `061c125ce02bd51524d2157d5e37831a568361a54802e67213b1e88f9d1e422d`, and new-kernel racecheck validation `523e1864f5a9a6c773f1aaa6a1a42913d5530bb9649367c755a6a1372f6fa069`. All execution receipts, logs, model records and pinned sources/binaries were rechecked after completion.

## Retained integration

The primary-checkout runtime sources exactly match the validated model worktree. The independent operator fixture adds the broader exponent/mantissa patterns from the separately validated packed-weight experiment, while retaining the precomputed implementation. Both rebuilt ordinary suites pass twenty-two tests (69.26/69.98 seconds), and another complete seven-file CLI comparison preserves all 1,656 hashes per repetition across modes and the 64 GiB allocation. The rebuilt CLI is byte-identical to the paired/sanitizer candidate `da73a3b4`; no new timing inference across different executables is necessary. The PRO default still disables head tuning; other hardware profiles and output counts keep their preceding dispatch.

`head-certified-retained-v1/summary.json` has SHA-256 `a031e5c0412e8e59391377b569b487ec18500822b3ff7fa072ecf4baf5be0488`, identity `0a6516831d7d4198f65d7572cf147deb59126e015513b9a5d5dae932cdb51868`, built identity `65b135a9cef7af8d466a8e81fac83c7536d3ef72bc1f8e17f573974bca4bcbca`, regression `8ea91158d8705fb7a0011b40a0584e96db9a60108dfad8d3657fb5e24afd94be`, and source patch `5cf1f749844c2b4330b36fbc1e947b8aca264b0cde8f85a961c8ec93cc58138d`. Rebuilt release/checked test executables are `e266bdd16a9e04fb6eac63b0cab482e5aceb1cd61ca5b697c4439a5932afa896` / `65cff6fbe0c86e8db4717c7218b5b35bd37db0d2cb0b07e1d978b824c2971eae`. Complete pinned inputs and result hashes were rechecked after completion. The earlier native checks apply to the identical runtime sources; they are not mislabeled as fresh native runs of these new test executables.

The allocation-saving packed-weight center passes its operator checks but is rejected: its complete screen is 4.258 ms versus 3.569 ms for precomputed digits. [Its separate record](../../research/exact-head-certification.md#packed-weight-center-follow-up-rejected) preserves the source and results. The original independent numerical discrepancy, fastest-stock requirement in every requested case, and final matrix/report remain unfinished.

## Broader output-count screen

An isolated follow-up screens nine output counts with the retained runtime kernels. Eighty independent comparisons per release/checked build cover ten partial matrix shapes, both center tiles, eager/graph execution and forced/normal fallback. Every integer center and original-order output matches; complete operator memcheck/racecheck report zero errors or warnings. The full-vocabulary screen checks every output and both integer-center tiles, using six captured normalized inputs repeated to the requested row count.

| Outputs | Preceding grouped head, ms | Complete 32-column center, ms | Complete 64-column center, ms |
|---|---:|---:|---:|
| 1 | 0.622624 | 1.554464 | 1.556512 |
| 2 | 0.856128 | 1.622048 | 1.617984 |
| 4 | 0.927808 | 1.794112 | 1.787968 |
| 8 | 1.380384 | 2.067520 | 2.060320 |
| 16 | 2.293792 | 2.492416 | 2.482176 |
| 24 | 3.809312 | 2.794240 | 2.782208 |
| 32 | 4.139008 | 3.552256 | 3.545088 |
| 48 | 6.475776 | 6.029312 | 6.050816 |
| 64 | 8.538112 | 6.895616 | 6.914048 |

These are unpaired operator screens with three excluded warmups and seven measured CUDA-event samples, cache eviction outside timing and no overlapping builds. The certified path loses for one through sixteen outputs. Twenty-four, forty-eight and sixty-four outputs merit model evaluation, but this screen does not extend production dispatch beyond exactly thirty-two. The thirty-two-output preceding grouped baseline is distinct from the already retained certified implementation.

Artifacts are `head-certified-shapes-v1/`: completion SHA-256 `751023bd3deb11b5b9989f98a8a93abbe111628ebf8c8ee221d7766a4cef58de`, identity `cf7a4bc6374f35e683d901c061b0dccf0e9944c5a683202e95eb8707d8e904f0`, source patch `1f300a02f04e7948c94b09f9f6c591d6bf51e78e105c38630641e527748ed0ab`, and profiles `ce51f33adbbd42a4786b5d2045bc1cf90e50a7b91485843d26b233b094ea4dc4`. Pinned sources/binaries, execution receipts, validation records, profiles and telemetry were rechecked remotely after completion.

## Grouped-bound operator follow-up

Partitioning the original-order error bound across contiguous coordinate groups reduces exact fallback. The [bound derivation](../../research/exact-head-certification.md#partitioned-weighted-bound) preserves the original output contract. A CPU screen of eleven layouts over six captured inputs finds no incorrect certifications. Its GPU implementation adds dynamic input-group norms and precomputed FP64 weight-group norms, while keeping the integer center, representation-error allowance and original-order fallback unchanged.

Four hundred independent comparisons per release/checked build cover all preceding shapes and execution modes with whole-row and 4/16/32/64-group bounds. Every integer center and original-order output matches; group metadata is separately checked against host upper bounds. Complete operator memcheck/racecheck pass. Forty full-vocabulary profile records check all original outputs, reset complete-stage output sentinels and compare every integer center.

| Bound | Complete, 16 outputs, ms | Fallback, 16 outputs | Complete, 32 outputs, ms | Fallback, 32 outputs | Certification alone, 32 outputs, ms |
|---|---:|---:|---:|---:|---:|
| Whole row, retained | 2.481152 | 19.9094% | 3.543776 | 20.5754% | 0.116704 |
| 4 contiguous groups | 2.500608 | 18.3778% | 3.496672 | 19.0056% | 0.184320 |
| 16 contiguous groups | 2.588672 | 16.6939% | 3.518464 | 17.2929% | 0.346112 |
| 32 contiguous groups | 2.768896 | 15.8883% | 3.670752 | 16.4619% | 0.555008 |
| 64 contiguous groups | 3.161792 | 15.0228% | 4.029152 | 15.5976% | 0.978944 |

The preceding grouped head measures 2.300960/4.153344 ms at sixteen/thirty-two outputs in this run. Grouped certification does not close the sixteen-output gap. At thirty-two outputs, four groups improve the unpaired complete operator by only about 1.3%; the extra FP64 accumulation cost offsets most fallback savings. Larger group counts are slower despite fewer ambiguous outputs. No grouped-bound model integration or dispatch change is retained. Complete timings include the additional dynamic group-norm kernel; phase timings are separate diagnostics and must not be added as paired measurements. Each group adds 1,215,488 bytes of persistent weight metadata, outside the event interval.

GPU artifacts are `head-certified-groups-v1/`: completion SHA-256 `a22e7b9de5f7f6272fbe40a10ffa1a139b5ae1150a4f3ca405ccadff47d10ad0`, identity `f1e88fc0aa21095d89e2a2a419a59ae827e230c5dfc5bbbcb159c637369d5cc6`, patch `cefeced773261a0e023d104f4be236e0233b1697e7d0b2b934dbf0ac3d900a7d`, profiles `83da9bc7ace20d8a69d41f19acce23529a0ae9b210ea04d9b893964f6eff6ac1`, and telemetry `95310fe3d1c6a958b2a0aba520ccbb2c47bdfe894f26a29542f49f18042cfa7a`. Release/checked executables are `f932721d71e83558d18bf392b0ba01bcf699068760916302282cb5bc3b382382` / `70ad2a46dc06eba29adaf4c5f4b2bd04cd8e20cb0a32e898e2ef25e9384e4350`. All pinned identities and validation receipts were rechecked after completion. CPU artifacts are `head-interval-groups-v1/`, report `77d8a8896219c73cb193ce5c1e37ab3c1f4af294802486dd783d8202415001d5` and receipt `663cce2e409a9b1de14349f731d59102c6684bb8fa94a1b35fbdef4aab161185`.

### Upward FP32 group accumulation

A separate variant replaces only the positive grouped dot product with outward FP32 conversions, products and additions, widening its result exactly before the original FP64 certificate arithmetic. Both builds pass four hundred independent comparisons, now including input/weight rows at both supported nonzero magnitude endpoints. Complete memcheck/racecheck pass, and all forty profile records preserve original outputs. Release and checked disassembly confirms `FMUL.RP` / `FADD.RP` without flush-to-zero modifiers for the group sum.

| Bound | Complete, 16 outputs, ms | Complete, 32 outputs, ms | Certification alone, 32 outputs, ms |
|---|---:|---:|---:|
| Whole row, retained | 2.484224 | 3.546112 | — |
| 4 groups, upward FP32 | 2.494464 | 3.494656 | 0.175840 |
| 16 groups, upward FP32 | 2.563840 | 3.491840 | 0.313344 |
| 32 groups, upward FP32 | 2.710528 | 3.611648 | 0.495616 |
| 64 groups, upward FP32 | 3.031040 | 3.919872 | 0.874496 |

The preceding grouped head measures 2.310176/4.157440 ms. The best thirty-two-output variant improves the retained operator by about 1.5% in this unpaired screen, while all sixteen-output variants and larger group counts still regress. Do not retain this implementation or infer a model gain. The slight widening of the four-group bound adds a few fallback outputs; larger-group fallback fractions round to the preceding table's values. Weight storage and dynamic preparation remain unchanged from the FP64 prototype.

Artifacts are `head-certified-groups-f32-v1/`: completion SHA-256 `09a051f16b663b479742ed4c65c37205d53eb7836f8243b0565351f594f7e396`, identity `e21a6b8b00a3fe0631b29edfdd689861c3a8889c15fc1f3de995302344a39b9a`, patch `5b961adef364930732a96c4f6f5995c66d39e9d72f58997a117fe33c63d3e7c3`, and profiles `14db421d56984087a281db1549bcecb87c7bc0854cb6b5c3da899b148e95b5c0`. Release/checked executables are `b7a9050feddbe950cc7aad3997d944507c4e853711a2c94b03dc2c771b881eca` / `b74429f8585cc9a318d4d2d5889d73fdd0d952a86b3e2ffec162a7e6e6d61532`; grouped-kernel disassembly hashes are `90d2218ee01cd72a3de127a994b956daa125a949e932f822faa9a5f21f0b402d` / `d13c34478cc1f4b1a05f5748babf9260c193d73aca6ed31654340afbabd52ae4`. Complete identities, binary hashes, execution/validation receipts and profile/telemetry hashes were rechecked remotely. Production remains at retained CLI `da73a3b4`; the original numerical acceptance gate and every-case stock target remain unresolved.
