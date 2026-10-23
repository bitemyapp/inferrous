# RTX PRO 6000 asynchronous projections and 32-row head

All builds, checks and timings ran on `ssh wx-workstation`. These kernels preserve integer projection arithmetic or the existing ordered FP32 head arithmetic. The 32-output head was retained for exactly thirty-two outputs on the PRO profile and now serves as the preceding baseline for the [certified head](pro6000-certified-head.md); the projection experiments are rejected. Operator screens use three warmups and seven measured CUDA events; the subsequent paired head model pilot uses twenty measured samples per binary/case. Neither is the final stock comparison matrix.

## Validation

Four asynchronous-copy 64-by-128 projections pass 160 independent integer-reference/guard/eager-graph comparisons in each release and checked build, the existing seventeen-fixture test in both builds, and complete release memcheck/racecheck with zero errors or hazards/warnings. The timing screen contains all 364 expected records across 44 shapes, rotates through 32 distinct weight allocations, and compares every output with DP4A.

The 32-output packed head passes twenty-two independent host FP32-order/guard/eager-graph comparisons in each build and complete memcheck/racecheck with zero errors or hazards/warnings. Its full-vocabulary screen compares all outputs with the sixteen-row staged head at eleven row counts, evicting caches before each timed event. It reconstructs weights once for thirty-two dots and uses one staging slot, trading away copy/compute overlap.

## Projection results: rejected

Every new projection variant loses to the best existing kernel at every eligible tested shape. The table chooses the fastest of the four new variants per shape; this favorable selection still produces no win. The asynchronous-copy proposals are rejected for retention.

| Rows | Output width | Input width | Best existing kernel | Existing median, ms | Best new kernel | New median, ms | New / existing |
|---:|---:|---:|---|---:|---|---:|---:|
| 16 | 1024 | 4096 | `splitk` | 0.008077 | `large-async` | 0.037530 | 4.647 |
| 16 | 4096 | 4096 | `splitk` | 0.012956 | `large-async` | 0.040392 | 3.118 |
| 16 | 4096 | 12288 | `narrow-pipeline` | 0.051718 | `large-async` | 0.122080 | 2.360 |
| 16 | 12288 | 4096 | `narrow-pipeline` | 0.033872 | `large-async` | 0.043210 | 1.276 |
| 32 | 1024 | 4096 | `splitk` | 0.010261 | `large-async` | 0.038112 | 3.714 |
| 32 | 4096 | 4096 | `narrow-pipeline` | 0.016963 | `large-async` | 0.041354 | 2.438 |
| 32 | 4096 | 12288 | `narrow-pipeline` | 0.051397 | `large-async` | 0.121010 | 2.354 |
| 32 | 12288 | 4096 | `tiled-pipeline` | 0.035725 | `large-async` | 0.044751 | 1.253 |
| 64 | 1024 | 4096 | `narrow-pipeline` | 0.014610 | `large-async` | 0.040033 | 2.740 |
| 64 | 4096 | 4096 | `tiled-pipeline` | 0.019528 | `large-async` | 0.042000 | 2.151 |
| 64 | 4096 | 12288 | `tiled-pipeline` | 0.057932 | `large-async` | 0.122224 | 2.110 |
| 64 | 12288 | 4096 | `large-pipeline` | 0.039239 | `large-async` | 0.047378 | 1.207 |
| 128 | 1024 | 4096 | `narrow-pipeline` | 0.015946 | `large-async` | 0.040263 | 2.525 |
| 128 | 4096 | 4096 | `tiled-pipeline` | 0.024655 | `large-async` | 0.042181 | 1.711 |
| 128 | 4096 | 12288 | `large-pipeline` | 0.067316 | `large-async` | 0.118953 | 1.767 |
| 128 | 12288 | 4096 | `large-pipeline` | 0.051587 | `large-async` | 0.059204 | 1.148 |
| 256 | 1024 | 4096 | `narrow-pipeline` | 0.018509 | `large-async` | 0.041931 | 2.265 |
| 256 | 4096 | 4096 | `large-pipeline` | 0.031501 | `large-async` | 0.042763 | 1.358 |
| 256 | 4096 | 12288 | `large-pipeline` | 0.092961 | `large-async` | 0.128114 | 1.378 |
| 256 | 12288 | 4096 | `large` | 0.093531 | `large-async` | 0.102171 | 1.092 |
| 512 | 1024 | 4096 | `tiled-pipeline` | 0.021584 | `large-async-pipeline` | 0.041705 | 1.932 |
| 512 | 4096 | 4096 | `large-pipeline` | 0.055559 | `large-async` | 0.060428 | 1.088 |
| 512 | 4096 | 12288 | `large-pipeline` | 0.161916 | `large-async` | 0.199213 | 1.230 |
| 512 | 12288 | 4096 | `large` | 0.158210 | `large-async` | 0.161593 | 1.021 |
| 1024 | 1024 | 4096 | `large-pipeline` | 0.028172 | `large-async` | 0.042061 | 1.493 |
| 1024 | 4096 | 4096 | `large` | 0.097702 | `large-async` | 0.105815 | 1.083 |
| 1024 | 4096 | 12288 | `large` | 0.278583 | `large-async` | 0.316554 | 1.136 |
| 1024 | 12288 | 4096 | `large` | 0.270325 | `large-async` | 0.291650 | 1.079 |

## Head operator results

The new head is 6.1–7.4% faster at the screened 17–32 and 64 row counts. It is slower for the smaller groups and for 33/48 rows. This compares two operators even where production would choose another head; it is not a claim that production slows down at those shapes. The paired model trial below selects the new kernel only for exactly thirty-two outputs on the PRO profile. Other row counts retain their preceding dispatch.

| Rows | Sixteen-row median, ms | Thirty-two-row median, ms | New / existing |
|---:|---:|---:|---:|
| 1 | 2.143232 | 3.690496 | 1.722 |
| 2 | 2.160672 | 3.859456 | 1.786 |
| 8 | 2.353184 | 3.990560 | 1.696 |
| 16 | 2.475040 | 4.109344 | 1.660 |
| 17 | 4.484128 | 4.161568 | 0.928 |
| 24 | 4.696096 | 4.349984 | 0.926 |
| 31 | 4.774944 | 4.466720 | 0.935 |
| 32 | 4.713504 | 4.391968 | 0.932 |
| 33 | 6.676512 | 8.228896 | 1.233 |
| 48 | 7.027744 | 8.473632 | 1.206 |
| 64 | 9.098272 | 8.538912 | 0.939 |

## Paired model trial and integrated validation

The isolated model candidate passes release and checked full-logit comparisons across chunks 64/512, eager/graph execution, original/reversed request order, cold/warm prefixes and recomputation. Thirty-two requests shrink to sixteen active requests during generation. Each build checks twelve combinations and 960 complete output hashes against the unchanged head arithmetic. Separate CLI verification compares 1,592 hashes in each of two repetitions across nine workloads, both binaries, all three cache modes and a 64 GiB candidate KV configuration; every hash and token matches.

The timing pilot uses the frozen production/candidate binaries in A/B/B/A order, three excluded warmups and ten measurements per resident suite, giving twenty measured samples per binary/case. It measures two batch-32 workloads and a batch-16 control in cold, prefix-cache-enabled and recompute modes. The five-token prompt cannot reuse a complete 64-token prefix page even when prefix caching is enabled. Builds finished before all timing runs. This is a focused model pilot, not the final stock comparison matrix.

| Mode | Batch / prompt / outputs | Baseline median, ms | Candidate median, ms | Candidate / baseline |
|---|---|---:|---:|---:|
| Cold | 16 / 129 / 8 | 213.342 | 213.344 | 1.0000 |
| Cold | 32 / 129 / 8 | 378.189 | 375.930 | 0.9940 |
| Cold | 32 / 5 / 32 | 451.429 | 442.502 | 0.9802 |
| Prefix | 16 / 129 / 8 | 82.564 | 82.569 | 1.0001 |
| Prefix | 32 / 129 / 8 | 114.202 | 111.819 | 0.9791 |
| Prefix | 32 / 5 / 32 | 451.093 | 441.894 | 0.9796 |
| Recompute | 16 / 129 / 8 | 1,182.876 | 1,189.090 | 1.0053 |
| Recompute | 32 / 129 / 8 | 2,342.612 | 2,346.636 | 1.0017 |
| Recompute | 32 / 5 / 32 | 1,690.406 | 1,680.535 | 0.9942 |

Both candidate arms improve batch-32 cold/prefix latency against both surrounding baseline arms. The batch-16 cold/prefix control is effectively unchanged. Recompute measurements have baseline drift: batch-32 prompt-129 baseline arm medians rise from 2,326.03 to 2,354.91 ms, while the unaffected batch-16 baseline rises from 1,171.26 to 1,192.40 ms. The pooled recompute numbers above do not establish a clear gain or explain that drift. Integrated sanitizer and native-context validation below pass, supporting retention of the exactly-thirty-two-output PRO dispatch. The original numerical gate and stock-performance target remain open.

The model source patch is `b43e46b5116c9641ce72b7406a85ac1d69e7ee4fb73a857480646e851f0f7dda`; frozen candidate SHA-256 is `d99fc2c187d5ccaff9b69f86b496398260ce6c7c2a26ebeb2257549992c28f42` against baseline `29d4ebb0e73a749748d91d68a9eef4871a0c8a8654c533a825e40c270a69e34d`. Artifacts are under `head32-model-v1/` and `head32-model-paired-v1/`. The release/checked model-check summary is `56c9a9da688c4683b166949d8f0a2a10c2608606e7e3cc1f38064e5889aebb43`, paired identity manifest `dcbb305b8b005d43a24d8c899824ff2c19831d55371e624223c00f7d259b46c0`, full-hash verification `5a28340914cd5356ebf73fa6d29b672697a6106d7c643fc1cdc1ec7536352f40`, paired summary `f55a2d20398456b49adff0b9197352fbc0e24f3d29709b8a4f34df3910d2fa79`, and telemetry `e758b5f4bd1ce44e782e7e62306089de2186c78d0b588677bae279592072e013`. All nineteen raw suite identities and successful command receipts were rechecked after completion.

The same frozen candidate passes unfiltered full-model memcheck on thirty-two five-token prompts with two outputs each. All 64 tokens and full-logit hashes agree directly with the original untuned `864a88df` executable. The complete 1,304-launch inventory contains two new-head launches, one eager and one graph, each covering all 18,992 vocabulary CTAs at 256 threads. Its exhaustive new-head racecheck partition passes with zero errors or warnings and preserves all original outputs. Other kernel families are filtered and retain their separately recorded coverage; this is not a new aggregate full-model racecheck pass. Under `head32-model-sanitizers-v1/`, memcheck validation SHA-256 is `46b920e3821eb22075c585cd332034501f376a72dd22a11c2a57dc7747ac7c87`, racecheck validation `52d3b4b7ecbb923b85baca54f152ca8f8eff2451a089c5e1cbd5e251243b2f6d`, and completion summary `a8bc5ae288849efac7c7d7f63d705a9b3f38a80c5a33a36f14ad7635cd56593a`.

Both release and checked builds pass all twenty ordinary tests and the explicit native-session test. The latter prepares thirty-two sessions individually, including 8K/16K/near-32K prompts, then continues all thirty-two together. It preserves 32 warmup and 64 continuation hashes for both graph/original-order and eager/reversed-order cases, reaches 32,767 tokens, allocates 64 GiB KV and releases every page. Native runs take 149.56/154.03 seconds. Completion is `head32-native-v1/validation-v3/summary.json`, SHA-256 `75685ac1cb861e2665b9b57889a85ddf866e0e81b5f5f029b5ced534cec8823a`. The first waiter was corrected before GPU execution to supply the operator-fixture path; a subsequent parser recovery reused the successful release logs after recognizing Rust's test-name prefix on the first result. Neither recovery changed kernels or reran successful GPU checks.

After integration and formatting, the rebuilt CLI has SHA-256 `d54b9f2e4c81a29cc180d40ea590deb5568884ed302cec0344dd799050eb0e1a`. Its new head retains 80 registers, zero stack/local storage and 33,792 shared bytes. All 1,592 full-logit hashes in each of two repetitions match the preceding baseline across nine workloads, all three cache modes and a 64 GiB candidate KV configuration. `head32-retained-v1/summary.json` is `dc29b57238d812fa9f58ea19673d899f3ef70677c243129ecd33d6f50c266325`; its regression report is `f97bff769461dbcee494240aad45251f16018e6299198bffa907ed92ab0b546a`. Performance numbers above remain attributed to the frozen pre-format candidate, not a new timing run of this rebuilt executable. The final stock matrix must be regenerated with the selected final executable identity.

## Evidence

Raw artifacts remain under `clean-work/pro6000-baseline/` on the workstation. Frozen binary/source identities and resource counts are recorded in [round 8](../optimization/round-8.md#asynchronous-copy-large-projections-compiled-and-queued). The first queues stopped before GPU work when the preceding numerical trace harness rejected 1,025 rows. Recovery preserves those failures and resumes the same frozen operator binaries after the corrected trace diagnostic. No build overlapped these timing screens.

| Evidence | Path below the artifact root | SHA-256 |
|---|---|---|
| Projection completion | `large-async-v2/validation-v3/summary.json` | `af1a6e92ebab6f16497509811ca459998ba75bf6e6e499eab56835a15ae5f3c2` |
| Projection profiles | `large-async-v2/validation-v3/profile-summary.json` | `0e7f01eeb16246225d901e4339bf25544a2b781d269752337a67241ab8778c47` |
| Projection recovery identity | `large-async-v2/identity-v3.json` | `c3f80f61201d8a26f01812de54797966d5b49f1f1015675558cbbbeba5c8e508` |
| Head completion | `head32-single-v2/validation-v2/summary.json` | `26182f8be0987d814f9eeb848c9bd7a89dd1e688ebec12e4dca36b77be9e8975` |
| Head profiles | `head32-single-v2/validation-v2/profile-summary.json` | `8f029288a2d2c8be89d3cd7030ce29e6dc0d0584390508dfbad8e1049deb3345` |
| Head recovery identity | `head32-single-v2/identity-v2.json` | `d90ba693e8ada34b9581814f77d760fed1c4fcc6aaea6fa2852c9770f8d54da2` |
