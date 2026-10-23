# RTX PRO 6000 asynchronous projections and 32-row head

All builds, checks and timings ran on `ssh wx-workstation`. These isolated kernels preserve integer projection arithmetic or the existing ordered FP32 head arithmetic. Production dispatch is unchanged. The measurements are unpaired operator screens with three warmups and seven measured CUDA events, not stock comparisons or model-level acceptance.

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

## Head results: model trial pending

The new head is 6.1–7.4% faster at the screened 17–32 and 64 row counts. It is slower for the smaller groups and for 33/48 rows. This compares two operators even where production would choose another head; it is not a claim that production slows down at those shapes. A batch-32 paired model trial is justified, with unchanged full-logit hashes required. No model gain, retention or stock win has been established.

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
