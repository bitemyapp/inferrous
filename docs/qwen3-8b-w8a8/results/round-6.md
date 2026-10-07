# Results summary: Round 6

Generated from the raw artifacts in `results/2026-10-06/round6` of the development repository, which are not published. The narrative, rationale and conclusions are in [round 6 record](../optimization/round-6.md). Timing columns are medians over measured runs; decode is tokens per second for tokens 2–32; API is the public generation call in milliseconds; first step is the prefill step in milliseconds.

## Measurements

| Variant | Measured runs | Decode p5 | Decode p129 | Decode p512 | API p5 | API p129 | API p512 | First step p512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| round6-checked | 0 | — | — | — | — | — | — | — |
| round6-paired-candidate-a | 20 | 182.5 | 178.4 | 170.7 | 178.1 | 192.6 | 225.8 | 44.01 |
| round6-paired-candidate-b | 20 | 182.1 | 177.9 | 170.2 | 178.5 | 193.3 | 226.7 | 44.28 |
| round6-paired-control-a | 20 | 170.9 | 165.8 | 160.7 | 189.6 | 205.8 | 237.1 | 43.89 |
| round6-paired-control-b | 20 | 169.6 | 164.5 | 159.6 | 191.2 | 207.6 | 238.8 | 44.26 |

## Regression checks

Each check compares every token ID and complete-vocabulary logit hash against the original oracle.

| Check | Result |
|---|---|
| round6-checked-regression.json | pass (3 requests, 3 workloads) |
| round6-final-regression.json | pass (138 requests, 3 workloads) |
| round6-paired-candidate-a-regression.json | pass (69 requests, 3 workloads) |
| round6-paired-candidate-b-regression.json | pass (69 requests, 3 workloads) |
| round6-paired-control-a-regression.json | pass (69 requests, 3 workloads) |
| round6-paired-control-b-regression.json | pass (69 requests, 3 workloads) |

## Tests and sanitizers

| Log | Result |
|---|---|
| round6-checked-test.log | test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 19.44s |
| round6-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round6-model-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round6-optimized-test.log | test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.26s |
| round6-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| round6-synccheck.log | ========= ERROR SUMMARY: 0 errors |

## Memory

Sampled process GPU memory in a separate instrumented run; a sampled maximum, not an exact peak.

| File | Workload | Samples | Max sampled MiB |
|---|---|---:|---:|
| round6-memory.json | p512 | 18 | 10256.0 |

## Profiles

Nsight Systems kernel summaries: the six kernels with the largest share of GPU time.

| Profile | Top kernels (share of GPU time) |
|---|---|
| round6-profile-summary | linear_large 52.4%, attention_prefill 29.8%, linear_tiled 5.7%, swiglu 2.8%, prepare_qk 2.2%, residual 2.2% |

## Microbenchmarks

CUDA C++ microbenchmarks of L2 prefetch and programmatic dependent launch on the RTX PRO 6000 (SM120). Times are CUPTI kernel durations in microseconds: `small` is a latency-bound kernel and `total` spans it plus a following cold streaming read of `read` MiB. The design and conclusions are in the round 6 record.

### prefetch_bench3 (CUDA graphs; no prefetch, 16-CTA bulk prefetch, PDL)

```text
read=16 spin=1000 pf=0 mode=0 ctas=0 pdl=0 graph=1 small   1.54 reader  11.52 ( 1456 GB/s) total  13.34
read=16 spin=1000 pf=0 mode=0 ctas=0 pdl=1 graph=1 small   1.54 reader  11.81 ( 1421 GB/s) total  13.44
read=16 spin=1000 pf=2 mode=2 ctas=16 pdl=0 graph=1 small   1.73 reader  10.43 ( 1608 GB/s) total  12.42
read=16 spin=1000 pf=4 mode=2 ctas=16 pdl=0 graph=1 small   2.59 reader   9.44 ( 1777 GB/s) total  12.29
read=16 spin=1000 pf=8 mode=2 ctas=16 pdl=0 graph=1 small   4.35 reader   7.97 ( 2106 GB/s) total  12.58
read=16 spin=1000 pf=4 mode=2 ctas=16 pdl=1 graph=1 small   2.56 reader   9.95 ( 1686 GB/s) total  12.54
read=16 spin=2000 pf=0 mode=0 ctas=0 pdl=0 graph=1 small   2.40 reader  11.52 ( 1456 GB/s) total  14.18
read=16 spin=2000 pf=0 mode=0 ctas=0 pdl=1 graph=1 small   2.40 reader  11.78 ( 1425 GB/s) total  14.27
read=16 spin=2000 pf=2 mode=2 ctas=16 pdl=0 graph=1 small   2.43 reader  10.34 ( 1623 GB/s) total  13.06
read=16 spin=2000 pf=4 mode=2 ctas=16 pdl=0 graph=1 small   2.59 reader   9.44 ( 1777 GB/s) total  12.29
read=16 spin=2000 pf=8 mode=2 ctas=16 pdl=0 graph=1 small   4.35 reader   7.90 ( 2123 GB/s) total  12.51
read=16 spin=2000 pf=4 mode=2 ctas=16 pdl=1 graph=1 small   2.59 reader   9.95 ( 1686 GB/s) total  12.58
read=16 spin=4000 pf=0 mode=0 ctas=0 pdl=0 graph=1 small   4.10 reader  11.58 ( 1448 GB/s) total  15.94
read=16 spin=4000 pf=0 mode=0 ctas=0 pdl=1 graph=1 small   4.10 reader  11.78 ( 1425 GB/s) total  16.00
read=16 spin=4000 pf=2 mode=2 ctas=16 pdl=0 graph=1 small   4.13 reader  10.24 ( 1638 GB/s) total  14.65
read=16 spin=4000 pf=4 mode=2 ctas=16 pdl=0 graph=1 small   4.13 reader   9.09 ( 1846 GB/s) total  13.50
read=16 spin=4000 pf=8 mode=2 ctas=16 pdl=0 graph=1 small   4.32 reader   7.94 ( 2114 GB/s) total  12.54
read=16 spin=4000 pf=4 mode=2 ctas=16 pdl=1 graph=1 small   4.13 reader   9.38 ( 1789 GB/s) total  13.57
read=50 spin=1000 pf=0 mode=0 ctas=0 pdl=0 graph=1 small   1.57 reader  33.38 ( 1571 GB/s) total  35.17
read=50 spin=1000 pf=0 mode=0 ctas=0 pdl=1 graph=1 small   1.54 reader  33.54 ( 1563 GB/s) total  35.17
read=50 spin=1000 pf=2 mode=2 ctas=16 pdl=0 graph=1 small   1.73 reader  32.35 ( 1621 GB/s) total  34.30
read=50 spin=1000 pf=4 mode=2 ctas=16 pdl=0 graph=1 small   2.59 reader  31.26 ( 1677 GB/s) total  34.14
read=50 spin=1000 pf=8 mode=2 ctas=16 pdl=0 graph=1 small   4.35 reader  29.92 ( 1752 GB/s) total  34.50
read=50 spin=1000 pf=4 mode=2 ctas=16 pdl=1 graph=1 small   2.59 reader  31.62 ( 1658 GB/s) total  34.24
read=50 spin=2000 pf=0 mode=0 ctas=0 pdl=0 graph=1 small   2.40 reader  33.44 ( 1568 GB/s) total  36.10
read=50 spin=2000 pf=0 mode=0 ctas=0 pdl=1 graph=1 small   2.37 reader  33.50 ( 1565 GB/s) total  36.00
read=50 spin=2000 pf=2 mode=2 ctas=16 pdl=0 graph=1 small   2.43 reader  32.13 ( 1632 GB/s) total  34.82
read=50 spin=2000 pf=4 mode=2 ctas=16 pdl=0 graph=1 small   2.56 reader  31.26 ( 1677 GB/s) total  34.11
read=50 spin=2000 pf=8 mode=2 ctas=16 pdl=0 graph=1 small   4.32 reader  29.82 ( 1758 GB/s) total  34.43
read=50 spin=2000 pf=4 mode=2 ctas=16 pdl=1 graph=1 small   2.56 reader  31.62 ( 1658 GB/s) total  34.21
read=50 spin=4000 pf=0 mode=0 ctas=0 pdl=0 graph=1 small   4.10 reader  33.47 ( 1566 GB/s) total  37.82
read=50 spin=4000 pf=0 mode=0 ctas=0 pdl=1 graph=1 small   4.10 reader  33.63 ( 1559 GB/s) total  37.82
read=50 spin=4000 pf=2 mode=2 ctas=16 pdl=0 graph=1 small   4.13 reader  32.13 ( 1632 GB/s) total  36.51
read=50 spin=4000 pf=4 mode=2 ctas=16 pdl=0 graph=1 small   4.13 reader  30.91 ( 1696 GB/s) total  35.33
read=50 spin=4000 pf=8 mode=2 ctas=16 pdl=0 graph=1 small   4.32 reader  29.89 ( 1754 GB/s) total  34.46
read=50 spin=4000 pf=4 mode=2 ctas=16 pdl=1 graph=1 small   4.13 reader  31.04 ( 1689 GB/s) total  35.26
```

### prefetch_bench4

```text
read=16 spin=1000 variant=0          small   1.54  total  13.38
read=16 spin=1000 variant=1          small   2.56  total  12.35
read=16 spin=1000 variant=2          small   1.95  total  12.32
read=16 spin=1000 variant=3          small   2.11  total  12.29
read=16 spin=1000 variant=4          small   1.50  total  12.90
read=16 spin=2000 variant=0          small   2.40  total  14.21
read=16 spin=2000 variant=1          small   2.56  total  12.26
read=16 spin=2000 variant=2          small   2.43  total  12.26
read=16 spin=2000 variant=3          small   2.43  total  12.19
read=16 spin=2000 variant=4          small   2.37  total  13.70
read=16 spin=4000 variant=0          small   4.13  total  15.94
read=16 spin=4000 variant=1          small   4.13  total  13.47
read=16 spin=4000 variant=2          small   4.13  total  13.44
read=16 spin=4000 variant=3          small   4.13  total  13.41
read=16 spin=4000 variant=4          small   4.06  total  15.39
read=25 spin=1000 variant=0          small   1.54  total  19.42
read=25 spin=1000 variant=1          small   2.59  total  18.37
read=25 spin=1000 variant=2          small   1.95  total  18.43
read=25 spin=1000 variant=3          small   2.85  total  18.37
read=25 spin=1000 variant=4          small   1.54  total  18.88
read=25 spin=2000 variant=0          small   2.40  total  20.26
read=25 spin=2000 variant=1          small   2.56  total  18.37
read=25 spin=2000 variant=2          small   2.85  total  18.37
read=25 spin=2000 variant=3          small   2.88  total  18.43
read=25 spin=2000 variant=4          small   2.40  total  19.81
read=25 spin=4000 variant=0          small   4.10  total  21.98
read=25 spin=4000 variant=1          small   4.13  total  19.36
read=25 spin=4000 variant=2          small   4.13  total  19.10
read=25 spin=4000 variant=3          small   4.13  total  19.04
read=25 spin=4000 variant=4          small   4.06  total  21.47
read=50 spin=1000 variant=0          small   1.54  total  35.30
read=50 spin=1000 variant=1          small   2.56  total  34.15
read=50 spin=1000 variant=2          small   1.95  total  34.27
read=50 spin=1000 variant=3          small   3.26  total  34.59
read=50 spin=1000 variant=4          small   1.50  total  34.69
read=50 spin=2000 variant=0          small   2.40  total  36.06
read=50 spin=2000 variant=1          small   2.56  total  34.21
read=50 spin=2000 variant=2          small   2.98  total  34.24
read=50 spin=2000 variant=3          small   3.39  total  34.59
read=50 spin=2000 variant=4          small   2.40  total  35.62
read=50 spin=4000 variant=0          small   4.10  total  37.79
read=50 spin=4000 variant=1          small   4.13  total  35.26
read=50 spin=4000 variant=2          small   4.16  total  34.37
read=50 spin=4000 variant=3          small   4.16  total  34.72
read=50 spin=4000 variant=4          small   4.10  total  37.28
```

## Screening sweeps

```text
== sweep1.jsonl
ctas=16 scale=0                                                        p5    n=10 median  170.94
ctas=16 scale=0                                                        p512  n=10 median  161.16
ctas=16 scale=1.0                                                      p5    n=10 median  176.66
ctas=16 scale=1.0                                                      p512  n=10 median  165.85
ctas=32 scale=1.0                                                      p5    n=10 median  175.66
ctas=32 scale=1.0                                                      p512  n=10 median  164.30
ctas=64 scale=1.0                                                      p5    n=10 median  174.16
ctas=64 scale=1.0                                                      p512  n=10 median  163.46
ctas=16 scale=0.5                                                      p5    n=10 median  175.48
ctas=16 scale=0.5                                                      p512  n=10 median  165.04
ctas=32 scale=0.5                                                      p5    n=10 median  174.23
ctas=32 scale=0.5                                                      p512  n=10 median  163.35
ctas=64 scale=0.5                                                      p5    n=10 median  172.90
ctas=64 scale=0.5                                                      p512  n=10 median  163.10
ctas=16 scale=0                                                        p5    n=10 median  169.48
ctas=16 scale=0                                                        p512  n=10 median  159.87
== sweep2.jsonl
ctas=16 scale=0                                                        p5    n=10 median  174.59
ctas=16 scale=0                                                        p512  n=10 median  163.80
ctas=16 scale=1.0                                                      p5    n=10 median  180.80
ctas=16 scale=1.0                                                      p512  n=10 median  168.24
ctas=16 scale=0                                                        p5    n=10 median  174.11
ctas=16 scale=0                                                        p512  n=10 median  163.37
== sweep3.jsonl
-                                                                      p5    n=10 median  180.82
-                                                                      p512  n=10 median  168.30
CLEAN_PREFETCH_MIB=6,4,6,3,3,2,14,0                                    p5    n=10 median  180.57
CLEAN_PREFETCH_MIB=6,4,6,3,3,2,14,0                                    p512  n=10 median  168.16
CLEAN_PREFETCH_MIB=6,4,6,3,3,2,0,16                                    p5    n=10 median  180.38
CLEAN_PREFETCH_MIB=6,4,6,3,3,2,0,16                                    p512  n=10 median  167.94
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,16                             p5    n=10 median  180.87
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,16                             p512  n=10 median  168.38
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,0                              p5    n=10 median  180.82
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,0                              p512  n=10 median  168.11
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,16;CLEAN_PREFETCH_CTAS=24      p5    n=10 median  180.61
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,16;CLEAN_PREFETCH_CTAS=24      p512  n=10 median  168.62
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,16;CLEAN_PREFETCH_CTAS=32      p5    n=10 median  180.91
CLEAN_PREFETCH_MIB=3.5,3,6,3.5,1.75,2,4,16;CLEAN_PREFETCH_CTAS=32      p512  n=10 median  168.08
CLEAN_PREFETCH_MIB=2.5,2,4,2.5,1.25,1.5,3,8                            p5    n=10 median  181.31
CLEAN_PREFETCH_MIB=2.5,2,4,2.5,1.25,1.5,3,8                            p512  n=10 median  169.10
-                                                                      p5    n=10 median  179.48
-                                                                      p512  n=10 median  167.15
== sweep4.jsonl
CLEAN_PDL=0                                                            p5    n=10 median  181.34
CLEAN_PDL=0                                                            p512  n=10 median  168.91
-                                                                      p5    n=10 median  180.85
-                                                                      p512  n=10 median  171.10
CLEAN_PDL=0                                                            p5    n=10 median  180.91
CLEAN_PDL=0                                                            p512  n=10 median  168.39
-                                                                      p5    n=10 median  180.41
-                                                                      p512  n=10 median  170.46
== sweep5.jsonl
CLEAN_PDL=0;CLEAN_PREFETCH_SCALE=0                                     p5    n=10 median  174.40
CLEAN_PDL=0;CLEAN_PREFETCH_SCALE=0                                     p512  n=10 median  163.64
CLEAN_PREFETCH_SCALE=0                                                 p5    n=10 median  177.44
CLEAN_PREFETCH_SCALE=0                                                 p512  n=10 median  167.31
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0                                     p5    n=10 median  181.64
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0                                     p512  n=10 median  171.03
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,16                                    p5    n=10 median  181.44
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,16                                    p512  n=10 median  170.87
CLEAN_PDL=0;CLEAN_PREFETCH_SCALE=0                                     p5    n=10 median  173.76
CLEAN_PDL=0;CLEAN_PREFETCH_SCALE=0                                     p512  n=10 median  163.02
== sweep6.jsonl
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=0,0,0,0,0        p5    n=10 median  180.62
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=0,0,0,0,0        p512  n=10 median  170.46
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0                                     p5    n=10 median  183.14
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0                                     p512  n=10 median  171.53
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=1024,1024,512,1024,1024 p5    n=10 median  183.28
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=1024,1024,512,1024,1024 p512  n=10 median  171.80
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=256,512,128,256,256 p5    n=10 median  182.21
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=256,512,128,256,256 p512  n=10 median  170.95
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=0,0,0,0,0        p5    n=10 median  179.98
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=0,0,0,0,0        p512  n=10 median  169.60
== sweep7.jsonl
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=1024,1024,512,1024,1024 p5    n=10 median  182.69
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=1024,1024,512,1024,1024 p512  n=10 median  171.22
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=2048,2048,1024,2048,2048 p5    n=10 median  182.37
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=2048,2048,1024,2048,2048 p512  n=10 median  170.91
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=4096,4096,2048,4096,6144 p5    n=10 median  179.54
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=4096,4096,2048,4096,6144 p512  n=10 median  168.86
CLEAN_PREFETCH_MIB=0,0,8,6,0,0,0,0;CLEAN_PREFETCH_ROW=2048,2048,1024,2048,2048 p5    n=10 median  179.86
CLEAN_PREFETCH_MIB=0,0,8,6,0,0,0,0;CLEAN_PREFETCH_ROW=2048,2048,1024,2048,2048 p512  n=10 median  170.00
CLEAN_PREFETCH_MIB=3,2,6,4,2,2,4,0;CLEAN_PREFETCH_ROW=2048,2048,1024,2048,2048 p5    n=10 median  181.91
CLEAN_PREFETCH_MIB=3,2,6,4,2,2,4,0;CLEAN_PREFETCH_ROW=2048,2048,1024,2048,2048 p512  n=10 median  170.98
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=1024,1024,512,1024,1024 p5    n=10 median  182.32
CLEAN_PREFETCH_MIB=0,0,6,4,0,0,0,0;CLEAN_PREFETCH_ROW=1024,1024,512,1024,1024 p512  n=10 median  170.69
e64f92bcf4fd978955658e5370fd0f1ce8cad4b19c3e4b2c50b4db56a24f4881  cand1
890f69db8d6a3717ad6a5aeb566f49653097473d58d4cd524a90afe3a6a357b4  cand2
cd00593685d37dbe6eebe34c58fea56d2d70ff8ce52ffdb812527652c4e27f18  cand3
96b901a2fbe68f0858d9c142e16196798d1822429a6b4aa9fdd71e8ef1152f9b  cand4
bd6c3cfb978745049ca04a5da44b8221de0e2bb2349e14e16a6bc3c539b023dc  cand5
e01c85e654b89913b83d0b7bd73274a78c52b015ee45949cbb0d8b2eb8ac77c2  control
```

## Screening profiles

```text
== p5-node
kernels 35592 steps 96
decode steps 31: kernels/step 364  wall 5799.0us  kernel-sum 5752.4us  gaps 46.6us
kernel                       n/step   us/step  us/call gap-before us/step
linear_swiglu                    36    2276.0    63.22                4.7
linear_residual                  72    1651.5    22.94                9.3
linear_triple                    36     608.8    16.91                4.7
head_packed                       1     574.1   574.10                0.1
norm_quantize                    72     265.7     3.69                9.0
attention                        36     140.5     3.90                4.6
prepare_qk                       36      91.3     2.54                4.7
quantize                         36      70.8     1.97                4.6
quantize_with_max                36      42.1     1.17                4.8
select_token                      1      21.3    21.31               -0.1
norm                              1       8.6     8.58                0.1
embedding                         1       1.7     1.69                0.0
inter-step gap (last kernel end -> next embedding start): mean 189.3us median 189.1us
step period mean 5988.3us -> 167.0 tok/s
O-proj+res  mean 11.94us median 11.90  -> 1405 GB/s
down+res    mean 33.94us median 33.92  -> 1483 GB/s
== c1off-node
kernels 35592 steps 96
decode steps 31: kernels/step 364  wall 5832.2us  kernel-sum 5785.6us  gaps 46.7us
kernel                       n/step   us/step  us/call gap-before us/step
linear_swiglu                    36    2275.7    63.21                4.7
linear_residual                  72    1652.1    22.95                9.3
linear_triple                    36     609.3    16.92                4.8
head_packed                       1     580.6   580.60                0.1
norm_quantize                    72     270.2     3.75                9.1
attention                        36     148.3     4.12                4.5
prepare_qk                       36      92.7     2.57                4.7
quantize                         36      77.3     2.15                4.6
quantize_with_max                36      46.1     1.28                4.8
select_token                      1      22.6    22.64               -0.1
norm                              1       8.9     8.88                0.1
embedding                         1       1.7     1.71                0.0
inter-step gap (last kernel end -> next embedding start): mean 195.1us median 194.8us
step period mean 6027.2us -> 165.9 tok/s
O-proj+res  mean 11.93us median 11.90  -> 1406 GB/s
down+res    mean 33.96us median 33.95  -> 1482 GB/s
== c1on-node
kernels 35592 steps 96
decode steps 31: kernels/step 364  wall 5638.7us  kernel-sum 5590.9us  gaps 47.8us
kernel                       n/step   us/step  us/call gap-before us/step
linear_swiglu                    36    2147.7    59.66                4.8
linear_residual                  72    1394.5    19.37                9.7
head_packed                       1     591.9   591.93                0.1
linear_triple                    36     479.5    13.32                4.8
norm_quantize                    72     443.8     6.16                9.2
attention                        36     211.6     5.88                4.7
prepare_qk                       36     120.5     3.35                4.8
quantize                         36      83.3     2.31                4.7
quantize_with_max                36      70.2     1.95                4.8
select_token                      1      32.3    32.35               -0.0
norm                              1      13.9    13.92                0.1
embedding                         1       1.7     1.72                0.0
inter-step gap (last kernel end -> next embedding start): mean 188.4us median 189.1us
step period mean 5826.9us -> 171.6 tok/s
O-proj+res  mean 5.93us median 5.89  -> 2830 GB/s
down+res    mean 32.81us median 32.80  -> 1534 GB/s
== c3off-node
kernels 35592 steps 96
decode steps 31: kernels/step 364  wall 5713.6us  kernel-sum 5666.7us  gaps 47.0us
kernel                       n/step   us/step  us/call gap-before us/step
linear_swiglu                    36    2277.5    63.26                4.6
linear_residual                  72    1655.2    22.99                9.4
linear_triple                    36     624.1    17.34                4.7
head_packed                       1     592.6   592.62                0.1
norm_quantize                    72     162.5     2.26                9.1
attention                        36     138.4     3.84                4.6
quantize                         36      77.5     2.15                4.7
prepare_qk                       36      64.8     1.80                4.7
quantize_with_max                36      46.3     1.29                4.9
select_token                      1      23.6    23.62               -0.1
norm                              1       2.5     2.49                0.1
embedding                         1       1.7     1.69                0.0
inter-step gap (last kernel end -> next embedding start): mean 199.1us median 199.6us
step period mean 5912.6us -> 169.1 tok/s
O-proj+res  mean 11.93us median 11.90  -> 1406 GB/s
down+res    mean 34.05us median 34.02  -> 1478 GB/s
== c3on-node
kernels 35592 steps 96
decode steps 31: kernels/step 364  wall 5523.2us  kernel-sum 5473.8us  gaps 49.4us
kernel                       n/step   us/step  us/call gap-before us/step
linear_swiglu                    36    2175.4    60.43                5.3
linear_residual                  72    1398.8    19.43               10.0
head_packed                       1     601.2   601.18                0.1
linear_triple                    36     517.0    14.36                5.0
norm_quantize                    72     274.9     3.82                9.5
attention                        36     207.3     5.76                4.9
prepare_qk                       36     100.6     2.80                5.0
quantize                         36      86.3     2.40                4.8
quantize_with_max                36      69.7     1.94                4.9
select_token                      1      33.1    33.11               -0.0
norm                              1       7.7     7.67                0.1
embedding                         1       1.7     1.69                0.0
inter-step gap (last kernel end -> next embedding start): mean 199.5us median 198.4us
step period mean 5722.7us -> 174.7 tok/s
O-proj+res  mean 6.02us median 5.98  -> 2788 GB/s
down+res    mean 32.84us median 32.83  -> 1533 GB/s
== p5-graph
graphs 31: exec mean 5754.5us median 5755.1; between graphs mean 38.9us median 38.1; period 5793.2us -> 172.6 tok/s
== c5-graph
graphs 31: exec mean 5430.2us median 5429.6; between graphs mean 40.7us median 40.5; period 5470.5us -> 182.8 tok/s
```

## Memcheck determinism

```text
# compute-sanitizer memcheck logit hashes, round4-memcheck-request.json (p512 prompt, 3 outputs)
oracle (baseline.jsonl p512 first three):
['0ea5be1b6c395a52a129bacd27191a2fd68dd632034ff28d67fe2604acf7098f', 'c26e01c5519189a5e22823de5bf744fa5218512a58e2cf99dad37615fcceba30', 'e2431307de7de706bba9fbdcd9317a8fa481e2ee902b088808084b2f6bc18188']
round5 control under memcheck (round5-final-model-memcheck.log):
"logit_sha256":["038c35573174ea70a0f6f695dac9b52952e973788ba77e95efb87f7f278e6488","84399f1801fb16a5f9c5bc908b7c98a4c4ea1fb7a59a6117c72cbb24c3366c23","37ac6e6f40de1e6bcddaf7a0d44d10332e7027f06d0f776a11f52122653ee43a"]
round5 control under memcheck, rerun 2026-10-06:
[138cfc90d8ff..., 1fc4c3e24024...] (differs from both)
round6 candidate under memcheck:
"logit_sha256":["f2155cfe96eb64550cc4400bc04c6f787caf595705375f4847670636b318c6f9","a00f4e6f0980a2639a097aa1bb0d254f449e5de1746330ba006fa287b61611a9","979c39e6e674d3cf7130be2420001362b55575820de2a9c00d2dd17ee57738b7"]
control and candidate without sanitizer: identical to oracle
p5/p129 prompts (3 outputs) under memcheck, control and candidate: identical to oracle first three hashes, 0 errors
```

## Procedures

The exact scripts used for the retained measurements and checks. `<workdir>` is the development checkout on the benchmark workstation.

### final-benchmarks.sh

```sh
#!/usr/bin/env bash
# A/B/B/A counters temperature and clock drift without changing machine policy.
# control is the preserved round-5 executable (sha256 e01c85e6...).
set -euo pipefail
cd <workdir>
R=${R:?set R to the run directory holding control and round6-candidate}
T=crates/inferrous-qwen3/tools
for label in control-a candidate-a candidate-b control-b; do
    binary=$R/control
    if [[ "$label" == candidate-* ]]; then binary=$R/round6-candidate; fi
    python3 $T/bench_clean.py --binary "$binary" --model clean-work/model-bundle --workloads clean-work/workloads --output "$R/round6-paired-$label.jsonl" > "$R/round6-paired-$label.log" 2>&1
    python3 $T/check_regression.py clean-work/baseline.jsonl "$R/round6-paired-$label.jsonl" > "$R/round6-paired-$label-regression.json"
done
cat "$R/round6-paired-control-a.jsonl" "$R/round6-paired-control-b.jsonl" > "$R/round6-control.jsonl"
cat "$R/round6-paired-candidate-a.jsonl" "$R/round6-paired-candidate-b.jsonl" > "$R/round6-final.jsonl"
python3 $T/summarize.py "$R/round6-paired-control-a.jsonl" "$R/round6-paired-candidate-a.jsonl" "$R/round6-paired-candidate-b.jsonl" "$R/round6-paired-control-b.jsonl" "$R/round6-control.jsonl" "$R/round6-final.jsonl" > "$R/round6-summary.json"
python3 $T/check_regression.py clean-work/baseline.jsonl "$R/round6-final.jsonl" > "$R/round6-final-regression.json"
```

### final-checks.sh

```sh
#!/usr/bin/env bash
# Round 6 build, test, sanitizer and profile procedure. R names a fresh run
# directory under clean-work; nothing here overwrites earlier evidence.
set -euo pipefail
cd <workdir>
R=${R:?set R to a fresh clean-work run directory}
export CUDA_OXIDE_BACKEND=<workdir>/vendor/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export RUST_TEST_THREADS=1
export CLEAN_OPERATOR_FIXTURES=$PWD/clean-work/operator-fixtures-64
export CLEAN_MODEL_BUNDLE=$PWD/clean-work/model-bundle
T=crates/inferrous-qwen3/tools
BIN=target/inferrous-gpu/release/inferrous-qwen3
$T/cuda.sh test > "$R/round6-optimized-test.log" 2>&1
tests=$(grep -o 'Running unittests src/lib.rs ([^)]*)' "$R/round6-optimized-test.log" | sed 's/.*(\(.*\))/\1/')
cp "$tests" "$R/round6-tests"
$T/cuda.sh build > "$R/round6-optimized-build.log" 2>&1
cp "$BIN" "$R/round6-candidate"
CLEAN_BOUNDS_CHECKS=1 $T/cuda.sh test > "$R/round6-checked-test.log" 2>&1
CLEAN_BOUNDS_CHECKS=1 $T/cuda.sh build > "$R/round6-checked-build.log" 2>&1
python3 $T/bench_clean.py --binary "$BIN" --model clean-work/model-bundle --workloads clean-work/workloads --output "$R/round6-checked.jsonl" --repeat 1 > "$R/round6-checked.log" 2>&1
python3 $T/check_regression.py clean-work/baseline.jsonl "$R/round6-checked.jsonl" > "$R/round6-checked-regression.json"
$T/cuda.sh build > "$R/round6-restored-build.log" 2>&1
cmp "$R/round6-candidate" "$BIN"
for check in memcheck racecheck synccheck; do
    /usr/local/cuda-13.1/bin/compute-sanitizer --tool "$check" --error-exitcode=1 "$R/round6-tests" conformance:: --test-threads=1 > "$R/round6-$check.log" 2>&1
done
# 512-token prefill plus three outputs replays programmatic decode graphs.
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck --error-exitcode=1 "$BIN" clean-work/model-bundle --request clean-work/round4-memcheck-request.json > "$R/round6-model-memcheck.log" 2>&1
nsys profile --trace=cuda --cuda-graph-trace=graph --sample=none --cpuctxsw=none --output="$R/round6-profile" "$BIN" clean-work/model-bundle --request clean-work/workloads/p512.json --repeat 2 > "$R/round6-profile.log" 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$R/round6-profile.nsys-rep" > "$R/round6-profile-summary.csv" 2>&1
R=$R python3 crates/inferrous-qwen3/results/2026-10-06/round6/memory.py
cargo +nightly-2026-04-03 fmt -p inferrous-qwen3 -- --check
python3 -m py_compile $T/*.py
sha256sum "$R/control" "$R/round6-candidate" "$BIN" > "$R/round6-binaries.sha256"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$R/round6-device.csv"
```

## Executable hashes

round6-binaries.sha256:

```text
e01c85e654b89913b83d0b7bd73274a78c52b015ee45949cbb0d8b2eb8ac77c2  <workdir>/clean-work/round6-20261006.NMTdSj/control
7affbce7a426e33b8b37b4f6e116a8c98c4bd130fb20c481b991f610b87f4fad  <workdir>/clean-work/round6-20261006.NMTdSj/round6-candidate
7affbce7a426e33b8b37b4f6e116a8c98c4bd130fb20c481b991f610b87f4fad  target/inferrous-gpu/release/inferrous-qwen3
```

## Other raw artifacts

Not summarized here: 1 `.csv` file; 3 `.cu` files; 1 `.json` file; 5 `.log` files; 10 `.py` files (build and benchmark logs, helper scripts, cubin resource listings and other intermediate data).
