# 128-row projection experiment on RTX PRO 6000

The isolated 128-row INT8 projection kernels pass independent correctness and sanitizer checks but do not justify retention. The eight-warp variant improves only the largest 12,288-output projection by about 1.5%; all other comparisons lose. The sixteen-warp variant loses at every tested shape. No production selector change or full-model trial is justified by this screen.

Both kernels compute exact signed INT8 dot products followed by the existing FP32 scales and BF16 rounding. They use 128×128 output tiles, 128-wide K tiles and TMA staging. The eight-warp and sixteen-warp alternatives use 256 and 512 threads respectively.

## Validation

All 80 independent integer-reference comparisons pass in each release and checked build, including partial row/output tiles, K=12,288 extremes and eager/graph execution. Memcheck reports zero errors and racecheck reports zero hazards, errors or warnings, with all 80 cases in each run. Racecheck completes in 6.87 seconds. Output guards and exact BF16 results are checked. The profile contains all 340 expected records across 44 shapes; each has seven event measurements after three warmups and 32 distinct weight allocations. Outputs also agree with the independent DP4A path. These are operator results, not model benchmarks.

## Timings

All execution is on `ssh wx-workstation`. The existing column selects the fastest eligible pre-existing operator at each shape. N denotes output width and K input width.

| Rows | N | K | Existing variant | Existing, ms | M128 eight warps, ms | M128 sixteen warps, ms |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1024 | 4096 | dp4a | 0.003548 | 0.054977 | 0.054818 |
| 2 | 1024 | 4096 | dp4a | 0.003730 | 0.054941 | 0.054872 |
| 4 | 1024 | 4096 | dp4a | 0.004622 | 0.055001 | 0.054944 |
| 8 | 1024 | 4096 | splitk | 0.006667 | 0.054878 | 0.054879 |
| 16 | 1024 | 4096 | splitk | 0.008392 | 0.055261 | 0.055184 |
| 32 | 1024 | 4096 | splitk | 0.010258 | 0.056154 | 0.056034 |
| 64 | 1024 | 4096 | narrow-pipeline | 0.014216 | 0.058593 | 0.056536 |
| 128 | 1024 | 4096 | narrow-pipeline | 0.016007 | 0.062367 | 0.057952 |
| 256 | 1024 | 4096 | narrow-pipeline | 0.018774 | 0.063238 | 0.059084 |
| 512 | 1024 | 4096 | tiled-pipeline | 0.022035 | 0.062794 | 0.058891 |
| 1024 | 1024 | 4096 | large-pipeline | 0.029896 | 0.062539 | 0.058018 |
| 1 | 4096 | 4096 | dp4a | 0.011721 | 0.056333 | 0.056396 |
| 2 | 4096 | 4096 | dp4a | 0.011789 | 0.056266 | 0.056331 |
| 4 | 4096 | 4096 | splitk | 0.012305 | 0.056135 | 0.056202 |
| 8 | 4096 | 4096 | splitk | 0.012434 | 0.056076 | 0.056013 |
| 16 | 4096 | 4096 | splitk | 0.013321 | 0.056075 | 0.056012 |
| 32 | 4096 | 4096 | narrow-pipeline | 0.017159 | 0.056595 | 0.056466 |
| 64 | 4096 | 4096 | tiled-pipeline | 0.019660 | 0.058773 | 0.056852 |
| 128 | 4096 | 4096 | tiled-pipeline | 0.025426 | 0.059981 | 0.057997 |
| 256 | 4096 | 4096 | large-pipeline | 0.033031 | 0.059917 | 0.057877 |
| 512 | 4096 | 4096 | large-pipeline | 0.057529 | 0.066091 | 0.065971 |
| 1024 | 4096 | 4096 | large | 0.099425 | 0.104799 | 0.126505 |
| 1 | 12288 | 4096 | dp4a | 0.032135 | 0.062728 | 0.062483 |
| 2 | 12288 | 4096 | narrow-pipeline | 0.033545 | 0.063497 | 0.064325 |
| 4 | 12288 | 4096 | narrow-pipeline | 0.033681 | 0.064585 | 0.065299 |
| 8 | 12288 | 4096 | narrow-pipeline | 0.033928 | 0.063570 | 0.064390 |
| 16 | 12288 | 4096 | narrow-pipeline | 0.034309 | 0.063251 | 0.063892 |
| 32 | 12288 | 4096 | tiled-pipeline | 0.037000 | 0.062412 | 0.063172 |
| 64 | 12288 | 4096 | large-pipeline | 0.039753 | 0.065226 | 0.064400 |
| 128 | 12288 | 4096 | large-pipeline | 0.053631 | 0.068642 | 0.067184 |
| 256 | 12288 | 4096 | large | 0.097051 | 0.101150 | 0.128814 |
| 512 | 12288 | 4096 | large | 0.162687 | 0.168912 | 0.194120 |
| 1024 | 12288 | 4096 | large | 0.279488 | 0.275395 | 0.331946 |
| 1 | 4096 | 12288 | dp4a | 0.033792 | 0.159500 | 0.159881 |
| 2 | 4096 | 12288 | dp4a | 0.034059 | 0.159313 | 0.159634 |
| 4 | 4096 | 12288 | dp4a | 0.047629 | 0.159117 | 0.159690 |
| 8 | 4096 | 12288 | narrow-pipeline | 0.051466 | 0.159186 | 0.159562 |
| 16 | 4096 | 12288 | narrow-pipeline | 0.051723 | 0.159250 | 0.159635 |
| 32 | 4096 | 12288 | narrow-pipeline | 0.052040 | 0.159887 | 0.160138 |
| 64 | 4096 | 12288 | tiled-pipeline | 0.059018 | 0.162185 | 0.160584 |
| 128 | 4096 | 12288 | large-pipeline | 0.073644 | 0.163016 | 0.161425 |
| 256 | 4096 | 12288 | large-pipeline | 0.096673 | 0.162442 | 0.160965 |
| 512 | 4096 | 12288 | large-pipeline | 0.167614 | 0.190023 | 0.190919 |
| 1024 | 4096 | 12288 | large | 0.285568 | 0.299006 | 0.369496 |

## Artifacts

`clean-work/pro6000-baseline/m128-projections-summary-v1.json` has SHA-256 `9081c146f3bccada3bfb46a42485750e3bba3b6bb6de02699e0a703f5bd649ce` and retains every timing sample and frozen identity. `m128-projections-runs-v1.jsonl` records commands, successful exits and log hashes; the audit rechecked those hashes. Raw files remain on the workstation. See [round 8](../optimization/round-8.md) for preparation and the unchanged production decision.
