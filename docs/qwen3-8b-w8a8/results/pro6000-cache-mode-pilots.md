# RTX PRO 6000 prefix and recomputation pilots

These completed eight-workload pilots still show performance gaps against the fastest supplied eligible stock configuration. They do not replace the required 99-scenario, three-mode, twenty-measurement matrix. All execution occurred on `ssh wx-workstation`, using its RTX PRO 6000 Blackwell Max-Q. See [round 8](../optimization/round-8.md) for the implementation and validation history.

Both modes use the pinned native-context bundle, 32,768-token context limit, batch capacity 32, 12 GiB KV budget, FP32 weight scales, BF16 KV/head, greedy output and identical request arrays. Prefix mode enables native prefix caching; recomputation resubmits the entire growing prefix for every output. Each of eight stock configurations completed 48 records per mode: eight workloads with three excluded warmups and three measurements. The stock configurations pair FlashInfer, FlashAttention, Triton and B12X attention with graph-only or compiled execution. Every prefix configuration reports positive cached-token observations; every recomputation configuration reports zero cached tokens. Submission-count validation separately checks genuine recomputation.

Local medians combine six measured samples from two processes. Prefix uses the validated counter-enabled executable `29d4ebb0e73a749748d91d68a9eef4871a0c8a8654c533a825e40c270a69e34d`; recomputation uses the completed packed-baseline arms of the direct-head A/B/B/A experiment. The comparison reports retain each binary/configuration identity. Local and stock samples are unpaired. CPU compilation overlapped prefix B12X compiled and recomputation FlashInfer graph-only/compiled; `projection-build-pilot-overlap-v1.json` records those intervals. GPU jobs remained serial. These limitations preclude final ranking claims.

## API latency medians

All values are milliseconds. Homogeneous short prompts have 129 tokens and eight outputs; solo 8K has 8,192 prompt tokens and eight outputs. Ragged 12 retains all 56 requested outputs.

| Workload | Local prefix | Fastest supplied stock prefix | Local recomputation | Fastest supplied stock recomputation |
|---|---:|---:|---:|---:|
| Batch 1, prompt 129 | 47.878 | 62.250 | 141.862 | 140.006 |
| Batch 2, prompt 129 | 55.663 | 61.130 | 185.380 | 167.776 |
| Batch 4, prompt 129 | 60.360 | 62.711 | 355.726 | 260.265 |
| Batch 8, prompt 129 | 68.596 | 69.802 | 649.381 | 500.982 |
| Batch 16, prompt 129 | 80.654 | 70.182 | 1206.558 | 806.934 |
| Batch 32, prompt 129 | 110.412 | 76.150 | 2362.755 | 1477.107 |
| Ragged 12 | 85.997 | 74.433 | 295.377 | 260.688 |
| Solo 8K | 104.283 | 72.133 | 17228.003 | 3336.513 |

For prefix reuse, graph-only Triton supplies the lowest stock median for batches 1/2/8/16 and ragged 12; graph-only FlashInfer supplies batches 4/32 and solo 8K. For recomputation, compiled FlashAttention supplies batches 1/2, graph-only Triton supplies ragged 12, and graph-only FlashInfer supplies the other cases. The latter recomputation samples include the recorded CPU-build overlap.

## Reproduction evidence

Artifacts remain under `clean-work/pro6000-baseline/` on the workstation. `cache-modes-summary-v1.json` records successful completion of all eight expected configurations in both modes. The two comparison reports validate matching contracts, complete samples and input identities, and include all supplied candidate series rather than only the fastest rows. `cache-modes-prefix-hash-verification-v1.json` separately verifies all 568 local prefix full-logit hashes against the original oracle. Timing probes disable verification overhead.

| Artifact | SHA-256 |
|---|---|
| `cache-modes-summary-v1.json` | `5e665e4c773ed988fe64f348af07ee3656a7c06917af6c0c6fbee60774c88ad9` |
| `cache-modes-prefix-comparison-v1.json` | `383d197b801022cf712f72101ab020dedd90cdb39ce7bf029e51fb3d0dab5e00` |
| `cache-modes-recompute-comparison-v1.json` | `ced8c4610081f70aa75ebcf88ad06dfcf3c7e98d5a16370a54b19ea7aabfe7c5` |
| `workloads.json` | `8cc4cb85278806aef3bac73302083f37703a55e4d31c42987f18ba2dc7109c70` |

No failed worker is counted as a stock win or loss. This pilot's all-local-medians-lower condition is false in both modes. Final execution still requires the selected validated implementation, matched stock configurations at the final 64 GiB budget, twenty measured samples per scenario, and the complete acceptance audit.
