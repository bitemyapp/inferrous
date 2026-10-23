# Fixed paged policy: native and sanitizer qualification

The isolated fixed FP32 paged policy passes all eight selected release/checked tests, unfiltered full-model memcheck, and racecheck partitions covering every launch of its nine changed paged kernel families in the sanitizer workload. All work runs on `ssh wx-workstation`. The candidate remains outside production: these checks do not resolve the [original independent numerical gate](../discrepancies.md), establish performance acceptance, or validate the full ordinary suite against legacy arithmetic.

## Candidate and test scope

The worktree `inferrous-fixed-policy-native` starts at `1f6a76c` and applies fifteen arithmetic hunks from the earlier independently referenced fixed-policy diagnostic. It retains the current session, admission, graph and certified-head implementation. Twelve changed/new function bodies are byte-identical to the historical fixed-policy kernel; unrelated segmented kernels and historical host/test changes are excluded. The [optimization record](../optimization/round-8.md#fixed-paged-policy-current-native-qualification) describes the arithmetic and original source audit.

| Check | Release | Checked | What is compared |
|---|---|---|---|
| Head-model invariance | Pass | Pass | Twelve chunk/graph/order/cache combinations, eighty full-logit hashes per combination |
| Tiled-prefill invariance | Pass | Pass | Tuned and untuned paged paths under the same fixed policy |
| Native ragged execution, 9 GiB KV | Pass | Pass | 8,192/16,384/32,766-token prompts, cold/warm reuse, genuine full-prefix recomputation, chunk and graph changes |
| Native continuation, 64 GiB KV | Pass | Pass | Thirty-two sessions, thirty-two warm hashes and sixty-four continuation hashes per comparison; tuned head versus untuned head, normal/reversed order, chunks 512/128, graph/eager execution |

The longest continued session reaches 32,767 tokens. Both native tests assert that all pages are released after their final owners disappear. Native ragged tests finish in 270.22/284.29 seconds and native continuation tests in 215.82/218.81 seconds for release/checked builds. These are test durations, not benchmark comparisons.

The test name `tiled_prefill_matches_original_across_chunks_graphs_and_cache_modes` uses “original” for the untuned paged path in this candidate. It does not mean production arithmetic or the independent Torch oracle. Legacy `prepare_qk` and `norm_rope` also receive historical arithmetic hunks but are not exercised by this paged qualification; cross-policy legacy equivalence is not claimed.

## Sanitizer workload and coverage

After all eight tests pass and their source/receipt hashes are audited, the runner builds and freezes a release CLI. One ragged workload contains thirty-two prompts of five through nine tokens, totaling 221 prefill rows. The first request generates three outputs; the other thirty-one generate two, yielding sixty-five full-vocabulary hashes. This produces tiled eager prefill, thirty-two-row grouped graph decode and a final solo paged graph decode. Configuration uses native context, thirty-two slots, 12 GiB KV, chunk 512, all tuning enabled, no prefix caching and full-logit verification.

The inventory contains 1,818 launches across twenty-seven families. Unfiltered full-model memcheck reports zero errors. Four racecheck partitions cover all 759 launches of the nine changed paged families exactly once:

| Partition | Families | Checked launches | Duration | Peak process-group RSS |
|---|---|---:|---:|---:|
| Normalization | `norm`, `norm_quantize` | 219 | 76.44 s | 33,875,386,368 bytes |
| Quantization | `quantize`, `quantize_with_max` | 216 | 40.04 s | 31,089,213,440 bytes |
| Q/K preparation and softmax | `prepare_qk_paged`, `attention_softmax_paged` | 216 | 61.16 s | 28,438,040,576 bytes |
| Values | `attention_values_paged`, `attention_values_tiled`, `attention_values_grouped` | 108 | 36.37 s | 31,095,898,112 bytes |

Every partition reports zero errors and warnings, preserves all sixty-five output hashes and tokens, and matches the inventory's complete launch order, shapes and graph/eager flags. The independently audited sanitizer's launch-dump reporting returns exit 99 for informational records; acceptance requires its strict complete footer and no diagnostics, not the exit code alone. Memcheck exits zero in 20.52 seconds. No memory guard triggers.

The final audit rechecks source/binary identities, recorded commands, receipts, logs, exact partition coverage and all seven model outputs against the unsanitized candidate. This is exhaustive coverage of changed paged families on this workload, not all 1,818 launches under racecheck, native-length sanitizer coverage, or a new independent numerical reference. Unchanged families retain separate historical evidence; unused legacy preparations are outside this campaign.

## Frozen artifacts

Artifacts remain under `clean-work/pro6000-baseline/` on the workstation. Both runners are terminal and the GPU compute queue is idle at completion.

| Artifact | SHA-256 |
|---|---|
| Candidate kernel | `302c7d5d7304d400f3ba0b30c78be49f73e767bc1dbcf48c8dad3230ced70135` |
| Native source patch | `5704344942a1e1ea54a3c589451f11b5607d6c3265d375847b92f4f3c4a37605` |
| `fixed-policy-native-v1/summary.json` | `166ceb7c3dce0ede75c4d4cf9fab1d5bb6bd5890204ced4f435f3f763318c5a2` |
| `fixed-policy-model-sanitizers-v1/run.py` | `f0e8a3054a4866295823b80420ae110c601bdc173d18469a138f3b6a5cb8cad5` |
| Sanitizer campaign identity | `fba68b880339f599601a1eef6984a65f86bb2ed8ea9655aaf2463876e87baeb6` |
| Frozen candidate CLI | `b60ae2056f10fc8abb2a7ecb705baef1d1bbbad1534abf5805c9caa3cb44248b` |
| Sanitizer campaign summary | `730d98d565c806f4871f319c8c86a25a26eb84ad1e33fbe814bbe8a2ecf25534` |
| Sanitizer completion audit | `93db67bd8bc9d257af7ce01b1c7816e6778b2c28d48101cb95e43f2a40f7a8ee` |

The production CLI remains `da73a3b486c9704b15254430d1d9eb9ff5bc4322bd7f50e59c979e0c3ee4d74b`. The final matrix plan and original-reference acceptance contract remain unchanged.
