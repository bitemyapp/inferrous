# Round 7: paged execution on the RTX 5090

This round is in progress on `feat/kv-cache-batching-rtx5090`. The [acceptance tracker](../../plans/kv-cache-batching-rtx5090.md) remains the completion checklist. These measurements establish individual changes and correctness boundaries; they are not a claim that the new engine beats the fastest stock alternative across the requested matrix.

## Model integration

Commit `820a0ff` introduces bounded paged sessions, ragged row batches, full-prefix recomputation, prefix reuse, capacity-based admission waves and a native RoPE extension helper. It retains the original short-context numerical path as an oracle. Release and checked model tests compare selected tokens and full-vocabulary logit hashes for different row shapes, batch order, stop boundaries, continuation and forks. A session's Rust borrow keeps its engine alive; only initialized complete pages can be shared, and every call fences GPU uses before caller-owned leases can be released.

The initial paged value kernel loaded each key's V values directly from global memory. The retained candidate stages two 64-token tiles in shared memory with asynchronous copies, overlapping the next page's load with ordered FP32 accumulation of the current page. It preserves every product, addition and rounding boundary. The query/key and softmax arithmetic are unchanged. Decode graphs use stable step allocations and ordinary completion edges, avoiding reliance on programmatic dependencies in multirow kernels that do not implement that protocol.

## Paired value-kernel experiment

An 8,192-token prompt generated eight tokens with prefix caching disabled and full-logit verification enabled. Both executables used a 6 GiB paged KV budget, prefill chunk 256 and eager execution. The preserved baseline is `clean-work/kv-batch-baseline/paged-eager-naive`; the candidate is `paged-eager-staged`. Four process runs use A/B/B/A order, each with one warmup and three measured repetitions. Model loading is outside generation timing. All sixteen runs match every generated logit hash and token.

| Measurement | Direct global V loads | Staged V pages | Change |
|---|---:|---:|---:|
| Median generation API latency | 5.4439 s | 4.4841 s | 17.6% lower |
| Median decode throughput | 29.75 tokens/s | 99.18 tokens/s | 3.33× |

Raw files are `clean-work/kv-batch-baseline/pv-abba-{a1,b1,b2,a2}.jsonl`, with process summaries in `pv-abba.log`. Each variant has six measured repetitions. This is a paired tuning experiment, not the full three-warmup/twenty-measurement benchmark matrix. Desktop activity was present; the second baseline block was slightly slower than the first. A complete clock/power time series was not collected. Short and batched performance still need the corresponding retained-change measurements.

## Native-context correctness

The explicit `native_context_ragged_recompute_and_chunk_invariance` model test processes three ragged requests with prompt lengths 8,192, 16,384 and 32,766 and two output tokens each. The last request reaches the configured 32,768-token total limit. A cold cached run with chunk 256 and graph decode, a warm-prefix run, and a fresh full-prefix recomputation run with chunk 128 and eager decode match every token and logit hash. It checks exact processed/reused token counts and confirms that all leases are reclaimed after clearing prefixes.

The release test passes in 305.58 seconds. Its diagnostic generation times are 94.570 seconds cold, 0.523 seconds warm and 201.577 seconds recomputed. These are different workloads and configurations, not a fair speed comparison between cache policies. They demonstrate substantial reusable work and also expose the remaining long-prefill cost. The separate checked run also passes in 321.73 seconds (100.185 seconds cold, 0.553 seconds warm, 211.669 seconds recomputed). Logs are `clean-work/kv-batch-baseline/native-context-tests.log` and `native-context-checked.log`. The test validates execution consistency, not independent long-context model quality. The existing external-model discrepancy remains applicable.

## Stock comparison discipline

Both harnesses accept named ragged batches. The stock recomputation mode submits a fresh one-token request for each complete growing prefix with prefix caching disabled. Unit tests verify those submitted token arrays and retirement of requests that finish earlier. Disabling prefix caching during an ordinary multi-token request is still autoregressive KV reuse, so it is not used as an uncached comparison.

The official SGLang 0.5.22 CUDA 13 image is pinned by digest in `tools/stock_container.sh`, alongside the existing vLLM 0.32.0 image. The version/tag comes from the [upstream release](https://github.com/sgl-project/sglang/releases/tag/v0.5.22). GPU timing remains serial.

Inspection of the installed Humming compressed-tensors schema confirms that channel weight scales are allocated in `param_dtype`, hence BF16 for these runs. Its earlier stock screen therefore does not preserve the exported FP32 weight scales. The installed vLLM compressed-tensors W8A8 and SGLang W8A8 definitions allocate them in FP32. Raw source inspection is preserved under `clean-work/stock/quantizer-scale-source.txt`, `humming-weight-schema.txt` and `sglang-api-source.txt`. The harness records its effective scale dtype and actual GPU name. Kernel arithmetic can still differ even when weights, scales and activation precision match; exact cross-engine equality is not assumed.

The initial current-version smoke screens expose two unsupported defaults on SM120: vLLM compressed-tensors with automatic linear dispatch selects a CUTLASS INT8 operation that rejects SM120, and SGLang's W8A8 path raises `No implemented int8_scaled_mm for current compute capability`. These failures are not timing results. SGLang's compressed-tensors W8A8 source calls the same CUDA entrypoint; a second runtime test of that adapter has not been performed. vLLM's explicit Triton linear backend completes two repetitions each of a single 5-token prompt and a four-sequence 129-token batch in full-recomputation mode. Its metrics report zero cached prefix tokens on all one-token calls, and submitted prefix lengths grow as expected. This is harness validation in eager mode, not the final stock configuration screen. Logs and raw results use `clean-work/stock/vllm032-ct-{auto,triton}-recompute-smoke.*` and `sglang0522-recompute-smoke.*`.

Compute Sanitizer 2026.3 (CUDA 13.4.92, now included in the Nix shell) reports zero memory errors for both paged attention operator tests and for the complete batch CLI with four 129-token prompts and eight generated tokens each. Racecheck reports zero hazards, errors or warnings for the paged operator tests, including the native-context case. Artifacts are `clean-work/kv-batch-baseline/paged-memcheck.log`, `paged-racecheck-attempt2.log` and `batch-cli-memcheck.log`. The initial racecheck invocation used an incorrect executable path and ran no test; the corrected invocation is the evidence. The CLI benchmark smoke also confirms identical full-logit hashes between cached and recomputed modes for all repeated requests. These checks cover the new path; they do not claim exhaustive sanitizer coverage of every requested benchmark shape.

## Output-head scheduling

Interleaving vocabulary rows across batch members lets adjacent warps reuse the same packed head weights in cache. The old scheduling processed an entire vocabulary for one output row before moving to the next, exceeding L2 capacity between uses. The arithmetic of each dot and the output matrix layout remain unchanged. A separate batch-32 verification matches all 256 generated full-logit hashes against the saved original executable. The ordinary ragged/session oracle test also passes.

Paired A/B/B/A runs use 129-token prompts, eight output tokens, no prefix caching, graph decode, prefill chunk 1024 and verification disabled. Each process has one warmup and three measured repetitions; the table combines six measurements per variant. Results are `clean-work/kv-batch-baseline/head-b{1,4,32}-{a1,b1,b2,a2}.jsonl`; complete-hash probes are `head-b32-head-{original,interleaved}-verify.jsonl`.

| Batch | Original median API | Interleaved median API | Original aggregate decode | Interleaved aggregate decode |
|---|---:|---:|---:|---:|
| 1 | 55.229 ms | 55.132 ms | 173.28 tokens/s | 174.51 tokens/s |
| 4 | 101.312 ms | 93.142 ms | 400.14 tokens/s | 447.37 tokens/s |
| 32 | 449.282 ms | 400.130 ms | 1163.42 tokens/s | 1494.97 tokens/s |

The aggregate decode estimate divides total post-first-token outputs by the slowest stream's measured decode interval. These are homogeneous batches; do not apply that estimator to ragged or differently stopped requests. The single-stream difference is small enough to treat as unchanged. Batch 4 and batch 32 improve generation latency by 8.1% and 10.9%, respectively.

## Adaptive prefill policy

A short recomputation pilot with four 129-token prompts and four outputs reduced median API latency from about 198 ms at chunk 256 to 165 ms at chunk 512 and 138 ms at chunk 1024, with identical complete logit hashes. This pilot is not paired evidence for a universal chunk setting. The scheduler now treats the configured chunk as an upper bound and also enforces a score scratch byte budget, based on the largest position that the next chunk could reach. The RTX 5090 capacity policy permits up to 1024 rows with a 1 GiB score limit; it drops to at most 256 rows when the visible context requires a 32K score stride. The PRO 6000 policy remains a separate unmeasured starting point. Full-model tests force a smaller scratch budget to verify scheduling invariance.

The adaptive policy passes all 16 ordinary GPU tests in release (26.57 seconds) and checked release (31.74 seconds), including forced 4 MiB scratch scheduling. The native-context test also passes (303.49 seconds; diagnostic cold/warm/recompute times 92.771/0.518/201.338 seconds), preserving every cross-mode logit hash. This timing does not establish a long-prefill speedup; that remains dominated by attention. The ordinary suite now explicitly ignores two opt-in tasks: native-context model validation and projection profiling.

## Projection measurements

`cuda.sh test --ignored projection_shapes --nocapture` measures projection kernels with CUDA events around captured launch sequences. It cycles through 32 distinct weight allocations, a 512 MiB or 1.5 GiB working set, and excludes three warmups before retaining seven timings. Every candidate's output is compared exactly against the DP4A implementation. This avoids a single small matrix remaining entirely in L2. It measures isolated projections, so candidate dispatch changes still require end-to-end paired validation. The host restricts hardware performance counters to administrators; CUDA-event timing requires no driver setting change.

The experiment passes in 7.21 seconds. Raw timings are `clean-work/kv-batch-baseline/projection-profile.log` and extracted `projection-profile.json`. The original thresholds are generally appropriate, but the RTX 5090 has clear candidates for small batched MLP projections:

| Shape (rows × input × output) | Original kernel and median | Candidate and median |
|---|---|---|
| 2 × 12288 × 4096 | Split-K, 80.1 µs | DP4A, 33.2 µs |
| 4 × 12288 × 4096 | Split-K, 80.2 µs | DP4A, 47.6 µs |
| 8 × 12288 × 4096 | Split-K, 80.1 µs | Tiled, 58.8 µs |
| 16 × 12288 × 4096 | Split-K, 81.8 µs | Tiled, 59.2 µs |
| 16 × 4096 × 12288 | Split-K, 51.8 µs | Tiled, 33.8 µs |

The candidates are retained after the paired model experiment below. The original PRO 6000 and legacy selections remain unchanged. The profile also confirms that replacing the existing 32-row tiled projection with Split-K would be slower, so that proposed direction is rejected before changing production dispatch.

## Paired projection dispatch

The same executable runs the original and tuned projection dispatch through `tuned_projections: false/true`. A/B/B/A suites use 129-token prompts, eight outputs, batches 1/2/4/8/16/32, chunk 1024, 4 GiB KV, no prefix caching, graph decode and verification disabled. Each process has three warmups and six measurements per workload, giving twelve measured samples per variant. Separate verification suites compare every full-vocabulary logit hash, in both cached and fully recomputed modes. All hashes match. `bench_batch.py` now loads one engine for the complete named suite and records the executable SHA-256 in every record.

| Batch | Original cached API | Tuned cached API | Original aggregate decode | Tuned aggregate decode |
|---|---:|---:|---:|---:|
| 1 | 55.654 ms | 55.938 ms | 173.57 tokens/s | 172.50 tokens/s |
| 2 | 78.303 ms | 66.457 ms | 238.98 tokens/s | 299.87 tokens/s |
| 4 | 94.651 ms | 85.095 ms | 439.18 tokens/s | 515.60 tokens/s |
| 8 | 144.188 ms | 134.611 ms | 731.26 tokens/s | 821.63 tokens/s |
| 16 | 241.151 ms | 224.098 ms | 1039.56 tokens/s | 1207.40 tokens/s |
| 32 | 404.493 ms | 404.065 ms | 1489.93 tokens/s | 1489.52 tokens/s |

Cached API latency improves 15.1%, 10.1%, 6.6% and 7.1% at batches 2, 4, 8 and 16. The batch-1 and batch-32 controls use unchanged dispatch and are treated as unchanged. Full-prefix recomputation also remains effectively unchanged: original/tuned median API latencies in batch order are 129.34/130.02, 168.12/167.85, 266.95/269.25, 572.67/565.40, 1082.33/1079.71 and 2098.15/2098.40 ms. Its large prefill row counts mostly use the existing tiled dispatch. These measurements support the small cached-batch change; they do not imply a recomputation improvement.

Raw records and full-logit probes are `clean-work/kv-batch-baseline/projection-{reuse,recompute}-{a1,b1,b2,a2,av,bv}.jsonl`; the summary and script are `projection-abba-summary.json` and `compare_projections.py`. One-second GPU temperature, power, clock, utilization and memory samples are retained in `projection-abba-telemetry.csv`. Among samples with at least 50% GPU utilization, SM clocks span 2557–2872 MHz (median 2625 MHz) and temperature spans 56–82°C. All GPU processes ran serially; desktop activity remained present. The regular release suite passes 16 tests in 30.83 seconds, with native-context and timing experiments explicitly ignored. The stock comparison matrix is still pending.

## Grouped paged attention

Four adjacent query heads share one KV head. The grouped score kernel stages that key tile once, then computes four independent ordered QK dots. The grouped value kernel likewise stages one V tile and four probability rows, preserving each sum's ascending key order. The softmax tree and rounding boundaries are unchanged. Shuffled-page/ragged operator tests match both the original paged probabilities and outputs exactly, then compare against the original contiguous implementation.

The opt-in `grouped_attention_shapes` experiment flushes 256 MiB before each event-timed graph, takes three warmups and seven measured samples, and compares complete outputs for original scores/values, grouped scores only, grouped values only, and both grouped. Refined shapes cover 64/512/4096/32768 score strides and 1/2/4/8/16/64/256 rows where rows fit the stride. Grouped PV loses occupancy below sixteen rows; it is not used there. Grouped scores are selected at sixteen or more rows, at eight or more rows with stride at least 512, or at stride at least 4096. The RTX PRO 6000 policy remains unchanged. Raw operator data are `grouped-attention-profile-refined.log/json` under `clean-work/kv-batch-baseline/`.

A/B/B/A model suites compare `tuned_attention: false/true` using one frozen executable, the extended bundle, 4 GiB KV, chunk 1024, no prefix caching, graph decode and verification disabled. Each process has one warmup and three measured repetitions, giving six measurements per variant. Separate full-logit probes match every output in both modes. Short prompts generate eight tokens; the 8192-token prompt generates eight cached outputs or two fully recomputed outputs. Comparisons within each mode have identical work; cached and recomputed columns are not interchangeable workloads.

| Workload | Original cached API | Grouped cached API | Original recomputed API | Grouped recomputed API |
|---|---:|---:|---:|---:|
| Batch 1, prompt 129 | 55.275 ms | 54.187 ms | 127.514 ms | 119.981 ms |
| Batch 2, prompt 129 | 66.002 ms | 64.227 ms | 166.998 ms | 151.476 ms |
| Batch 4, prompt 129 | 84.875 ms | 81.081 ms | 269.610 ms | 238.115 ms |
| Batch 8, prompt 129 | 134.384 ms | 127.304 ms | 564.269 ms | 506.810 ms |
| Batch 16, prompt 129 | 222.073 ms | 207.204 ms | 1081.584 ms | 960.250 ms |
| Batch 32, prompt 129 | 403.244 ms | 375.633 ms | 2107.165 ms | 1864.230 ms |
| Batch 1, prompt 8192 | 4308.044 ms | 2056.210 ms | 8623.893 ms | 4046.580 ms |

The 8K API improvements are 52.3% cached and 53.1% recomputed. Solo 8K cached decode remains approximately unchanged at 103.95 versus 105.80 tokens/s; most of the improvement is prefill. Short cached batches improve 2.0–6.8%, and short recomputation improves 5.9–11.7%. Artifacts are `grouped-attention-{reuse,recompute}-{a1,b1,b2,a2,av,bv}.jsonl`, `grouped-attention-abba-summary.json`, `compare_grouped_attention.py` and `grouped-attention-abba-telemetry.csv`. Native-context checked and sanitizer validation of the combined dispatch are recorded below.

## Stock batch and native-context screening

The first 4K-capacity screen covers 129-token prompts, eight outputs and batches 1/2/4/8/16/32 with FP32 weight scales and the Triton INT8 linear backend. Three warmups and six measured samples per workload favor FlashAttention with graphs and compilation disabled over compiled FlashAttention or compiled Triton attention. The graph-only capture list covers the batch sizes instead of inheriting the old solo-only setting. Its median API times are 71.95/72.47/89.32/109.36/155.09/224.84 ms. These are unpaired screening results, not final comparison samples. Artifacts are `clean-work/stock/vllm032-ct-triton-*-batch-screen.*`.

A subsequent screen uses the extended bundle, 32768 total context, an explicit 12 GiB stock KV budget, and an added solo 8K prompt. Graph-only Triton attention leads the sampled short batches 1/2/4/32, while FlashInfer leads batches 8/16 and the 8K prompt. The 8K API medians are 451.71 ms for FlashAttention, 548.78 ms for Triton attention and 436.06 ms for FlashInfer. Inferrous's current grouped attention is still slower on this long-prefill case; no universal stock win is claimed. These results also have three warmups and six measured samples and need the final matched comparison runs. Artifacts are `vllm032-ct-triton-{fa,triton,flashinfer}-native-screen.*` and `vllm032-ct-triton-native-screen-summary.json`.

B12X is absent from the official image. The separate Dockerfile installs vLLM 0.32.0's declared `b12x==1.5.1` optional dependency without changing other packages, and verifies all of its active dependency constraints. Both its default and graph-only native-context runs initially fail during the legacy CuTeDSL precompile hook because a real tensor lacks `fake_mode`. The supported `enable_cutedsl_warmup: false` setting avoids that hook and both retries complete all 63 records. Runtime compilation remains enabled; JIT warnings occur in the excluded initial warmup. The graph-only retry's median API times for short batches 1/2/4/8/16/32 and solo 8K are 80.81/80.85/81.74/93.74/143.75/227.70/451.71 ms; compiled B12X gives 102.35/102.05/103.71/130.60/167.39/260.26/489.01 ms. It does not lead these sampled workloads. Successful files are `vllm032-ct-triton-b12x-lazy{,-compiled}-native-screen.jsonl/log`; their immutable image ID is recorded in each row. Initial failures remain separate evidence and are not performance results.

## Packed head groups

The next head kernel decodes each losslessly packed BF16 weight once for four or eight independent output rows. Each output preserves the original lane-strided FP32 sum, warp reduction and final BF16 rounding. Two const specializations share one implementation; padded group members read an admitted row but never publish an output. The opt-in `grouped_head_shapes` test checks every complete vocabulary output for every batch size from 1 through 64, using reversed input-row selection and the real model head. All comparisons match exactly. The operator profile uses 256-thread blocks for all variants, three warmups and seven measured CUDA-event samples after cache eviction. The production original path retains its existing 128-thread launch; end-to-end comparison below tests that actual dispatch.

The 32-row isolated head drops from roughly 11.6 ms with one dot per warp to 6.1 ms with four and 4.9 ms with eight. Group width eight is slower for 3–4 and 9–12 outputs, where padding dominates; those use width four. One and two outputs retain the original path. The other admitted output counts use width eight. `tuned_head: false` keeps the original interleaved head for comparison, and the PRO 6000 policy remains unchanged. The complete 64-shape test passes in 25.69 seconds; raw files are `grouped-head-all-profile.log` and the earlier four/eight-head profile files under `clean-work/kv-batch-baseline/`.

A/B/B/A suites compare that final dispatch using one frozen executable, the extended bundle, 12 GiB KV, chunk 1024, no prefix caching, graph decode and verification disabled. Three warmups plus four measured samples per process give eight measured samples per variant. Each homogeneous request has 129 prompt tokens and eight outputs. A twelve-request ragged workload varies prompt and output lengths, exercising shrinking output groups and graph shapes. Separate verification probes match every full-logit hash across original/grouped heads and cached/recomputed modes. Cached work counts and full-prefix recomputation work counts also match exactly.

| Workload | Original cached API | Grouped cached API | Original recomputed API | Grouped recomputed API |
|---|---:|---:|---:|---:|
| Batch 1 | 54.376 ms | 54.326 ms | 119.392 ms | 119.590 ms |
| Batch 2 | 64.020 ms | 64.015 ms | 149.442 ms | 149.610 ms |
| Batch 4 | 81.031 ms | 77.002 ms | 234.288 ms | 231.257 ms |
| Batch 8 | 126.846 ms | 116.058 ms | 502.504 ms | 491.882 ms |
| Batch 16 | 206.631 ms | 182.026 ms | 954.270 ms | 928.715 ms |
| Batch 32 | 372.467 ms | 316.569 ms | 1848.993 ms | 1784.461 ms |
| Ragged 12 | 124.569 ms | 116.958 ms | 222.747 ms | 214.488 ms |

The unchanged one/two-row controls remain effectively unchanged. Cached improvements at batches 4/8/16/32 are 5.0/8.5/11.9/15.0%, and the ragged case improves 6.1%. Recomputed improvements are smaller because prefill dominates. Files are `grouped-head-{reuse,recompute}-{a1,b1,b2,a2,av,bv}.jsonl`, `grouped-head-abba-summary.json`, `compare_grouped_head.py` and `grouped-head-abba-telemetry.csv`. The ordinary release and checked suites both pass 16 tests (26.55/27.08 seconds), with four opt-in long-context/profiling tests explicitly ignored.

The expanded projection profile also tests the existing TMA kernel below its production threshold. It is slower at 16–64 rows for the sampled shapes, and slower for Q/O/down at 128 rows. The 128-row gate/up case is promising (about 42.5 µs versus 75.1 µs with tiled projection), but is not yet retained; it requires a separate end-to-end experiment. Raw data are `projection-small-tma-profile.log/json`.

## Combined validation and remaining profile

The native-context model test passes with grouped attention and heads in release (159.08 seconds) and checked release (215.81 seconds). Diagnostic cold/warm/recomputed times are 45.947/0.291/103.867 seconds and 64.051/0.373/142.394 seconds respectively. All full-logit hashes remain identical across cached, prefix-reused and fully recomputed ragged 8K/16K/near-32K requests and across chunk sizes. These different test workloads are correctness checks, not a paired performance comparison. Logs are `grouped-native-{release,checked}.log`.

Compute Sanitizer reports zero memory errors and zero race hazards/warnings for both paged operator tests, including the grouped native-context path. A full ragged twelve-request model run also reports zero memory errors and matches all 56 full-logit hashes and selected tokens against the uninstrumented verification run. Logs are `grouped-paged-memcheck.log`, `grouped-paged-racecheck.log` and `grouped-ragged-memcheck.log`.

The [CUPTI collector](../profiling.md) provides kernel timestamps without changing driver settings. Its first three-repetition batch-32 trace has 23,436 records with zero dropped or invalid intervals. Of the eager kernel time, `linear_large` consumes 69.1%, `linear_tiled` 5.6%, SwiGLU 5.3% and the two grouped attention kernels 7.6% combined. Of graph kernel time, `linear_tiled` consumes 56.0% and the eight-row packed head 33.9%. These phases describe launch mechanisms: the graph also runs the final 32-row prefill step in this workload. The profile includes all traced repetitions, and its instrumented API times are not benchmark results. All these kernels report zero local-memory bytes per thread. Artifacts are `grouped-b32-kernels.csv` and `grouped-b32-kernel-summary.json`.

The remaining short-batch gap is therefore concentrated in projections and the head. Candidate experiments are overlapping TMA loads with projection compute and improving input reuse in grouped heads. The longer-context stock gap also requires further attention work. The full scenario matrix and universal stock-performance requirement remain open.
