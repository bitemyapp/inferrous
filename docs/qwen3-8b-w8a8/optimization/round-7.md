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

These candidates have not yet been retained in the model dispatch. The original PRO 6000 selection remains unchanged. The profile also confirms that replacing the existing 32-row tiled projection with Split-K would be slower, so that proposed direction is rejected before changing production dispatch.
