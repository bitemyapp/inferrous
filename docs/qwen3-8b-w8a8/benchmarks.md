# Qwen3-8B INT8 benchmark comparison

Measured on one RTX PRO 6000 Blackwell Max-Q (96 GB, 300 W), 2026-10-02–03. The independent cuda-oxide baseline is `bd6e4448`; the retained optimized runtime is `b88d941f`. Final measurements are in `round5/round5-final.jsonl`, with A/B/B/A controls in the same directory; the two stock winners were rerun in `round5/round5-stock-fa2.jsonl` and `round5/round5-stock-b12x.jsonl`. Current raw measurements, percentile summaries, environment details, profiles and regression reports are in [2026-10-03 results summary](results/round-3.md); earlier comparisons remain in [2026-10-02 results summary](results/rounds-1-2.md).

**Accuracy limitation:** independent operator tests pass, and every optimization preserves the original Rust baseline's complete vocabulary logits. However, the full-model Torch numerical comparison fails its declared tolerance. Matching three greedy tokens does not establish quality. See [discrepancy notes](discrepancies.md). These are performance measurements of an experimental implementation.

**Precision scope:** the historical Humming tables below predate the harness's effective weight-scale dtype check. The later [round 7 audit](optimization/round-7.md) found that the screened Humming schema allocates channel scales in the model parameter dtype, BF16, while the export contains FP32 scales. That later inspection does not establish the exact runtime dtype of every older version. The older timings remain historical observations; they are not evidence of eligibility or a win under the current strict FP32-scale comparison contract. Current matched reports require recorded FP32 stock scales.

## Results

All rows start from the same Qwen3-8B revision and checkpoint containing exported INT8 weights and FP32 channel scales; this does not establish the stock backend's runtime scale dtype. The body has per-channel symmetric INT8 weights and dynamic per-token INT8 activations; embeddings, output-head values and KV cache remain BF16. The GPU head uses lossless 12-bit storage plus exceptions, reconstructing the original BF16 bits before arithmetic; it introduces no additional quantization. No FP8 substitution, smoothing, rotations, prefix caching or speculative decoding. One request at a time, greedy generation, exactly 32 output tokens, EOS ignored, context limit 4096. Each run uses three warmups and twenty measured resident requests. The sixth pass runs original/candidate/candidate/original serially and pools forty measured requests per implementation and shape; stock columns retain twenty per shape and come from the fifth pass. GPU jobs run serially, without fixed application clocks.

| Prompt tokens | Original decode tok/s | Optimized decode tok/s | Stock FA2 decode tok/s | Stock B12X decode tok/s |
|---:|---:|---:|---:|---:|
| 5 | 66.0 | **182.3** | 136.1 | 144.4 |
| 129 | 57.1 | **178.2** | 140.4 | 143.3 |
| 512 | 40.3 | **170.5** | 140.2 | 142.9 |

Both stock columns use vLLM + Humming INT8 + native full CUDA graphs. FA2 is FlashAttention; B12X is an alternative Blackwell attention backend. B12X wins stock decode, but FA2 wins stock end-to-end latency for these 32-token requests:

| Prompt tokens | Original API ms | Optimized API ms | Stock FA2 API ms | Stock B12X API ms |
|---:|---:|---:|---:|---:|
| 5 | 502.8 | **178.3** | 244.6 | 252.4 |
| 129 | 743.4 | **192.9** | 237.4 | 254.0 |
| 512 | 1572.6 | **226.2** | 243.3 | 254.8 |

The clean implementation exceeds the stock decode winner by **26.2% / 24.4% / 19.3%** and reduces request latency versus FA2 by **27.1% / 18.8% / 7.0%**. Stock was not rerun in the sixth pass; these compare against the fifth pass's fresh stock runs. Long requests remain the closest comparison: the 512-token median is 17.1 ms below that stock run and 16.5 ms (6.8%) below the best stock median observed, 242.7 ms. Long prefill (first step 44.1 ms versus stock's roughly 22 ms first token) is the main remaining gap.


The fourth pass adds gate/up/activation fusion and CTA-local bulk tensor copies for large prefill projections. A/B/B/A runs of the preceding and new binaries pool forty measured requests per shape, retaining every individual run:

| Prompt tokens | Previous decode tok/s | New decode tok/s | Previous API ms | New API ms |
|---:|---:|---:|---:|---:|
| 5 | 165.95 | 167.09 | 195.22 | 193.96 |
| 129 | 161.36 | 162.18 | 211.26 | 210.37 |
| 512 | 156.28 | 157.34 | 244.54 | 241.91 |

This is another **0.5–0.7% decode gain** and **0.4–1.1% lower request latency**. The long first step drops from 45.91 to 44.61 ms; the short/medium first steps are effectively unchanged at 8.16 / 18.96 ms. Recorded stock FA2 first-token latencies are 15.9 / 16.0 / 22.3 ms, so long prefill remains the main gap. The opening cold baseline was faster than later controls; no improvement is attributed to that clock/temperature difference. All logit hashes are unchanged. See [round 4 record](optimization/round-4.md) for rejected variants, compiler routing, the removed cluster-copy memory regression, and validation.

The fifth pass retains pinned double-buffered result transfers, native nearest-even integer conversion, and hardware integer warp reductions. It also evaluates and removes lossless INT8 compression, an overlapped TMA variant, four-step graphs, a transposed tensor-core decode and two exact BF16 reciprocal algorithms. All preserve hashes, but those larger experiments did not establish a speed win.

| Prompt tokens | Previous decode tok/s | New decode tok/s | Previous API ms | New API ms |
|---:|---:|---:|---:|---:|
| 5 | 167.134 | 169.625 | 193.908 | 191.081 |
| 129 | 162.258 | 164.593 | 210.222 | 207.475 |
| 512 | 157.350 | 159.470 | 241.843 | 238.948 |

A/B/B/A gives **1.3–1.5% more decode throughput** and **1.2–1.5% lower API latency**. Both new runs beat both controls on all three shapes. First-step medians are 8.064 / 18.836 / 44.271 ms. These are ordinary single-step measurements; the rejected multi-step timing representation is not retained. See [round 5 record](optimization/round-5.md) for proofs, rejected implementations and the full validation record.

The sixth pass keeps DRAM busy between projections. It adds L2 prefetch of upcoming weights (from attention-window CTAs and from each projection warp before it waits), a KV-prefix prefetch during QKV, hoisted cold loads in the latency-bound kernels, and programmatic graph edges with `griddepcontrol`. Prefill is unchanged. Serial A/B/B/A against the round-5 executable pools forty measured requests per shape:

| Prompt tokens | Previous decode tok/s | New decode tok/s | Previous API ms | New API ms |
|---:|---:|---:|---:|---:|
| 5 | 170.278 | 182.270 | 190.356 | 178.290 |
| 129 | 165.292 | 178.205 | 206.480 | 192.860 |
| 512 | 160.138 | 170.496 | 237.988 | 226.197 |

That is **6.5–7.8% more decode throughput** and **5.0–6.6% lower API latency**, with every hash unchanged. Both new runs beat both controls on all three shapes. First-step medians are unchanged at 7.95 / 18.64 / 44.12 ms. See [round 6 record](optimization/round-6.md) for the step-time accounting, microbenchmarks, rejected variants and validation.

Timing boundaries matter: Rust's first step excludes request input/scratch allocation; stock first-token latency includes dispatch. Decode is 31 divided by the interval for tokens 2–32. Rust step timing includes graph launches, full logit transfer and GPU greedy selection; vLLM uses first/last token timestamps. Public API timing brackets each generation call. Rust additionally includes full-vocabulary checksumming and detokenization; hashes overlap the next GPU step, with the final hash inside API timing. vLLM disables detokenization. Both use persistent allocation pools; Rust resizes prefill scratch when the prompt length changes. Every prompt is recomputed, with no prefix reuse. Loading/JIT/decode capture is outside warm measurements. These boundaries show the usable implementations, not isolated kernel throughput. Output sequences need not agree across engines; shape and generated-token count do. No accuracy equivalence is inferred from shared weights.

The separate memory measurements are instrumented runs and do not contribute to throughput timings. Original and first-pass Rust versions sampled 10,286 MiB of process GPU memory (`memory.json`). The second pass sampled **10,548 MiB** (`round2-memory.json`). The third pass retains those full-capacity KV caches and scratch, stores the head losslessly, and samples **10,256 MiB** (`round3-memory.json`). The fourth pass remains at **10,256 MiB** (`round4/round4-memory.json`); the rejected cluster-copy variant increased memory and is not retained. These are sampled maxima, not exact peaks. The fifth pass remains at **10,256 MiB** (`round5/round5-memory.json`); the new pinned transfer buffers reside on the host.

## Which stock harness

The fastest **locally tested** stock configuration in these historical solo workloads was vLLM 0.30.0 with unmodified upstream Humming, dynamic INT8 activations and native CUDA graphs. FlashAttention had the lowest measured stock request latency; B12X had the highest measured stock decode throughput. The missing runtime-scale verification described above prevents using those runs as the current precision-matched baseline. These observations also do not establish rankings for other engines, batch sizes or context lengths. Humming supports INT8 activation and weight types; its other activation types must not be substituted in this comparison. See [Humming's support matrix](https://github.com/vllm-project/humming#support-matrix).

| vLLM configuration | Decode tok/s, 5 / 129 / 512 | API ms, 5 / 129 / 512 |
|---|---|---|
| Triton INT8, default compilation/graphs, FA2 | 77.9 / 77.8 / 77.8 | 414.9 / 416.9 / 424.7 |
| Triton INT8, eager, FA2 | 113.3 / 110.9 / 110.2 | 292.3 / 298.5 / 306.5 |
| Triton INT8, graph capture size 1, FA2 | 77.9 / 77.8 / 77.7 | 415.1 / 417.0 / 425.2 |
| Triton INT8, V1 runner, FA2 | 78.2 / 78.2 / 78.1 | 413.1 / 415.4 / 423.5 |
| Humming INT8, eager, FA2 | 78.1 / 76.9 / 77.2 | 423.8 / 430.1 / 428.5 |
| Triton INT8, native full graphs, FA2 | 126.6 / 130.1 / 129.3 | 257.7 / 256.5 / 267.2 |
| Humming INT8, native full graphs, FA2 (previous pass) | 140.6 / 140.5 / 140.3 | 236.9 / 237.4 / 242.7 |
| Humming INT8, native full graphs, FA2 (third pass) | 136.1 / 140.4 / 140.0 | 243.8 / 237.2 / 244.0 |
| **Humming INT8, native full graphs, FA2 (fifth pass)** | **136.1 / 140.4 / 140.2** | **244.6 / 237.4 / 243.3** |
| Triton INT8, eager, B12X | 37.8 / 37.7 / 37.7 | 877.9 / 878.9 / 878.8 |
| Humming INT8, native full graphs, B12X (previous pass) | 144.4 / 143.4 / 142.9 | 251.7 / 253.5 / 253.8 |
| Humming INT8, native full graphs, B12X (third pass) | 144.4 / 143.3 / 142.9 | 252.3 / 254.1 / 254.8 |
| **Humming INT8, native full graphs, B12X (fifth pass)** | **144.4 / 143.3 / 142.9** | **252.4 / 254.0 / 254.8** |

Native graphs use supported settings: compilation mode 0, graph mode FULL, capture sizes `[1]`, custom operations `['all']`. This avoids the slower default compiled configuration here. Default compiled Humming also encountered a Torch `RecursionError`; disabling compilation while retaining native CUDA graphs resolved it without source patches. B12X requires block size 128 in this setup.

Other candidates screened:

- vLLM's default Cutlass INT8 path rejects SM120. The stock Triton path and Humming work, as do the Rust INT8 tensor-core kernels; this is a backend limitation, not absence of INT8 hardware. See the pinned [Cutlass](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py) and [Triton](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/triton.py) implementations.
- SGLang 0.5.21 / sgl-kernel 0.4.7: the required `int8_scaled_mm` probe rejects this device capability. No successful engine timing is claimed. Its [W8A8 implementation](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/layers/quantization/w8a8_int8.py) routes through that operator.
- TensorRT-LLM: the current PyTorch quantization matrix did not provide a matching INT8 W8A8 path for this experiment; the old TensorRT backend has been removed. This was documentation screening, not a benchmark or proof that every historical version is incapable. Sources: [quantization matrix](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html), [backend removal](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html).
- LMDeploy: its model matrix did not verify Qwen3 CUDA W8A8 support; no timing is claimed. See [supported models](https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html).

## Reproduce the stock winner

Commands run from the repository root, with scratch files under `clean-work/`. Environment: `clean-work/vllm-venv`. Pinned components:

| Component | Version |
|---|---|
| vLLM | 0.30.0 |
| Torch | 2.13.0+cu130 |
| Transformers | 5.18.0 |
| compressed-tensors | 0.17.0 |
| humming-kernels | 0.1.16.post13+g5d6e603a0 |
| Humming source | `5d6e603a0332f85ba1d4c3da05298092d9a4a448` |
| B12X | 1.3.0 |
| Driver | 580.159.04 |
| CUDA_HOME for stock engines | /usr/local/cuda-13.1 |

The Humming package was built from the pinned official source without patches. The normal compressed-tensors **linear-backend adapter** does not carry the activation schema into Humming in this version. Use the **Humming quantization method**, which reads the checkpoint's INT8 input schema. The harness asserts that schema before running. This distinction prevents accidentally timing W8A16.

```sh
CUDA_HOME=/usr/local/cuda-13.1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  clean-work/vllm-venv/bin/python crates/inferrous-qwen3/tools/bench_engines.py \
  vllm --model clean-work/model-bundle/hf \
  --hardware-profile rtx-pro-6000 \
  --requests clean-work/workloads/requests.json \
  --quantization humming --graphs-without-compile --attention-backend FLASH_ATTN \
  --output clean-work/new-stock-run.jsonl --repeat 23
```

For the decode winner, change `FLASH_ATTN` to `B12X` and use another output path. For the best Triton run, change quantization to `compressed-tensors` and add `--linear-backend triton`. All harnesses preserve raw timestamps and token IDs. `tools/summarize.py` reproduces p50/p95/p99 API latency and throughput from the JSONL files. These twenty- or forty-sample tails are descriptive, not production p99s.

## Validation and remaining gap

The second pass adds shared-memory INT8 prefill tiles, exact norm/quantization fusion, coordinate-major K caches, shared and pipelined attention tiles, grouped query-head prefill, CUDA decode graphs, GPU greedy selection, overlapping logit hashes, producer-computed MLP maxima, combined Q/K preparation, and grouped decode projections. Each retained lever has its own commit, measurements and behavior argument in [round 2 record](optimization/round-2.md). The first pass is documented in [round 1 record](optimization/round-1.md). Slower experiments were removed; their measurements remain available. The third pass adds lossless BF16 head packing and fused decode projection/residual sums, documented in [round 3 record](optimization/round-3.md). The fourth pass fuses gate/up with SwiGLU and stages large prefill tiles using CTA-local TMA, documented in [round 4 record](optimization/round-4.md). The fifth pass adds pinned readback, native rounding and integer reductions, documented in [round 5 record](optimization/round-5.md).

Nine tests pass in optimized and checked builds: independent operators, distinct-weight fused gate/up/activation, 1,328,443 native-rounding boundary and bit-pattern inputs against independent f64 nearest-even, exhaustive host/GPU BF16 reconstruction, admission/ties, capture cleanup and nineteen changing-input eager/graph comparisons. The latter cover seventeen prompt lengths through 4093, every graph capacity, long-to-short cache reuse and stop handling. All 138 final candidate requests preserve the original token IDs and all 4,416 complete-vocabulary hashes. The checked build matches all three workloads. Operator/argmax/packing memcheck, racecheck and synccheck, plus 512-token prefill and three-output full-model memcheck, report zero errors or hazards. Rust formatting and Python syntax checks pass. The restored optimized binary has the same SHA-256 as the measured binary. Results are retained beside the raw measurements. None of these replaces the still-failing independent full-model numerical gate.

The final 512-token profile attributes 61.4% of GPU time to decode projections, 8.2% to the packed BF16 head, 9.0% to large prefill matrices and 5.1% to prefill attention. Longer prefill is the remaining performance target; preserving exact baseline arithmetic rules out replacing ordered floating-point reductions with a different reduction schedule without a separate numerical/quality decision.

## Paged and stock batch harnesses

Generate the complete native-context matrix on `wx-workstation` with the selected hardware configuration:

```sh
python3 crates/inferrous-qwen3/tools/make_batch_matrix.py \
  --config configs/qwen3-8b-w8a8/rtx-pro-6000.json \
  --output clean-work/new-comparison-matrix
```

The new directory contains one shared workload file, cold/prefix-enabled configurations, an explicit manifest and capacity-rejection probes. The 96 homogeneous cases cross prompts 5/129/512/2K/4K/8K/16K/near-32K with batches 1/2/4/8/16/32 and output counts 2/32. Near-native prompts reserve their requested outputs within 32,768 tokens. Two ragged cases and one page-growth case bring the total to 99 scenarios, each required in cold reuse, prefix-enabled reuse and true recomputation. Cases exceeding simultaneous residency are retained and annotated with predicted cold admission waves, rather than silently omitted; actual admission and memory behavior still require runtime evidence. Prefix-enabled results must report real cache reuse, including eviction or zero-hit cases. The generator uses the provided KV budget and kernel flags, sets batch capacity 32 and disables logit hashing for timing. It does not select a winning kernel policy or run a benchmark.

Both harnesses traverse workload names in sorted order, matching Rust's `BTreeMap`, so input JSON insertion order cannot change prefix-cache history between engines. Three warmups and twenty measured repetitions are required for final timing tables; generated inputs, passing harness tests and tuning pilots do not satisfy that measurement gate.

`tools/bench_batch.py` runs the paged engine using a JSON object mapping workload names to request arrays. Each array may contain homogeneous or ragged prompts and different positive output limits; timing workloads use empty stop lists. It records every repetition, verifies complete deterministic outputs, and checks the exact processed-token count for full recomputation. The [session guide](sessions.md) describes configuration, native-context preparation and timing boundaries.

Run separate correctness probes with `verify_logits: true`, then compare all binaries and cache modes with `tools/check_batch_regression.py`. Supply the exact workload manifest and expected repetition count. The checker rejects missing workloads, duplicate/reordered runs, mixed model/device/configuration identities within a file, disabled or incomplete hashes, changed requests and incorrect cached/recomputed token accounting. It compares every token, full-vocabulary logit hash and finish reason across all inputs; a matching selected token cannot hide changed logits. The report includes input-file and executable SHA-256 identities. Run this and all other tests/benchmarks on `ssh wx-workstation` for the current PRO 6000 work.

```sh
python3 crates/inferrous-qwen3/tools/check_batch_regression.py \
  path/to/original-reuse.jsonl path/to/candidate-reuse.jsonl \
  path/to/original-recompute.jsonl path/to/candidate-recompute.jsonl \
  --workloads path/to/verification-workloads.json --repeat 1 \
  --output path/to/new-hash-verification.json
```

`tools/bench_engines.py` accepts the same named arrays, while retaining compatibility with the earlier single-request workload objects. `--cache-mode reuse` uses ordinary multi-token generation. `--cache-mode recompute` submits the entire growing prefix as a new one-token request at each step, retires finished sequences, and requires prefix caching disabled. It retains each step's submitted lengths, elapsed times and backend metrics. Two host unit tests verify those submission sequences independently of a GPU. Run them with `python -m unittest discover -s crates/inferrous-qwen3/tools/tests -v` inside the Nix shell.

The pinned stock containers are launched with `tools/stock_container.sh`, which supports `vllm` and `sglang` and mounts the host driver on NixOS without installing a system container runtime. Images that expect the NVIDIA container runtime to supply `nvidia-smi` can instead use `NVIDIA_SMI_BINARY=/usr/bin/nvidia-smi` to bind a compatible host executable read-only. This is needed by the pinned SGLang image on the PRO workstation; a missing utility is a launch failure, not evidence of an unsupported inference kernel. The executable's loader and library dependencies must be available inside the image. The current vLLM compressed-tensors automatic INT8 backend rejects SM120; select its supported Triton backend explicitly. For example, from the repository root:

```sh
crates/inferrous-qwen3/tools/stock_container.sh vllm python3 \
  /workspace/crates/inferrous-qwen3/tools/bench_engines.py vllm \
  --model /workspace/clean-work/model-bundle/hf \
  --requests /workspace/clean-work/batch-workloads.json \
  --output /workspace/clean-work/stock-batch-results.jsonl \
  --hardware-profile rtx5090 --max-context 4096 --max-num-seqs 4 \
  --quantization compressed-tensors --linear-backend triton \
  --cache-mode reuse --repeat 23
```

Use a new output path for every run. Increasing context requires a matching workload, engine limit and memory budget; cold/warm prefix policy and verification overhead must be stated separately. Preserve failed engine logs and verify the expected number of JSONL records before accepting a result: a stock worker failure can terminate its parent without a useful process exit status. The current SGLang W8A8 kernel also rejects SM120. Humming runs but converts these FP32 per-channel weight scales to BF16; its effective precision is recorded separately. The [active round 7 record](optimization/round-7.md) distinguishes tested configurations, unsupported paths and the unfinished comparison matrix.

For graph-only stock screening, add `--graphs-without-compile`. Its default capture sizes now cover powers of two through the configured sequence capacity, plus that capacity itself; it no longer silently captures only batch 1 for a batched run. `--graph-capture-sizes 1 2 4 8 16 32` makes the list explicit, while `--graph-size-one` retains the original solo experiment. Raw records include the chosen sizes and immutable container image identifier.

For matched KV capacity, vLLM accepts `--kv-cache-bytes 12884901888` for the RTX 5090 profile's 12 GiB budget. This explicitly controls KV allocation instead of deriving it from the memory fraction; the raw record retains both arguments. `--disable-cutedsl-warmup` selects vLLM's supported `enable_cutedsl_warmup: false` setting for diagnosing its legacy precompile hook. It does not disable runtime kernel compilation or the benchmark's excluded warmup repetitions. Record the flag and check whether JIT work remains in measured repetitions.

The official vLLM image omits B12X. This optional stock variant installs exactly the `b12x==1.5.1` extra declared by vLLM 0.32.0, retaining the base image's other packages and unmodified engine code:

```sh
docker build --tag inferrous-vllm-b12x:0.32.0-1.5.1 \
  - < crates/inferrous-qwen3/tools/stock-vllm-b12x.Dockerfile
# Use stock_container.sh vllm-b12x for this image and --attention-backend B12X.
```

Use `tools/summarize_batch.py` for the new batch schemas:

```sh
python crates/inferrous-qwen3/tools/summarize_batch.py \
  clean-work/batch-results.jsonl clean-work/stock-batch-results.jsonl \
  --output clean-work/batch-summary.json
```

It excludes three warmups per workload per file and requires twenty measured samples per distinct executable/configuration/workload by default. Tuning pilots must explicitly lower `--min-samples` and label the smaller sample count. Complete request arrays, model roots, hardware, executable/image identities and settings separate series; reordered, missing or duplicate repetitions and incomplete outputs fail validation. The common host-call metrics are API latency and total output tokens divided by that latency, which also applies to ragged batches. Per-stream first-token/completion/post-first-output metrics retain the engine's native boundary: local generation-call timing, vLLM's native latency and monotonic timestamps, or recomputation-harness step completions. Do not treat those boundaries as interchangeable host-call measurements. Three additional host tests cover ragged completion accounting, timestamp domains and incomplete outputs. Percentile tails at these sample counts remain descriptive.

Timing validation checks individual local steps before summing them, requires finite positive API durations, rejects nonfinite or negative native timings, and requires monotonic recomputation completions and ordered first/last-token timestamps. Invalid intermediate values cannot be hidden by a positive total or silently omitted from decode-rate samples. Missing optional native metrics remain unknown.

When local records include `prefix_cache_stats`, the summary reports per-call differences of the lifetime lookup, hit, miss, matched-token, eviction, replacement and clear counters, along with after-call resident entry/token counts. `prefix_hit_fraction` is hits divided by lookups for each call; calls with no lookups contribute no fraction sample. Decreasing counters, inconsistent lookup totals and lookups under a disabled prefix policy fail validation. Older records and stock engines without these counters contribute unknown values, not zeros. These metrics describe the automatic prefix cache; `reused_tokens` also reflects the actual token rows avoided by execution. See the [session metrics contract](sessions.md#rust-ownership) for retries, concurrent callers and shared physical pages.

The `reused_tokens` distribution includes the local engine's reused-token counter and vLLM's native `num_cached_tokens` values summed across the batch. vLLM defines this field as prefix-cache hits, distinct from within-request decode KV. Recomputation checks every one-token submission and rejects any nonzero prefix reuse; cold stock records also reject hits when prefix caching is disabled. Missing native counters remain unknown (an empty sample list, summarized as `null`), never implicit zero-hit evidence. Invalid counts, counts exceeding the submitted prefix and incomplete per-step metadata fail validation. Cache-count availability for other stock engines must be established separately; timing success alone does not prove warm-prefix reuse.

Use `tools/compare_batch_performance.py` to compare complete matched series against every supplied stock candidate, separately for each cache/prefix mode:

```sh
python3 crates/inferrous-qwen3/tools/compare_batch_performance.py \
  --local path/to/local-a1.jsonl path/to/local-a2.jsonl \
  --stock path/to/stock-triton.jsonl path/to/stock-flashinfer.jsonl \
  --workloads path/to/workloads.json \
  --output path/to/new-performance-comparison.json
```

Every input must contain the complete workload set with matching request arrays and contiguous repetitions; distinct files can contribute samples to the same executable/configuration. The default excludes three warmups per file and requires twenty measurements per candidate and workload. The checker rejects duplicate file contents, changed identities within an input, verification-enabled local timings, precision changes, unmatched model/device/context/sequence/KV capacities, prefix-policy differences and incomplete recomputation submissions. It selects the lowest stock API median independently for each workload and reports local losses as well as gains. File hashes, candidate identities, sample counts and descriptive percentiles remain in the report. This establishes only the fastest supplied eligible candidate: exhaustive stock screening, numerical agreement, sanitizer gates and coverage of every required mode remain separate acceptance requirements. A stock engine whose actual KV capacity is not established cannot enter this explicitly matched-capacity comparison.
