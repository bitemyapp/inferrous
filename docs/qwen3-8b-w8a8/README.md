# Qwen3-8B W8A8 engine (`inferrous-qwen3`)

Independent Rust/cuda-oxide implementation of standard Qwen3-8B inference. The body uses symmetric per-output-channel INT8 weights and dynamic per-token INT8 activations. Norm, RoPE, SiLU and causal softmax attention retain ordinary floating-point model semantics. Embeddings, residuals, KV cache and output-head weights are BF16. INT8 projections accumulate in INT32; floating-point arithmetic uses FP32. There are no Hadamard rotations, integer attention, lookup-table activations, Q14 RoPE coefficients or 15-bit output head.

Quantization is an explicit baseline choice: max-absolute scale divided by 127, round-to-nearest-even, no calibration or smoothing. This is not a claim that there is one universal W8A8 format. Preparation exports the exact same quantized weights and FP32 channel scales in a cuda-oxide bundle and a compressed-tensors checkpoint usable by comparison engines. The original checkpoint is pinned to `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`.

All preparation, builds, correctness checks and benchmarks were run on one RTX PRO 6000 Blackwell Max-Q workstation. The initial scope is one request at a time, greedy generation, no prompt caching or speculative decoding, at most 4096 prompt-plus-generated tokens. Standard floating-point kernels can differ in reduction order across engines; numerical agreement and performance are reported separately from bitwise regression checks.

The original unoptimized baseline is commit `bd6e4448`. Commit identifiers in these records refer to the private development history this code was extracted from. Start with the [repository README](../../README.md) for setup. See [round 1 record](optimization/round-1.md), [round 2 record](optimization/round-2.md), [round 3 record](optimization/round-3.md), [round 4 record](optimization/round-4.md), [round 5 record](optimization/round-5.md), and [round 6 record](optimization/round-6.md) for timings, profiling and proofs. The [production plan](../plans/production.md) lists the decisions and workstreams needed before this engine can serve traffic, and the [Qwen3.6-35B-A3B FP8 plan](../plans/qwen3.6-35b-a3b-fp8.md) describes the next model. [coverage notes](coverage.md) and [discrepancy notes](discrepancies.md) describe the validation boundary. In particular, the full-model numerical comparison against Torch currently fails its declared tolerance even though the short greedy sequence and independent operator checks pass. This is experimental inference code, not an accuracy-certified model release.

On the workstation, optimized decode is **182.3 / 178.2 / 170.5 tokens/s** for 5 / 129 / 512 prompt tokens (32 generated tokens), versus **144.4 / 143.3 / 142.9** for the fastest tested stock W8A8 decode configuration (measured in round 5). Total request latency is **178 / 193 / 226 ms**; the best stock latency observed across runs is **237 / 237 / 243 ms**. Long prefill remains the 512-token request's largest remaining gap versus stock. See [benchmarks](benchmarks.md) for workloads, timing boundaries, accuracy limits, other configurations and reproduction.

## Reproduce

Commands run from the repository root on Linux with an SM120 GPU, using `clean-work/` as scratch space for bundles, fixtures and results. The cuda-oxide revision is `3d7cbe18f58981b88401ea70dee21e7996427956`; the toolchain is `nightly-2026-04-03`. No CUDA C++ inference kernels or vendor BLAS calls are used by this crate.

```sh
# Build the codegen backend once from the pinned vendor/cuda-oxide.
(cd vendor/cuda-oxide/crates/rustc-codegen-cuda && cargo build --release)
export CUDA_OXIDE_BACKEND=$PWD/vendor/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
crates/inferrous-qwen3/tools/cuda.sh build
export CLEAN_OPERATOR_FIXTURES="$PWD/clean-work/operator-fixtures-64"
export CLEAN_MODEL_BUNDLE="$PWD/clean-work/model-bundle"
crates/inferrous-qwen3/tools/cuda.sh test
target/inferrous-gpu/release/inferrous-qwen3 \
  clean-work/model-bundle --prompt 'The capital of France is' --max-tokens 32
```

`tools/prepare.py --help` describes export of the original pinned BF16 checkpoint to raw Rust tensors plus a compressed-tensors checkpoint under `BUNDLE/hf`. Preparation is offline: weights are quantized once, and only activations are quantized during inference. At model load, the GPU head is packed losslessly into 12-bit codes plus exceptions; every original BF16 bit is restored before computation. This saves 24.2% of head storage without changing weight values. Export refuses an existing destination. The current bundle root is `e51d2517784b99de3802110542e831061489bf00df33db1b412ba3b3d01d21d7`.

The resident model owns eight decode graphs, fixed KV caches and decode scratch. Generation is serialized by a mutex; prefill scratch is reused for the current prompt shape. Every request recomputes its prompt and overwrites the visible KV prefix. Tracing uses eager launches. Graph capture and model loading are outside warm generation timing; first-use allocation still appears in API timing. GPU greedy selection checks all logits, and complete vocabulary hashes remain in every result. Hashing overlaps the next GPU step without changing returned bytes.

Generate operator fixtures with `tools/operator_fixtures.py NEW_DIRECTORY`, using the workstation's Torch environment. Full-model reference traces use `tools/reference.py --model BUNDLE --request REQUEST.json --output NEW_DIRECTORY`. The Rust CLI's `--trace DIR` dumps every layer, final norm and vocabulary logits, plus layer-4 intermediate values used to investigate DISC-001; traced runs are not benchmarks. `tools/compare_traces.py` checks these traces without silently updating thresholds or expected output.

Create requests with `tools/workloads.py BUNDLE NEW_DIRECTORY`. Then run:

```sh
python3 crates/inferrous-qwen3/tools/bench_clean.py \
  --binary target/inferrous-gpu/release/inferrous-qwen3 \
  --model clean-work/model-bundle --workloads clean-work/workloads \
  --output clean-work/new-measurements.jsonl
python3 crates/inferrous-qwen3/tools/check_regression.py \
  clean-work/baseline.jsonl clean-work/new-measurements.jsonl
```

Use `tools/summarize.py` on result files for p50/p95/p99 API latency, first-token latency and decode throughput. It excludes the first three repetitions and requires at least twenty measurements. The percentile tails from twenty samples are descriptive, not production tail-latency estimates.
