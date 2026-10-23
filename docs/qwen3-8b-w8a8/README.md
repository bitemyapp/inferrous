# Qwen3-8B W8A8 engine (`inferrous-qwen3`)

Independent Rust/cuda-oxide implementation of standard Qwen3-8B inference. The body uses symmetric per-output-channel INT8 weights and dynamic per-token INT8 activations. Norm, RoPE, SiLU and causal softmax attention retain ordinary floating-point model semantics. Embeddings, residuals, KV cache and output-head weights are BF16. INT8 projections accumulate in INT32; floating-point arithmetic uses FP32. There are no Hadamard rotations, integer attention, lookup-table activations, Q14 RoPE coefficients or 15-bit output head.

Quantization is an explicit baseline choice: max-absolute scale divided by 127, round-to-nearest-even, no calibration or smoothing. This is not a claim that there is one universal W8A8 format. Preparation exports the exact same quantized weights and FP32 channel scales in a cuda-oxide bundle and a compressed-tensors checkpoint usable by comparison engines. The original checkpoint is pinned to `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`.

The published optimization rounds through round 6 were measured on one RTX PRO 6000 Blackwell Max-Q workstation. Local RTX 5090 setup and validation are covered in the [Nix guide](../migrations/local-nix.md). The original engine handles one request at a time with greedy generation and at most 4096 prompt-plus-generated tokens. The [paged runtime](sessions.md) adds reusable sessions, prefix caching, ragged batches and full-prefix recomputation through the native 32,768-token total context; its RTX 5090 performance work is tracked in [round 7](optimization/round-7.md). Standard floating-point kernels can differ in reduction order across engines; numerical agreement and performance are reported separately from bitwise regression checks.

The original unoptimized baseline is commit `bd6e4448`. Commit identifiers in these records refer to the private development history this code was extracted from. Start with the [repository README](../../README.md) for setup. See [round 1 record](optimization/round-1.md), [round 2 record](optimization/round-2.md), [round 3 record](optimization/round-3.md), [round 4 record](optimization/round-4.md), [round 5 record](optimization/round-5.md), and [round 6 record](optimization/round-6.md) for timings, profiling and proofs. The [production plan](../plans/production.md) lists the decisions and workstreams needed before this engine can serve traffic, and the [Qwen3.6-35B-A3B FP8 plan](../plans/qwen3.6-35b-a3b-fp8.md) describes the next model. [coverage notes](coverage.md) and [discrepancy notes](discrepancies.md) describe the validation boundary. In particular, the full-model numerical comparison against Torch currently fails its declared tolerance even though the short greedy sequence and independent operator checks pass. This is experimental inference code, not an accuracy-certified model release.

On the workstation, optimized decode is **182.3 / 178.2 / 170.5 tokens/s** for 5 / 129 / 512 prompt tokens (32 generated tokens), versus **144.4 / 143.3 / 142.9** for the fastest tested stock W8A8 decode configuration (measured in round 5). Total request latency is **178 / 193 / 226 ms**; the best stock latency observed across runs is **237 / 237 / 243 ms**. Long prefill remains the 512-token request's largest remaining gap versus stock. See [benchmarks](benchmarks.md) for workloads, timing boundaries, accuracy limits, other configurations and reproduction.

## Reproduce

Commands run from the repository root on Linux with an SM120 GPU, using `clean-work/` as scratch space for bundles, fixtures and results. The vendored NVIDIA cuda-rust revision is `6921d3ee07e28ec62895ea40fc2fef39977ca411`; the toolchain is `nightly-2026-08-28`. The optimization records through round 6 were measured with the earlier cuda-oxide fork (`3d7cbe18`, `nightly-2026-04-03`); the [cuda-rust migration record](../migrations/cuda-rust.md) compares the two. No CUDA C++ inference kernels or vendor BLAS calls are used by this crate.

```sh
nix develop
# Download/export the pinned model and create independent operator fixtures.
crates/inferrous-qwen3/tools/cuda.sh prepare
# Builds the pinned codegen backend automatically.
crates/inferrous-qwen3/tools/cuda.sh build
crates/inferrous-qwen3/tools/cuda.sh test
target/inferrous-gpu/release/inferrous-qwen3 \
  clean-work/model-bundle --prompt 'The capital of France is' --max-tokens 32
```

See [local Nix development](../migrations/local-nix.md) for the locked toolchain, existing checkpoint reuse, checked release builds and the `cuda.sh bench` wrapper. GPU tests require both the independent fixtures and the model bundle; the full-model replay test fails if its bundle is missing.

`tools/prepare.py --help` describes export of the original pinned BF16 checkpoint to raw Rust tensors plus a compressed-tensors checkpoint under `BUNDLE/hf`. Preparation is offline: weights are quantized once, and only activations are quantized during inference. At model load, the GPU head is packed losslessly into 12-bit codes plus exceptions; every original BF16 bit is restored before computation. This saves 24.2% of head storage without changing weight values. Export refuses an existing destination. The published workstation bundle root is `e51d2517784b99de3802110542e831061489bf00df33db1b412ba3b3d01d21d7`; a locally regenerated bundle must use a baseline with its own matching root.

The original resident model owns eight decode graphs, fixed KV caches and decode scratch. Its generation is serialized by a mutex; prefill scratch is reused for the current prompt shape. Every legacy request recomputes its prompt and overwrites the visible KV prefix. Tracing uses eager launches. Graph capture and model loading are outside warm generation timing; first-use allocation still appears in API timing. GPU greedy selection checks all logits, and complete vocabulary hashes remain in every legacy result. Hashing overlaps the next GPU step without changing returned bytes. The separate paged runtime's ownership, graph replacement, prefix reuse and optional verification are described in the [session guide](sessions.md).

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

The [kernel profiling guide](profiling.md) describes opt-in CUPTI collection through the Nix shell. Kernel traces identify bottlenecks; their instrumented timings remain separate from paired benchmark results.
