# inferrous

LLM inference written in Rust, with every GPU kernel also written in Rust and compiled by [cuda-oxide](https://github.com/bitemyapp/cuda-oxide). There are no CUDA C++ kernels and no vendor BLAS or attention libraries.

The first engine, [`inferrous-qwen3`](crates/inferrous-qwen3), runs Qwen3-8B with INT8 weights and dynamic INT8 activations (W8A8) on one NVIDIA RTX PRO 6000 Blackwell (SM120). It decodes at **182.3 / 178.2 / 170.5 tokens/s** for 5 / 129 / 512-token prompts, faster than the best stock W8A8 configuration we measured (vLLM with Humming INT8 kernels, 144.4 / 143.3 / 142.9). See [benchmarks](docs/qwen3-8b-w8a8/benchmarks.md) for workloads, timing boundaries and caveats.

The project's discipline is measurement and exactness:

- **Regression oracle.** Every optimization preserves the complete-vocabulary logit hashes and token IDs of the original implementation, checked on every benchmark run.
- **Independent checks.** PyTorch and CPU-integer operator fixtures, checked and unchecked device builds, graph versus eager replay, and Compute Sanitizer.
- **Paired timing.** Retained changes need serial A/B/B/A measurements, and rejected experiments are recorded with their data. The [optimization records](docs/qwen3-8b-w8a8/optimization) explain each change.

## Status

This is research-grade code, not an accuracy-certified model server.

- One request at a time, greedy decoding, up to 4096 tokens of context.
- The full-model numerical comparison against an independent PyTorch composition currently fails its declared tolerance ([discrepancy notes](docs/qwen3-8b-w8a8/discrepancies.md)). Model quality has not been evaluated.
- The [production plan](docs/plans/production.md) lists what serving traffic would require: paged KV, continuous batching, sampling, an API and a quality gate.
- [Qwen3.6-35B-A3B FP8](docs/plans/qwen3.6-35b-a3b-fp8.md) is the next model: Gated DeltaNet hybrid attention, fine-grained MoE and MTP.

## Requirements

- Linux with an SM120 NVIDIA GPU (developed on an RTX PRO 6000 Blackwell Max-Q, driver 580.159.04) and CUDA 12.8. CUDA 13.1 is used only for Compute Sanitizer and the stock-engine comparisons.
- The pinned Rust toolchain (`rust-toolchain.toml`, nightly-2026-04-03).
- The cuda-oxide submodule at the pinned revision. Its own documentation covers codegen-backend prerequisites.
- Python with PyTorch, `safetensors` and `numpy` for model preparation and reference fixtures.
- The `Qwen/Qwen3-8B` checkpoint at revision `b968826d9c46dd6066d109eabc6255188de91218`.

## Quick start

```sh
git clone --recurse-submodules https://github.com/bitemyapp/inferrous.git
cd inferrous
# Export the pinned checkpoint into a cuda-oxide bundle (refuses existing output).
python3 crates/inferrous-qwen3/tools/prepare.py --source <hf-snapshot>/b968826d9c46dd6066d109eabc6255188de91218 \
  --output clean-work/model-bundle
# Build the cuda-oxide codegen backend once, from the pinned submodule.
(cd vendor/cuda-oxide/crates/rustc-codegen-cuda && cargo build --release)
export CUDA_OXIDE_BACKEND=$PWD/vendor/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
crates/inferrous-qwen3/tools/cuda.sh build
target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle \
  --prompt 'The capital of France is' --max-tokens 32
```

The [engine guide](docs/qwen3-8b-w8a8/README.md) covers tests, operator fixtures, reference traces and benchmarks. The prepared bundle root, recorded in the [benchmarks](docs/qwen3-8b-w8a8/benchmarks.md), identifies an exact reproduction of the benchmarked weights.

## Layout

| Path | Contents |
|---|---|
| `crates/inferrous-qwen3` | The Qwen3-8B W8A8 engine: host runtime, device kernels, tests and tools |
| `docs/` | All documentation, indexed in [docs/README.md](docs/README.md): the engine guide, benchmarks, validation, safety contract, optimization records and result summaries, plus plans and research |
| `vendor/cuda-oxide` | Rust-to-PTX compiler and runtime (submodule; Apache-2.0) |

## License

Copyright (c) 2026 Chris Allen <cma@bitemyapp.com>. Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option. cuda-oxide is Apache-2.0 (NVIDIA CORPORATION & AFFILIATES); model weights carry their own licences (Qwen3: Apache-2.0).
