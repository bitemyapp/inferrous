# inferrous

LLM inference written in Rust, with every GPU kernel also written in Rust and compiled by cuda-oxide, the SIMT kernel compiler in [NVIDIA cuda-rust](https://github.com/NVIDIA/cuda-rust). There are no CUDA C++ kernels and no vendor BLAS or attention libraries.

The first engine, [`inferrous-qwen3`](crates/inferrous-qwen3), runs Qwen3-8B with INT8 weights and dynamic INT8 activations (W8A8) on one NVIDIA RTX PRO 6000 Blackwell (SM120). It decodes at **182.3 / 178.2 / 170.5 tokens/s** for 5 / 129 / 512-token prompts, faster than the best stock W8A8 configuration we measured (vLLM with Humming INT8 kernels, 144.4 / 143.3 / 142.9). See [benchmarks](docs/qwen3-8b-w8a8/benchmarks.md) for workloads, timing boundaries and caveats.

The project's discipline is measurement and exactness:

- **Regression oracle.** Every optimization preserves the complete-vocabulary logit hashes and token IDs of the original implementation, checked on every benchmark run.
- **Independent checks.** PyTorch and CPU-integer operator fixtures, checked and unchecked device builds, graph versus eager replay, and Compute Sanitizer.
- **Paired timing.** Retained changes need serial A/B/B/A measurements, and rejected experiments are recorded with their data. The [optimization records](docs/qwen3-8b-w8a8/optimization) explain each change.

## Status

This is research-grade code, not an accuracy-certified model server.

- Greedy decoding with [paged sessions, ragged batches and full-prefix recomputation](docs/qwen3-8b-w8a8/sessions.md), validated through 32,768 total tokens. The original single-request 4K engine remains the short-context regression oracle.
- Separate RTX 5090 and RTX PRO 6000 configurations. The [RTX 5090 batch performance comparison](docs/plans/kv-cache-batching-rtx5090.md) is in progress; the workstation timings above describe the original engine.
- The full-model numerical comparison against an independent PyTorch composition currently fails its declared tolerance ([discrepancy notes](docs/qwen3-8b-w8a8/discrepancies.md)). Model quality has not been evaluated.
- The [production plan](docs/plans/production.md) lists what serving traffic would require: paged KV, continuous batching, sampling, an API and a quality gate.
- [Qwen3.6-35B-A3B FP8](docs/plans/qwen3.6-35b-a3b-fp8.md) is the next model: Gated DeltaNet hybrid attention, fine-grained MoE and MTP.

## Requirements

- Linux with an SM120 NVIDIA GPU (including the RTX 5090) and a working NVIDIA driver. The pinned cuda-rust bindings require CUDA 13.0 or newer; the Nix shell supplies CUDA 13.4. Historical benchmark environments are recorded separately in the results.
- The pinned Rust toolchain (`rust-toolchain.toml`, nightly-2026-08-28).
- The NVIDIA cuda-rust submodule (`vendor/cuda-rust`) at the pinned revision. Its cuda-oxide documentation covers codegen-backend prerequisites.
- Python with PyTorch, `safetensors` and `numpy` for model preparation and reference fixtures.
- The `Qwen/Qwen3-8B` checkpoint at revision `b968826d9c46dd6066d109eabc6255188de91218`.

## Quick start

```sh
git clone --recurse-submodules https://github.com/bitemyapp/inferrous.git
cd inferrous
nix develop
# Download the pinned checkpoint and prepare the bundle, fixtures, and workloads.
crates/inferrous-qwen3/tools/cuda.sh prepare
# Build the pinned backend and SM120 engine in release mode.
crates/inferrous-qwen3/tools/cuda.sh build
crates/inferrous-qwen3/tools/cuda.sh test
target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle \
  --prompt 'The capital of France is' --max-tokens 32
# Three warmups and twenty measured requests for each of three prompt lengths.
crates/inferrous-qwen3/tools/cuda.sh bench clean-work/local.jsonl
```

The [local Nix setup](docs/migrations/local-nix.md) covers prerequisites, existing checkpoints, checked builds and regression comparisons. The [engine guide](docs/qwen3-8b-w8a8/README.md) covers tests, operator fixtures, reference traces and benchmarks. The prepared bundle root, recorded in the [benchmarks](docs/qwen3-8b-w8a8/benchmarks.md), identifies an exact reproduction of the benchmarked weights.

## Layout

| Path | Contents |
|---|---|
| `crates/inferrous-qwen3` | The Qwen3-8B W8A8 engine: host runtime, device kernels, tests and tools |
| `docs/` | All documentation, indexed in [docs/README.md](docs/README.md): the engine guide, benchmarks, validation, safety contract, optimization records and result summaries, plus plans and research |
| `vendor/cuda-rust` | NVIDIA cuda-rust: host runtime crates and the cuda-oxide Rust-to-PTX compiler (submodule; Apache-2.0) |

## License

Copyright (c) 2026 Chris Allen <cma@bitemyapp.com>. Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option. NVIDIA cuda-rust, including cuda-oxide, is Apache-2.0 (NVIDIA CORPORATION & AFFILIATES); model weights carry their own licences (Qwen3: Apache-2.0).
