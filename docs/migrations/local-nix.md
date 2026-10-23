# Local Nix development and RTX 5090 validation

The root `flake.nix` and `flake.lock` provide an `x86_64-linux` development shell with the Rust nightly from `rust-toolchain.toml`, CUDA 13.4, Clang/libclang 22, native libraries needed by the Rust codegen backend, and Python with CPU PyTorch, NumPy, safetensors, tokenizers and Hugging Face Hub. CUDA includes Compute Sanitizer, NVVM, nvJitLink, libdevice and the cuRAND headers required by the pinned cuda-rust bindings. The bindings require CUDA 13.0 or newer; the historical CUDA 12.8 instructions are insufficient for this submodule revision.

The RTX 5090 uses compute capability 12.0, matching the engine's `sm_120` target ([NVIDIA GPU table](https://developer.nvidia.com/cuda/gpus)). On NixOS the shell uses the installed driver in `/run/opengl-driver/lib`; it does not install or change the kernel driver. `nvidia-smi` must work on the host. On other Linux distributions, expose the host NVIDIA driver libraries through `LD_LIBRARY_PATH`. CUDA toolkit libraries and the pinned Rust compiler come from Nix.

The [RTX 5090 validation record](../qwen3-8b-w8a8/results/rtx5090-nix.md) records successful release and checked test suites, matching benchmark outputs, timings and artifact locations.

## Enter the environment

```sh
git submodule update --init --recursive
nix develop
nvcc --version
rustc --version
nvidia-smi
```

Use `nix develop --command COMMAND ...` for noninteractive runs. The flake is a development environment; GPU tests and benchmarks run on the host through `cuda.sh`, outside the Nix build sandbox. The script respects the shell's `RUSTUP_TOOLCHAIN`, uses locked Cargo dependencies, and builds the backend directly from the pinned submodule. It needs no separately installed CUDA toolkit or Rust toolchain inside this shell.

## Prepare data once

```sh
crates/inferrous-qwen3/tools/cuda.sh prepare
```

This downloads `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` into `clean-work/hf-cache`, exports the INT8 bundle, generates independent CPU PyTorch operator fixtures, and creates the 5/129/512-token workloads. Budget about 40 GB of disk for the original checkpoint and the exported raw/Hugging Face bundles, plus build artifacts and Nix dependencies. All generated data lives in ignored `clean-work/`. Existing completed outputs are reused; incomplete output directories must be moved aside before retrying.

To reuse an existing checkpoint, pass its snapshot directory (whose final path component is the pinned revision):

```sh
crates/inferrous-qwen3/tools/cuda.sh prepare --source /path/to/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

`CLEAN_MODEL_BUNDLE`, `CLEAN_OPERATOR_FIXTURES` and `CLEAN_WORKLOADS` override the default `clean-work/model-bundle`, `clean-work/operator-fixtures` and `clean-work/workloads` paths for both preparation and execution. Prepared bundles validate tensor and tokenizer hashes when loaded. Regenerating data with another preparation environment can change the bundle root; use a matching baseline when checking regression hashes.

## Build and test release mode

```sh
crates/inferrous-qwen3/tools/cuda.sh build
crates/inferrous-qwen3/tools/cuda.sh test
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh test
```

The script builds the Rust CUDA backend and `cargo-oxide` in release mode, then builds/tests the engine with `--release --features gpu --materialize-cubin --no-fmad --arch sm_120`. It keeps the backend in `target/cuda-oxide-backend`, the build tool in `target/cargo-oxide`, and the engine in `target/inferrous-gpu`. An explicit `CUDA_OXIDE_BACKEND` can select an already-built backend instead. `CARGO_BUILD_JOBS` defaults to 16 and can be reduced on machines with less free memory.

The default engine uses the existing validated unchecked-indexing policy. `CLEAN_BOUNDS_CHECKS=1` enables device bounds checks and uses a separate `target/inferrous-gpu-checked` directory, preserving the unchecked executable for comparison. All GPU tests run serially on device 0. Missing fixtures or model data cause a failure; the full-model eager/graph comparison is never silently skipped. Plain `cargo test --release` only runs host tests because the GPU feature is disabled by default; it does not validate GPU inference.

## Benchmark and compare

```sh
crates/inferrous-qwen3/tools/cuda.sh bench clean-work/rtx5090.jsonl
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh bench \
  clean-work/rtx5090-checked.jsonl clean-work/rtx5090.jsonl
```

Each command first ensures the release executable is current, then runs three warmups and twenty measured requests for each prompt length, generating 32 tokens per request. Outputs must be new files. The wrapper verifies deterministic token IDs and complete-vocabulary logit hashes across all repetitions, then prints p50/p95/p99 API latency and median decode throughput. Passing a second JSONL path additionally requires exact agreement with that baseline, including its model root. Without a supplied baseline this is a local determinism check, not a comparison against the historical workstation results. Keep GPU workloads serial and account for other desktop/GPU activity when interpreting timings.

Raw timings and regression checks do not establish model quality. The independent full-model PyTorch tolerance discrepancy remains documented in [the engine discrepancy notes](../qwen3-8b-w8a8/discrepancies.md).

For reusable sessions, ragged batches, full-prefix recomputation and native 32K context, see the [session guide](../qwen3-8b-w8a8/sessions.md). `cuda.sh test` also forwards test-runner arguments, for example `cuda.sh test --ignored native_context --nocapture`; that explicit long test requires `CLEAN_LONG_MODEL_BUNDLE` pointing to a 32K bundle.
