# Migration to NVIDIA cuda-rust

Recorded 2026-10-07. inferrous moved from the `bitemyapp/cuda-oxide` fork at `3d7cbe18` (toolchain `nightly-2026-04-03`) to upstream [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) at `6921d3ee07e28ec62895ea40fc2fef39977ca411` (2026-10-06, toolchain `nightly-2026-08-28`, rustc 1.100.0-nightly). Upstream renamed the repository from cuda-oxide to cuda-rust and turned it into an umbrella for NVIDIA's CUDA platform for Rust: shared host runtime crates plus two kernel programming models, Tile (`cutile-rs`) and SIMT (`cuda-oxide`). The fork's merge base with upstream was `b22efa99` (2026-08-24); upstream has 359 commits since.

**Outcome:** one source change, bit-identical outputs, faster builds, and decode throughput unchanged within measurement drift (about 0.25% lower, traced mostly to one kernel's code generation).

## What changed

| Area | Before | After |
|---|---|---|
| Submodule | `vendor/cuda-oxide` → `bitemyapp/cuda-oxide` @ `3d7cbe18` | `vendor/cuda-rust` → `NVIDIA/cuda-rust` @ `6921d3ee` |
| Toolchain | `nightly-2026-04-03` | `nightly-2026-08-28` (the cuda-oxide pin inside cuda-rust) |
| Host runtime crate path | `vendor/cuda-oxide/crates/cuda-core` | `vendor/cuda-rust/cuda-core` (moved to the repository root, shared with cutile-rs) |
| Kernel crates and compiler | `vendor/cuda-oxide/crates/{cuda-device,cuda-host,cargo-oxide,rustc-codegen-cuda}` | `vendor/cuda-rust/cuda-oxide/crates/…` |
| Launch configuration type | `cuda_core::LaunchConfig` | `cuda_core::simt::LaunchConfig`; the root `LaunchConfig` now belongs to the Tile runtime. This import was the only source change |
| `Cargo.lock` | 117 packages | 137 packages |
| Release executable | 7,203,920 bytes | 7,404,608 bytes (+2.8%) |

Every `cargo-oxide` flag the build uses (`--materialize-cubin`, `--no-fmad`, `--arch sm_120`, `--unchecked-indexing`, `--cargo-target-dir`, `--device-codegen-crate`) and the `CUDA_OXIDE_BACKEND` override still exist upstream. Of the fork's seven upstream-absent commits, the warp-uniform load and persisting-L2 access policy now exist upstream, and none of the others (shared-memory carveout, artifact-only builds, literal-include kernel discovery, host-build body stripping, convergent-attribute cleanup) is needed by inferrous. `cuda-host` reaches `cuda-core` through the same sibling path inferrous uses, so the graph contains a single `cuda-core`.

## Correctness

- Optimized and `CLEAN_BOUNDS_CHECKS=1` builds pass all nine tests, including the independent operator fixtures and the full-model replay that compares programmatic decode graphs with eager launches.
- Every benchmark run reproduces the original complete-vocabulary logit hashes and token IDs: four paired runs of 69 requests and a checked-build run of three.
- Compute Sanitizer memcheck, racecheck and synccheck report zero errors or hazards on the operator tests; full-model memcheck on the 5-token workload reports zero errors.

## Compile times

Measured on the benchmark workstation (64-core Threadripper PRO 9985WX, `CARGO_BUILD_JOBS=16`, no compiler cache). "Cold" removes the backend, `cargo-oxide` and engine target directories; crate downloads were already cached for both. Engine builds are `tools/cuda.sh build` (release, unchecked indexing, materialized cubin).

| Step | Before | After | Change |
|---|---:|---:|---:|
| Codegen backend, cold release build | 57.5 s | 32.0 s | −44% |
| Engine, cold (two runs) | 28.0 / 27.8 s | 26.3 / 26.1 s | −6% |
| … of which `cargo-oxide` (dev build) | 8.2 / 8.0 s | 8.0 / 8.1 s | — |
| … of which engine release build | 19.5 / 19.5 s | 18.0 / 17.8 s | −8% |
| No-op rebuild (three runs) | 0.2 / 0.1 / 0.1 s | 0.1 / 0.1 / 0.1 s | — |
| Rebuild after touching `src/kernels.rs` (three runs) | 3.9 / 3.9 / 4.0 s | 3.7 / 3.6 / 3.6 s | −7% |
| Rebuild after touching `src/gpu.rs` (three runs) | 3.9 / 3.9 / 3.9 s | 3.6 / 3.6 / 3.6 s | −8% |
| `tools/cuda.sh test`: release test build | 13.2 s | 13.1 s | — |
| `tools/cuda.sh test`: running nine tests | 18.7 s | 17.9 s | −4% |

## Runtime

Serial A/B/B/A on one RTX PRO 6000 Blackwell Max-Q: before, after, after, before, each run 23 requests per workload (three warmups, twenty measured), pooled to forty measured requests per binary and shape.

| Prompt tokens | Decode before | Decode after | Change | API p50 before | API p50 after | First step before | First step after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 183.20 tok/s | 182.71 tok/s | −0.27% | 177.38 ms | 177.84 ms | 7.90 ms | 7.89 ms |
| 129 | 179.11 | 178.74 | −0.21% | 191.82 | 192.23 | 18.47 | 18.51 |
| 512 | 171.44 | 171.01 | −0.25% | 224.79 | 225.37 | 43.68 | 43.87 |

The two control runs drifted by about 0.9% over the session (p5: 183.99 then 182.41 tok/s), and both "after" runs (182.97, 182.44) fall between them, so the difference is within run-to-run drift. A graph-level profile of the 5-token workload agrees on direction: decode-graph execution is 5435.9 µs per step after versus 5421.8 µs before (+14 µs, +0.26%). Kernel-level profiles locate most of it in the fused gate/up kernel, 63.80 µs versus 63.53 µs per call (+10 µs per step over 36 layers); the O/down, QKV and head kernels match to within 0.05 µs. Register and shared-memory usage are unchanged for every hot kernel; the new compiler emits that kernel with 696 SASS instructions instead of 672 (same 90 global loads and 58 DP4A instructions, one more branch).

## Follow-ups

1. **Adopt upstream PDL support.** cuda-rust now provides `griddepcontrol` intrinsics (`grid_dependency::wait` and `trigger_dependents`) and a `cuda-core` launch helper with programmatic stream serialization. Use them instead of inline PTX; the post-capture graph-edge rewrite stays until the generated SIMT launch methods expose the attribute.
2. **Recover the gate/up kernel's 0.4%.** Diff its SASS between the two toolchains and restore the previous control flow if the extra branch is avoidable.
3. **Re-check workarounds.** The integer warp-reduction PTX workaround (the earlier backend misclassified integer `redux` as an SM100-only float operation) was not re-evaluated against upstream.
