#!/usr/bin/env bash
set -euo pipefail
mode=${1:-build}
[[ "$mode" == build || "$mode" == test ]] || exit 2
crate_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd -- "$crate_dir/../.." && pwd)
cd "$crate_dir"
# Use the backend built from the pinned vendor/cuda-oxide; outside that repository
# cargo-oxide would otherwise use a cached or freshly fetched, unpinned backend.
: "${CUDA_OXIDE_BACKEND:?build vendor/cuda-oxide/crates/rustc-codegen-cuda and export CUDA_OXIDE_BACKEND (see README)}"
export RUSTUP_TOOLCHAIN=nightly-2026-04-03 RUST_MIN_STACK=33554432
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16}
# GPU integration tests share device 0 and must not race their workloads.
export RUST_TEST_THREADS=${RUST_TEST_THREADS:-1}
# Kernels have fixed shapes, enforced by private validated model data and the
# admitted request graph. Keep a checked build available for differential runs.
# The complete index/lifetime argument is in docs/qwen3-8b-w8a8/safety.md.
index_options=(--unchecked-indexing)
if [[ ${CLEAN_BOUNDS_CHECKS:-0} == 1 ]]; then index_options=(); fi
cargo run --manifest-path "$repo_dir/vendor/cuda-oxide/Cargo.toml" -p cargo-oxide -- \
  "$mode" --materialize-cubin --no-fmad --arch sm_120 \
  "${index_options[@]}" \
  --cargo-target-dir "$repo_dir/target/inferrous-gpu" \
  --device-codegen-crate inferrous-qwen3 \
  -- --release --features gpu -p inferrous-qwen3
