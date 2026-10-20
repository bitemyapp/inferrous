#!/usr/bin/env bash
set -euo pipefail
mode=${1:-build}
if [[ $# -gt 0 ]]; then shift; fi
crate_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd -- "$crate_dir/../.." && pwd)
tools_dir="$crate_dir/tools"
case "$mode" in
  prepare) exec python3 "$tools_dir/prepare_local.py" "$@" ;;
  build)
    [[ $# == 0 ]] || { echo "$mode does not accept arguments" >&2; exit 2; }
    ;;
  test) ;;
  bench)
    if [[ $# -lt 1 || $# -gt 2 ]]; then
      echo "Usage: $0 bench OUTPUT.jsonl [BASELINE.jsonl]" >&2
      exit 2
    fi
    output=$(realpath -m -- "$1")
    baseline=$(realpath -m -- "${2:-$1}")
    [[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 1; }
    if [[ $# == 2 && ! -f "$baseline" ]]; then
      echo "Missing baseline: $baseline" >&2
      exit 1
    fi
    ;;
  *) echo "Usage: $0 {prepare [--source SNAPSHOT]|build|test|bench OUTPUT.jsonl [BASELINE.jsonl]}" >&2; exit 2 ;;
esac

CLEAN_MODEL_BUNDLE=$(realpath -m -- "${CLEAN_MODEL_BUNDLE:-$repo_dir/clean-work/model-bundle}")
CLEAN_OPERATOR_FIXTURES=$(realpath -m -- "${CLEAN_OPERATOR_FIXTURES:-$repo_dir/clean-work/operator-fixtures}")
export CLEAN_MODEL_BUNDLE CLEAN_OPERATOR_FIXTURES
workloads=$(realpath -m -- "${CLEAN_WORKLOADS:-$repo_dir/clean-work/workloads}")
if [[ "$mode" == test || "$mode" == bench ]]; then
  required=("$CLEAN_MODEL_BUNDLE/manifest.json")
  if [[ "$mode" == test ]]; then required+=("$CLEAN_OPERATOR_FIXTURES/manifest.json"); fi
  if [[ "$mode" == bench ]]; then required+=("$workloads/requests.json"); fi
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || { echo "Missing $path; run $0 prepare first." >&2; exit 1; }
  done
fi

oxide_dir="$repo_dir/vendor/cuda-rust/cuda-oxide"
[[ -f "$oxide_dir/Cargo.toml" ]] || {
  echo "Missing cuda-rust submodule; run git submodule update --init --recursive." >&2
  exit 1
}
export RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-nightly-2026-08-28} RUST_MIN_STACK=33554432
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16}
# Use the backend built from the pinned vendor/cuda-rust; outside that repository
# cargo-oxide would otherwise use a cached or freshly fetched, unpinned backend.
if [[ -z ${CUDA_OXIDE_BACKEND:-} ]]; then
  cargo build --locked --release \
    --manifest-path "$oxide_dir/crates/rustc-codegen-cuda/Cargo.toml" \
    --target-dir "$repo_dir/target/cuda-oxide-backend"
  export CUDA_OXIDE_BACKEND="$repo_dir/target/cuda-oxide-backend/release/librustc_codegen_cuda.so"
fi
[[ -f "$CUDA_OXIDE_BACKEND" ]] || { echo "Missing CUDA_OXIDE_BACKEND: $CUDA_OXIDE_BACKEND" >&2; exit 1; }
# GPU integration tests share device 0 and must not race their workloads.
export RUST_TEST_THREADS=1
# Kernels have fixed shapes, enforced by private validated model data and the
# admitted request graph. Keep a checked build available for differential runs.
# The complete index/lifetime argument is in docs/qwen3-8b-w8a8/safety.md.
index_options=(--unchecked-indexing)
if [[ ${CLEAN_BOUNDS_CHECKS:-0} == 1 ]]; then index_options=(); fi
# Keep checked artifacts separate, so differential runs cannot overwrite release.
target_dir="$repo_dir/target/inferrous-gpu"
if [[ ${CLEAN_BOUNDS_CHECKS:-0} == 1 ]]; then target_dir+="-checked"; fi
build_mode=$mode
if [[ "$mode" == bench ]]; then build_mode=build; fi
test_args=()
if [[ "$mode" == test && $# -gt 0 ]]; then test_args=(-- "$@"); fi
cd "$crate_dir"
cargo run --locked --release --manifest-path "$oxide_dir/Cargo.toml" \
  --target-dir "$repo_dir/target/cargo-oxide" -p cargo-oxide -- \
  "$build_mode" --materialize-cubin --no-fmad --arch sm_120 \
  "${index_options[@]}" \
  --cargo-target-dir "$target_dir" \
  --device-codegen-crate inferrous-qwen3 \
  -- --locked --release --features gpu -p inferrous-qwen3 "${test_args[@]}"

if [[ "$mode" == bench ]]; then
  python3 "$tools_dir/bench_clean.py" --binary "$target_dir/release/inferrous-qwen3" \
    --model "$CLEAN_MODEL_BUNDLE" --workloads "$workloads" --output "$output"
  python3 "$tools_dir/check_regression.py" "$baseline" "$output"
  python3 "$tools_dir/summarize.py" "$output"
fi
