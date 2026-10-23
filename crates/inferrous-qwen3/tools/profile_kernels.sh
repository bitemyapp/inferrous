#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo "Usage: $0 NEW_TRACE.csv BINARY [ARGUMENTS...]" >&2; exit 2; }
output=$(realpath -m -- "$1")
shift
[[ ! -e "$output" ]] || { echo "Refusing existing trace: $output" >&2; exit 1; }
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$tools_dir/../../.." && pwd)
library="$repo_dir/target/inferrous-profile/libkernel_trace.so"
binary=$(type -P -- "$1")
binary_sha=$(sha256sum -- "$binary")
binary_sha=${binary_sha%% *}
mkdir -p -- "$(dirname -- "$library")"
# Nix exposes libraries in lib; NVIDIA's Linux toolkit uses lib64.
cuda_lib="$CUDA_HOME/lib"
if [[ ! -f "$cuda_lib/libcupti.so" ]]; then cuda_lib="$CUDA_HOME/lib64"; fi
[[ -f "$cuda_lib/libcupti.so" ]] || { echo "Missing CUPTI library under CUDA_HOME=$CUDA_HOME" >&2; exit 1; }
# The primary header names the current activity layout; deprecated layouts live
# in a separate header. CUDA 13.1 uses Kernel11 and CUDA 13.4 uses Kernel13.
activity_kernel=$(sed -n 's/^} \(CUpti_ActivityKernel[0-9][0-9]*\);$/\1/p' "$CUDA_HOME/include/cupti_activity.h")
[[ "$activity_kernel" =~ ^CUpti_ActivityKernel[0-9]+$ ]] || { echo "Cannot identify CUPTI kernel activity layout" >&2; exit 1; }
clang -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread \
  -DINFERROUS_ACTIVITY_KERNEL="$activity_kernel" \
  -I "$CUDA_HOME/include" -L "$cuda_lib" -Wl,-rpath,"$cuda_lib" \
  "$tools_dir/kernel_trace.c" -lcupti -o "$library"
exec env INFERROUS_KERNEL_TRACE="$output" INFERROUS_TRACE_BINARY_SHA256="$binary_sha" \
  LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}" "$@"
