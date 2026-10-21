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
clang -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread \
  -I "$CUDA_HOME/include" -L "$CUDA_HOME/lib" -Wl,-rpath,"$CUDA_HOME/lib" \
  "$tools_dir/kernel_trace.c" -lcupti -o "$library"
exec env INFERROUS_KERNEL_TRACE="$output" INFERROUS_TRACE_BINARY_SHA256="$binary_sha" \
  LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}" "$@"
