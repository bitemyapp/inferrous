#!/usr/bin/env bash
# Run an unmodified stock engine with the host GPU and driver, including NixOS.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
engine=${1:?Usage: stock_container.sh ENGINE COMMAND [ARGUMENTS...]}
shift
case "$engine" in
  vllm) image=vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34 ;;
  vllm-b12x) image=$(docker image inspect --format '{{.Id}}' inferrous-vllm-b12x:0.32.0-1.5.1) ;;
  sglang) image=lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae ;;
  *) echo "Unsupported stock image: $engine" >&2; exit 2 ;;
esac
[[ $# -gt 0 ]] || { echo "Supply the container command, for example python3." >&2; exit 2; }
driver_library=${NVIDIA_DRIVER_LIBRARY:-/run/opengl-driver/lib/libcuda.so.1}
[[ -f "$driver_library" ]] || { echo "Set NVIDIA_DRIVER_LIBRARY to the host libcuda.so.1." >&2; exit 1; }
driver_dir=$(dirname "$(readlink -f -- "$driver_library")")
container_user="$(id -u):$(id -g)"
# In rootless Docker, container root is the invoking host user; UID 1000 in
# the container would instead map to a subordinate UID and lose file access.
security_options=$(docker info --format '{{json .SecurityOptions}}')
if [[ "$security_options" == *name=rootless* ]]; then container_user=0:0; fi
mkdir -p "$repo_dir/clean-work/stock/cache"
exec docker run --rm \
  --user "$container_user" \
  --shm-size 8g \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --volume "$driver_dir:/usr/local/nvidia/lib64:ro" \
  --volume "$repo_dir:/workspace:ro" \
  --volume "$repo_dir/clean-work:/workspace/clean-work" \
  --workdir /workspace \
  --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env HOME=/workspace/clean-work/stock/cache \
  --env XDG_CACHE_HOME=/workspace/clean-work/stock/cache \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_NO_USAGE_STATS=1 \
  --env INFERROUS_STOCK_IMAGE="$image" \
  --entrypoint "$1" "$image" "${@:2}"
