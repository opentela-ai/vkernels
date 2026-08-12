#!/usr/bin/env bash
# Build the profiling toolchain image and exercise nvcc / nsys / ncu on any
# NVIDIA GPU, fully inside Docker — no sudo required.
#
# Usage:
#   CUDA_ARCHS=90 bash meta/docker/run-profile.sh   # target sm_90
#   bash meta/docker/run-profile.sh                 # auto-detect (or native)
#
# GPU access WITHOUT editing the host's toolkit config:
#   * The host's NVIDIA Container Toolkit has `no-cgroups = true`, which makes
#     `--gpus all` inject the device nodes AND driver libs but SKIPS the
#     device-cgroup allow-list.  NVML then fails ("Failed to initialize
#     NVML") and CUDA reports "no CUDA-capable device is detected".
#   * `--device` alone fixes the cgroup but injects NO driver libs (the
#     toolkit hook is what bind-mounts libcuda.so.13 from the host), so CUDA
#     reports "CUDA driver version is insufficient".
#   * Combining `--gpus all` (nodes + libs) with explicit
#     `--device-cgroup-rule` entries (allow-list for the injected nodes) gives
#     both at once and needs no root.  The major:minor numbers below come from
#     `ls -l /dev/nvidia*` on this host.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

# Auto-detect or accept override.
if [ -z "${CUDA_ARCHS:-}" ]; then
  if command -v nvidia-smi &>/dev/null; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    CUDA_ARCHS=${CC:-native}
  else
    CUDA_ARCHS=native
  fi
fi

ARCH_TAG=$(echo "$CUDA_ARCHS" | tr ',' '_' | tr ';' '_')
IMG="vkernels/prof:cuda13-sm${ARCH_TAG}"

# --gpus all injects nodes+libs; the rules below re-add the allow-list that
# `no-cgroups = true` dropped.  Profiling tools also need SYS_ADMIN + an
# unconfined seccomp profile to read counters / trace.
GPU=(
  --gpus all
  --cap-add=SYS_ADMIN
  --security-opt seccomp=unconfined
  --device-cgroup-rule "c 195:0 rwm"
  --device-cgroup-rule "c 195:254 rwm"
  --device-cgroup-rule "c 195:255 rwm"
  --device-cgroup-rule "c 500:0 rwm"
  --device-cgroup-rule "c 500:1 rwm"
  --device-cgroup-rule "c 503:1 rwm"
  --device-cgroup-rule "c 503:2 rwm"
)

echo "==> building image $IMG (CUDA_ARCHS=$CUDA_ARCHS)"
docker build --build-arg CUDA_ARCHS="$CUDA_ARCHS" \
  -t "$IMG" -f meta/docker/Dockerfile.prof meta/docker

echo "==> compile bench.cu for sm_${CUDA_ARCHS} in the container (no GPU needed to compile)"
docker run --rm -v "$PWD/meta/docker:/work" -w /work "$IMG" \
  nvcc -arch="sm_${CUDA_ARCHS}" -lineinfo -O2 --ptxas-options=-v -o bench bench.cu

echo "==> nvidia-smi + run the binary in the container"
docker run --rm "${GPU[@]}" -v "$PWD/meta/docker:/work" -w /work "$IMG" \
  bash -lc 'nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv && ./bench 8388608 1024'

echo "==> nsys timeline (writes meta/docker/bench.nsys-rep)"
docker run --rm "${GPU[@]}" -v "$PWD/meta/docker:/work" -w /work "$IMG" \
  nsys profile --stats=true --force-overwrite=true -o bench ./bench 8388608 1024 || true

echo "==> ncu per-kernel profile (writes meta/docker/bench.ncu-rep)"
docker run --rm "${GPU[@]}" -v "$PWD/meta/docker:/work" -w /work "$IMG" \
  ncu --set full --import-source yes --force-overwrite \
      --kernel-name regex:vec_add_k\|matmul_k \
      --launch-skip 1 --launch-count 2 -o bench ./bench 8388608 1024 || true

echo "==> reports on disk"
ls -la meta/docker/bench.nsys-rep* meta/docker/bench.ncu-rep* 2>/dev/null || true

echo "==> nsys GPU kernel summary (from the captured report)"
docker run --rm "${GPU[@]}" -v "$PWD/meta/docker:/work" -w /work "$IMG" \
  nsys stats --report cuda_gpu_kern_sum bench.nsys-rep 2>/dev/null | sed -n "1,12p" || true

echo "==> ncu per-kernel summary (from the captured report)"
docker run --rm -v "$PWD/meta/docker:/work" -w /work "$IMG" \
  ncu --import bench.ncu-rep --print-summary per-kernel 2>/dev/null | grep -E "Device 0|SM Busy|Issue Slots|Achieved Occupancy" | head -12 || true
