#!/bin/bash
#SBATCH --partition=mi300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vkernels-moeaux_%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/vkernels-moeaux_%j.err
set -euo pipefail
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
date
export TMPDIR=/tmp
PR=/tmp/podman-moeaux-$$
mkdir -p "$PR/root" "$PR/run"
echo "=== loading container image (20.8 GB) ==="
podman --root "$PR/root" --runroot "$PR/run" load -i /capstor/scratch/cscs/xyao/vkernels-full-v2.tar 2>&1 | tail -1

SRC=/capstor/scratch/cscs/xyao/vkernels-moeaux
CMD=(podman --root "$PR/root" --runroot "$PR/run" run --rm
     --device=/dev/kfd --device=/dev/dri
     --security-opt seccomp=unconfined
     -v "$SRC:/workspace/vkernels:ro"
     vkernels:hip-full)

echo "=== configure (gfx942, benchmarks ON, tests OFF) ==="
"${CMD[@]}" bash -c '
  set -e
  cmake -B /tmp/build -S /workspace/vkernels -G Ninja \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DVKERNELS_BUILD_HIP=ON \
    -DVKERNELS_BUILD_BENCHMARKS=ON \
    -DVKERNELS_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -25
  echo "=== build test_moe_aux_correct (+ vkernels lib) ==="
  cmake --build /tmp/build --target test_moe_aux_correct -j$(nproc) 2>&1 | tail -30
  echo "=== run harness ==="
  /tmp/build/meta/benchmarks/test_moe_aux_correct
'
rc=$?
echo "=== exit code: $rc ==="
rm -rf "$PR" 2>/dev/null || true
date
