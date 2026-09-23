#!/bin/bash
#SBATCH --partition=mi300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/issue145_%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/issue145_%j.err
# Issue #145 — token-major gather scatter_reduce: correctness + bench on MI300A (native ROCm 6.3).
set -euo pipefail
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
date
export HIPCC=/opt/rocm-6.3.0/lib/llvm/bin/hipcc
export PATH=/opt/rocm-6.3.0/bin:$PATH
SRC=/capstor/scratch/cscs/xyao/vkernels-issue145
BLD=$SRC/build-gfx942

echo "=== configure (gfx942, benchmarks ON, tests OFF) ==="
cmake -B "$BLD" -S "$SRC" -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DVKERNELS_BUILD_HIP=ON \
  -DVKERNELS_BUILD_BENCHMARKS=ON \
  -DVKERNELS_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5

echo "=== build test_moe_aux_correct + moe_aux_bench ==="
cmake --build "$BLD" --target test_moe_aux_correct moe_aux_bench -j16 2>&1 | tail -12

echo "=== correctness harness (small + K3) ==="
"$BLD/meta/benchmarks/test_moe_aux_correct"

echo "=== bench: mxfp4_moe_aux (K3 + M sweep) ==="
"$BLD/meta/benchmarks/moe_aux_bench"
echo "ALL_DONE"
date
