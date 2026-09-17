#!/bin/bash
#SBATCH --job-name=vk-i80
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Issue #80: double-buffered dsa_sparse_fwd_split partial (LDS key staging)
# -- 17/17 correctness rows + decode split sweep + split_for revalidation.
# sbatch-safe (lands on mi300); also runnable interactively:
#   srun --partition=mi300 -N1 -G1 --time=00:45:00 \
#     bash meta/scripts/run_issue80_db_mi300.sh
set -euo pipefail
ulimit -c 0
: "${SRC:=/capstor/scratch/cscs/xyao/vk-issue-80}"
B="$SRC/build-i80"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release, benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3

echo "=== build (test_dsa_correct + dsa_bench) ==="
cmake --build "$B" --target test_dsa_correct dsa_bench -j 64 2>&1 \
  | grep -E "error:|Built target" | tail -8

echo
echo "############ 1. CORRECTNESS (test_dsa_correct) ############"
"$B/meta/benchmarks/test_dsa_correct"

echo
echo "############ 2. SPLIT SWEEP (dsa_bench) ############"
"$B/meta/benchmarks/dsa_bench"

echo
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
