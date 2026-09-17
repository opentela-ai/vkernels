#!/bin/bash
#SBATCH --job-name=vk-i79
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Issue #79: mhc_pre_gemm_sqrsum column-split across blocks (grid
# (num_tokens, hc_mult3), 256 threads, coalesced fn-row walk). Build +
# correctness gate (test_mhc_correct vs CPU oracle) + mhc_bench on an
# MI300A (gfx942) -- CSCS beverin `mi300` partition.
#
# Run interactively:
#   srun --partition=mi300 -N1 -G1 --time=00:45:00 bash meta/scripts/run_issue79_mhc_mi300.sh
# Or batch (from the synced scratch checkout):
#   sbatch -p mi300 -N1 --gres=gpu:1 --time=00:45:00 -o run.%j.out meta/scripts/run_issue79_mhc_mi300.sh
set -euo pipefail
: "${SRC:=$(pwd)}"
B="$SRC/build-i79"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3 || true
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release, tests + benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -4

echo "=== build (vkernels + hip C ABI + mhc correctness test + mhc bench) ==="
cmake --build "$B" --target vkernels vkernels_hip test_mhc_correct mhc_bench \
  -j 64 \
  2>&1 | grep -E "error:|warning: |Built target|hip_capi" | tail -12

echo
echo "############ 1. MHC CORRECTNESS (test_mhc_correct) ############"
"$B/meta/benchmarks/test_mhc_correct" 2>&1

echo
echo "############ 2. BENCHMARK: MHC pre/post (mhc_bench) ############"
"$B/meta/benchmarks/mhc_bench" 2>&1

echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
