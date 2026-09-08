#!/bin/bash
# Build + benchmark dsa_kpool (issue #60) on an MI300A (gfx942) compute node.
# The kpool kernels are HIP/gfx942 (AMD) and run on CSCS beverin's `mi300`
# partition -- NOT on bristen (NVIDIA GH200, no ROCm).
#
# Run interactively:
#   srun --partition=mi300 -N1 -G1 --time=00:30:00 bash meta/scripts/run_dsa_kpool_bench_mi300.sh
# Or batch:
#   sbatch -p mi300 -N1 --gres=gpu:1 --time=00:30:00 \
#     -o kpool_bench.%j.out meta/scripts/run_dsa_kpool_bench_mi300.sh
set -euo pipefail
: "${SRC:=$HOME/vkernels-60}"
B="$SRC/build_kpool"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release, tests + benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -4

echo "=== build (vkernels + test_dsa_kpool_correct + dsa_kpool_bench + host oracle) ==="
cmake --build "$B" --target vkernels test_dsa_kpool_correct dsa_kpool_bench \
                                        vkernels_test_dsa_kpool -j 64 \
  2>&1 | grep -E "error:|Built target|dsa_kpool|kpool_correct" | tail -12

echo
echo "############ 1. HOST ORACLE (vkernels_test_dsa_kpool) ############"
"$B/tests/vkernels_test_dsa_kpool" 2>&1 | tail -6

echo
echo "############ 2. DEVICE CORRECTNESS (test_dsa_kpool_correct) ############"
"$B/meta/benchmarks/test_dsa_kpool_correct" 2>&1 | tail -24

echo
echo "############ 3. BENCHMARK (dsa_kpool_bench) ############"
"$B/meta/benchmarks/dsa_kpool_bench" 2>&1 | tail -64

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
