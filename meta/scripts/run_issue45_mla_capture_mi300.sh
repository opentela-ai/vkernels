#!/bin/bash
#SBATCH --job-name=vk-i45
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
# Issue #45 (vLLM serving integration): validate the stream/capture-safe MLA
# C ABI on an MI300A (gfx942) -- CSCS beverin `mi300` partition.
#   1. test_mla_correct   -- MLA numerics vs CPU oracle (incl. split-K decode)
#   2. test_mla_capture   -- stream/graph-capture safety (cold + warm capture,
#                            re-feed replay, legacy/stream parity)
#   3. test_kda_correct   -- KDA oracle tests (no-regression guard)
# Run: sbatch -p mi300 -N1 --gres=gpu:1 --time=00:40:00 -o run.%j.out meta/scripts/run_issue45_mla_capture_mi300.sh
set -euo pipefail
: "${SRC:=$(pwd)}"
B="$SRC/build-i45"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== commit: $(git rev-parse --short HEAD) ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3 || true

echo "=== configure (HIP, gfx942, Release, tests + benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -4

echo "=== build (vkernels + hip C ABI + host C ABI + capture/correct tests) ==="
cmake --build "$B" --target vkernels vkernels_hip vkernels_hostc \
  test_mla_correct test_mla_capture test_kda_correct \
  -j 64 2>&1 | grep -E "error:|warning: |Built target" | tail -12

echo
echo "############ 1. MLA CORRECTNESS (test_mla_correct) ############"
"$B/meta/benchmarks/test_mla_correct" 2>&1

echo
echo "############ 2. MLA STREAM/GRAPH CAPTURE (test_mla_capture) ############"
"$B/meta/benchmarks/test_mla_capture" 2>&1

echo
echo "############ 3. KDA NO-REGRESSION (test_kda_correct) ############"
"$B/meta/benchmarks/test_kda_correct" 2>&1

echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
