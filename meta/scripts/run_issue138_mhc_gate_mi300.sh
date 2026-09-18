#!/bin/bash
#SBATCH --job-name=vk-i138
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
# Issue #138: mhc_pre gate revision — fp64-reference comparison so the
# blocked-order (NSLICE-split) kernel is admissible at NSLICE >= 16.
#
# On an MI300A (gfx942) -- CSCS beverin `mi300` partition. Build + run:
#   1. test_mhc_correct  — multi-seed (5) NSLICE ladder {256..1} on the GLM
#      shapes under BOTH gates (default fp64-reference; VK_MHC_STRICT_GATE=1
#      legacy 1e-4 oracle-chain for the record) + the corrupted-kernel
#      (fn row off-by-one) negative control, which must FAIL.
#   2. mhc_bench         — decode-shape latency (n=1, hc_hidden=16384,
#      hc_mult3=24), strict + blocked NSLICE ladder.
#
# Run interactively:
#   srun --partition=mi300 -N1 -G1 --time=00:40:00 bash meta/scripts/run_issue138_mhc_gate_mi300.sh
# Or batch (from the synced scratch checkout):
#   sbatch -p mi300 -N1 --gres=gpu:1 --time=00:40:00 -o run.%j.out meta/scripts/run_issue138_mhc_gate_mi300.sh
set -euo pipefail
: "${SRC:=$(pwd)}"
B="$SRC/build-i138"
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
echo "############ 1a. MHC CORRECTNESS — issue #138 fp64 gate (default) ######"
"$B/meta/benchmarks/test_mhc_correct" 2>&1 | tee "$B/test_mhc_correct_f64gate.log" || \
  echo "(test exited non-zero — clean envelope above the provisional 2e-5 threshold; survey data still recorded above)"
echo

echo "############ 1b. Same rows under the LEGACY 1e-4 oracle-chain gate #####"
VK_MHC_STRICT_GATE=1 "$B/meta/benchmarks/test_mhc_correct" 2>&1 \
  | tee "$B/test_mhc_correct_strictgate.log" || \
  echo "(legacy-gate run exits non-zero when blocked NSLICE rows exceed 1e-4 — EXPECTED, see issue #138; continuing)"

echo
echo "############ 2. BENCHMARK: MHC pre/post (mhc_bench) #####################"
"$B/meta/benchmarks/mhc_bench" 2>&1

echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
