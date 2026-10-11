#!/bin/bash
#SBATCH --job-name=issue148
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
# Issue #148: hardware-fp8 MFMA decode on the fnuz storage (beverin MI300A).
#
#   1. BUILD  the HIP baseline (test_glm_fp8_gemv + glm_fp8_gemv_bench) from
#             the synced worktree copy so the before/after numbers share a
#             node and clock state.
#   2. #64    glm_fp8_gemv_bench: re-measure the manual-decode fused kernel
#             records (22.4 us gate/up M=1, 16.4 us down M=1, job 629781).
#   3. #148   bench_glm_issue148_decode.py: parity (fp32 dequant reference +
#             torch blockwise oracle on the same quantized operands), config
#             sweep (BN x warps x stages x split-K), end-to-end timing, and
#             the expert_gemv manual-vs-FNUZ_NATIVE A/B.
#
# Run: SRC=/capstor/scratch/cscs/xyao/issue148 sbatch -o issue148.%j.out \
#        meta/scripts/issue148_decode_mfma_mi300.sh
set -euo pipefail
: "${SRC:=/capstor/scratch/cscs/xyao/issue148}"
B="$SRC/build_issue148"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-interactive} ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3 || true
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"

echo "=== configure + build HIP baseline (gfx942, Release) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
cmake --build "$B" --target vkernels test_glm_fp8_gemv glm_fp8_gemv_bench probe_mfma_fp8 -j 32 \
  2>&1 | grep -E "error:|Built target" | tail -6
ulimit -c 0

echo
echo "############ 1. #64 HIP manual-decode GEMV: correctness (sanity) ############"
"$B/meta/benchmarks/test_glm_fp8_gemv" 2>&1

echo
echo "############ 2. #64 HIP manual-decode GEMV: BEFORE numbers (same node) ############"
"$B/meta/benchmarks/glm_fp8_gemv_bench" 200 2>&1 | tail -12

echo
echo "############ 2b. #148 MFMA probe (debug-hook ground truth) ############"
"$B/meta/benchmarks/probe_mfma_fp8" 2>&1

echo
"$B/meta/benchmarks/probe_mfma_fp8" dbg 2>&1

echo
echo "############ 3. #148 python env ############"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SRC}/src/python:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/tmp}"
: "${PY:=python3}"
"$PY" -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"

echo
echo "############ 4. #148 decode MFMA: parity + sweep + A/B ############"
OUT="$SRC/work/glm53"; mkdir -p "$OUT"
"$PY" "$SRC/meta/benchmarks/bench_glm_issue148_decode.py" --sweep \
  --output "$OUT/issue148_decode.json" 2>&1

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
