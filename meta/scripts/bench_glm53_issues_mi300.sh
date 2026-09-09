#!/bin/bash
#SBATCH --job-name=glm53opt
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Build + validate + benchmark the new #64/#66/#68 work on a beverin MI300A
# (gfx942) node:
#
#   1. BUILD  vkernels + test_glm_fp8_gemv + glm_fp8_gemv_bench (cmake).
#   2. #64    test_glm_fp8_gemv: block-FP8 decode-fused GEMV vs the CPU
#             oracle (expect PASS, max_rel << 2e-2), then the fused-vs-
#             materialized-BF16 microbenchmark at the GLM expert shapes.
#   3. #66/#68 pytest test_glm_projection.py (contract + GPU parity/graph
#             tests) then bench_glm_projection.py: default BLAS vs
#             TunableOp-selected BLAS vs the vkernels Triton split-K
#             operator at [64,4096] [128,4096] [512,4096] [1536,4096],
#             M=1/2. Python 3.11 + torch/triton come from the system
#             interpreter + ~/.local; Triton's host helper needs the
#             bootstrapped Python.h (see run_issue63_torchops_mi300.sh).
#
# Run batch:
#   SRC=$SCRATCH/vkernels sbatch -o glm53opt.%j.out \
#     meta/scripts/bench_glm53_issues_mi300.sh
set -euo pipefail
: "${SRC:=${SCRATCH:-$HOME}/vkernels}"
B="$SRC/build_glm53"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-interactive} ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3 || true
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"

echo "=== configure + build (HIP, gfx942, Release) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
cmake --build "$B" --target vkernels test_glm_fp8_gemv glm_fp8_gemv_bench -j 64 \
  2>&1 | grep -E "error:|Built target" | tail -6

ulimit -c 0

echo
echo "############ 1. #64 block-FP8 GEMV: correctness vs CPU oracle ############"
"$B/meta/benchmarks/test_glm_fp8_gemv" 2>&1

echo
echo "############ 2. #64 block-FP8 GEMV: fused vs materialized BF16 ############"
"$B/meta/benchmarks/glm_fp8_gemv_bench" 2>&1

echo
echo "############ 3. #66/#68 python env ############"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SRC}/src/python:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/tmp}"
PYINC="${SRC}/.pybuild/python/include/python3.11"
mkdir -p "${SRC}/.pybuild"
if [ ! -f "${PYINC}/Python.h" ]; then
  echo "bootstrapping Python 3.11 headers (python-build-standalone)..."
  url="https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz"
  curl -fsSL "${url}" | tar -C "${SRC}/.pybuild" -xz && test -f "${PYINC}/Python.h"
fi
export CPATH="${PYINC}${CPATH:+:${CPATH}}"
PY=/usr/bin/python3.11
"$PY" -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"

echo
echo "############ 4. #66/#68 projection: pytest (GPU tests incl. graph) ############"
"$PY" -m pytest "$SRC/tests/python/test_glm_projection.py" -q -rN 2>&1 | tail -6

echo
echo "############ 5. #66/#68 projection: BLAS vs TunableOp vs Triton ############"
OUT="$SRC/work/glm53"; mkdir -p "$OUT"
"$PY" "$SRC/meta/benchmarks/bench_glm_projection.py" --output "$OUT/glm_projection.json" 2>&1 | tail -12

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
