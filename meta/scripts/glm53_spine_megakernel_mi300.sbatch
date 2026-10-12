#!/bin/bash
#SBATCH --job-name=glm53spine
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vkernels-e1/logs/%x-%j.out
#
# GLM-5.3-Flash KDA-spine megakernel -- MI300A (gfx942) bring-up + E1 pricing.
#
# The four MI300A levers this job turns, in order:
#
#   L0  CDNA3 spin barrier (Milestone 0).  The megakernel's grid barrier is a
#       monotonic-counter spin; on CDNA3 it runs on wave64 across 228 CUs, so
#       the producer-barrier-consumer visibility microprogram is re-verified
#       here (240 rounds, worker counts up to the CU count) rather than
#       assumed from the GB10/48-SM attestation.
#   L1  Residency-gated co-residency sweep.  validate_residency() used to
#       raise "not implemented for HIP"; it now re-materialises the kernel's
#       own HSACO through libamdhip64 and queries
#       hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, so the P sweep can
#       climb to the CU-count bound instead of hanging the spin at the wall
#       clock.  The bench prints that bound and caps the sweep on it.
#   L2  ROCm validity gate.  Gate 1 is the nvcc-JIT fused CUDA kernel, so it
#       is skipped here; ROCm validity of the Triton KDA spine is carried by
#       gate 2/2b (megakernel vs the pure-torch decomposed eager chain) and
#       gate 3 (barrier-counter soak, no drift), and re-checked at *every*
#       P arm -- layer/head partitioning at 228 CUs is the MI300A-specific
#       risk, not GB10's.
#   L3  E1 pricing on MI300A: M vs graph-of-kernels A' (and B' if the fused
#       CUDA chain happens to exist) to decide whether E2 is funded here.
#
# Run (from the MI300A checkout):
#   SRC=/capstor/scratch/cscs/xyao/vkernels-e1 \
#     sbatch meta/scripts/glm53_spine_megakernel_mi300.sbatch
set -uo pipefail
: "${SRC:=${SCRATCH:-$HOME}/vkernels}"
OUT="$SRC/work/glm53"; mkdir -p "$OUT"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-interactive} ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3 || true
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
ulimit -c 0

echo "=== python env (system 3.11 + ~/.local torch/triton; triton's host helper needs Python.h) ==="
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
"$PY" - <<'EOF'
import torch, triton
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
p = torch.cuda.get_device_properties(0)
print("device", p.name, "CUs", p.multi_processor_count)
EOF

echo
echo "############ L0: CDNA3 Milestone-0 spin barrier (wave64, up to the CU count) ############"
timeout 1200 "$PY" - <<'EOF'
import torch
from vkernels.compiler.runtime import synchronization as sync

cus = torch.cuda.get_device_properties(0).multi_processor_count
counts = tuple(sorted({1, 8, 48, min(128, cus), cus}))
report = sync.verify_milestone0(rounds=240, worker_counts=counts, invocations=3)
for k, v in report.items():
    print(f"  M0 {k:16s} counter={v}")
print("  M0 GRID_SYNC_BACKEND_VERIFIED =", sync.GRID_SYNC_BACKEND_VERIFIED)
assert sync.GRID_SYNC_BACKEND_VERIFIED, "M0 did not flip the verified flag"
print(f"L0 PASS: producer-barrier-consumer visibility holds on {cus} CUs")
EOF
echo "[L0 exit=$?]"

echo
echo "############ L1-L3: E1 KDA-spine megakernel + residency-bounded P sweep ############"
timeout 2000 "$PY" "$SRC/bench/glm53_e1_spine_bench.py" 2>&1 \
  | tee "$OUT/e1_spine_mi300_${SLURM_JOB_ID:-manual}.log"
echo "[bench exit=$?]"

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
