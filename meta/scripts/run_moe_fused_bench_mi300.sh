#!/bin/bash
#SBATCH --job-name=vkmoe
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
# Reproducible on-harness run for the MXFP4 fused-MoE work (issue #41) on a
# beverin MI300A / gfx942 node.  Builds the three relevant host/GPU benches
# from the repo checkout into node-local /tmp (not quota'd) and runs:
#   1. the GPU kernel validated against the CPU oracle at the K3 routing
#      shape (acceptance criterion #1), then the default-harness single
#      decode (criterion #2, ~0.45 ms baseline);
#   2. the host-only dequant A/B (dequant_weight_tile vs the bit-identical
#      _ref), whose ~2.8x is the speedup the CPU oracle sees on this node;
#   3. the host oracle bench (full fused-MoE CPU reference, after-only).
#
# Source must be readable from the node (the repo checkout, or an
# rsync of it).  Override SRC to point at your checkout, e.g.
#   SRC=$HOME/vkernels sbatch meta/scripts/run_moe_fused_bench_mi300.sh
set -euo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vkmoe-$$
mkdir -p "$TMPDIR"
B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
date
test -d "$SRC" || { echo "SRC=$SRC not found (set it to your checkout)"; exit 1; }

echo "=== configure (HIP, gfx942, benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3

echo "=== build benches ==="
cmake --build "$B" --target moe_fused_bench bench_dequant_ab moe_fused_oracle_bench -j 64 \
  2>&1 | grep -E "error:|Built target" | tail -6

GPU="$B/meta/benchmarks/moe_fused_bench"
DEQ="$B/meta/benchmarks/bench_dequant_ab"
ORC="$B/meta/benchmarks/moe_fused_oracle_bench"

echo; echo "############ 1. GPU vs CPU oracle — K3 routing (criterion #1) ############"
"$GPU" situ --E 256 --hidden 7168 --ispp 512 --topk 16 --ms 1,2,4 --kmajor
echo; echo "############ 1b. default harness single decode (criterion #2) ############"
"$GPU" situ --ispp 512 --topk 6 --ms 1,2,4,8

echo; echo "############ 2. host dequant A/B (bit-exact, ~2.8x) ############"
"$DEQ" --hidden 7168 --ispp 512 --iters 9
"$DEQ" --hidden 4096 --ispp 512 --iters 9

echo; echo "############ 3. host oracle (full fused-MoE CPU reference) ############"
"$ORC" --ms 1 --hidden 512 --ispp 1024 --topk 16 --E 256 --iters 5
"$ORC" --ms 1 --hidden 7168 --ispp 128 --topk 16 --E 256 --iters 5

echo; echo "===== ALL DONE ====="
