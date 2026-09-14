#!/bin/bash
#SBATCH --job-name=vkmoe41
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
# On-hardware validation for the issue #41 "memory access fault at
# ispp >= 4096" fix: the fused-MoE HIP kernels computed per-expert weight
# byte offsets (expert * w13_bytes / expert * w2_bytes) in INT32, which
# wraps negative once expert * hidden * ispp/2 >= 2^31.  This job builds
# the HIP preset from SRC and runs, on one MI300A:
#   1. test_moe_fused_bigshape_correct — GPU vs CPU oracle at and beyond
#      the old wrap thresholds (ispp=4096/E=256 decode+prefill, all-slots
#      expert 255, and the E=32/ispp=33792 serving-shard shape), with the
#      routing pinned to the previously-wrapping expert ids;
#   2. the previously-faulting bench invocations (docs/performance/
#      moe-fused/gfx942.md "GPU memory-access fault at ispp >= 4096"):
#      E=256 ispp=4096 with the CPU oracle, then --dummy --no-cpu latency
#      at ispp=8192, ispp=33792 (E=32 shard) and ispp=33792 (E=256 full).
# Source must be readable from the node:
#   SRC=$HOME/vkernels sbatch meta/scripts/test_moe_fused_bigshape_mi300.sh
set -euo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vkmoe41-$$
mkdir -p "$TMPDIR"
B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== source: $SRC ($(git -C "$SRC" log --oneline -1 2>/dev/null || echo 'no git')) ==="
date
test -d "$SRC" || { echo "SRC=$SRC not found"; exit 1; }

echo "=== configure (HIP, gfx942, benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3

echo "=== build ==="
cmake --build "$B" --target test_moe_fused_bigshape_correct moe_fused_bench -j 64 \
  2>&1 | grep -E "error:|Built target" | tail -6

T="$B/meta/benchmarks/test_moe_fused_bigshape_correct"
GPU="$B/meta/benchmarks/moe_fused_bench"

echo; echo "############ 1. big-shape correctness (GPU vs CPU oracle) ############"
export VK_MOE_KTIME=1   # per-stage [ktime] lines localize any device fault
CASE_RC=0
case_n() { echo "--- case $1: $2 ---"; "$T" "$1" || CASE_RC=1; }
case_n 1 "decode ispp=4096, wrap-threshold experts"
case_n 2 "decode ispp=4096, all-slots expert 255"
case_n 3 "prefill ispp=4096, wrap-threshold experts"
case_n 4 "decode ispp=33792 E=32 shard"
if [ "$CASE_RC" -ne 0 ]; then echo "cases failed rc=$CASE_RC"; exit "$CASE_RC"; fi

echo; echo "###### 2. previously-faulting bench: E=256 ispp=4096 (with CPU oracle) ######"
"$GPU" situ --E 256 --hidden 7168 --ispp 4096 --topk 16 --ms 1,2,4 --kmajor
"$GPU" situ --E 256 --hidden 7168 --ispp 4096 --topk 16 --ms 1

echo; echo "###### 3. latency at larger shapes (--dummy --no-cpu) ######"
echo "--- ispp=8192, E=256 ---"
"$GPU" situ --E 256 --hidden 7168 --ispp 8192 --topk 16 --ms 1 --dummy --no-cpu --kmajor
echo "--- ispp=33792, E=32 (per-GPU TP=8 shard, ~12 GB) ---"
"$GPU" situ --E 32 --hidden 7168 --ispp 33792 --topk 16 --ms 1 --dummy --no-cpu --kmajor
echo "--- ispp=33792, E=256 (full K3, ~97 GB) ---"
"$GPU" situ --E 256 --hidden 7168 --ispp 33792 --topk 16 --ms 1 --dummy --no-cpu --kmajor

echo; echo "===== ALL DONE ====="
