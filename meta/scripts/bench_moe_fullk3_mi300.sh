#!/bin/bash
#SBATCH --job-name=vkmoe41lat
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
# Issue #41 acceptance criterion #2: decode latency at the full-K3 shapes.
# These invocations previously died with "Memory access fault by GPU node-N
# ... Reason: Unknown" (INT32 per-expert offset wrap, fixed in 5b89cf9), so
# the real-shape M=1 latency has never been measured.  Runs, on one MI300A:
#   1. the default-harness reference point (criterion #2's ~0.45 ms baseline),
#   2. the K3 routing shape with the CPU oracle (E=256 h7168 ispp=512 k16),
#   3. the full-K3 per-GPU TP=8 shard  (E=32  h7168 ispp=33792, ~12 GB),
#   4. the full-K3 whole-model shape   (E=256 h7168 ispp=33792, ~97 GB),
#   5. the wrap-threshold shape with oracle (E=256 h7168 ispp=4096),
#   6. one VK_MOE_KTIME=1 pass at the shard shape for the stage split.
# SRC must be a readable checkout/rsync of the repo (default $HOME/vkernels):
#   SRC=$HOME/vkernels sbatch meta/scripts/bench_moe_fullk3_mi300.sh
set -euo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vkmoe41lat-$$
mkdir -p "$TMPDIR"
B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src: $(cat "$SRC/src/c/vkernels/kernels/moe_fused.hip" | md5sum | cut -c1-8) ==="
date
test -d "$SRC" || { echo "SRC=$SRC not found"; exit 1; }

echo "=== configure (HIP, gfx942, benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3

echo "=== build ==="
cmake --build "$B" --target moe_fused_bench -j 64 \
  2>&1 | grep -E "error:|Built target" | tail -4

GPU="$B/meta/benchmarks/moe_fused_bench"

echo; echo "###### 1. default harness reference (E=256 h4096 i512 k6, oracle-checked) ######"
"$GPU" situ --ispp 512 --topk 6 --ms 1,2,4

echo; echo "###### 2. K3 routing shape (E=256 h7168 i512 k16, oracle-checked) ######"
"$GPU" situ --E 256 --hidden 7168 --ispp 512 --topk 16 --ms 1,2,4 --kmajor

echo; echo "###### 3. full-K3 per-GPU TP=8 shard: E=32 h7168 ispp=33792 (~12 GB) ######"
"$GPU" situ --E 32 --hidden 7168 --ispp 33792 --topk 16 --ms 1,2,4 --kmajor --dummy --no-cpu

echo; echo "###### 4. full-K3 whole-model: E=256 h7168 ispp=33792 (~97 GB) ######"
"$GPU" situ --E 256 --hidden 7168 --ispp 33792 --topk 16 --ms 1,2 --kmajor --dummy --no-cpu

echo; echo "###### 5. wrap-threshold shape (E=256 h7168 ispp=4096, oracle-checked) ######"
"$GPU" situ --E 256 --hidden 7168 --ispp 4096 --topk 16 --ms 1,2 --kmajor

echo; echo "###### 6. stage split at the shard shape (ktime syncs between stages) ######"
VK_MOE_KTIME=1 "$GPU" situ --E 32 --hidden 7168 --ispp 33792 --topk 16 --ms 1 --kmajor --dummy --no-cpu

echo; echo "===== ALL DONE ====="
