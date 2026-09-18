#!/bin/bash
#SBATCH --job-name=vk-i78
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
# Issue #78 correctness gate on MI300A (gfx942): on-device moe_align_block_size
# parity vs the CPU reference (test_capi_moe_align: skewed / hot-expert /
# distinct-expert / TP-skip / empty routings + end-to-end GPU-aligned GEMM feed),
# plus the standalone C++ MoE GPU test suites and an align-latency micro-measure.
# SRC must be a readable checkout/rsync of the repo:
#   SRC=/capstor/scratch/cscs/xyao/vk-issue-78 sbatch meta/scripts/test_issue78_mi300.sh
set -euo pipefail
ulimit -c 0
# SRC must be a readable checkout/rsync of the repo (job 640262 died on the
# old $HOME/vkernels default: no such checkout exists on beverin).
: "${SRC:?usage: SRC=/capstor/scratch/cscs/xyao/vk-issue-78 sbatch meta/scripts/test_issue78_mi300.sh}"
B="$SRC/build-i78"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
test -d "$SRC" || { echo "SRC=$SRC not found"; exit 1; }
test -f "$SRC/src/c/vkernels/kernels/moe_fused.hip" || { echo "SRC=$SRC missing src/c/vkernels/kernels/moe_fused.hip"; exit 1; }
echo "=== src: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
date

echo "=== configure (HIP, gfx942, tests+benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_TESTS=ON -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3

echo "=== build ==="
cmake --build "$B" --target test_capi_moe_align test_capi_moe \
  test_moe_fused_correct test_moe_fused_bigshape_correct moe_fused_bench \
  -j 64 2>&1 | grep -E "error|Error|Built target" | tail -8

GPU="$B/meta/benchmarks"
echo; echo "###### test_capi_moe_align (issue #78 parity gate) ######"
"$GPU/test_capi_moe_align"

echo; echo "###### test_capi_moe ######"
"$GPU/test_capi_moe"

echo; echo "###### test_moe_fused_correct ######"
"$GPU/test_moe_fused_correct"

echo; echo "###### test_moe_fused_bigshape_correct ######"
"$GPU/test_moe_fused_bigshape_correct"

echo; echo "###### align latency micro-measure (K3 decode shapes) ######"
# vk_hip_moe_align_block_size latency at the K3 PP decode routing shapes
# (M*top_k well under the 1024 single-block limit), via the bench binary's
# default harness (also re-verifies CPU-oracle parity per shape).
"$B/meta/benchmarks/moe_fused_bench" situ --E 32 --hidden 7168 --ispp 512 \
  --topk 8 --ms 1,4,16 --kmajor 2>&1 | tail -12

echo; echo "===== ALL DONE ====="
date
