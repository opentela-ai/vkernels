#!/bin/bash
#SBATCH --job-name=vk150ab
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
# Issue #150 A/B + correctness (gfx942 / MI300A): the new 128-row prefill
# M-tile config (VK_MOE_PF_VARIANT=6, block_size=128) against the shipped
# v5 default (64-row tiles), plus the full correctness gates for both.
#
#   SRC=/iopsstor/scratch/cscs/xyao/vk150/src sbatch meta/scripts/issue150_ab_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk150ab-$$
mkdir -p "$TMPDIR"; B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "src hip md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8)"
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | grep -Ei "error|warning: GPU" | head
cmake --build "$B" --target test_moe_fused_correct test_moe_fused_prefill_correct \
  test_moe_fused_bigshape_correct moe_fused_prefill_bench moe_fused_bench -j 64 2>&1 \
  | grep -E "error|Error|Built target" | tail -12

echo; echo "############ decode correctness (default path) ############"
env -u VK_MOE_PF_VARIANT "$B/meta/benchmarks/test_moe_fused_correct" || echo "(exit $?)"

echo; echo "############ prefill correctness, VARIANT=5 (default) ############"
VK_MOE_PF_VARIANT=5 "$B/meta/benchmarks/test_moe_fused_prefill_correct" situ \
  || echo "(exit $?)"

echo; echo "############ prefill correctness, VARIANT=6 (BM=128 + fallback) ############"
VK_MOE_PF_VARIANT=6 "$B/meta/benchmarks/test_moe_fused_prefill_correct" situ \
  || echo "(exit $?)"

echo; echo "############ big-shape correctness, VARIANT=5 ############"
VK_MOE_PF_VARIANT=5 "$B/meta/benchmarks/test_moe_fused_bigshape_correct" \
  || echo "(exit $?)"

echo; echo "############ big-shape correctness, VARIANT=6 (incl. block=128 case) ############"
VK_MOE_PF_VARIANT=6 "$B/meta/benchmarks/test_moe_fused_bigshape_correct" \
  || echo "(exit $?)"

echo; echo "############ prefill bench, VARIANT=5 (shipped default) ############"
VK_MOE_PF_VARIANT=5 "$B/meta/benchmarks/moe_fused_prefill_bench" situ || echo "(exit $?)"

echo; echo "############ prefill bench, VARIANT=6 (BM=128) ############"
VK_MOE_PF_VARIANT=6 "$B/meta/benchmarks/moe_fused_prefill_bench" situ || echo "(exit $?)"

echo; echo "############ decode regression guard (E=256 h4096 i512 topk=6) ############"
env -u VK_MOE_PF_VARIANT "$B/meta/benchmarks/moe_fused_bench" situ \
  --ispp 512 --topk 6 --ms 8,32,48 || echo "(exit $?)"

echo; echo "===== ALL DONE ====="
