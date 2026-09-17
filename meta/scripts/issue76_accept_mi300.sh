#!/bin/bash
#SBATCH --job-name=vk76acc
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
# Issue #76 acceptance run.  On the shipped default path (VK_MOE_PF_VARIANT
# unset): the decode and prefill correctness oracles, the big-shape oracle,
# then the prefill bench for the perf record and a large-shape decode bench
# as the regression guard for the untouched decode kernels.
#
#   SRC=/iopsstor/scratch/cscs/xyao/vk76/src sbatch -A <acct> \
#       meta/scripts/issue76_accept_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk76acc-$$
mkdir -p "$TMPDIR"; B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "src hip md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8)"
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$B" --target test_moe_fused_correct test_moe_fused_prefill_correct \
  test_moe_fused_bigshape_correct moe_fused_prefill_bench moe_fused_bench -j 64 2>&1 \
  | grep -E "error|Error|Built target" | tail -10

echo; echo "############ decode correctness (default path) ############"
env -u VK_MOE_PF_VARIANT "$B/meta/benchmarks/test_moe_fused_correct" \
  || echo "(exit $?)"

echo; echo "############ prefill correctness (default path) ############"
env -u VK_MOE_PF_VARIANT "$B/meta/benchmarks/test_moe_fused_prefill_correct" \
  || echo "(exit $?)"

echo; echo "############ big-shape correctness (default path) ############"
env -u VK_MOE_PF_VARIANT "$B/meta/benchmarks/test_moe_fused_bigshape_correct" \
  || echo "(exit $?)"

echo; echo "############ perf, default vs baseline ############"
for V in 0 5; do
  echo "--- VK_MOE_PF_VARIANT=$V"
  VK_MOE_PF_VARIANT=$V "$B/meta/benchmarks/moe_fused_prefill_bench" situ \
    || echo "(exit $?)"
done
echo; echo "############ decode regression guard (E=256 h4096 i512 topk=6) ############"
env -u VK_MOE_PF_VARIANT "$B/meta/benchmarks/moe_fused_bench" situ \
  --ispp 512 --topk 6 --ms 8,32,48 || echo "(exit $?)"

echo; echo "===== ALL DONE ====="
