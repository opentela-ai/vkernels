#!/bin/bash
#SBATCH --job-name=vk76ab90
#SBATCH --partition=mi200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
# Issue #76 prefill A/B (gfx90a / MI250X): the three variants that were still in
# contention, swept on one build -- register-staged pipeline + 32-wide gateup
# N tile (2), pipeline + 16-wide gateup N tile (4), and 4 plus a 32-wide down
# N tile (5, the shipped default) -- followed by the prefill correctness
# oracle on 4 and 5.  For the variants that were already eliminated
# (0/1/3) see issue76_ab_*.sh history and beverin-issue76-prefill-variants.txt.
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk76ab-$$
mkdir -p "$TMPDIR"; B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "src hip md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8)"
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$B" --target moe_fused_prefill_bench test_moe_fused_prefill_correct \
  -j 64 2>&1 | grep -E "error|Error|Built target" | tail -8
PF="$B/meta/benchmarks/moe_fused_prefill_bench"
PFC="$B/meta/benchmarks/test_moe_fused_prefill_correct"

for V in 2 4 5; do
  echo; echo "############ VK_MOE_PF_VARIANT=$V ############"
  VK_MOE_PF_VARIANT=$V "$PF" situ || echo "(bench exit $?)"
done

for V in 4 5; do
echo; echo "############ correctness, VARIANT=$V ############"
VK_MOE_PF_VARIANT=$V "$PFC" || echo "(correct exit $?)"
done
echo; echo "===== ALL DONE ====="
