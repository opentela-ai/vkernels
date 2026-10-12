#!/bin/bash
#SBATCH --job-name=vk150bis
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
# Issue #150 fault bisect: the VK_MOE_PF_VARIANT=6 aperture fault.
# Builds three variants of the same source and runs the prefill correct
# suite under VARIANT=6 on each:
#   A. unpatched (control, expected fault)
#   B. -O2 device code (optimizer-bug probe)
#   C. __launch_bounds__(kThreads_pf, 2) -> (kThreads_pf) on the two
#      128-row tile kernels (occupancy-hint probe)
#   SRC=/iopsstor/scratch/cscs/xyao/vk150lane/src sbatch meta/scripts/issue150_bisect_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
W=/iopsstor/scratch/cscs/xyao/vk150lane
export TMPDIR=/tmp/vk150bis-$$
mkdir -p "$TMPDIR"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="

build_and_run () {
  local name="$1" src="$2" extra="$3"
  local B="$TMPDIR/build_$name"
  echo; echo "############ variant $name ############"
  cmake -S "$src" -B "$B" -DVKERNELS_BUILD_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
    -DVKERNELS_BUILD_BENCHMARKS=ON $extra >/dev/null 2>&1
  cmake --build "$B" --target test_moe_fused_prefill_correct -j 32 2>&1 \
    | grep -E "error|Error|Built target" | tail -2
  VK_MOE_PF_VARIANT=6 "$B/meta/benchmarks/test_moe_fused_prefill_correct" situ 2>&1 | tail -12
  echo "($name exit ${PIPESTATUS[0]})"
}

# A: control
build_and_run ctrl "$SRC" ""

# B: -O2 device code
build_and_run o2 "$SRC" "-DCMAKE_HIP_FLAGS=-O2"

# C: launch_bounds without the 2-blocks/CU min hint
sed 's/__launch_bounds__(kThreads_pf, 2)/__launch_bounds__(kThreads_pf)/' \
  "$SRC/src/c/vkernels/kernels/moe_fused.hip" > "$TMPDIR/moe_fused_lb.hip"
cp -r "$SRC/src" "$TMPDIR/src_lb"
cp "$TMPDIR/moe_fused_lb.hip" "$TMPDIR/src_lb/src/c/vkernels/kernels/moe_fused.hip"
grep -c "__launch_bounds__(kThreads_pf, 2)" "$TMPDIR/src_lb/src/c/vkernels/kernels/moe_fused.hip" \
  | xargs echo "remaining (kThreads_pf, 2) occurrences in patched src:"
build_and_run lb "$TMPDIR/src_lb" ""

echo; echo "===== ALL DONE ====="
