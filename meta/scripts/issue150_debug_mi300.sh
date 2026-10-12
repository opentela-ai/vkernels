#!/bin/bash
#SBATCH --job-name=vk150dbg
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
# Issue #150 debug pass: confirm the VK_MOE_PF_VARIANT=6 aperture fault and
# dump per-kernel code-object resource stats (VGPR / SGPR / scratch / LDS)
# for the kBM=128 vs kBM=64 prefill instantiations.
#
#   SRC=/iopsstor/scratch/cscs/xyao/vk150lane/src sbatch meta/scripts/issue150_debug_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk150dbg-$$
mkdir -p "$TMPDIR"; B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "src hip md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8)"
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$B" --target test_moe_fused_prefill_correct -j 64 2>&1 \
  | grep -E "error|Error|Built target" | tail -4

echo; echo "############ confirm VARIANT=6 fault ############"
VK_MOE_PF_VARIANT=6 "$B/meta/benchmarks/test_moe_fused_prefill_correct" situ 2>&1 | head -40
echo "(exit ${PIPESTATUS[0]})"

echo; echo "############ VARIANT=5 sanity (must be green) ############"
VK_MOE_PF_VARIANT=5 "$B/meta/benchmarks/test_moe_fused_prefill_correct" situ 2>&1 | tail -5

echo; echo "############ code-object resource stats ############"
OBJ=$(find "$B" -name "moe_fused.hip.o" -o -name "*.hipfb" -o -name "moe_fused.o" 2>/dev/null | head -3)
echo "objs: $OBJ"
for o in $OBJ; do
  llvm-objdump --triple=amdgcn-amd-amdhsa -d "$o" 2>/dev/null | grep -E "\.vgpr_count|\.sgpr_count|\.scratch_size_bytes|\.lds_size|\.workgroup_processor_mode|gateup_swiglu_kernel_prefill_pipe|down_combine_kernel_prefill_pipe|amdgcn_hsail" | head -80
done
echo "===== ALL DONE ====="
