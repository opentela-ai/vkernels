#!/bin/bash
#SBATCH --job-name=vk150b2
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
# Issue #150 fault bisect, round 2:
#   A. launch_bounds probe, fixed copy step (was broken in bisect round 1)
#   B. standalone repro (block=64 control, then block=128 fault case)
#   C. same repro under rocgdb (live, not core) if the fault reproduces
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
W=/iopsstor/scratch/cscs/xyao/vk150lane
export TMPDIR=/tmp/vk150b2-$$
mkdir -p "$TMPDIR"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="

echo; echo "############ A. launch_bounds probe ############"
R="$TMPDIR/src_lb"
cp -r "$SRC" "$R"
rm -rf "$R/build"
sed -i 's/__launch_bounds__(kThreads_pf, 2)/__launch_bounds__(kThreads_pf)/' \
  "$R/src/c/vkernels/kernels/moe_fused.hip"
echo "patched occurrences left: $(grep -c '__launch_bounds__(kThreads_pf, 2)' "$R/src/c/vkernels/kernels/moe_fused.hip" || true)"
BA="$TMPDIR/build_lb"
cmake -S "$R" -B "$BA" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$BA" --target test_moe_fused_prefill_correct -j 32 2>&1 \
  | grep -E "error|Error|Built target" | tail -2
VK_MOE_PF_VARIANT=6 "$BA/meta/benchmarks/test_moe_fused_prefill_correct" situ 2>&1 | tail -10
echo "(lb exit ${PIPESTATUS[0]})"

echo; echo "############ B. standalone repro ############"
R2="$TMPDIR/repro"
mkdir -p "$R2"
hipcc -O3 -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 \
  -I"$SRC/src/c" "$SRC/meta/benchmarks/repro_issue150_fault.hip" \
  -o "$R2/repro150" 2>&1 | grep -E "error" | head
echo "--- block=64 (v5 control) ---"
"$R2/repro150" 64 2>&1 | tail -4
echo "--- block=128 (fault case) ---"
VK_MOE_PF_VARIANT=6 "$R2/repro150" 128 2>&1 | tail -6
echo "(repro exit ${PIPESTATUS[0]})"

echo; echo "############ C. repro under rocgdb ############"
VK_MOE_PF_VARIANT=6 timeout 900 rocgdb -batch \
  -ex 'set pagination off' \
  -ex run \
  -ex 'bt' \
  -ex 'info amdgpu wavefronts' \
  --args "$R2/repro150" 128 2>&1 | tail -40
echo "(rocgdb exit ${PIPESTATUS[0]})"

echo; echo "===== ALL DONE ====="
