#!/bin/bash
#SBATCH --job-name=vk150b3
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
# Issue #150 fault bisect, round 3: standalone repro (HIP include fixed)
#   A. block=64 control run (v5 path, must pass)
#   B. block=128 fault case
#   C. block=128 under rocgdb (live fault localization)
#   D. block=128 with AMD_SERIALIZE_KERNEL + HSA debug env, fallback probes
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk150b3-$$
mkdir -p "$TMPDIR"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="

R="$TMPDIR/repro"; mkdir -p "$R"
SRC_INC="$SRC/src/c"
hipcc -O3 -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -fgpu-rdc -c \
  -I"$SRC_INC" "$SRC_INC/vkernels/kernels/moe_fused.hip" -o "$R/moe_fused_hip.o" 2>&1 | grep -E "error" | head
hipcc -x hip -O3 -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -fgpu-rdc -c \
  "$SRC/meta/benchmarks/repro_issue150_fault.hip" -I"$SRC_INC" -o "$R/repro.o" 2>&1 | grep -E "error" | head
g++ -O2 -std=c++17 -I"$SRC_INC" -c "$SRC_INC/vkernels/kernels/moe_fused.cpp" \
  -o "$R/moe_fused_cpp.o" 2>&1 | grep -E "error" | head
ls -la "$R"/*.o || { echo "BUILD FAILED (objects)"; exit 1; }
hipcc -fgpu-rdc --hip-link "$R/repro.o" "$R/moe_fused_hip.o" "$R/moe_fused_cpp.o" \
  -lamdhip64 -o "$R/repro150" 2>&1 | grep -E "error|undefined" | head
ls -la "$R/repro150" || { echo "BUILD FAILED (link)"; exit 1; }

echo; echo "############ A. block=64 control ############"
"$R/repro150" 64 2>&1 | tail -4
echo "(exit ${PIPESTATUS[0]})"

echo; echo "############ B. block=128 ############"
VK_MOE_PF_VARIANT=6 "$R/repro150" 128 2>&1 | tail -6
echo "(exit ${PIPESTATUS[0]})"

echo; echo "############ C. rocgdb live ############"
VK_MOE_PF_VARIANT=6 timeout 900 rocgdb -batch \
  -ex 'set pagination off' \
  -ex run \
  -ex 'bt' \
  -ex 'info threads' \
  --args "$R/repro150" 128 2>&1 | tail -45
echo "(rocgdb exit ${PIPESTATUS[0]})"

echo; echo "############ D. env probes ############"
echo "--- HSA_ENABLE_SDMA=0, AMD_SERIALIZE_KERNEL=3 ---"
HSA_ENABLE_SDMA=0 AMD_SERIALIZE_KERNEL=3 VK_MOE_PF_VARIANT=6 "$R/repro150" 128 2>&1 | tail -4
echo "--- HSA_FORCE_FINE_GRAIN_PCIE=1 ---"
HSA_FORCE_FINE_GRAIN_PCIE=1 VK_MOE_PF_VARIANT=6 "$R/repro150" 128 2>&1 | tail -4

echo; echo "===== ALL DONE ====="
