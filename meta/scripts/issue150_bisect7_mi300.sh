#!/bin/bash
#SBATCH --job-name=vk150b7
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:55:00
# Issue #150 round 7.  Round-6 finding: CMake HIP_FLAGS = -O3 -DNDEBUG
# -std=c++17 --offload-arch=gfx942 -fPIC (+ HIP_SEPARABLE_COMPILATION adds
# -fgpu-rdc at compile); the test links libvkernels.a (all kernel TUs).
# Matrix:
#   A. repro main linked against FULL libvkernels.a        (fault?)
#   B. repro linked with CMake-built moe_fused objects only (fault?)
#   C. manual moe_fused.hip builds: -fPIC / -fgpu-rdc / both (fault?)
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk150b7-$$
mkdir -p "$TMPDIR"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
SRC_INC="$SRC/src/c"

BC="$TMPDIR/build_c"
cmake -S "$SRC" -B "$BC" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$BC" --target test_moe_fused_prefill_correct -j 32 2>&1 \
  | grep -E "error|Built target" | tail -2

R="$TMPDIR/repro"; mkdir -p "$R"
CLANGXX=/opt/rocm-6.3.0/lib/llvm/bin/clang++
# repro main object (device-code-free, plain compile)
hipcc -x hip -O3 -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -c \
  -I"$SRC_INC" "$SRC/meta/benchmarks/repro_issue150_fault.hip" -o "$R/main.o" 2>&1 | grep error | head -3
g++ -O2 -std=c++17 -I"$SRC_INC" -c "$SRC_INC/vkernels/kernels/moe_fused.cpp" -o "$R/cpp.o" 2>&1 | grep error | head -3

echo; echo "############ A. full libvkernels.a ############"
CLANGXX -O3 -DNDEBUG --offload-arch=gfx942 --hip-link "$R/main.o" \
  "$BC/src/c/libvkernels.a" -lamdhip64 -o "$R/repro_full" 2>&1 | grep -E "error|undefined" | head -5
VK_MOE_PF_VARIANT=6 "$R/repro_full" 128 2>&1 | tail -3
echo "(full exit ${PIPESTATUS[0]})"

echo; echo "############ B. CMake moe_fused objects only ############"
cd "$R" && ar x "$BC/src/c/libvkernels.a" moe_fused.hip.o && cd - >/dev/null
CLANGXX -O3 -DNDEBUG --offload-arch=gfx942 --hip-link "$R/main.o" "$R/moe_fused.hip.o" "$R/cpp.o" \
  -lamdhip64 -o "$R/repro_mf" 2>&1 | grep -E "error|undefined" | head -5
VK_MOE_PF_VARIANT=6 "$R/repro_mf" 128 2>&1 | tail -3
echo "(mf-only exit ${PIPESTATUS[0]})"

echo; echo "############ C. manual flag matrix ############"
build_link_run () {  # $1 label, $2 extra compile flags
  local LBL="$1" F="$2"
  hipcc $F -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -c \
    -I"$SRC_INC" "$SRC_INC/vkernels/kernels/moe_fused.hip" -o "$R/mf_$LBL.o" 2>&1 | grep -E "error" | head -2
  CLANGXX -O3 -DNDEBUG --offload-arch=gfx942 --hip-link "$R/main.o" "$R/mf_$LBL.o" "$R/cpp.o" \
    -lamdhip64 -o "$R/repro_$LBL" 2>&1 | grep -E "error|undefined" | head -3
  [ -x "$R/repro_$LBL" ] || { echo "$LBL: BUILD FAIL"; return; }
  VK_MOE_PF_VARIANT=6 "$R/repro_$LBL" 128 2>&1 | tail -2
  echo "($LBL exit ${PIPESTATUS[0]})"
}
echo "--- pic only ---"
build_link_run pic "-O3 -DNDEBUG -std=c++17 -fPIC"
echo "--- rdc only ---"
build_link_run rdc "-O3 -DNDEBUG -std=c++17 -fgpu-rdc"
echo "--- pic+rdc (CMake equivalent) ---"
build_link_run picrdc "-O3 -DNDEBUG -std=c++17 -fPIC -fgpu-rdc"

echo; echo "===== ALL DONE ====="
