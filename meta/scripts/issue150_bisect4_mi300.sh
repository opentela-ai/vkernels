#!/bin/bash
#SBATCH --job-name=vk150b4
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
# Issue #150 fault bisect, round 4.  Round-3 finding: an -fgpu-rdc standalone
# repro at block=128 on FIRST launch does NOT fault.  Discriminate:
#   A. CMake control build (expect fault) + capture the hip compile command
#   B. standalone NON-rdc build: 64-control, 128-first-launch
#   C. standalone rdc build: seq (64-v5-launches then 128-first-launch)
#   D. standalone rdc build: alloc pressure then 128
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk150b4-$$
mkdir -p "$TMPDIR"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
SRC_INC="$SRC/src/c"

echo; echo "############ A. CMake control ############"
BC="$TMPDIR/build_c"
cmake -S "$SRC" -B "$BC" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$BC" --target test_moe_fused_prefill_correct -j 32 2>&1 \
  | grep -E "error|Built target" | tail -2
python3 - "$BC/compile_commands.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for e in d:
    if "moe_fused.hip" in e["file"] and "vkernels.dir" in e["command"] or "moe_fused.hip" in e["file"]:
        print("HIP COMPILE FLAGS:", " ".join(c for c in e["command"].split() if c.startswith("-") and "I/" not in c)[:600])
        break
PYEOF
VK_MOE_PF_VARIANT=6 "$BC/meta/benchmarks/test_moe_fused_prefill_correct" situ 2>&1 | tail -6
echo "(cmake exit ${PIPESTATUS[0]})"

R="$TMPDIR/repro"; mkdir -p "$R"
build_repro () {  # $1 = extra flags, $2 = suffix
  local F="$1" S="$2"
  hipcc $F -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -c \
    -I"$SRC_INC" "$SRC_INC/vkernels/kernels/moe_fused.hip" -o "$R/mf_$S.o" 2>&1 | grep -E "error" | head -3
  hipcc -x hip $F -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -c \
    -I"$SRC_INC" "$SRC/meta/benchmarks/repro_issue150_fault.hip" -o "$R/repro_$S.o" 2>&1 | grep -E "error" | head -3
  g++ -O2 -std=c++17 -I"$SRC_INC" -c "$SRC_INC/vkernels/kernels/moe_fused.cpp" -o "$R/cpp_$S.o" 2>&1 | grep -E "error" | head -3
  hipcc $F --hip-link "$R/repro_$S.o" "$R/mf_$S.o" "$R/cpp_$S.o" -lamdhip64 -o "$R/repro_$S" 2>&1 | grep -E "error|undefined" | head -3
  ls "$R/repro_$S" >/dev/null 2>&1 || echo "build $S FAILED"
}

build_repro "" nordc
build_repro "-fgpu-rdc" rdc

echo; echo "############ B. non-rdc: 64 / 128 ############"
"$R/repro_nordc" 64 2>&1 | tail -2
VK_MOE_PF_VARIANT=6 "$R/repro_nordc" 128 2>&1 | tail -3
echo "(nordc 128 exit ${PIPESTATUS[0]})"

echo; echo "############ C. rdc: seq (64 first, then 128) ############"
VK_MOE_PF_VARIANT=6 "$R/repro_rdc" seq 2>&1 | tail -6
echo "(rdc seq exit ${PIPESTATUS[0]})"

echo; echo "############ D. rdc: alloc pressure then 128 ############"
VK_MOE_PF_VARIANT=6 "$R/repro_rdc" alloc 2>&1 | tail -4
echo "(rdc alloc exit ${PIPESTATUS[0]})"

echo; echo "===== ALL DONE ====="
