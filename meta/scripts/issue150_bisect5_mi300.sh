#!/bin/bash
#SBATCH --job-name=vk150b5
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:55:00
# Issue #150 fault bisect, round 5.  Round-4 finding: standalone repros never
# fault; only the CMake-built test does.  Reproduce the CMake build EXACTLY:
#   A. configure with compile_commands.json, dump hip compile + link flags
#   B. rebuild the standalone repro with those exact flags -> fault?
#   C. if B faults, drop-flag bisection
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk150b5-$$
mkdir -p "$TMPDIR"; K="$TMPDIR/keep"
mkdir -p "$K"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
SRC_INC="$SRC/src/c"

BC="$TMPDIR/build_c"
cmake -S "$SRC" -B "$BC" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null 2>&1
cmake --build "$BC" --target test_moe_fused_prefill_correct -j 32 2>&1 \
  | grep -E "error|Built target" | tail -2

python3 - <<'PYEOF' > "$K/flags.txt"
import json
d = json.load(open("/tmp/vk150b5-$SLURM_JOB_ID/build_c/compile_commands.json"))
for e in d:
    if "moe_fused.hip" in e["file"]:
        print("COMPILE:", e["command"]); break
PYEOF
cat "$BC/src/c/CMakeFiles/vkernels.dir/link.txt" 2>/dev/null | head -2 > "$K/link.txt"
cp "$K/flags.txt" "$K/link.txt" /iopsstor/scratch/cscs/xyao/vk150lane/ 2>/dev/null
echo "--- hip compile command ---"; cat "$K/flags.txt"
echo "--- vkernels link ---"; cat "$K/link.txt"

VK_MOE_PF_VARIANT=6 "$BC/meta/benchmarks/test_moe_fused_prefill_correct" situ 2>&1 | tail -4
echo "(cmake control exit ${PIPESTATUS[0]})"

echo; echo "############ B. repro with exact CMake hip flags ############"
# Extract just the flag words from the compile command and rebuild the repro.
python3 - <<'PYEOF' > "$K/hipflags.txt"
import json, shlex
d = json.load(open("/tmp/vk150b5-$SLURM_JOB_ID/build_c/compile_commands.json"))
for e in d:
    if "moe_fused.hip" in e["file"]:
        toks = shlex.split(e["command"])
        out = []
        for t in toks:
            if t in ("-c", "-o") or t.endswith(".hip") or t.endswith(".o"):
                if t == "-c": continue
                if t.endswith(".hip") or t.endswith(".o"): 
                    if t == "-o": out.append(t)
                    continue
            if out and out[-1] == "-o": out.append(t); continue
            if t in ("-c",): continue
            if t.endswith(".hip") or t.endswith(".o"): continue
            if t.startswith("-D") or t.startswith("-O") or t.startswith("-mllvm") \
               or t.startswith("-f") or t.startswith("-m") or t.startswith("--"):
                out.append(t)
        print(" ".join(out))
        break
PYEOF
HF=$(cat "$K/hipflags.txt")
echo "extracted flags: $HF"
R="$TMPDIR/repro"; mkdir -p "$R"
hipcc $HF -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -c \
  -I"$SRC_INC" "$SRC_INC/vkernels/kernels/moe_fused.hip" -o "$R/mf.o" 2>&1 | grep -E "error" | head -3
hipcc -x hip $HF -DVKERNELS_HAS_HIP=1 --offload-arch=gfx942 -c \
  -I"$SRC_INC" "$SRC/meta/benchmarks/repro_issue150_fault.hip" -o "$R/repro.o" 2>&1 | grep -E "error" | head -3
g++ -O2 -std=c++17 -I"$SRC_INC" -c "$SRC_INC/vkernels/kernels/moe_fused.cpp" -o "$R/cpp.o" 2>&1 | grep -E "error" | head -3
hipcc $HF --hip-link "$R/repro.o" "$R/mf.o" "$R/cpp.o" -lamdhip64 -o "$R/repro_exact" 2>&1 | grep -E "error|undefined" | head -3
ls -la "$R/repro_exact" || { echo "exact build FAILED"; }
VK_MOE_PF_VARIANT=6 "$R/repro_exact" 128 2>&1 | tail -4
echo "(exact-flags 128 exit ${PIPESTATUS[0]})"
VK_MOE_PF_VARIANT=6 "$R/repro_exact" seq 2>&1 | tail -5
echo "(exact-flags seq exit ${PIPESTATUS[0]})"

cp -r "$R" "$K/repro" 2>/dev/null
echo; echo "===== ALL DONE ====="
