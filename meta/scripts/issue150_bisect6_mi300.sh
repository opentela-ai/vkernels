#!/bin/bash
#SBATCH --job-name=vk150b6
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
# Issue #150 round 6: dump the EXACT hip compile command for moe_fused.hip
# and the test link command, save them to scratch, confirm the control fault.
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk150b6-$$
mkdir -p "$TMPDIR"
K=/iopsstor/scratch/cscs/xyao/vk150lane/flags
mkdir -p "$K"
echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="

BC="$TMPDIR/build_c"
cmake -S "$SRC" -B "$BC" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null 2>&1
cmake --build "$BC" --target test_moe_fused_prefill_correct -j 32 2>&1 \
  | grep -E "error|Built target" | tail -2

export BUILDDIR="$BC"
python3 <<'PYEOF' | tee /iopsstor/scratch/cscs/xyao/vk150lane/flags/moe_fused_cmd.txt
import json, os, shlex
d = json.load(open(os.path.join(os.environ["BUILDDIR"], "compile_commands.json")))
for e in d:
    if "moe_fused.hip" in e["file"]:
        print(shlex.join(shlex.split(e["command"])))
        break
PYEOF

echo "--- test link.txt ---"
cat "$BC"/meta/benchmarks/CMakeFiles/test_moe_fused_prefill_correct.dir/link.txt 2>/dev/null \
  | tee /iopsstor/scratch/cscs/xyao/vk150lane/flags/test_link.txt
echo "--- hip lld/hipcc flags from flags.make ---"
grep -E "HIP_FLAGS|_FLAGS" "$BC/src/c/CMakeFiles/vkernels.dir/flags.make" 2>/dev/null \
  | tee /iopsstor/scratch/cscs/xyao/vk150lane/flags/vkernels_flags.txt

VK_MOE_PF_VARIANT=6 "$BC/meta/benchmarks/test_moe_fused_prefill_correct" situ 2>&1 | tail -4
echo "(cmake control exit ${PIPESTATUS[0]})"
echo "===== ALL DONE ====="
