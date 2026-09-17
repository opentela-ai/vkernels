#!/bin/bash
#SBATCH --job-name=vk76cnt
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
# Issue #76: hardware-counter evidence for the "the prefill gateup kernel is
# latency-bound with an ALU-heavy dequant, not MFMA-throughput-bound"
# argument.  Runs the prefill bench twice (variant 0 and the shipped default
# variant 5) under rocprof and prints, per kernel, the wave count, occupancy,
# instruction mix and stall split.
#
#   SRC=/iopsstor/scratch/cscs/xyao/vk76/src sbatch -A <acct> \
#       meta/scripts/issue76_counters_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk76cnt-$$
mkdir -p "$TMPDIR"; B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "src hip md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8)"

cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$B" --target moe_fused_prefill_bench -j 64 2>&1 \
  | grep -E "error|Error|Built target" | tail -5
PF="$B/meta/benchmarks/moe_fused_prefill_bench"

# rocprof v1 reruns the app once per "pmc :" line, so each line must be a
# counter group that fits the hardware's simultaneous-counter budget.
CAT="$TMPDIR/cnt.txt"
cat > "$CAT" <<'EOF'
pmc : Wavefronts VALUInsts LDSInsts FlatVMemInsts VALUUtilization
pmc : OccupancyPercent MeanOccupancyPerCU ALUStalledByLDS MemUnitStalled MemUnitBusy
pmc : SQ_INSTS_VALU_sum SQ_INSTS_VALU_MFMA_BF16_sum SQ_BUSY_CYCLES GRBM_COUNT
EOF

for V in 0 5; do
  echo; echo "############ VK_MOE_PF_VARIANT=$V (rocprof) ############"
  ( cd "$TMPDIR" && VK_MOE_PF_VARIANT=$V rocprof -i "$CAT" -o "cnt_v$V" \
      "$PF" situ ) 2>&1 | tail -5
done

echo; echo "############ per-kernel counters ############"
ls "$TMPDIR"
for V in 0 5; do
  for F in "$TMPDIR"/cnt_v$V*.csv; do
    [ -f "$F" ] || continue
    echo "--- $F"
    head -1 "$F"
    grep -E "gateup|down_combine" "$F" | grep -v KernelName | head -6
  done
done
echo; echo "===== ALL DONE ====="
