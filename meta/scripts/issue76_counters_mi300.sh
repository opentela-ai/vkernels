#!/bin/bash
#SBATCH --job-name=vk76cnt
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
# Issue #76: hardware-counter evidence for the mechanism behind the prefill
# win -- "variant 5 raises measured occupancy by shrinking the tile's LDS and
# VGPR footprint and issuing 4x the gateup grid, it does not make the MFMA
# pipe busier".  Runs the prefill bench twice (v0 baseline and the shipped
# default v5) under rocprof and prints, per kernel, the profile-lock data
# (grid, LDS, VGPR, wave size), occupancy and the stall split.
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
# OccupancyPercent is MeanOccupancyPerCU over "Max Waves Per CU", so record
# the device shape that those two columns are relative to.
rocminfo 2>/dev/null | awk '
  /Marketing Name/ && !seen++ {print "gpu:", $0}
  /Max Waves Per CU/ {print "shape:", $0}
  /Wavefront Size/ {print "shape:", $0}
  /^ *Compute Unit:/ {cus++}
  END {print "compute units:", cus}'

cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$B" --target moe_fused_prefill_bench -j 64 2>&1 \
  | grep -E "error|Error|Built target" | tail -5
PF="$B/meta/benchmarks/moe_fused_prefill_bench"

# rocprof v1 reruns the app once per "pmc :" line, so each line must be a
# counter group that fits the hardware's simultaneous-counter budget.  Use
# one input file per group: rocprof names the profile after the input file
# base, so distinct input names give distinct CSVs instead of overwriting.
#
# Do NOT pass -o: on ROCm 6.3 rocprof v1 rejects its own default output name
# with "error: file name must have .CSV extension" whenever -o is given (it
# fails the same way with -o out.CSV, so it is not about the extension).
# Without -o, rocprof writes <cwd>/<input base>.csv, hence the cd.
#
# Counter names are the raw gfx942 ones from `rocprof --list-basic`; the
# `SQ_INSTS_VALU_sum`-style derived spelling is rejected on this part.
# The group list is one "name counters..." line per pmc pass.  Do not iterate
# it with an unquoted for-loop: that splits on every space, not on lines.
groups() {
  cat <<'EOF'
occ OccupancyPercent MeanOccupancyPerCU ALUStalledByLDS MemUnitStalled MemUnitBusy
valu SQ_INSTS_VALU SQ_INSTS_VALU_CVT SQ_INSTS_MFMA SQ_WAVES
cyc SQ_BUSY_CYCLES GRBM_COUNT SQ_WAIT_INST_LDS
EOF
}

for V in 0 5; do
  groups | while read -r g ctrs; do
    printf 'pmc : %s\n' "$ctrs" > "$TMPDIR/cnt_v${V}_${g}.txt"
  done
done

for V in 0 5; do
  groups | while read -r g ctrs; do
    echo; echo "############ VK_MOE_PF_VARIANT=$V pmc-group=$g (rocprof) ############"
    ( cd "$TMPDIR" && VK_MOE_PF_VARIANT=$V rocprof -i "cnt_v${V}_${g}.txt" \
        "$PF" situ ) 2>&1 | tail -4
  done
done

echo; echo "############ per-kernel counters ############"
for V in 0 5; do
  groups | while read -r g ctrs; do
    F="$TMPDIR/cnt_v${V}_${g}.csv"
    echo; echo "--- variant $V group $g ($F)"
    [ -f "$F" ] || { echo "(missing)"; continue; }
    head -1 "$F"
    grep -E "gateup|down_combine" "$F" | grep -v KernelName | head -4
  done
done
echo; echo "===== ALL DONE ====="
