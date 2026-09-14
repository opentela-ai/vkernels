#!/bin/bash
#SBATCH --job-name=vkmoe41prof
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
# Issue #41: per-kernel rocprof profile of the fused-MoE decode at
# (a) the acceptance default-harness shape (E=256 h4096 ispp=512 k6, M=1,
#     oracle-checked — target <=0.30 ms, currently ~0.32 ms), and
# (b) the full-K3 shard shape (E=32 h7168 ispp=33792, M=1, --dummy).
# The --stats summary gives per-kernel mean durations (gateup vs down vs
# align vs combine) so the remaining latency can be attributed to a stage
# before optimizing further.
#   SRC=$HOME/vkernels sbatch meta/scripts/prof_moe_rocprof_mi300.sh
set -euo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vkmoe41prof-$$
mkdir -p "$TMPDIR"
B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
date
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -2
cmake --build "$B" --target moe_fused_bench -j 64 \
  2>&1 | grep -E "error:|Built target" | tail -3

GPU="$B/meta/benchmarks/moe_fused_bench"
PROF=$(command -v rocprof)

echo; echo "###### default harness M=1 (per-kernel durations) ######"
"$PROF" --stats -o "$TMPDIR/rp_default.csv" "$GPU" situ --ispp 512 --topk 6 --ms 1 \
  2>&1 | grep -vE "^\[|ROCProfiler" | tail -5
echo "--- stats:"; cat "$TMPDIR/rp_default"*.csv 2>/dev/null | head -20

echo; echo "###### full-K3 shard M=1 (per-kernel durations) ######"
"$PROF" --stats -o "$TMPDIR/rp_shard.csv" \
  "$GPU" situ --E 32 --hidden 7168 --ispp 33792 --topk 16 --ms 1 \
  --kmajor --dummy --no-cpu 2>&1 | grep -vE "^\[|ROCProfiler" | tail -5
echo "--- stats:"; cat "$TMPDIR/rp_shard"*.csv 2>/dev/null | head -20

echo; echo "###### unit-busy counters, default harness M=1 ######"
cat > "$TMPDIR/pmc.txt" <<'PMC'
pmc: SQ_WAVES SQ_PERCENT_BUSY SQ_INST_CYCLE_VAL
pmc: SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE
pmc: TCP_TOTAL_CACHE_ACCESSES_pmc_TCC_TOTAL_READ_SECTORS_pmc_TCC_TOTAL_WRITE_SECTORS_pmc_TCC_MC_RD_REQ_sum_TCC_MC_WR_REQ_sum
PMC
"$PROF" -i "$TMPDIR/pmc.txt" -o "$TMPDIR/rp_pmc.csv" "$GPU" situ --ispp 512 --topk 6 --ms 1 \
  2>&1 | grep -vE "^\[|ROCProfiler" | tail -3
echo "--- pmc:"; cat "$TMPDIR/rp_pmc"*.csv 2>/dev/null | grep -E "kernel|SQ_|TCP_|TCC_" | head -12

echo; echo "===== ALL DONE ====="
