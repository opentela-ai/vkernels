#!/bin/bash
#SBATCH --job-name=vki86c
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
# Issue #86 stage-split probe: localizes whether the large-ispp oracle
# mismatch comes from gateup_swiglu or down_combine, and prints the
# per-column badness map at the ispp=4096 boundary. Incremental build
# against the sweep job's build-sweep dir is fast.
#   SRC=/capstor/scratch/cscs/xyao/vk-issue-86 sbatch -A a-infra02 \
#     -o run.%j.out meta/scripts/issue86_probe_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
echo "=== node: $(hostname) job=${SLURM_JOB_ID:-interactive} date: $(date -u) ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
ulimit -c 0
B="$SRC/build-sweep"
echo; echo "=== build probe_moe_largeispp (incremental) ==="
cmake --build "$B" --target probe_moe_largeispp -j 64 2>&1 | grep -E "error:|Built target" | tail -4
PROBE="$B/meta/benchmarks/probe_moe_largeispp"
test -x "$PROBE" || { echo "probe not built"; exit 1; }

echo; echo "###### A. repro ispp=4096 ######"
"$PROBE" || echo "EXIT=$?"
echo; echo "###### B. control ispp=3072 ######"
"$PROBE" 32 7168 3072 || echo "EXIT=$?"
echo; echo "###### C. shard ispp=33792 (CPU oracle ON) ######"
"$PROBE" 32 7168 33792 || echo "EXIT=$?"
echo; echo "===== DONE ====="
