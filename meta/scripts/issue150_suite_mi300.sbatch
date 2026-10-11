#!/bin/bash
# Issue #150 regression arm (adjusted resubmit of terminal job 676611/676635
# pair).  Runs the full ctest suite after the verify arm's build, with:
#   - sbatch -o persistent output so a wall-timeout kill still leaves every
#     completed test row behind (the previous in-band SSH pipes lost it),
#   - ctest --timeout per test so one hung test cannot eat the wall,
#   - steady-state monitor samples every 60s,
#   - self-cancel at 900 GB RSS — respects the lane's 1 TB memory guard.
#SBATCH --job-name=vk150suite
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --dependency=afterany:%d
set -uo pipefail
LANE=/iopsstor/scratch/cscs/xyao/vk150lane
BC=$LANE/cmbuild
export PATH=/opt/rocm-6.3.0/bin:/opt/rocm-6.3.0/lib/llvm/bin:$PATH

echo "=== vk150 suite: node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ) ==="
echo "=== fixed-source md5: $(md5sum "$LANE/src/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) (expect dcb95d56) ==="

# In-band monitor: steady-state samples every 60s; self-cancel at 900 GB.
(
  while true; do
    RSS=$(sstat -j "${SLURM_JOB_ID}.batch" -n -o MaxRSS 2>/dev/null | tr -d ' ')
    MEM=$(free | awk '/^Mem:/{printf "%u/%uGB", $3, $2}')
    PROG=$(tail -1 "$BC/Testing/Temporary/LastTest.log.tmp" 2>/dev/null | head -c 80)
    echo "[mon $(date -u +%H:%M:%SZ)] rss=${RSS:-n/a} node-mem=${MEM} last=${PROG:-n/a}"
    KB=$(echo "$RSS" | awk '{v=$1; if (v ~ /K$/) {gsub(/K/,"",v); print v} else if (v ~ /M$/) {gsub(/M/,"",v); print v*1024} else if (v ~ /G$/) {gsub(/G/,"",v); print v*1048576}}')
    if [ -n "${KB:-}" ] && [ "${KB:-0}" -gt 921600000 ]; then
      echo "[mon] RSS exceeded 900 GB — respecting the 1 TB lane guard, cancelling"
      scancel "$SLURM_JOB_ID"
      exit 0
    fi
    sleep 60
  done
) &
MON_PID=$!

echo; echo "########## Phase 1: full incremental build (-j 16, guard-friendly) ##########"
cmake --build "$BC" -j 16 2>&1 | grep -E " error|error:" | head -5
echo "BUILD_EXIT=${PIPESTATUS[0]}"

echo; echo "########## Phase 2: full ctest suite (per-test timeout 300s) ##########"
cd "$BC" || exit 1
ctest --timeout 300 --output-on-failure -LE "multi|node" 2>&1

kill "$MON_PID" 2>/dev/null
echo; echo "===== suite done $(date -u +%FT%TZ) ====="
