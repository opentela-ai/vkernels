#!/bin/bash
# Issue #150 verification arm (adjusted resubmit of terminal job 676611/676635
# pair).  Persistent-output design: sbatch -o streams every phase to the lane
# artifact so a wall-timeout kill still leaves the monitor samples behind.
#   Phase 1  incremental rebuild (-j 16, stays under the 1 TB memory guard)
#   Phase 2  variant-6 verdict sweep (swiglu/situ, repeats, variants 0-6)
#   Phase 3  MOE ctest family with per-test timeout
#SBATCH --job-name=vk150verify
#SBATCH --partition=mi300
#SBATCH --account=a-infra01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
set -uo pipefail
LANE=/iopsstor/scratch/cscs/xyao/vk150lane
SRC=$LANE/src
BC=$LANE/cmbuild
export PATH=/opt/rocm-6.3.0/bin:/opt/rocm-6.3.0/lib/llvm/bin:$PATH

echo "=== vk150 verify-fix: node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ) ==="
echo "=== fixed-source md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) (expect dcb95d56) ==="

# In-band monitor: steady-state samples every 60s; self-cancel at 900 GB so the
# job stays under the lane's 1 TB memory guard by design.
(
  while true; do
    RSS=$(sstat -j "${SLURM_JOB_ID}.batch" -n -o MaxRSS 2>/dev/null | tr -d ' ')
    MEM=$(free | awk '/^Mem:/{printf "%u/%uGB", $3, $2}')
    echo "[mon $(date -u +%H:%M:%SZ)] rss=${RSS:-n/a} node-mem=${MEM}"
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

echo; echo "########## Phase 1: incremental rebuild (vkernels + lane test targets) ##########"
cmake --build "$BC" -j 16 \
  --target vkernels test_moe_fused_prefill_correct \
  vkernels_test_moe vkernels_test_moe_fused vkernels_test_moe_align_oracle \
  vkernels_test_moe_aux vkernels_test_moe_aux_fused vkernels_test_glm_moe \
  vkernels_test_dist_moe 2>&1 | grep -E " error|error:|Built target" | tail -12
echo "BUILD_EXIT=$? (0 = up to date with fixed source)"

echo; echo "########## Phase 2: variant-6 verdict sweep ##########"
T=$BC/meta/benchmarks/test_moe_fused_prefill_correct
echo "--- default (v5 fallback path) swiglu ---"
"$T" 2>&1 | tail -3
for v in 0 1 2 3 4; do
  echo "--- variant $v swiglu ---"
  VK_MOE_PF_VARIANT=$v "$T" 2>&1 | tail -2
done
for rep in 1 2 3; do
  echo "--- variant 6 swiglu rep $rep ---"
  VK_MOE_PF_VARIANT=6 "$T" 2>&1 | tail -4
  echo "--- variant 6 situ rep $rep ---"
  VK_MOE_PF_VARIANT=6 "$T" situ 2>&1 | tail -2
done

echo; echo "########## Phase 3: MOE ctest family (per-test timeout 300s) ##########"
cd "$BC" || exit 1
ctest -R "^moe$|^moe_fused$|^moe_align_oracle$|^moe_aux$|^moe_aux_fused$|^glm_moe$|^dist_moe$" \
  --timeout 300 --output-on-failure 2>&1 | tail -25

kill "$MON_PID" 2>/dev/null
echo; echo "===== verify-fix done $(date -u +%FT%TZ) ====="
