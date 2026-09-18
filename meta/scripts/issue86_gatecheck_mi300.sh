#!/bin/bash
#SBATCH --job-name=vk-i86-gate
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
# Issue #86: verify the retuned bigshape determinism gate (bound on the
# float-atomicAdd accumulation-order spread instead of bitwise equality)
# builds and passes all cases on MI300A.
#   SRC=/capstor/scratch/cscs/xyao/vk-issue-86 sbatch -A a-infra02 \
#     -o run.%j.out meta/scripts/issue86_gatecheck_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
echo "=== node: $(hostname) job=${SLURM_JOB_ID:-interactive} date: $(date -u) ==="
echo "=== src md5: $(md5sum "$SRC/meta/benchmarks/test_moe_fused_bigshape_correct.hip" | cut -c1-8) ==="
ulimit -c 0
B="$SRC/build-sweep"
echo; echo "=== build test_moe_fused_bigshape_correct (incremental) ==="
cmake --build "$B" --target test_moe_fused_bigshape_correct -j 64 2>&1 | grep -E "error:|Built target" | tail -4
TS="$B/meta/benchmarks/test_moe_fused_bigshape_correct"
test -x "$TS" || { echo "bigshape test not built"; exit 1; }
"$TS"; RC=$?
echo; echo "===== DONE rc=$RC ====="
