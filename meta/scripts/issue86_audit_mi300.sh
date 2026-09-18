#!/bin/bash
#SBATCH --job-name=vk-i86-audit
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Issue #86 audit: independent re-validation of fe8b5ec's two claims on a
# clean beverin MI300A node with current code:
#   (a) the ispp>=4096 "memory access fault" does not reproduce on the
#       current build (recorded fault shapes, dummy/no-cpu, exit-code checked);
#   (b) the residual oracle MISMATCH is a bounded bf16-act-ULP artifact:
#       probe_moe_largeispp across seeds 0..4 / M=1,2 / ispp 3072,4096,33792
#       with the bf16-ULP act distance and the GPU-vs-GPU rerun spread
#       (atomicAdd accumulation-order noise floor) printed per rep.
#   SRC=/capstor/scratch/cscs/xyao/vk-issue-86 sbatch -A a-infra02 \
#     -o run.%j.out meta/scripts/issue86_audit_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
echo "=== node: $(hostname) job=${SLURM_JOB_ID:-interactive} date: $(date -u) ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
test -d "$SRC" || { echo "SRC=$SRC not found"; exit 1; }
ulimit -c 0
B="$SRC/build-sweep"
mkdir -p "$B"
if [ ! -f "$B/CMakeCache.txt" ]; then
  echo; echo "=== configure (HIP gfx942 Release, benchmarks ON) ==="
  cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_BENCHMARKS=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
fi
echo; echo "=== build moe_fused_bench + probe_moe_largeispp (incremental) ==="
cmake --build "$B" --target moe_fused_bench probe_moe_largeispp -j 64 2>&1 | grep -E "error:|warning:|Built target" | tail -8
GPU="$B/meta/benchmarks/moe_fused_bench"
PROBE="$B/meta/benchmarks/probe_moe_largeispp"
test -x "$GPU" || { echo "moe_fused_bench not built"; exit 1; }
test -x "$PROBE" || { echo "probe not built"; exit 1; }

FAULT_FREE=1

echo; echo "###### A. fault repro on CURRENT code (exit codes checked) ######"
echo "--- E=32 h7168 ispp=4096 kmajor M=1,2 dummy no-cpu (minimal recorded shape) ---"
"$GPU" situ --E 32 --hidden 7168 --ispp 4096 --topk 16 --ms 1,2 --kmajor --dummy --no-cpu || FAULT_FREE=0
echo "--- E=32 h7168 ispp=4096 nmajor M=1,2 dummy no-cpu ---"
"$GPU" situ --E 32 --hidden 7168 --ispp 4096 --topk 16 --ms 1,2 --dummy --no-cpu || FAULT_FREE=0
echo "--- E=256 h7168 ispp=4096 kmajor M=1,2 dummy no-cpu (recorded fault shape) ---"
"$GPU" situ --E 256 --hidden 7168 --ispp 4096 --topk 16 --ms 1,2 --kmajor --dummy --no-cpu || FAULT_FREE=0
echo "--- E=32 h7168 ispp=33792 kmajor M=1,2 dummy no-cpu (TP-shard) ---"
"$GPU" situ --E 32 --hidden 7168 --ispp 33792 --topk 16 --ms 1,2 --kmajor --dummy --no-cpu || FAULT_FREE=0
echo "--- E=256 h7168 ispp=33792 kmajor M=1,2 dummy no-cpu (whole-model K3) ---"
"$GPU" situ --E 256 --hidden 7168 --ispp 33792 --topk 16 --ms 1,2 --kmajor --dummy --no-cpu || FAULT_FREE=0
echo "[audit] fault-repro block FAULT_FREE=$FAULT_FREE (1 = all shapes ran clean)"

echo; echo "###### B. oracle-mismatch quantification across seeds/M/ispp ######"
for SEED in 0 1 2 3 4; do
  echo "--- ispp=4096 kmajor M=1 seed=$SEED ---"
  "$PROBE" 32 7168 4096 kmajor 1 "$SEED" || echo "EXIT=$?"
done
for SEED in 0 1 2; do
  echo "--- ispp=4096 nmajor M=2 seed=$SEED ---"
  "$PROBE" 32 7168 4096 nmajor 2 "$SEED" || echo "EXIT=$?"
done
echo "--- control ispp=3072 kmajor M=1 seed=0 ---"
"$PROBE" 32 7168 3072 kmajor 1 0 || echo "EXIT=$?"
echo "--- control ispp=3072 kmajor M=1 seed=3 ---"
"$PROBE" 32 7168 3072 kmajor 1 3 || echo "EXIT=$?"
for SEED in 0 1 2; do
  echo "--- shard ispp=33792 kmajor M=1 seed=$SEED ---"
  "$PROBE" 32 7168 33792 kmajor 1 "$SEED" || echo "EXIT=$?"
done

echo; echo "===== DONE (FAULT_FREE=$FAULT_FREE) ====="
