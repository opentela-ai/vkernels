#!/bin/bash
#SBATCH --job-name=vki86b
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Issue #86: bracket the ispp>=4096 oracle MISMATCH (cpu-rel 0.166 at E=32
# ispp=4096 M=1 kmajor) across ispp x M x kmajor; re-run the recorded fault
# shapes at M=1,2 (the records used --ms 1,2; the first repro pass used
# --ms 1 only); the 97GB whole-model acceptance shape; and the bigshape
# correctness test as an independent cross-check.
#   SRC=/capstor/scratch/cscs/xyao/vk-issue-86 sbatch -A a-infra02 \
#     -o run.%j.out meta/scripts/issue86_sweep_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
echo "=== node: $(hostname) job=${SLURM_JOB_ID:-interactive} date: $(date -u) ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
test -d "$SRC" || { echo "SRC=$SRC not found"; exit 1; }
ulimit -c 0
B="$SRC/build-sweep"
mkdir -p "$B"
if [ ! -f "$B/CMakeCache.txt" ]; then
  echo; echo "=== configure (HIP gfx942 Release, benchmarks+tests ON) ==="
  cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
    -DVKERNELS_BUILD_BENCHMARKS=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
fi
echo; echo "=== build (moe_fused_bench) ==="
cmake --build "$B" --target moe_fused_bench -j 64 2>&1 | grep -E "error:|Built target" | tail -4
GPU="$B/meta/benchmarks/moe_fused_bench"
test -x "$GPU" || { echo "moe_fused_bench not built"; exit 1; }

echo; echo "###### 1. ORACLE sweep — bracket the MISMATCH (E=32, real weights) ######"
for IPP in 512 1024 2048 2560 3072 4096; do
  echo "--- kmajor ispp=$IPP ---"
  "$GPU" situ --E 32 --hidden 7168 --ispp $IPP --topk 16 --ms 1,2 --kmajor || echo "EXIT=$?"
  echo "--- nmajor ispp=$IPP ---"
  "$GPU" situ --E 32 --hidden 7168 --ispp $IPP --topk 16 --ms 1,2 || echo "EXIT=$?"
done

echo; echo "###### 2. recorded fault shapes re-run (--no-cpu, dummy) ######"
echo "--- E=256 ispp=4096 kmajor M=1,2 (recorded fault shape) ---"
"$GPU" situ --E 256 --hidden 7168 --ispp 4096 --topk 16 --ms 1,2 --kmajor --dummy --no-cpu || echo "EXIT=$?"
echo "--- E=32 ispp=33792 kmajor M=1,2 (TP-shard) ---"
"$GPU" situ --E 32 --hidden 7168 --ispp 33792 --topk 16 --ms 1,2 --kmajor --dummy --no-cpu || echo "EXIT=$?"

echo; echo "###### 3. acceptance shape: E=256 ispp=33792 whole model (~97 GB) ######"
"$GPU" situ --E 256 --hidden 7168 --ispp 33792 --topk 16 --ms 1 --kmajor --dummy --no-cpu || echo "EXIT=$?"

echo; echo "###### 4. bigshape correctness test (independent cross-check) ######"
cmake --build "$B" --target test_moe_fused_bigshape_correct -j 64 2>&1 | grep -E "error:|Built target" | tail -3 || true
TS="$B/meta/benchmarks/test_moe_fused_bigshape_correct"
test -x "$TS" && "$TS" || echo "(bigshape test unavailable or failed)"
echo; echo "===== DONE ====="
