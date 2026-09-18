#!/bin/bash
#SBATCH --job-name=vki86
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Issue #86: minimal repro + bisect for the fused_moe_mxfp4 memory fault at
# ispp >= 4096. Every bench invocation is its own short-lived process (the
# MI300A per-context fault workaround) and core dumps are disabled.
#   SRC=/capstor/scratch/cscs/xyao/vk-issue-86 sbatch -o run.%j.out \
#     meta/scripts/issue86_repro_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vki86-$$
mkdir -p "$TMPDIR"
B="$TMPDIR/build"
echo "=== node: $(hostname) job=${SLURM_JOB_ID:-interactive} date: $(date -u) ==="
echo "=== src md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8) ==="
test -d "$SRC" || { echo "SRC=$SRC not found"; exit 1; }
ulimit -c 0

echo; echo "=== configure (HIP gfx942 Release, benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
echo "=== build (moe_fused_bench) ==="
cmake --build "$B" --target moe_fused_bench -j 64 2>&1 | grep -E "error:|Built target" | tail -4
GPU="$B/meta/benchmarks/moe_fused_bench"
test -x "$GPU" || { echo "moe_fused_bench not built"; exit 1; }

echo; echo "###### A. control: ispp=2048 E=32 --no-cpu (expect OK) ######"
"$GPU" situ --E 32 --hidden 7168 --ispp 2048 --topk 16 --ms 1 --kmajor --dummy --no-cpu || echo "EXIT=$?"
echo; echo "###### B. minimal repro: ispp=4096 E=32 --no-cpu ######"
"$GPU" situ --E 32 --hidden 7168 --ispp 4096 --topk 16 --ms 1 --kmajor --dummy --no-cpu || echo "EXIT=$?"
echo; echo "###### C. oracle ON at the same shape (ispp=4096 E=32) ######"
"$GPU" situ --E 32 --hidden 7168 --ispp 4096 --topk 16 --ms 1 --kmajor || echo "EXIT=$?"
echo; echo "###### D. recorded fault: ispp=4096 E=256 --no-cpu ######"
"$GPU" situ --E 256 --hidden 7168 --ispp 4096 --topk 16 --ms 1 --kmajor --dummy --no-cpu || echo "EXIT=$?"
echo; echo "###### E. full-K3 shard: ispp=33792 E=32 --no-cpu ######"
"$GPU" situ --E 32 --hidden 7168 --ispp 33792 --topk 16 --ms 1 --kmajor --dummy --no-cpu || echo "EXIT=$?"
echo; echo "###### F. compute-sanitizer memcheck at shape B ######"
if command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --tool memcheck --print-level info \
    "$GPU" situ --E 32 --hidden 7168 --ispp 4096 --topk 16 --ms 1 --kmajor --dummy --no-cpu 2>&1 | tail -60 || true
else
  echo "(compute-sanitizer not on PATH)"
fi
echo; echo "###### G. ktime stage split at shape B (which stage faults?) ######"
VK_MOE_KTIME=1 "$GPU" situ --E 32 --hidden 7168 --ispp 4096 --topk 16 --ms 1 --kmajor --dummy --no-cpu || echo "EXIT=$?"
echo; echo "===== DONE ====="
