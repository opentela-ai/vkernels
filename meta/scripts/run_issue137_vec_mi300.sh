#!/bin/bash
#SBATCH --job-name=vk-i137c
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
# Issue #137, final confirmation: shipped defaults are now the UNFUSED
# vectorized partial (G=1, no gate) + separate combine, split_for =
# 32 (topk>=1024) / min(topk,16) -- jobs 641060/641089 measured the
# unfused vec kernel >= db + serial at every admitted (shape, split).
# 18-row device-vs-oracle suite under every dispatch variant + the decode
# split sweep for the shipped path vs each legacy/diagnostic variant.
# sbatch-safe (lands on mi300); also runnable interactively:
#   srun --partition=mi300 -N1 -G1 --time=00:55:00 \
#     bash meta/scripts/run_issue137_vec_mi300.sh
set -euo pipefail
ulimit -c 0
: "${SRC:=/capstor/scratch/cscs/xyao/vkernels-i137-agent}"
B="$SRC/build-i137"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release, benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -3

echo "=== build (test_dsa_correct + dsa_bench_split137) ==="
cmake --build "$B" --target test_dsa_correct dsa_bench_split137 -j 64 2>&1 \
  | grep -E "error:|Built target" | tail -8

T="$B/meta/benchmarks/test_dsa_correct"
SW="$B/meta/benchmarks/dsa_bench_split137"

echo
echo "###### 1. CORRECTNESS (test_dsa_correct) — every dispatch variant ######"
echo "--- 1a. default (unfused vec partial + separate combine) ---"
"$T"
echo "--- 1b. VK_DSA_VEC=0 (legacy serial/db dispatch, pre-#137 paths) ---"
VK_DSA_VEC=0 "$T"
echo "--- 1c. VK_DSA_DB=2 (db kernel forced at every split) ---"
VK_DSA_DB=2 "$T"
echo "--- 1d. VK_DSA_FUSE=1 (fused combine: fence+atomic+last-warp merge) ---"
VK_DSA_FUSE=1 "$T"
echo "--- 1e. VK_DSA_VECG=4 (register batch G=4) ---"
VK_DSA_VECG=4 "$T"
echo "--- 1f. VK_DSA_VECG=8 (deeper register batch, tail-group masking) ---"
VK_DSA_VECG=8 "$T"
echo "--- 1g. fused + G=4 combined (641089 2f config) ---"
VK_DSA_FUSE=1 VK_DSA_VECG=4 "$T"
echo "--- 1h. nondeterminism probe: default env, 3 reps ---"
VK_DSA_TEST_REPS=3 "$T"

echo
echo "###### 2. SPLIT SWEEP (dsa_bench_split137) — serial vs db vs vec ######"
echo "--- 2a. serial baseline (VK_DSA_DB=0 VK_DSA_VEC=0) ---"
VK_DSA_DB=0 VK_DSA_VEC=0 "$SW"
echo "--- 2b. db kernel forced at every split (VK_DSA_DB=2 VK_DSA_VEC=0) ---"
VK_DSA_DB=2 VK_DSA_VEC=0 "$SW"
echo "--- 2c. shipped default dispatch (unfused vec, split_for 32/16) ---"
"$SW"
echo "--- 2d. fused combine (VK_DSA_FUSE=1; expected ~20-25% slower) ---"
VK_DSA_FUSE=1 "$SW"
echo "--- 2e. vec G=4, unfused (VK_DSA_VECG=4) ---"
VK_DSA_VECG=4 "$SW"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
