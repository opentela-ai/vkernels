#!/bin/bash
#SBATCH --job-name=vk-i137b
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
# Issue #137, confirmation run: shipped three-way dispatch (vec gate ->
# db occupancy gate -> serial), re-derived split_for = min(topk, 16),
# default G=1 (vectorization only; G>1 measured slower). Re-runs the
# 17-row device-vs-oracle suite under every dispatch variant + the decode
# split sweep serial vs db vs vec, and fills the missing fused/unfused
# cells at G=1 (the 641060 log showed a large fused-vs-unfused delta at
# G=4 that needs a repro at the default G).
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
echo "--- 1a. default (auto: vec gate -> db gate -> serial) ---"
"$T"
echo "--- 1b. VK_DSA_VEC=0 (legacy serial/db dispatch, pre-#137 paths) ---"
VK_DSA_VEC=0 "$T"
echo "--- 1c. VK_DSA_DB=2 (db kernel forced at every split) ---"
VK_DSA_DB=2 "$T"
echo "--- 1d. VK_DSA_FUSE=0 (vec partial, separate combine launch) ---"
VK_DSA_FUSE=0 "$T"
echo "--- 1e. VK_DSA_VECG=4 (register batch G=4) ---"
VK_DSA_VECG=4 "$T"
echo "--- 1f. VK_DSA_VECG=8 (deeper register batch, tail-group masking) ---"
VK_DSA_VECG=8 "$T"
echo "--- 1g. nondeterminism probe: default env, 3 reps ---"
VK_DSA_TEST_REPS=3 "$T"

echo
echo "###### 2. SPLIT SWEEP (dsa_bench_split137) — serial vs db vs vec ######"
echo "--- 2a. serial baseline (VK_DSA_DB=0 VK_DSA_VEC=0) ---"
VK_DSA_DB=0 VK_DSA_VEC=0 "$SW"
echo "--- 2b. db kernel forced at every split (VK_DSA_DB=2 VK_DSA_VEC=0) ---"
VK_DSA_DB=2 VK_DSA_VEC=0 "$SW"
echo "--- 2c. shipped default dispatch (auto gate, split_for=min(topk,16)) ---"
"$SW"
echo "--- 2d. vec G=1, fused combine, forced over the gate (VK_DSA_VEC=2) ---"
VK_DSA_VEC=2 "$SW"
echo "--- 2e. vec G=1, separate combine (VK_DSA_VEC=2 VK_DSA_FUSE=0) ---"
VK_DSA_VEC=2 VK_DSA_FUSE=0 "$SW"
echo "--- 2f. vec G=4, fused (VK_DSA_VEC=2 VK_DSA_VECG=4) ---"
VK_DSA_VEC=2 VK_DSA_VECG=4 "$SW"
echo "--- 2g. vec G=4, separate combine (641060 2d anomaly repro) ---"
VK_DSA_VEC=2 VK_DSA_FUSE=0 VK_DSA_VECG=4 "$SW"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
