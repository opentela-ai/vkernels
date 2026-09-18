#!/bin/bash
#SBATCH --job-name=vk-i137
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
# Issue #137: vectorized + register-batched dsa_sparse_fwd_split partial
# (lever a) with the optional FUSED combine (lever b) -- 17-row
# device-vs-oracle correctness under every dispatch variant + the decode
# split sweep (H=64 topk=2048 split {16,32,64,128} is the acceptance row)
# serial vs vec, same methodology as run_issue80_db_mi300.sh.
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
echo "--- 1a. default (auto: db gate low-split, vec+fuse high-split) ---"
"$T"
echo "--- 1b. VK_DSA_VEC=0 (legacy serial/db dispatch, pre-#137 paths) ---"
VK_DSA_VEC=0 "$T"
echo "--- 1c. VK_DSA_DB=2 (db kernel forced at every split) ---"
VK_DSA_DB=2 "$T"
echo "--- 1d. VK_DSA_FUSE=0 (vec partial, separate combine launch) ---"
VK_DSA_FUSE=0 "$T"
echo "--- 1e. VK_DSA_VECG=8 (deeper register batch, tail-group masking) ---"
VK_DSA_VECG=8 "$T"
echo "--- 1f. VK_DSA_VECG=1 (vectorization only, no batching) ---"
VK_DSA_VECG=1 "$T"
echo "--- 1g. nondeterminism probe: default env, 3 reps ---"
VK_DSA_TEST_REPS=3 "$T"

echo
echo "###### 2. SPLIT SWEEP (dsa_bench_split137) — serial vs vec variants ######"
echo "--- 2a. serial baseline (VK_DSA_DB=0 VK_DSA_VEC=0) ---"
VK_DSA_DB=0 VK_DSA_VEC=0 "$SW"
echo "--- 2b. vec, vectorization only (G=1) ---"
VK_DSA_DB=0 VK_DSA_VEC=2 VK_DSA_VECG=1 "$SW"
echo "--- 2c. vec G=4, fused combine (default settings, forced over db) ---"
VK_DSA_VEC=2 "$SW"
echo "--- 2d. vec G=4, separate combine (isolates lever b) ---"
VK_DSA_VEC=2 VK_DSA_FUSE=0 "$SW"
echo "--- 2e. vec G=8 ---"
VK_DSA_VEC=2 VK_DSA_VECG=8 "$SW"
echo "--- 2f. shipped default dispatch (auto gate) ---"
"$SW"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
