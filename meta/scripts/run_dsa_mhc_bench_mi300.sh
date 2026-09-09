#!/bin/bash
#SBATCH --job-name=vkdsamhc
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
# Build + benchmark the GLM-5.3-Flash / DeepSeek-V3 sparse-attention stack
# (issue #51) on an MI300A (gfx942) compute node -- CSCS beverin `mi300`
# partition (NOT bristen, which is NVIDIA GH200 with no ROCm).
#
# Covers the two issue-#51 kernel groups that have no perf numbers yet:
#   * dsa_bench        -- hip::dsa_sparse_fwd (sparse-MLA forward), GLM-5.3-Flash
#                         (tail_dim == 0) + DeepSeek-V3 (tail_dim > 0) shapes.
#   * mhc_bench        -- hip::mhc_pre_gemm_sqrsum + hip::mhc_post.
# plus, for same-build consistency with docs/performance/dsa-topk/gfx942.md:
#   * dsa_topk_logits_bench -- the indexer stage that feeds dsa_sparse_fwd.
# Correctness gates run first (device vs CPU oracle):
#   * test_dsa_correct, test_mhc_correct
#
# Run interactively:
#   srun --partition=mi300 -N1 -G1 --time=00:40:00 bash meta/scripts/run_dsa_mhc_bench_mi300.sh
# Or batch:
#   sbatch -p mi300 -N1 --gres=gpu:1 --time=00:40:00 \
#     -o dsa_mhc_bench.%j.out meta/scripts/run_dsa_mhc_bench_mi300.sh
set -euo pipefail
: "${SRC:=$HOME/vkernels}"
B="$SRC/build_dsamhc"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release, tests + benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -4

echo "=== build (vkernels + hip C ABI + correctness tests + the three benches) ==="
cmake --build "$B" --target vkernels vkernels_hip test_dsa_correct test_mhc_correct \
                                        dsa_bench mhc_bench dsa_topk_logits_bench \
                                        -j 64 \
  2>&1 | grep -E "error:|warning: |Built target|dsa_bench|mhc_bench|logits_bench|correct|hip_capi" | tail -14

echo
echo "############ 1. DSA CORRECTNESS (test_dsa_correct) ############"
"$B/meta/benchmarks/test_dsa_correct" 2>&1

echo
echo "############ 2. MHC CORRECTNESS (test_mhc_correct) ############"
"$B/meta/benchmarks/test_mhc_correct" 2>&1

echo
echo "############ 3. BENCHMARK: DSA sparse-MLA forward (dsa_bench) ############"
"$B/meta/benchmarks/dsa_bench" 2>&1

echo
echo "############ 4. BENCHMARK: MHC pre/post (mhc_bench) ############"
"$B/meta/benchmarks/mhc_bench" 2>&1

echo
echo "############ 5. BENCHMARK: DSA indexer top-k logits (dsa_topk_logits_bench) ############"
"$B/meta/benchmarks/dsa_topk_logits_bench" 2>&1

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
