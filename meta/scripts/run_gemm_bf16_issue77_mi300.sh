#!/bin/bash
#SBATCH --job-name=vk-i77
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Build + test + benchmark the gemm_bf16 cross-tile B-reuse + LDS
# double-buffer kernel (issue #77, gfx942/MI300A track) on CSCS beverin.
#
#   1. CORRECTNESS (test_gemm_bf16_correct) -- public hip::gemm_bf16 at
#      serving + warmup M (warmup now routes to gemm_bf16_kernel_reuse),
#      the explicit-tile sweep, and the new cross-tile B-reuse sweep
#      (every compiled (BM,BN,RM) config, beta != 0, unaligned N).
#      Tolerance unchanged: max_rel < 2e-2.
#   2. BENCHMARK (gemm_bf16_bench) -- config-selected shapes incl. warmup
#      M = 8192 (roofline table), then the cross-tile B-reuse autotuner
#      sweep (VK_BENCH_ONLY=reuse) over (16,64,R4), (32,64,R2), (32,64,R4).
#
# Run: sbatch -o run.%j.out meta/scripts/run_gemm_bf16_issue77_mi300.sh
set -euo pipefail
: "${SRC:=${SCRATCH:-$HOME}/vk-issue-77}"
B="$SRC/build"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-interactive} ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release, tests + benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -4

echo "=== build (vkernels + test_gemm_bf16_correct + gemm_bf16_bench) ==="
cmake --build "$B" --target vkernels test_gemm_bf16_correct gemm_bf16_bench -j 64 \
  2>&1 | grep -E "error|warning: |Built target" | tail -12

echo
echo "############ 1. CORRECTNESS (test_gemm_bf16_correct) — reuse kernel vs CPU oracle ############"
"$B/meta/benchmarks/test_gemm_bf16_correct" 2>&1

echo
echo "############ 2. BENCH: config-selected shapes (warmup M=8192 -> reuse kernel) ############"
ulimit -c 0
VK_BENCH_ONLY=shapes "$B/meta/benchmarks/gemm_bf16_bench" 2>&1

echo
echo "############ 3. BENCH: cross-tile B-reuse autotuner sweep ############"
VK_BENCH_ONLY=reuse "$B/meta/benchmarks/gemm_bf16_bench" 2>&1

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
