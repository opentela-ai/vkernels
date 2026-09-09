#!/bin/bash
#SBATCH --job-name=vkkda
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
# Build + test + benchmark the Kimi Delta Attention kernels (issue #21, the
# reusable primitive for GLM-5.3-Flash per-key-gated chunked prefill -- issue
# #70) on an MI300A (gfx942) compute node -- CSCS beverin `mi300` partition.
#
# #70 is *efficient GLM-compatible chunked prefill and its device-native
# integration*, NOT a claim no KDA implementation exists. This runner
# establishes the current state of the reusable native HIP primitive
# (hip::kda_delta_rule_fwd + layer_norm_gated + gate_chunk_cumsum, the
# per-key-dim gated delta-rule that hip_dsa_mhc.py exposes as
# kda_delta_rule_fwd / kda_delta_rule_fwd_with_scratch) on a beverin node:
#
#   1. CORRECTNESS (test_kda_correct) -- end-to-end KDA forward vs the CPU
#      oracle (max_rel < 2e-2, same convention as test_gemm_bf16_correct /
#      test_mla_correct). This is the acceptance bar for #21 and the
#      "CPU/Torch oracle" that #70's work/acceptance calls for.
#   2. BENCHMARK (bench_kda.sh kda_bench) -- hip::kda_delta_rule_fwd at the
#      documented (H,S,D) shapes, each as a SEPARATE short-lived process (the
#      MI300A runtime faults a long-lived single-context sweep non-
#      deterministically; see bench_kda.sh). Plus layer_norm_gated +
#      gate_chunk_cumsum. Reports us(min/med), TFLOP/s, GB/s, AI, bound.
#
# GLM-specific scope note (#70): GLM uses per-key gates [B,H,S,D], FP32
# recurrent state/accumulation, causal ordering, and BF16 projection/output
# boundaries. The native forward here is K3-shaped (H<=128, S in {64,512});
# the GLM block/shape contract (issue #64: E4M3FN + per-128x128 FP32 scales;
# H=4096) is NOT exercised by this primitive and remains the #70 integration
# gap. floe's Torch _kda_chunk (1.125 s / 22.98% of 1024-token-prefill GPU
# time, profile 628645) is the cost this primitive would replace; it is not
# present in this checkout (see issue #65).
#
# Run interactively:
#   SRC=/capstor/scratch/cscs/xyao/vkernels srun --partition=mi300 -N1 -G1 --time=00:30:00 \
#     bash meta/scripts/run_kda_bench_mi300.sh
# Or batch:
#   SRC=/capstor/scratch/cscs/xyao/vkernels sbatch -o kda_bench.%j.out \
#     meta/scripts/run_kda_bench_mi300.sh
set -euo pipefail
: "${SRC:=${SCRATCH:-$HOME}/vkernels}"
B="$SRC/build_kda"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-interactive} ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release, tests + benchmarks ON) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -4

echo "=== build (vkernels + test_kda_correct + kda_bench) ==="
cmake --build "$B" --target vkernels test_kda_correct kda_bench -j 64 \
  2>&1 | grep -E "error:|warning: |Built target|kda_bench|test_kda" | tail -10

echo
echo "############ 1. CORRECTNESS (test_kda_correct) — KDA fwd vs CPU oracle ############"
"$B/meta/benchmarks/test_kda_correct" 2>&1

echo
echo "############ 2. BENCHMARK (kda_delta_rule_fwd + layer_norm_gated + gate_chunk_cumsum) ############"
# bench_kda.sh runs each (H,S,D) as its own short-lived process (MI300A
# per-context fault workaround) and prints the table. ulimit -c 0 stops the
# faulting runs dumping multi-hundred-MB GPU cores.
ulimit -c 0
bash "$SRC/meta/benchmarks/bench_kda.sh" "$B/meta/benchmarks/kda_bench" 2>&1

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
