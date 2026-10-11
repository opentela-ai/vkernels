#!/bin/bash
#SBATCH --partition=mi300
#SBATCH --exclusive
#SBATCH -N1
#SBATCH --gres=gpu:1
#SBATCH --account=a-infra02
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vk-156v2-%j.out
# Issue #156 follow-up (lane issue-156/gemv-bw): decode-GEMV v2 gates +
# batched probe on MI300A (gfx942). Build is done on the login node before
# submission; this job is measurement only.
#   1. test_gemm_bf16_correct  (incl. new decode-GEMV v2 section 6b)
#   2. probe156 v2             (batched >=1000-launch event pairs, checksums)
#   3. prefill V5 after-check  (M=1024 row must hold ~617-630 us / ~20.5 TF)
#   4. moe_aux                 (16/16 incl. bit-exact quant must not regress)
# Run: sbatch -o ... meta/scripts/issue156_v2_mi300.sh
set -uo pipefail
cd /capstor/scratch/cscs/xyao/vk-156v2
echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-interactive} ==="
rocm-smi --showmclk --showsclk 2>/dev/null | tail -4 || true

echo '=== gates: test_gemm_bf16_correct ==='
./build/meta/benchmarks/test_gemm_bf16_correct 2>&1 | tee /capstor/scratch/cscs/xyao/vk-156v2-gates.txt | tail -45

echo '=== probe (v2 phase) ==='
./probe156 v2 2>&1 | tee /capstor/scratch/cscs/xyao/vk-156v2-probe.txt | grep -E "shape|med|GB/s|max_rel|PASS|FAIL|dequant" | tail -120

echo '=== prefill V5 after-check ==='
cd build && ./meta/benchmarks/moe_fused_prefill_bench 2>&1 | tail -8

echo '=== moe_aux ==='
./meta/benchmarks/moe_aux_bench 2>&1 | tail -25
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
