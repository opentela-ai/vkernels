#!/bin/bash
#SBATCH --job-name=kdachunk
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
# Build + correctness + A/B for the chunked WY K3 forward (#70) on a
# beverin MI300A (gfx942) node. The chunked kernels ship IN src
# kda.hip (kda_delta_rule_fwd_chunked_with_scratch), so unlike
# ab_kda_pardot.sh there is NO kda.hip variant swapping: one build.
#
#   1. BUILD  vkernels + test_kda_chunked + kda_chunked_bench (cmake).
#   2. CORRECTNESS  test_kda_chunked: chunked vs kda_naive_delta_rule_fwd_cpu
#      (the K3 oracle) at 10 shapes incl. H=128 S=512 D=128, plus the
#      nonzero-initial-state contract vs the cooperative kernel. Expect
#      PASS (max_rel ~1e-5..1e-3 << 2e-2).
#   3. A/B  kda_chunked_bench: committed cooperative per-token kernel vs
#      chunked WY, same inputs/events, H sweep at S=512 D=128 + long-S
#      (1024, 2048). One short-lived process per shape (MI300A per-context
#      fault workaround).
#
# The math was CPU-verified first (tests/kernels/attn/test_kda_k3_chunked.cpp,
# k3_wy_chunked_fwd vs oracle <=1e-6 incl. S=512 D=128 cs=64); this runner
# validates the HIP port and measures it.
#
# Run interactively:
#   SRC=/capstor/scratch/cscs/xyao/vkernels srun --partition=mi300 -N1 -G1 --time=00:40:00 \
#     bash meta/scripts/ab_kda_chunked_mi300.sh
# Or batch:
#   SRC=/capstor/scratch/cscs/xyao/vkernels sbatch -o kdachunk.%j.out \
#     meta/scripts/ab_kda_chunked_mi300.sh
set -euo pipefail
: "${SRC:=${SCRATCH:-$HOME}/vkernels}"
B="$SRC/build_kda"
mkdir -p "$B"

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-interactive} ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
echo "=== lock perf level high ==="
rocm-smi --setperflevel high 2>/dev/null || echo "(setperflevel not permitted; continuing)"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4 || true

echo "=== configure (HIP, gfx942, Release) ==="
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON -DVKERNELS_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -4

echo "=== build (vkernels + test_kda_chunked + kda_chunked_bench) ==="
cmake --build "$B" --target vkernels test_kda_chunked kda_chunked_bench -j 64 \
  2>&1 | grep -E "error:|Built target" | tail -8

echo
echo "############ 1. CORRECTNESS (test_kda_chunked) — chunked WY vs K3 oracle ############"
"$B/meta/benchmarks/test_kda_chunked" 2>&1

echo
echo "############ 2. A/B (cooperative per-token vs chunked WY, #70) ############"
# One short-lived process per shape (MI300A per-context fault workaround;
# see bench_kda.sh). ulimit -c 0 stops faulting runs dumping GPU cores.
ulimit -c 0
echo "--- phase decomposition (which launch dominates) ---"
for cfg in "1 512 128" "128 512 128" "32 2048 128"; do
  set -- $cfg
  "$B/meta/benchmarks/kda_chunked_bench" phases "$1" "$2" "$3" 2>/dev/null \
    | grep '^phases' || echo "phases H=$1 S=$2 D=$3 (failed)"
done
echo "--- full A/B table ---"
for H in 1 8 16 32 64 128; do
  "$B/meta/benchmarks/kda_chunked_bench" "$H" 512 128 2>/dev/null \
    | grep '^H=' || echo "H=$H S=512 D=128 coop=? chunked=? (run failed)"
done
for S in 1024 2048; do
  "$B/meta/benchmarks/kda_chunked_bench" 32 "$S" 128 2>/dev/null \
    | grep '^H=' || echo "H=32 S=$S D=128 coop=? chunked=? (run failed)"
done

echo
echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
