#!/bin/bash
# Build + benchmark dsa_topk on an MI300A (gfx942) compute node.
# Run interactively:
#   srun --partition=mi300 -N1 -G1 --time=00:25:00 bash meta/scripts/run_dsa_topk_bench_mi300.sh
set -euo pipefail
: "${SRC:=$HOME/vkernels}"
B="$SRC/build_topk"
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

echo "=== build (test_dsa_topk_correct + dsa_topk_logits_bench) ==="
cmake --build "$B" --target test_dsa_topk_correct dsa_topk_logits_bench -j 64 \
  2>&1 | grep -E "error:|Built target|dsa_topk" | tail -8

echo
echo "############ 1. CORRECTNESS (test_dsa_topk_correct) ############"
"$B/meta/benchmarks/test_dsa_topk_correct"

echo
echo "############ 2. BENCHMARK (dsa_topk_logits_bench) ############"
"$B/meta/benchmarks/dsa_topk_logits_bench"

echo
echo "===== DONE ====="
