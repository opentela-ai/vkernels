#!/usr/bin/env bash
# A/B: 4-barrier/token (committed LDS, binary kda_bench_LDS) vs 3-barrier/token
# (post-OUTPUT barrier removed, binary kda_bench_3B). Same S=512 D=128, sweep H.
# The 4-barrier binary is the one already built from the committed kda.hip; the
# 3-barrier binary is rebuilt here. Only the trailing __syncthreads() after the
# OUTPUT loop differs, so this isolates the barrier cost.
set -uo pipefail
B=/capstor/scratch/cscs/xyao/vkernels/build_kda
OLD=$B/meta/benchmarks/kda_bench_4B     # 4 barriers/token (committed kda.hip, fresh gfx942 build)
NEW=$B/meta/benchmarks/kda_bench_3B     # 3 barriers/token (post-OUTPUT barrier removed)
printf "%5s %9s %9s %8s\n" H B4_us B3_us speedup
for H in 1 16 128; do
  o=$("$OLD" "$H" 512 128 2>/dev/null | sed -n '1p')
  n=$("$NEW" "$H" 512 128 2>/dev/null | sed -n '1p')
  ou=$(echo "$o" | awk '{print $4}')
  nu=$(echo "$n" | awk '{print $4}')
  if [ -n "$ou" ] && [ -n "$nu" ] && [ "$nu" != "0.0" ]; then
    sp=$(awk -v a="$ou" -v b="$nu" 'BEGIN{printf "%.2f", a/b}')
  else
    sp="-"
  fi
  printf "%5d %9s %9s %8s\n" "$H" "$ou" "$nu" "$sp"
done
