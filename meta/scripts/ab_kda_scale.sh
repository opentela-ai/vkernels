#!/usr/bin/env bash
# A/B at realistic GLM occupancy: sweep H for S=512 D=128, GMEM vs LDS.
# At H=1 the grid is only 4 blocks (D/Db=4) on a 304-CU MI300A -> 98.7%
# idle; at H>=64 the grid (H*4 blocks) fills the GPU. The LDS state-cache
# win should hold or grow at scale (gmem state traffic scales with H).
set -uo pipefail
B=/capstor/scratch/cscs/xyao/vkernels/build_kda
GMEM=$B/meta/benchmarks/kda_bench_GMEM
LDS=$B/meta/benchmarks/kda_bench_LDS
printf "%5s %9s %9s %8s %9s %9s\n" H GMEMus LDSus speedup GMEMtf LDStf
for H in 1 8 16 32 64 128; do
  g=$("$GMEM" "$H" 512 128 2>/dev/null | sed -n '1p')
  l=$("$LDS"  "$H" 512 128 2>/dev/null | sed -n '1p')
  gu=$(echo "$g" | awk '{print $4}')
  lu=$(echo "$l" | awk '{print $4}')
  gt=$(echo "$g" | awk '{print $5}')
  lt=$(echo "$l" | awk '{print $5}')
  if [ -n "$gu" ] && [ -n "$lu" ] && [ "$lu" != "0.0" ]; then
    sp=$(awk -v a="$gu" -v b="$lu" 'BEGIN{printf "%.2f", a/b}')
  else
    sp="-"
  fi
  printf "%5d %9s %9s %8s %9s %9s\n" "$H" "$gu" "$lu" "$sp" "$gt" "$lt"
done
