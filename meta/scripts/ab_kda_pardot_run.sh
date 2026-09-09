#!/usr/bin/env bash
# Run INSIDE srun --partition=mi300 (gfx942). Correctness + A/B for the
# parallel-dot experiment. Builds (ab_kda_pardot.sh) must already exist:
#   test_kda_4B  / kda_bench_4B   (oracle-identical, Db-thread dots)
#   test_kda_PAR / kda_bench_PAR  (r-way parallel-dot, all 256 threads)
# Prints correctness (11/11 expected for both; parallel-dot max_rel 1-2e-6
# from the reordered summation, still ≪ the 2e-2 threshold) then the A/B
# table at S=512 D=128 sweeping H. See docs/kda-lds-optimization.md.
set -uo pipefail
B=/users/xyao/vkernels-issue63/build_kda
OLD=$B/meta/benchmarks/kda_bench_4B
NEW=$B/meta/benchmarks/kda_bench_PAR
ulimit -c 0

echo "############ CORRECTNESS (both must be 11/11 PASS) ############"
echo "--- 4-barrier (oracle-identical) ---"; "$B/test_kda_4B"  2>&1 | tail -1
echo "--- parallel-dot (reordered sum) ---"; "$B/test_kda_PAR" 2>&1 | tail -1

echo "############ A/B  S=512 D=128  (H sweep) ############"
printf "%5s %9s %9s %8s\n" H B4_us PAR_us speedup
for H in 1 8 16 32 64 128; do
  o=$("$OLD" "$H" 512 128 2>/dev/null | sed -n '1p' | awk '{print $4}')
  n=$("$NEW" "$H" 512 128 2>/dev/null | sed -n '1p' | awk '{print $4}')
  sp="-"
  if [ -n "$o" ] && [ -n "$n" ] && [ "$n" != "0.0" ]; then sp=$(awk -v a="$o" -v b="$n" 'BEGIN{printf "%.2f", a/b}'); fi
  printf "%5d %9s %9s %8s\n" "$H" "$o" "$n" "$sp"
done
