#!/usr/bin/env bash
# Self-contained A/B build for the parallel-dot experiment. Builds BOTH sides
# from committed kernel variants in meta/benchmarks/ (no beverin-local temps,
# no git-object dependency), into test_kda_4B / kda_bench_4B (oracle-identical,
# Db threads do the length-D dots) and test_kda_PAR / kda_bench_PAR (r=kTh/Db
# threads/row, __shfl_xor reduce, all 256 threads work, per-dot order changed).
# Restore src kda.hip to the committed 4-barrier at the end. Run on the login
# node (compile only); A/B is run separately via srun --partition=mi300.
#
# RESULT (beverin MI300A, gfx942): parallel-dot is 0.88x (12% SLOWER) at
# H<=8 and 1.07x (7% faster) at H>=16, stable across runs. The low-H
# regression is decisive: if per-block PREDICT/OUTPUT compute were the
# binding resource, parallelising the length-D dots (128->16 dependent
# adds) would speed up EVERY H, especially low H (no occupancy fallback).
# It did not — the per-token critical path is bound by the serial
# recurrence + 4 barrier latencies, not thread starvation. The H>=16 win
# is occupancy-mediated (more blocks fill the GPU), not a per-block
# latency breakthrough. Do NOT commit the parallel-dot kernel as a
# replacement (regresses the latency-critical low-H case); the committed
# kda.hip stays. See docs/kda-lds-optimization.md.
set -uo pipefail
SRC=/users/xyao/vkernels-issue63; B=$SRC/build_kda
CLANG=/opt/rocm-6.3.0/lib/llvm/bin/clang++
COMMON="-O3 -std=c++17 --offload-arch=gfx942 -DVKERNELS_HAS_HIP=1 -D__HIP_ROCclr__=1 -I$SRC/src/c -L/usr/lib64/gcc/x86_64-suse-linux/7 $B/src/c/libvkernels.a /opt/rocm-6.3.0/lib/libamdhip64.so.6.3.60300 -lgcc"
KDA=$SRC/src/c/vkernels/kernels/kda.hip
V4=$SRC/meta/benchmarks/kda_4barrier.hip   # committed oracle-identical (Db-thread dots)
VP=$SRC/meta/benchmarks/kda_pardot.hip     # committed r-way parallel-dot variant

build_one () {   # $1 = variant src, $2 = test binary, $3 = bench binary
  cp "$1" "$KDA"; touch "$KDA"
  make -C "$B" vkernels 2>&1 | grep -E "Built target vkernels|error:" | tail -1
  $CLANG $COMMON "$SRC/meta/benchmarks/test_kda_correct.hip" -o "$2" 2>/dev/null && echo "built $(basename "$2")"
  $CLANG $COMMON "$SRC/meta/benchmarks/bench_kda.hip"        -o "$3" 2>/dev/null && echo "built $(basename "$3")"
}

echo "### 1. 4-barrier (oracle-identical, Db-thread dots) ###"
build_one "$V4" "$B/test_kda_4B" "$B/meta/benchmarks/kda_bench_4B"
echo "### 2. parallel-dot (r-way, __shfl_xor reduce) ###"
build_one "$VP" "$B/test_kda_PAR" "$B/meta/benchmarks/kda_bench_PAR"
cp "$V4" "$KDA"   # restore committed 4-barrier kda.hip
echo "### BUILD DONE ###"
