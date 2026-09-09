#!/usr/bin/env bash
# Definitive 4-barrier-LDS vs 3-barrier-LDS build + CORRECTNESS, both from the
# FIXED test_kda_correct.hip (rgate unsigned, g alloc [B,H,S,D]). Prior run was
# bogus: beverin HEAD=0ef9966 so 'git checkout HEAD -- kda.hip' restored the
# GMEM kernel, not 4-barrier LDS. Here we use the git-extracted 4-barrier LDS
# (kda.4barrier.hip, from commit 5028461) directly. Builds only.
set -uo pipefail
SRC=/users/xyao/vkernels-issue63; B=$SRC/build_kda
CLANG=/opt/rocm-6.3.0/lib/llvm/bin/clang++
COMMON="-O3 -std=c++17 --offload-arch=gfx942 -DVKERNELS_HAS_HIP=1 -D__HIP_ROCclr__=1 -I$SRC/src/c -L/usr/lib64/gcc/x86_64-suse-linux/7 $B/src/c/libvkernels.a /opt/rocm-6.3.0/lib/libamdhip64.so.6.3.60300 -lgcc"
KDA=$SRC/src/c/vkernels/kernels/kda.hip

echo "### 1. build REAL 4-barrier-LDS: lib + test_kda_4B + kda_bench_4B ###"
cp "$SRC/kda.4barrier.hip" "$KDA"
grep -c "__syncthreads()" "$KDA" | sed 's/^/4barrier syncthreads: /'
touch "$KDA"; make -C "$B" vkernels 2>&1 | grep -E "Built target vkernels|error:" | tail -1
$CLANG $COMMON "$SRC/meta/benchmarks/test_kda_correct.hip" -o "$B/test_kda_4B" 2>/dev/null && echo "built test_kda_4B"
$CLANG $COMMON "$SRC/meta/benchmarks/bench_kda.hip" -o "$B/meta/benchmarks/kda_bench_4B" 2>/dev/null && echo "built kda_bench_4B"

echo "### 2. build 3-barrier-LDS: lib + test_kda_3B + kda_bench_3B ###"
cp "$SRC/kda.3barrier.hip" "$KDA"
grep -c "__syncthreads()" "$KDA" | sed 's/^/3barrier syncthreads: /'
touch "$KDA"; make -C "$B" vkernels 2>&1 | grep -E "Built target vkernels|error:" | tail -1
$CLANG $COMMON "$SRC/meta/benchmarks/test_kda_correct.hip" -o "$B/test_kda_3B" 2>/dev/null && echo "built test_kda_3B"
$CLANG $COMMON "$SRC/meta/benchmarks/bench_kda.hip" -o "$B/meta/benchmarks/kda_bench_3B" 2>/dev/null && echo "built kda_bench_3B"
echo "### BUILD DONE ###"
