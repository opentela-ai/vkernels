#!/usr/bin/env bash
# A/B: kda_delta_rule_fwd gmem (committed) vs LDS-cache (optimized).
# Both built + run against the SAME fixed-g bench_kda.hip so the only
# variable is the state-residence kernel. Saves tables to old/new .out.
set -euo pipefail
SRC=/users/xyao/vkernels-issue63
B="$SRC/build_kda"
CLANG=/opt/rocm-6.3.0/lib/llvm/bin/clang++
LIB="$B/src/c/libvkernels.a"
COMMON="-O3 -std=c++17 --offload-arch=gfx942 -DVKERNELS_HAS_HIP=1 -D__HIP_ROCclr__=1 -I$SRC/src/c -L/usr/lib64/gcc/x86_64-suse-linux/7 $LIB /opt/rocm-6.3.0/lib/libamdhip64.so.6.3.60300 -lgcc"
KDA="$SRC/src/c/vkernels/kernels/kda.hip"

build_bench() {  # arg = output binary name
  $CLANG $COMMON "$SRC/meta/benchmarks/bench_kda.hip" -o "$B/meta/benchmarks/$1" 2>/dev/null
}

echo "########## 0. save current LDS kda.hip (optimization) ##########"
cp "$KDA" "$SRC/kda.hip.LDS.bak"
grep -c "srow\[Db \* D\]" "$KDA" | sed 's/^/LDS srow markers: /'

echo "########## 1. build + run NEW (LDS) table ##########"
build_bench kda_bench_LDS
ulimit -c 0
bash "$SRC/meta/benchmarks/bench_kda.sh" "$B/meta/benchmarks/kda_bench_LDS" 2>&1 \
  | tee "$SRC/new_table.out"

echo "########## 2. restore committed gmem kda.hip, rebuild lib + GMEM bench ##########"
( cd "$SRC" && git checkout HEAD -- src/c/vkernels/kernels/kda.hip )
grep -c "Sstate\[(size_t)d \* D + e\] \*= gt\[e\]" "$KDA" | sed 's/^/GMEM gmem-gate markers: /'
touch "$KDA"; make -C "$B" vkernels 2>&1 | grep -E "Built target vkernels|error:" | tail -3
build_bench kda_bench_GMEM
echo "built kda_bench_GMEM vs $(ls -la $LIB | awk '{print $5,$6,$7,$8}')"

echo "########## 3. run OLD (GMEM) table ##########"
bash "$SRC/meta/benchmarks/bench_kda.sh" "$B/meta/benchmarks/kda_bench_GMEM" 2>&1 \
  | tee "$SRC/old_table.out"

echo "########## 4. restore LDS optimization, rebuild lib (final) ##########"
cp "$SRC/kda.hip.LDS.bak" "$KDA"
touch "$KDA"; make -C "$B" vkernels 2>&1 | grep -E "Built target vkernels|error:" | tail -2
grep -c "srow\[Db \* D\]" "$KDA" | sed 's/^/final LDS srow markers: /'
echo "########## A/B DONE ##########"
