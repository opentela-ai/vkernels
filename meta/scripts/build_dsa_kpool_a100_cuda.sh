#!/bin/bash
# ============================================================================
# Build + run the dsa_kpool device sources on NVIDIA A100 (sm_80).
#
# The device and executable .hip sources are compiled by nvcc with the HIP
# compat shim (src/c/vkernels/kernels/cuda_compat) providing
# <hip/hip_runtime.h> — through the normal CMake build, not by hand: on a
# CUDA-only node CudaSupport.cmake finds nvcc, and src/c/CMakeLists.txt +
# meta/benchmarks/CMakeLists.txt add the kpool sources with
# LANGUAGE CUDA / VKERNELS_HAS_HIP=1. Requires the prgenv-gnu uenv (nvcc)
# and an A100 GPU node; the whole script runs INSIDE the uenv session.
#
#   srun -A infra02 -N1 --gres=gpu:1 -t 20 \
#       uenv run prgenv-gnu/24.11:v1 --view=default -- \
#       bash meta/scripts/build_dsa_kpool_a100_cuda.sh
# ============================================================================
set -euo pipefail
: "${SRC:=$HOME/vkernels-60}"   # checkout to build; override with SRC=/path
cd "$SRC"
B="$SRC/build-a100"
mkdir -p "$B"

echo "== nvcc =="
nvcc --version | tail -1
echo

echo "== cmake configure (CUDA-only node: ROCm absent, nvcc present) =="
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DVKERNELS_BUILD_HIP=OFF
echo

echo "== build =="
cmake --build "$B" --target test_dsa_kpool_correct dsa_kpool_bench -j 16
echo

echo "== device correctness (device kernels vs host oracle) =="
"$B/meta/benchmarks/test_dsa_kpool_correct"
echo

echo "== benchmark =="
"$B/meta/benchmarks/dsa_kpool_bench"
echo
echo "===== DONE"
