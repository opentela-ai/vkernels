#!/bin/bash
#SBATCH --job-name=vk-i151
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Build + correctness + benchmark for the mla_fwd fused-head tiled prefill
# kernel and the BQ>1 chunked-prefill split path (issue #151) on an MI300A
# (gfx942) node -- CSCS beverin `mi300` partition.
#
# Correctness gate first (test_mla_correct.hip: the CPU-oracle cross-check,
# now incl. fused-head prefill shapes and BQ=4 prefill split cases), then the
# host unit tests (split-for + head-group rules), then the acceptance
# numbers (mla_bench: the 44%-of-HBM prefill shapes and the 107 GB/s
# chunked-prefill shape).
# Mirrors the meta/scripts/*_mi300.sh convention (SRC= scratch checkout).
set -euo pipefail

SRC=${SRC:-/capstor/scratch/cscs/xyao/vk-issue-151}
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "ERROR: SRC scratch checkout missing at $SRC (rsync the worktree there)" >&2
  exit 1
fi

cmake -S "$SRC" -B "$SRC/build-i151" \
  -DVKERNELS_BUILD_HIP=ON \
  -DVKERNELS_BUILD_TESTS=ON \
  -DVKERNELS_BUILD_BENCHMARKS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build-i151" -j 64 --target vkernels test_mla_correct mla_bench vkernels_test_mla

echo "=============== correctness gate (test_mla_correct) ==============="
"$SRC/build-i151/meta/benchmarks/test_mla_correct"

echo "=============== host unit tests (split-for / head-groups) ==============="
"$SRC/build-i151/tests/vkernels_test_mla" || true

echo "=============== acceptance numbers (mla_bench) ==============="
"$SRC/build-i151/meta/benchmarks/mla_bench"
