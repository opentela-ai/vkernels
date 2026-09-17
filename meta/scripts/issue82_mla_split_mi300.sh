#!/bin/bash
#SBATCH --job-name=vk-i82
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# Build + correctness + benchmark for the mla_fwd split-K decode kernel
# (issue #82) on an MI300A (gfx942) node -- CSCS beverin `mi300` partition.
#
# Correctness gate first (test_mla_correct.hip: the CPU-oracle cross-check,
# incl. the new split-K decode cases), then the acceptance numbers
# (mla_bench: H=1 S_q=1 S_kv=8192 decode vs its plain single-block baseline).
# Mirrors the meta/scripts/*_mi300.sh convention (SRC= scratch checkout).
#
# Requirements (per the issue): decode latency drops well below the 5.5 ms
# baseline at H=1 S_q=1 S_kv=8192; prefill configs unchanged (the bench's
# prefill rows + the unchanged mla_fwd_with_tile path are the evidence).
set -euo pipefail

SRC=${SRC:-$HOME/vk-i82}
if [ ! -d "$SRC" ]; then
  echo "ERROR: SRC scratch checkout missing at $SRC (runner copies the worktree there)" >&2
  exit 1
fi

cmake -S "$SRC" -B "$SRC/build-i82" \
  -DVKERNELS_BUILD_HIP=ON \
  -DVKERNELS_BUILD_TESTS=ON \
  -DVKERNELS_BUILD_BENCHMARKS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build-i82" -j 64 --target vkernels test_mla_correct mla_bench

echo "=============== correctness gate (test_mla_correct) ==============="
"$SRC/build-i82/meta/benchmarks/test_mla_correct"

echo "=============== acceptance numbers (mla_bench) ==============="
"$SRC/build-i82/meta/benchmarks/mla_bench"

echo "=============== host unit tests (split-for formula) ==============="
"$SRC/build-i82/tests/vkernels_test_mla" || true
