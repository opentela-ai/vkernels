#!/bin/bash
# cross_node_nccl_bench_jsc.sbatch -- issue #152 on-fabric measurement of the
# striped multi-HCA KV striping variant vs the nccl-xfer / nccl-pipe baseline.
#
# Target testbed: JSC 2x GH200 120 GB over InfiniBand HDR-200 -- the exact
# setup of the issue #49 measurement recorded in docs/comm-cross-node-kv.md
# section 8 (4x mlx5 HCAs, GPUDirect RDMA, NCCL 2.29.7+cuda13.0, CUDA driver
# 13020, GPFS /e/scratch visible on both nodes).
#
# Submit (from the repo root on the JSC login node, after the build step in
# bench_cross_node_nccl_jsc.sh has produced build/cuda/meta/benchmarks/
# cross_node_nccl_bench):
#
#     sbatch meta/scripts/cross_node_nccl_bench_jsc.sbatch
#
# What runs: one full size sweep (256 -> 65536 toks) of the bench, which
# prints nccl-xfer / nccl-pipe / nccl-allgthr / nccl-allred / nccl-stripe /
# d2d-local rows under identical conditions, followed by a --quick
# NCCL_DEBUG=INFO pass to verify each stripe communicator actually landed
# on a distinct mlx5_<l> HCA (the NCCL_IB_HCA bench-only pinning trick).
#
# Success criterion (issue #152): nccl-stripe usefGB >= 3x nccl-xfer usefGB
# at 16384+ toks (~24 -> ~70-90 GB/s), with all rows reporting check=ok.
# Do NOT present results from a non-HDR fabric as the JSC result.
#SBATCH --job-name=vk152-stripe
#SBATCH --partition=boot            # FIXME: verify the GH200 partition name on the JSC system at submission time
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=vk152-stripe-%j.out

set -euo pipefail

BENCH=${BENCH:-build/cuda/meta/benchmarks/cross_node_nccl_bench}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"

echo "== hostmap =="; srun --mpi=pmix bash -lc 'hostname; nvidia-smi -L' || true

echo "== full sweep: striped vs baselines (iters=101 after 5 warmups) =="
srun --mpi=pmix -N2 -n2 --ntasks-per-node=1 --gres=gpu:1 \
    "$BENCH" --iters 101 --stripe 4

echo "== HCA-pinning check (NCCL_DEBUG=INFO, quick sweep) =="
srun --mpi=pmix -N2 -n2 --ntasks-per-node=1 --gres=gpu:1 \
    env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
    "$BENCH" --quick --stripe 4 2>&1 | grep -E "NET/IB|Using|stripe|mlx5|NCCL INFO" | head -80

echo "== done =="
