#!/bin/bash
# bench_cross_node_nccl_jsc.sh -- build the issue #152 striped-striping bench
# on the JSC 2x GH200 HDR testbed, then submit the measurement job.
#
# Context: the striped multi-lane plan implementation is already merged
# (cad17f6); all that remains for #152 is the on-fabric number. This script
# encodes the exact build + submission steps so anyone with JSC access can
# reproduce. Run it from a clone of the repo on the JSC login node.
#
# Environment from the docs/comm-cross-node-kv.md section 8 record:
#   GH200 120 GB (SM 9.0), CUDA driver 13020, NCCL 2.29.7+cuda13.0,
#   EasyBuild modules (EBROOTNCCL), shared GPFS /e/scratch (the NCCL
#   non-MPI bootstrap file exchange uses it).
#
# Usage:
#     bash meta/scripts/bench_cross_node_nccl_jsc.sh          # build + submit
#     bash meta/scripts/bench_cross_node_nccl_jsc.sh --build  # build only
set -euo pipefail

cd "$(dirname "$0")/../.."

# 1. Toolchain. Module names below follow the versions recorded in the
#    section 8 measurement; verify against `module spider NCCL` on the
#    target system before running (EasyBuild at JSC exposes the install
#    root via EBROOTNCCL, which meta/cmake/NcclSupport.cmake picks up).
module purge
module load CUDA/13.0
module load NCCL/2.29.7-CUDA-13.0 || module load NCCL  # FIXME: exact name via module spider
module list

echo "NCCL root: ${EBROOTNCCL:-<unset>}"

# 2. Configure + build only the bench target (NCCL is opt-in; the bench is
#    built only when VKERNELS_HAS_NCCL is found).
cmake --preset cuda
cmake --build --preset cuda --target cross_node_nccl_bench -j

BENCH=build/cuda/meta/benchmarks/cross_node_nccl_bench
test -x "$BENCH" || { echo "ERROR: $BENCH not built (NCCL not found?)" >&2; exit 1; }
echo "built: $BENCH"

# 3. Submit the 2-node job unless --build was requested.
if [[ "${1:-}" != "--build" ]]; then
    sbatch meta/scripts/cross_node_nccl_bench_jsc.sbatch
fi
