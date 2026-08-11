# Micro-benchmarks for the P2P run-list gather.
#
# CUDA-only: the primitive issues peer reads from inside one kernel, so there
# is nothing to time on the host. Built only when both VKERNELS_BUILD_BENCHMARKS
# (opt-in) and a CUDA toolkit (VKERNELS_HAS_CUDA) are available.
if(NOT VKERNELS_BUILD_BENCHMARKS)
  return()
endif()

if(NOT VKERNELS_HAS_CUDA)
  message(STATUS "vkernels: benchmarks skipped (no CUDA toolkit)")
  return()
endif()

# The benchmark links the same static library (which carries the
# VKERNELS_HAS_CUDA macro publicly) and times the cuda:: entry points against
# per-run cudaMemcpyAsync / cudaMemcpy2DAsync with raw cudaEvent.
add_executable(p2p_gather_bench p2p_gather_bench.cu)
target_link_libraries(p2p_gather_bench PRIVATE vkernels::vkernels)
target_compile_features(p2p_gather_bench PRIVATE cxx_std_17)
vkernels_set_warnings(p2p_gather_bench)
