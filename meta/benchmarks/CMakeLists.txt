# Micro-benchmarks for GPU kernel primitives.
#
# Built only when both VKERNELS_BUILD_BENCHMARKS (opt-in) and a GPU toolkit
# (CUDA or HIP) are available.
if(NOT VKERNELS_BUILD_BENCHMARKS)
  return()
endif()

if(NOT VKERNELS_HAS_CUDA AND NOT VKERNELS_HAS_HIP)
  message(STATUS "vkernels: benchmarks skipped (no GPU toolkit)")
  return()
endif()

# --- CUDA benchmarks ---
if(VKERNELS_HAS_CUDA)
  add_executable(p2p_gather_bench p2p_gather_bench.cu)
  target_link_libraries(p2p_gather_bench PRIVATE vkernels::vkernels)
  target_compile_features(p2p_gather_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(p2p_gather_bench)

  add_executable(p2p_kv_restore_bench p2p_kv_restore_bench.cu)
  target_link_libraries(p2p_kv_restore_bench PRIVATE vkernels::vkernels)
  target_compile_features(p2p_kv_restore_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(p2p_kv_restore_bench)
endif()

# --- HIP benchmarks ---
if(VKERNELS_HAS_HIP)
  add_executable(moe_bench bench_moe.hip)
  target_link_libraries(moe_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_bench)
endif()
