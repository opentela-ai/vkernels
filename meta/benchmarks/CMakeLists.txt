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

  add_executable(moe_fused_bench bench_moe_fused.hip)
  target_link_libraries(moe_fused_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_fused_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_fused_bench)

  add_executable(moe_fused_prefill_bench bench_moe_fused_prefill.hip)
  target_link_libraries(moe_fused_prefill_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_fused_prefill_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_fused_prefill_bench)

  add_executable(moe_aux_bench bench_moe_aux.hip)
  target_link_libraries(moe_aux_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_aux_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_aux_bench)

  add_executable(test_moe_fused_correct test_moe_fused_correct.hip)
  target_link_libraries(test_moe_fused_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_fused_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_fused_correct)

  add_executable(test_moe_aux_correct test_moe_aux_correct.hip)
  target_link_libraries(test_moe_aux_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_aux_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_aux_correct)

  add_executable(test_moe_fused_prefill_correct test_moe_fused_prefill_correct.hip)
  target_link_libraries(test_moe_fused_prefill_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_fused_prefill_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_fused_prefill_correct)

  add_executable(test_moe_fused_dist_correct test_moe_fused_dist_correct.hip)
  target_link_libraries(test_moe_fused_dist_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_fused_dist_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_fused_dist_correct)
endif()
