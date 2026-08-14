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

  add_executable(test_gemm_bf16_correct test_gemm_bf16_correct.hip)
  target_link_libraries(test_gemm_bf16_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_gemm_bf16_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_gemm_bf16_correct)

  add_executable(gemm_bf16_bench bench_gemm_bf16.hip)
  target_link_libraries(gemm_bf16_bench PRIVATE vkernels::vkernels)
  target_compile_features(gemm_bf16_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(gemm_bf16_bench)

  add_executable(test_mla_correct test_mla_correct.hip)
  target_link_libraries(test_mla_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_mla_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_mla_correct)

  add_executable(test_kda_correct test_kda_correct.hip)
  target_link_libraries(test_kda_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_kda_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_kda_correct)

  add_executable(mla_bench bench_mla.hip)
  target_link_libraries(mla_bench PRIVATE vkernels::vkernels)
  target_compile_features(mla_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(mla_bench)

  add_executable(kda_bench bench_kda.hip)
  target_link_libraries(kda_bench PRIVATE vkernels::vkernels)
  target_compile_features(kda_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(kda_bench)
endif()
