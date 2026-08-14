# tests/
add_library(vkernels_test_main OBJECT test_main.cpp)
target_include_directories(vkernels_test_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(VKERNELS_ENABLE_COVERAGE)
  vkernels_apply_coverage(vkernels_test_main)
endif()

vkernels_add_test(NAME span        SOURCES core/test_span.cpp)
vkernels_add_test(NAME error       SOURCES core/test_error.cpp)
vkernels_add_test(NAME logging     SOURCES core/test_logging.cpp)
vkernels_add_test(NAME stream      SOURCES core/test_stream.cpp)
vkernels_add_test(NAME device      SOURCES core/test_device.cpp)
vkernels_add_test(NAME allocator   SOURCES core/test_allocator.cpp)
vkernels_add_test(NAME elementwise SOURCES kernels/elementwise/test_elementwise.cpp)
vkernels_add_test(NAME reduce      SOURCES kernels/reduce/test_reduce.cpp)
vkernels_add_test(NAME gemm        SOURCES kernels/gemm/test_gemm.cpp)
vkernels_add_test(NAME gemm_bf16   SOURCES kernels/gemm/test_gemm_bf16.cpp)
vkernels_add_test(NAME mla         SOURCES kernels/attn/test_mla.cpp)
vkernels_add_test(NAME kda         SOURCES kernels/attn/test_kda.cpp)
vkernels_add_test(NAME moe         SOURCES kernels/moe/test_moe.cpp)
vkernels_add_test(NAME moe_fused   SOURCES kernels/moe/test_moe_fused.cpp)
vkernels_add_test(NAME moe_aux     SOURCES kernels/moe/test_moe_aux.cpp)
vkernels_add_test(NAME dist_moe    SOURCES kernels/moe/test_dist_moe.cpp)
vkernels_add_test(NAME topology    SOURCES comm/test_topology.cpp)
vkernels_add_test(NAME channel     SOURCES comm/test_channel.cpp)
vkernels_add_test(NAME allreduce   SOURCES comm/test_allreduce.cpp)
vkernels_add_test(NAME rccl        SOURCES comm/test_rccl.cpp)
vkernels_add_test(NAME rccl_c      SOURCES comm/test_rccl_c.cpp)
vkernels_add_test(NAME overlap     SOURCES comm/test_overlap.cpp)
vkernels_add_test(NAME p2p_gather SOURCES comm/test_p2p_gather.cpp)
vkernels_add_test(NAME p2p_kv_restore SOURCES comm/test_p2p_kv_restore.cpp)
vkernels_add_test(NAME pipeline_boundary SOURCES comm/test_pipeline_boundary.cpp)
vkernels_add_test(NAME pipeline_boundary_c SOURCES comm/test_pipeline_boundary_c.cpp)
vkernels_add_test(NAME capi         SOURCES capi/test_capi.cpp)

# The `extern "C"` ABI is CUDA-only (the entry points take a raw
# cudaStream_t); its runtime test is a .cu compiled when a toolkit is
# present. It links the shared `vkernels_c` boundary plus the CUDA runtime
# (pulled in automatically for .cu translation units).
if(VKERNELS_HAS_CUDA)
  vkernels_add_test(NAME p2p_gather_c SOURCES comm/test_p2p_gather_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME p2p_kv_restore_c SOURCES comm/test_p2p_kv_restore_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME pipeline_boundary_cuda_c
                    SOURCES comm/test_pipeline_boundary_cuda_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
endif()
