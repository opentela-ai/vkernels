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
vkernels_add_test(NAME tuning      SOURCES core/test_tuning.cpp)
vkernels_add_test(NAME allocator   SOURCES core/test_allocator.cpp)
vkernels_add_test(NAME elementwise SOURCES kernels/elementwise/test_elementwise.cpp)
vkernels_add_test(NAME reduce      SOURCES kernels/reduce/test_reduce.cpp)
vkernels_add_test(NAME gemm        SOURCES kernels/gemm/test_gemm.cpp)
vkernels_add_test(NAME gemm_bf16   SOURCES kernels/gemm/test_gemm_bf16.cpp)
vkernels_add_test(NAME mla         SOURCES kernels/attn/test_mla.cpp)
vkernels_add_test(NAME dsa         SOURCES kernels/attn/test_dsa.cpp)
vkernels_add_test(NAME dsa_topk    SOURCES kernels/attn/test_dsa_topk.cpp)
vkernels_add_test(NAME dsa_kpool      SOURCES kernels/attn/test_dsa_kpool.cpp)
vkernels_add_test(NAME dsa_kpool_fp8  SOURCES kernels/attn/test_dsa_kpool_fp8.cpp)
vkernels_add_test(NAME mhc         SOURCES kernels/attn/test_mhc.cpp)
vkernels_add_test(NAME kda         SOURCES kernels/attn/test_kda.cpp)
vkernels_add_test(NAME kda_k3_chunked SOURCES kernels/attn/test_kda_k3_chunked.cpp)
vkernels_add_test(NAME kda_cp SOURCES kernels/attn/test_kda_cp.cpp)
vkernels_add_test(NAME moe         SOURCES kernels/moe/test_moe.cpp)
vkernels_add_test(NAME moe_fused   SOURCES kernels/moe/test_moe_fused.cpp)
vkernels_add_test(NAME moe_align_oracle SOURCES kernels/moe/test_moe_align_oracle.cpp)
vkernels_add_test(NAME moe_aux     SOURCES kernels/moe/test_moe_aux.cpp)
vkernels_add_test(NAME moe_aux_fused SOURCES kernels/moe/test_moe_aux_fused.cpp)
# The NaN-parity device section of moe_aux_fused needs the shim's hip/
# runtime mapping on CUDA builds (real HIP builds define VKERNELS_HAS_HIP
# globally). Host-only builds compile the section out and skip these flags.
if(VKERNELS_HAS_CUDA AND NOT VKERNELS_HAS_HIP)
  set_source_files_properties(kernels/moe/test_moe_aux_fused.cpp
    PROPERTIES COMPILE_DEFINITIONS "VKERNELS_HAS_HIP=1")
  set_property(TARGET vkernels_test_moe_aux_fused APPEND PROPERTY
    INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/src/c/vkernels/kernels/cuda_compat)
elseif(VKERNELS_HAS_HIP)
  # Real HIP: the device section includes <hip/hip_runtime.h> directly; the
  # toolkit include dirs are PRIVATE to the vkernels target, so give the
  # test target the exported VKERNELS_HIP_RUNTIME_INCLUDE_DIRS. Host-compiled
  # TUs also need the platform macro hipcc would inject (the shim branch
  # must NOT get it: the cuda_compat shim #errors when it is set).
  set_property(TARGET vkernels_test_moe_aux_fused APPEND PROPERTY
    INCLUDE_DIRECTORIES ${VKERNELS_HIP_RUNTIME_INCLUDE_DIRS})
  set_property(TARGET vkernels_test_moe_aux_fused APPEND PROPERTY
    COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__=1)
endif()
vkernels_add_test(NAME glm_moe     SOURCES kernels/moe/test_glm_moe.cpp)
vkernels_add_test(NAME dist_moe    SOURCES kernels/moe/test_dist_moe.cpp)
vkernels_add_test(NAME topology    SOURCES comm/test_topology.cpp)
vkernels_add_test(NAME channel     SOURCES comm/test_channel.cpp)
vkernels_add_test(NAME allreduce   SOURCES comm/test_allreduce.cpp)
vkernels_add_test(NAME rccl        SOURCES comm/test_rccl.cpp)
vkernels_add_test(NAME rccl_c      SOURCES comm/test_rccl_c.cpp)
vkernels_add_test(NAME overlap     SOURCES comm/test_overlap.cpp)
vkernels_add_test(NAME p2p_gather SOURCES comm/test_p2p_gather.cpp)
vkernels_add_test(NAME p2p_kv_restore SOURCES comm/test_p2p_kv_restore.cpp)
vkernels_add_test(NAME p2p_kv_donate SOURCES comm/test_p2p_kv_donate.cpp)
vkernels_add_test(NAME kv_gather SOURCES comm/test_kv_gather.cpp)
vkernels_add_test(NAME kv_scatter SOURCES comm/test_kv_scatter.cpp)
vkernels_add_test(NAME pipeline_boundary SOURCES comm/test_pipeline_boundary.cpp)
vkernels_add_test(NAME cross_node_kv SOURCES comm/test_cross_node_kv.cpp)
vkernels_add_test(NAME cross_node_kv_striped SOURCES comm/test_cross_node_kv_striped.cpp)
vkernels_add_test(NAME pipeline_boundary_c SOURCES comm/test_pipeline_boundary_c.cpp)
vkernels_add_test(NAME fabric_import_c SOURCES comm/test_fabric_import_c.cpp)
vkernels_add_test(NAME capi         SOURCES capi/test_capi.cpp)
vkernels_add_test(
  NAME serving_c
  SOURCES capi/test_serving_c.cpp
          ${CMAKE_SOURCE_DIR}/src/c/vkernels/capi/serving_c.cpp
  INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
  NO_VKERNELS_LIB)

# KDA chunked-WY fused-launch GPU test: the device path lives in kda.hip,
# compiled under VKERNELS_HAS_HIP (real HIP on gfx942, or the HIP-on-NVIDIA
# shim in the CUDA build). On CUDA builds the TU is compiled with
# VKERNELS_HAS_HIP=1 (like cuda_capi_kpool.cpp) so it links the shim-compiled
# hip:: symbols from libvkernels; on host-only builds it self-reports a skip.
if(VKERNELS_HAS_CUDA AND NOT VKERNELS_HAS_HIP)
  set_source_files_properties(kernels/attn/test_kda_chunked_fused_gpu.cpp
    PROPERTIES COMPILE_DEFINITIONS "VKERNELS_HAS_HIP=1;VKERNELS_KDA_CUDA_SHIM=1")
endif()
# NOTE (fusion lane dsa-kda-launches): the set_source_files_properties above
# + this test line are a likely merge-conflict point with sibling lanes
# adding tests here (additive; last-writer-wins is fine).
vkernels_add_test(NAME kda_chunked_fused_gpu
                  SOURCES kernels/attn/test_kda_chunked_fused_gpu.cpp)

# DSA top-k fused-chain GPU test (fusion lane dsa-topk-tailfold): the device
# path lives in dsa_topk.hip, shim-compiled under VKERNELS_HAS_HIP in the
# CUDA build. Same pattern as moe_aux_fused (register the test first, then
# the shim properties; the test TU dequants through the SHARED
# device_numeric.cuh, so the CPU oracle sees bit-exact fp32 inputs).
# Host-only builds self-report a skip.
vkernels_add_test(NAME dsa_topk_fused_gpu
                  SOURCES kernels/attn/test_dsa_topk_fused_gpu.cpp)
if(VKERNELS_HAS_CUDA AND NOT VKERNELS_HAS_HIP)
  set_source_files_properties(kernels/attn/test_dsa_topk_fused_gpu.cpp
    PROPERTIES COMPILE_DEFINITIONS "VKERNELS_HAS_HIP=1;VKERNELS_DSA_TOPK_CUDA_SHIM=1")
  set_property(TARGET vkernels_test_dsa_topk_fused_gpu APPEND PROPERTY
    INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/src/c/vkernels/kernels/cuda_compat)
elseif(VKERNELS_HAS_HIP)
  # Real HIP: direct <hip/hip_runtime.h> include, see moe_aux_fused above.
  set_property(TARGET vkernels_test_dsa_topk_fused_gpu APPEND PROPERTY
    INCLUDE_DIRECTORIES ${VKERNELS_HIP_RUNTIME_INCLUDE_DIRS})
  set_property(TARGET vkernels_test_dsa_topk_fused_gpu APPEND PROPERTY
    COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__=1)
endif()

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
  vkernels_add_test(NAME p2p_kv_donate_c SOURCES comm/test_p2p_kv_donate_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME kv_gather_c SOURCES comm/test_kv_gather_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME kv_scatter_c SOURCES comm/test_kv_scatter_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME pipeline_boundary_cuda_c
                    SOURCES comm/test_pipeline_boundary_cuda_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME fabric_import_cuda_c SOURCES comm/test_fabric_import_cuda_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME cross_node_kv_c SOURCES comm/test_cross_node_kv_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
  vkernels_add_test(NAME cross_node_kv_allgather_c
                    SOURCES comm/test_cross_node_kv_allgather_c.cu
                    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
                    LIBS vkernels::vkernels_c NO_VKERNELS_LIB)
endif()
