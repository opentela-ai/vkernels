# vkernels library
find_package(Threads REQUIRED)

add_library(vkernels STATIC)
add_library(vkernels::vkernels ALIAS vkernels)

target_compile_features(vkernels PUBLIC cxx_std_17)
target_include_directories(vkernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The CPU reference sources are always compiled.
target_sources(vkernels
  PRIVATE
    vkernels/core/stream.cpp
    vkernels/core/tuning.cpp
    vkernels/kernels/elementwise.cpp
    vkernels/kernels/reduce.cpp
    vkernels/kernels/gemm.cpp
    vkernels/kernels/gemm_bf16.cpp
    vkernels/kernels/moe.cpp
    vkernels/kernels/moe_fused.cpp
    vkernels/kernels/moe_aux.cpp
    vkernels/kernels/mla.cpp
    vkernels/kernels/dsa.cpp
    vkernels/kernels/dsa_topk.cpp
    vkernels/kernels/dsa_kpool.cpp
    vkernels/kernels/mhc.cpp
    vkernels/kernels/kda.cpp
    vkernels/kernels/glm_moe.cpp
    vkernels/comm/channel.cpp
    vkernels/comm/rccl.cpp
    vkernels/comm/rccl_c.cpp
    vkernels/comm/allreduce.cpp
    vkernels/comm/overlap.cpp
    vkernels/comm/p2p_gather.cpp
    vkernels/comm/p2p_kv_restore.cpp
    vkernels/comm/p2p_kv_donate.cpp
    vkernels/comm/kv_gather.cpp
    vkernels/comm/kv_scatter.cpp
    vkernels/comm/pipeline_boundary.cpp
    vkernels/comm/pipeline_boundary_c.cpp
    vkernels/comm/fabric_import_c.cpp
    vkernels/comm/fabric_import.cpp
    vkernels/comm/cross_node_kv.cpp
    vkernels/dist/dist_moe.cpp
    # C ABI implementation, one domain per TU (see capi/capi_internal.hpp
    # for the shared error plumbing):
    vkernels/capi/capi.cpp
    vkernels/capi/capi_kernels.cpp
    vkernels/capi/capi_attn.cpp
    vkernels/capi/capi_moe.cpp
)

# The CUDA sources compile only when a toolkit is present.
if(VKERNELS_HAS_CUDA)
  # Host translation units (e.g. device.cpp) include <cuda_runtime.h> under
  # VKERNELS_HAS_CUDA; expose the toolkit headers to the host compiler too,
  # not just to nvcc.
  target_include_directories(vkernels PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_sources(vkernels
    PRIVATE
      vkernels/core/stream.cu
      vkernels/core/device.cpp
      vkernels/kernels/elementwise.cu
      vkernels/kernels/reduce.cu
      vkernels/kernels/gemm.cu
      vkernels/kernels/gemm_bf16.cu
      vkernels/kernels/dsa.cu
      vkernels/comm/allreduce.cu
      vkernels/comm/p2p_gather.cu
      vkernels/comm/p2p_kv_restore.cu
      vkernels/comm/p2p_kv_donate.cu
      vkernels/comm/kv_gather.cu
      vkernels/comm/kv_scatter.cu
      vkernels/comm/pipeline_boundary.cu
      vkernels/comm/fabric_import.cu
      vkernels/comm/cross_node_kv.cu
      vkernels/comm/cross_node_kv_allgather.cu)
  set_target_properties(vkernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # HIP-on-NVIDIA: attention device kernels written as .hip sources, compiled
  # here with nvcc through the tiny HIP compat shim at kernels/cuda_compat/
  # hip/hip_runtime.h (issue #60 A100 port). Only sources proven on NVIDIA
  # hardware are listed — extend as more kernels are ported. The macro stays
  # VKERNELS_HAS_HIP (per-source, so it does not leak into host TUs that
  # select CUDA headers via VKERNELS_HAS_CUDA) and the include dir is the
  # shim root so the sources' <hip/hip_runtime.h> resolves to the shim.
  # Guarded on !HIP: on a machine with both toolkits the HIP branch below
  # already owns these sources.
  if(NOT VKERNELS_HAS_HIP)
    set(VKERNELS_CUDA_HIP_SOURCES vkernels/kernels/dsa_kpool.hip
                                  vkernels/kernels/moe_aux.hip
                                  vkernels/kernels/kda.hip
                                  vkernels/kernels/dsa_topk.hip)
    set_source_files_properties(${VKERNELS_CUDA_HIP_SOURCES} PROPERTIES
      LANGUAGE CUDA COMPILE_DEFINITIONS VKERNELS_HAS_HIP=1)
    # kda.hip: the D=128 gram kernels stage 2 x 64 x 128 x 4 B = 64 KB of
    # STATIC shared (within gfx942's budget), which exceeds NVIDIA's 48 KB
    # static limit — the shim build serves D <= 64 only (VKERNELS_KDA_CUDA_SHIM).
    set_source_files_properties(vkernels/kernels/kda.hip PROPERTIES
      COMPILE_DEFINITIONS "VKERNELS_HAS_HIP=1;VKERNELS_KDA_CUDA_SHIM=1")
    # dsa_topk.hip: the fused cooperative path (VK_DSA_TOPK_FUSED) queries
    # the CUDA opt-in dynamic-shared ceiling
    # (cudaDevAttrMaxSharedMemoryPerBlockOptin -- 101,376 B on GB10).
    # gfx942 has no opt-in (the 64 KB non-optin cap IS the ceiling), so the
    # query is shim-build-only (VKERNELS_DSA_TOPK_CUDA_SHIM); real HIP builds
    # run the fits_lds arithmetic against kDsaTopkLdsCapGfx942.
    set_source_files_properties(vkernels/kernels/dsa_topk.hip PROPERTIES
      COMPILE_DEFINITIONS "VKERNELS_HAS_HIP=1;VKERNELS_DSA_TOPK_CUDA_SHIM=1")
    target_sources(vkernels PRIVATE ${VKERNELS_CUDA_HIP_SOURCES})
    target_include_directories(vkernels PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/vkernels/kernels/cuda_compat)
    # Extern "C" entry points so ctypes consumers (the sglang serving image)
    # can reach the nvcc-compiled device kernels: same names/contracts as
    # hip_capi.hpp (one loader serves both backends). See
    # capi/cuda_capi_kpool.cpp. Goes into the SHARED `vkernels_c` (added
    # below); the per-source VKERNELS_HAS_HIP=1 exposes the
    # vkernels::kernels::hip declarations from dsa_kpool.hpp to this host TU.
    set(VKERNELS_CUDA_KPOOL_CAPI_SRC vkernels/capi/cuda_capi_kpool.cpp)
    set_source_files_properties(${VKERNELS_CUDA_KPOOL_CAPI_SRC} PROPERTIES
      COMPILE_DEFINITIONS VKERNELS_HAS_HIP=1)
  else()
    unset(VKERNELS_CUDA_KPOOL_CAPI_SRC)
  endif()

  # fabric_import.cu is the only translation unit using the CUDA *driver*
  # API (cuMemAddressReserve / cuMemImportFromShareableHandle / cuMemMap /
  # cuMemSetAccess / cuMemUnmap / cuCtx*). Those symbols live in libcuda
  # (the CUDA::cuda_driver imported target from CUDAToolkit), not the
  # runtime that enable_language(CUDA) links automatically. Make the link
  # PUBLIC so every test / bench executable that links libvkernels.a also
  # pulls in libcuda and resolves the driver symbols.
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(vkernels PUBLIC CUDA::cuda_driver)
  endif()

  # C ABI shared library: `extern "C"` entry points over the CUDA primitive,
  # for non-C++ consumers (e.g. a serving runtime). The wrapper is the sole
  # source; it links the static `vkernels` for the C++ implementation and
  # folds every C++ exception into a status code so nothing is thrown across
  # the boundary. Built only with a CUDA toolkit — the entry points take a
  # raw cudaStream_t.
  add_library(vkernels_c SHARED
    vkernels/capi/serving_c.cpp
    vkernels/comm/p2p_gather_c.cu
    vkernels/comm/p2p_kv_restore_c.cu
    vkernels/comm/p2p_kv_donate_c.cu
    vkernels/comm/kv_gather_c.cu
    vkernels/comm/kv_scatter_c.cu
    vkernels/comm/pipeline_boundary_c.cu
    # Pure planning symbols (transport/access-pattern routing) must be part of
    # the shared ABI directly.  Merely linking the static archive does not pull
    # an unreferenced C wrapper object into libvkernels_c.so.
    vkernels/comm/fabric_import_c.cpp
    vkernels/comm/fabric_import_c.cu
    vkernels/comm/cross_node_kv_c.cu
    vkernels/comm/cross_node_kv_allgather_c.cu
    # kpool device C ABI (issue #60 A100 serving path; empty on ROCm builds,
    # where hip_capi.cpp owns the vk_hip_* symbols).
    ${VKERNELS_CUDA_KPOOL_CAPI_SRC})
  add_library(vkernels::vkernels_c ALIAS vkernels_c)
  target_link_libraries(vkernels_c PRIVATE vkernels)
  target_include_directories(vkernels_c PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_features(vkernels_c PRIVATE cxx_std_17)
  vkernels_set_warnings(vkernels_c)
endif()

# The all-gather translation unit always builds with CUDA so the stable C ABI
# can report UNSUPPORTED when NCCL is absent. When NCCL is present, link the
# real communicator/collective implementation into both static and shared
# consumers through vkernels' public dependency.
if(VKERNELS_HAS_NCCL)
  target_include_directories(vkernels PRIVATE ${VKERNELS_NCCL_INCLUDE_DIRS})
  target_link_libraries(vkernels PUBLIC vkernels::nccl)
endif()

# HIP sources compile only when ROCm is present.
if(VKERNELS_HAS_HIP)
  # Host translation units (core/tuning.cpp) include <hip/hip_runtime.h>
  # under VKERNELS_HAS_HIP; expose the ROCm headers to the host compiler
  # too, mirroring the CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES treatment
  # of the CUDA path above. CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES exists
  # once enable_language(HIP) has run (CMake 3.21+) but is left EMPTY on
  # current clang/hipcc toolchains (verified: CMake 3.28 + ROCm 6.3), so
  # derive the ROCm root from the compiler path (…/rocm-X.Y/lib/llvm/bin
  # -> …/rocm-X.Y/include) with the canonical /opt/rocm/include symlink
  # as the fallback hint.
  set(VKERNELS_HIP_RUNTIME_INCLUDE_DIRS "${CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES}")
  # Export to the tests/ directory scope too (test TUs that include
  # <hip/hip_runtime.h> directly -- moe_aux_fused, dsa_topk_fused_gpu --
  # need the toolkit include dirs, which the vkernels target keeps PRIVATE).
  set(VKERNELS_HIP_RUNTIME_INCLUDE_DIRS "${VKERNELS_HIP_RUNTIME_INCLUDE_DIRS}"
      CACHE INTERNAL "HIP runtime include dirs for test targets")
  if(NOT VKERNELS_HIP_RUNTIME_INCLUDE_DIRS)
    get_filename_component(_vk_hip_root "${CMAKE_HIP_COMPILER}" DIRECTORY)
    get_filename_component(_vk_hip_root "${_vk_hip_root}/../../.." ABSOLUTE)
    find_path(VKERNELS_HIP_RUNTIME_INCLUDE_DIR hip/hip_runtime.h
      HINTS "${_vk_hip_root}/include" "/opt/rocm/include")
    if(VKERNELS_HIP_RUNTIME_INCLUDE_DIR)
      list(APPEND VKERNELS_HIP_RUNTIME_INCLUDE_DIRS
        "${VKERNELS_HIP_RUNTIME_INCLUDE_DIR}")
    endif()
    unset(_vk_hip_root)
  endif()
  target_include_directories(vkernels SYSTEM PRIVATE
    ${VKERNELS_HIP_RUNTIME_INCLUDE_DIRS})
  target_sources(vkernels
    PRIVATE
      vkernels/kernels/moe.hip
      vkernels/kernels/moe_fused.hip
      vkernels/kernels/glm_moe.hip
      vkernels/kernels/moe_aux.hip
      vkernels/kernels/gemm_bf16.hip
      vkernels/kernels/mla.hip
      vkernels/kernels/dsa.hip
      vkernels/kernels/dsa_topk.hip
      vkernels/kernels/dsa_kpool.hip
      vkernels/kernels/mhc.hip
      vkernels/kernels/kda.hip)
  set_target_properties(vkernels PROPERTIES HIP_SEPARABLE_COMPILATION ON)

  # C ABI shared library: `extern "C"` entry points over the HIP compute
  # kernels, for non-C++ consumers (vLLM, ctypes, torch.ops.load_library,
  # Rust). The wrapper (capi/hip_capi.cpp) is the sole source; it links the
  # static `vkernels` for the C++ host launchers (the device kernels are
  # compiled there under VKERNELS_HAS_HIP). See capi/hip_capi.hpp.
  #
  # Built as a plain shared lib — no pybind11, no -fvisibility=hidden, no
  # LTO. pybind11 v3.1.0 auto-enables -flto=auto (pybind11Common.cmake:
  # 379-389) and HIP_SEPARABLE_COMPILATION propagates the same flag; both
  # prune the extern "C" entry points from the final .so. -fno-lto on both
  # compile and link keeps them (issue #43).
  add_library(vkernels_hip SHARED vkernels/capi/hip_capi.cpp)
  add_library(vkernels::vkernels_hip ALIAS vkernels_hip)
  target_link_libraries(vkernels_hip PRIVATE vkernels)
  target_include_directories(vkernels_hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(vkernels_hip PRIVATE cxx_std_17)
  target_compile_options(vkernels_hip PRIVATE -fno-lto)
  target_link_options(vkernels_hip PRIVATE -fno-lto)
  vkernels_set_warnings(vkernels_hip)

  # Host (CPU-reference) C ABI shared library for HIP builds. The CUDA-only
  # `vkernels_c` above needs .cu sources, but the attention CPU oracles
  # (vk_mla_fwd, vk_kda_naive_delta_rule_fwd in capi/capi_attn.cpp) are pure
  # host C++ already compiled into the static `vkernels` — expose them as
  # libvkernels.so so serving validators can cross-check device output
  # against the CPU oracle in-process (cookbook vkernels_attn.py,
  # VKERNELS_MLA_VALIDATE / VKERNELS_KDA_VALIDATE, issues #42/#45). Output
  # name `vkernels` -> build/hip/src/c/libvkernels.so, which is where the
  # serving-side resolver globs for it.
  add_library(vkernels_hostc SHARED
    vkernels/capi/capi.cpp
    vkernels/capi/capi_kernels.cpp
    vkernels/capi/capi_attn.cpp
    vkernels/capi/capi_moe.cpp)
  set_target_properties(vkernels_hostc PROPERTIES OUTPUT_NAME vkernels)
  target_link_libraries(vkernels_hostc PRIVATE vkernels)
  target_include_directories(vkernels_hostc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(vkernels_hostc PRIVATE cxx_std_17)
  target_compile_options(vkernels_hostc PRIVATE -fno-lto)
  target_link_options(vkernels_hostc PRIVATE -fno-lto)
  vkernels_set_warnings(vkernels_hostc)
endif()

# RCCL transport (issue #19): compile the HIP/RCCL path only when a usable
# librccl was found (RcclSupport.cmake). The host reference (rccl.cpp) and
# the full test suite build and pass without RCCL, exactly as the CI host
# job does; this block only adds the performance implementation.
if(VKERNELS_HAS_RCCL)
  target_sources(vkernels PRIVATE vkernels/comm/rccl.hip)
  target_include_directories(vkernels PRIVATE ${VKERNELS_RCCL_INCLUDE_DIRS})
  target_link_libraries(vkernels PRIVATE ${VKERNELS_RCCL_LIBRARIES})
  set_target_properties(vkernels PROPERTIES HIP_SEPARABLE_COMPILATION ON)
endif()

# Propagate the CUDA and HIP availability flags to consumers and to the
# library's own translation units.
target_compile_definitions(vkernels PUBLIC
  $<$<BOOL:${VKERNELS_HAS_CUDA}>:VKERNELS_HAS_CUDA=1>
  $<$<BOOL:${VKERNELS_HAS_NCCL}>:VKERNELS_HAS_NCCL=1>
  $<$<BOOL:${VKERNELS_HAS_HIP}>:VKERNELS_HAS_HIP=1>
  $<$<BOOL:${VKERNELS_HAS_RCCL}>:VKERNELS_HAS_RCCL=1>
  $<$<BOOL:${VKERNELS_HAS_OFI}>:VKERNELS_HAS_OFI=1>)

target_link_libraries(vkernels PUBLIC Threads::Threads)

# Allow linking the static library into a shared consumer (e.g. the
# CUDA-gated `vkernels_c` C ABI shared library below).
set_target_properties(vkernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

vkernels_set_warnings(vkernels)
vkernels_apply_sanitizers(vkernels)
if(VKERNELS_ENABLE_COVERAGE)
  vkernels_apply_coverage(vkernels)
endif()
