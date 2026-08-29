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
    vkernels/kernels/elementwise.cpp
    vkernels/kernels/reduce.cpp
    vkernels/kernels/gemm.cpp
    vkernels/kernels/gemm_bf16.cpp
    vkernels/kernels/moe.cpp
    vkernels/kernels/moe_fused.cpp
    vkernels/kernels/moe_aux.cpp
    vkernels/kernels/mla.cpp
    vkernels/kernels/kda.cpp
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
    vkernels/capi/capi.cpp
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
    vkernels/comm/cross_node_kv_allgather_c.cu)
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
  target_sources(vkernels
    PRIVATE
      vkernels/kernels/moe.hip
      vkernels/kernels/moe_fused.hip
      vkernels/kernels/moe_aux.hip
      vkernels/kernels/gemm_bf16.hip
      vkernels/kernels/mla.hip
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
