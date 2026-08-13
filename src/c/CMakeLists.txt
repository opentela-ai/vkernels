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
    vkernels/kernels/moe.cpp
    vkernels/kernels/moe_fused.cpp
    vkernels/comm/channel.cpp
    vkernels/comm/allreduce.cpp
    vkernels/comm/overlap.cpp
    vkernels/comm/p2p_gather.cpp
    vkernels/comm/p2p_kv_restore.cpp
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
      vkernels/comm/p2p_kv_restore.cu)
  set_target_properties(vkernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # C ABI shared library: `extern "C"` entry points over the CUDA primitive,
  # for non-C++ consumers (e.g. a serving runtime). The wrapper is the sole
  # source; it links the static `vkernels` for the C++ implementation and
  # folds every C++ exception into a status code so nothing is thrown across
  # the boundary. Built only with a CUDA toolkit — the entry points take a
  # raw cudaStream_t.
  add_library(vkernels_c SHARED vkernels/comm/p2p_gather_c.cu vkernels/comm/p2p_kv_restore_c.cu)
  add_library(vkernels::vkernels_c ALIAS vkernels_c)
  target_link_libraries(vkernels_c PRIVATE vkernels)
  target_include_directories(vkernels_c PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_features(vkernels_c PRIVATE cxx_std_17)
  vkernels_set_warnings(vkernels_c)
endif()

# HIP sources compile only when ROCm is present.
if(VKERNELS_HAS_HIP)
  target_sources(vkernels
    PRIVATE
      vkernels/kernels/moe.hip
      vkernels/kernels/moe_fused.hip)
  set_target_properties(vkernels PROPERTIES HIP_SEPARABLE_COMPILATION ON)
endif()

# Propagate the CUDA and HIP availability flags to consumers and to the
# library's own translation units.
target_compile_definitions(vkernels PUBLIC
  $<$<BOOL:${VKERNELS_HAS_CUDA}>:VKERNELS_HAS_CUDA=1>
  $<$<BOOL:${VKERNELS_HAS_HIP}>:VKERNELS_HAS_HIP=1>)

target_link_libraries(vkernels PUBLIC Threads::Threads)

# Allow linking the static library into a shared consumer (e.g. the
# CUDA-gated `vkernels_c` C ABI shared library below).
set_target_properties(vkernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

vkernels_set_warnings(vkernels)
vkernels_apply_sanitizers(vkernels)
if(VKERNELS_ENABLE_COVERAGE)
  vkernels_apply_coverage(vkernels)
endif()
