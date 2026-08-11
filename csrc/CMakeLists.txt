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
    vkernels/comm/channel.cpp
    vkernels/comm/allreduce.cpp
    vkernels/comm/overlap.cpp
)

# The CUDA sources compile only when a toolkit is present.
if(VKERNELS_HAS_CUDA)
  target_sources(vkernels
    PRIVATE
      vkernels/core/stream.cu
      vkernels/core/device.cpp
      vkernels/kernels/elementwise.cu
      vkernels/kernels/reduce.cu
      vkernels/kernels/gemm.cu
      vkernels/comm/allreduce.cu)
  set_target_properties(vkernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Propagate the CUDA availability flag to consumers and to the library's
# own translation units.
target_compile_definitions(vkernels PUBLIC $<$<BOOL:${VKERNELS_HAS_CUDA}>:VKERNELS_HAS_CUDA=1>)

target_link_libraries(vkernels PUBLIC Threads::Threads)

vkernels_set_warnings(vkernels)
vkernels_apply_sanitizers(vkernels)
if(VKERNELS_ENABLE_COVERAGE)
  vkernels_apply_coverage(vkernels)
endif()
