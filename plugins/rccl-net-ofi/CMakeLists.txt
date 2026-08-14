# plugins/rccl-net-ofi — HIP-aware OFI/CXI net plugin for RCCL (issue #19)
#
# Builds `librccl-net-ofi.so` against libfabric + the HIP runtime, gated by
# VKERNELS_HAS_OFI (libfabric found by RcclSupport.cmake). Never affects the
# host-only library or its tests: the top-level CMakeLists adds this
# subdirectory only when HIP + libfabric are both present.
#
# The plugin is a single C translation unit (the standard aws_ofi_nccl layout
# is C). The top-level project is CXX-only, so this directory explicitly
# enables C; the source includes <hip/hip_runtime.h> only for ABI types, so
# it needs the ROCm include dir and links libamdhip64 (the HIP runtime).
cmake_minimum_required(VERSION 3.18)

if(NOT VKERNELS_HAS_HIP)
  message(STATUS "vkernels/net-ofi: HIP not enabled; plugin not built")
  return()
endif()
if(NOT VKERNELS_HAS_OFI)
  message(STATUS "vkernels/net-ofi: libfabric not found; plugin not built")
  return()
endif()

enable_language(C)

# HIP runtime headers (hip/hip_runtime.h) + libamdhip64 (the loader-visible
# RCCL plugin links the HIP runtime alongside libfabric). Found from the same
# ROCm root the HIP compiler lives under (ROCM_PATH / _ROCM_PATH_FROM_HIP).
find_path(VKERNELS_HIP_INCLUDE_DIR
  NAMES hip/hip_runtime.h
  HINTS ${ROCM_PATH} ${_ROCM_PATH_FROM_HIP} $ENV{ROCM_PATH} $ENV{ROCM_ROOT} /opt/rocm
  PATH_SUFFIXES include)
find_library(VKERNELS_AMDHIP64_LIBRARY
  NAMES amdhip64
  HINTS ${ROCM_PATH} ${_ROCM_PATH_FROM_HIP} $ENV{ROCM_PATH} $ENV{ROCM_ROOT} /opt/rocm
  PATH_SUFFIXES lib lib64)

add_library(rccl-net-ofi SHARED rccl_net_ofi.c)
add_library(vkernels::rccl-net-ofi ALIAS rccl-net-ofi)

# Portable symbol export (the loader dlsym's `nccl_ofi_net`).
add_compile_definitions($<$<COMPILE_LANGUAGE:C>:
  DLL_EXPORT=__attribute__\(\(visibility\("default"\)\)\)>)
if(APPLE)
  set_property(TARGET rccl-net-ofi APPEND PROPERTY
    LINK_FLAGS "-Wl,-exported_symbol,_nccl_ofi_net")
else()
  target_compile_options(rccl-net-ofi PRIVATE -fvisibility=hidden)
endif()

target_include_directories(rccl-net-ofi PRIVATE
  ${VKERNELS_OFI_INCLUDE_DIRS} ${VKERNELS_HIP_INCLUDE_DIR})
target_link_libraries(rccl-net-ofi PRIVATE
  ${VKERNELS_OFI_LIBRARIES} ${VKERNELS_AMDHIP64_LIBRARY})
set_target_properties(rccl-net-ofi PROPERTIES
  OUTPUT_NAME "rccl-net-ofi"
  LINKER_LANGUAGE C
  POSITION_INDEPENDENT_CODE ON)

# Install the .so next to the RCCL loader's search path on a Slingshot node.
include(GNUInstallDirs)
install(TARGETS rccl-net-ofi LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
