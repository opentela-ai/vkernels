# plugins/rccl-net-ofi — HIP-aware OFI/CXI net plugin for RCCL (issue #19)
#
# Builds `librccl-net-ofi.so` against libfabric + the HIP runtime, gated by
# VKERNELS_HAS_OFI (libfabric found by RcclSupport.cmake). Never affects the
# host-only library or its tests: the top-level CMakeLists adds this
# subdirectory only when HIP + libfabric are both present.
cmake_minimum_required(VERSION 3.18)

if(NOT VKERNELS_HAS_HIP)
  message(STATUS "vkernels/net-ofi: HIP not enabled; plugin not built")
  return()
endif()
if(NOT VKERNELS_HAS_OFI)
  message(STATUS "vkernels/net-ofi: libfabric not found; plugin not built")
  return()
endif()

# The plugin is a single C translation unit compiled with hipcc (it links
# librt.so-equivalent HIP bits; a plain cc cannot link hiprt).
enable_language(HIP)

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

target_include_directories(rccl-net-ofi PRIVATE ${VKERNELS_OFI_INCLUDE_DIRS})
target_link_libraries(rccl-net-ofi PRIVATE ${VKERNELS_OFI_LIBRARIES})
set_target_properties(rccl-net-ofi PROPERTIES
  OUTPUT_NAME "rccl-net-ofi"
  POSITION_INDEPENDENT_CODE ON)

# Install the .so next to the RCCL loader's search path on a Slingshot node.
include(GNUInstallDirs)
install(TARGETS rccl-net-ofi LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
