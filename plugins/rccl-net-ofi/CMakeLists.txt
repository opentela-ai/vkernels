# plugins/rccl-net-ofi — OFI/CXI net plugin for RCCL (issue #19)
#
# Builds `librccl-net-ofi.so` against libfabric, gated by VKERNELS_HAS_OFI
# (libfabric found by RcclSupport.cmake). Never affects the host-only library
# or its tests: the top-level CMakeLists adds this subdirectory only when
# HIP + libfabric are both present.
#
# The plugin is a single C translation unit (the standard aws_ofi_nccl layout
# is C) compiled with the host C compiler. Its v-table uses only `void*`
# request handles (the RCCL net-plugin ABI never takes a hipStream_t), so it
# links libfabric alone — no HIP runtime, no libcudart.
cmake_minimum_required(VERSION 3.18)

if(NOT VKERNELS_HAS_HIP)
  message(STATUS "vkernels/net-ofi: HIP not enabled; plugin not built")
  return()
endif()
if(NOT VKERNELS_HAS_OFI)
  message(STATUS "vkernels/net-ofi: libfabric not found; plugin not built")
  return()
endif()

# The transport callbacks are placeholders. Never install a discoverable
# RCCL plugin until send/receive contract tests pass on a fabric runner.
message(STATUS "vkernels/net-ofi: disabled (transport not implemented)")
return()
