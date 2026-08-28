# Optional NCCL (NVIDIA Collective Communications) support for the
# vkernels comm layer (issue #49 on-site step).
#
# NCCL is NVIDIA's RCCL: the all-reduce / send-recv primitive a cross-node
# TP group runs over InfiniBand (what vLLM / real serving use; CUDA-native,
# no MPI). It is enabled only when both:
#   * VKERNELS_HAS_CUDA is ON (NCCL needs a CUDA runtime), and
#   * the NCCL headers (`nccl.h`) and library (`libnccl`) are found.
#
# Results exposed to the rest of the build:
#   * VKERNELS_HAS_NCCL -- a usable NCCL installation was found.
#   * VKERNELS_NCCL_INCLUDE_DIRS / VKERNELS_NCCL_LIBRARIES
#   * vkernels::nccl -- an imported target the cross-node bench links to.
#
# Nothing here is fatal when absent: the host-only library, the full test
# suite, and every other benchmark build and pass without NCCL, exactly as
# the CI host job does. The real-RDMA-fabric per-hop number is an opt-in
# on-site step (bench_cross_node_nccl.cu); the cost model in
# bench_cross_node_kv.cpp is the CI-verifiable surface.

set(VKERNELS_HAS_NCCL OFF CACHE BOOL "Whether a usable NCCL installation was found" FORCE)

if(NOT VKERNELS_HAS_CUDA)
  message(STATUS "vkernels: NCCL not searched (CUDA not enabled)")
  return()
endif()

# NCCL ships its public header at <prefix>/include/nccl.h and its library
# at <prefix>/lib/libnccl.so. EasyBuild exposes the install root in
# EBROOTNCCL (JSC), and a standard NCCL_ROOT / NCCL_DIR works too.
foreach(_root
    "${NCCL_ROOT}" "${NCCL_DIR}"
    "$ENV{NCCL_ROOT}" "$ENV{NCCL_DIR}" "$ENV{EBROOTNCCL}")
  if(_root)
    list(APPEND _nccl_roots "${_root}")
  endif()
endforeach()

find_path(VKERNELS_NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${_nccl_roots} /usr /usr/local
  PATH_SUFFIXES include)
find_library(VKERNELS_NCCL_LIBRARY
  NAMES nccl
  HINTS ${_nccl_roots} /usr /usr/local
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nccl
  "NCCL (nccl.h / libnccl) not found; cross-node NCCL bench disabled (issue #49)."
  VKERNELS_NCCL_LIBRARY VKERNELS_NCCL_INCLUDE_DIR)

# Drive VKERNELS_HAS_NCCL off the concrete required variables rather than
# the *_FOUND name emitted by find_package_handle_standard_args (derived
# from the first argument's case and brittle across CMake versions), the
# same rationale as RcclSupport.cmake. Both nccl.h and libnccl must be
# present.
if(VKERNELS_NCCL_INCLUDE_DIR AND VKERNELS_NCCL_LIBRARY)
  set(VKERNELS_HAS_NCCL ON CACHE BOOL "Whether a usable NCCL installation was found" FORCE)
  set(VKERNELS_NCCL_INCLUDE_DIRS "${VKERNELS_NCCL_INCLUDE_DIR}")
  set(VKERNELS_NCCL_LIBRARIES "${VKERNELS_NCCL_LIBRARY}")
  if(NOT TARGET vkernels::nccl)
    add_library(vkernels::nccl UNKNOWN IMPORTED)
    set_target_properties(vkernels::nccl PROPERTIES
      IMPORTED_LOCATION "${VKERNELS_NCCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${VKERNELS_NCCL_INCLUDE_DIR}")
  endif()
  message(STATUS "vkernels: NCCL enabled (include=${VKERNELS_NCCL_INCLUDE_DIRS} lib=${VKERNELS_NCCL_LIBRARIES})")
else()
  message(STATUS "vkernels: NCCL not found; cross-node NCCL bench disabled (host reference still builds)")
endif()
