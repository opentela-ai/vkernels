# Optional RCCL (ROCm Communication Collectives) + libfabric/OFI support for
# the vkernels comm layer (issue #19).
#
# RCCL is AMD's NCCL: the all-reduce primitive a cross-node TP group runs
# over. It is enabled only when both:
#   * VKERNELS_HAS_HIP is ON (RCCL needs a HIP runtime), and
#   * the RCCL headers (`rccl.h`) and library (`librccl`) are found.
#
# libfabric/OFI is what the HIP-aware OFI/CXI net plugin
# (`plugins/rccl-net-ofi`) links against so RCCL selects Slingshot RDMA
# instead of its built-in Socket transport (`NCCL_SOCKET_IFNAME=hsn0`,
# `NCCL_IB_DISABLE=1` on beverin). It is enabled when the libfabric headers
# (`rdma/fabric.h`) and library (`libfabric`) are found; absence degrades the
# plugin build but never the host-only library.
#
# Results exposed to the rest of the build:
#   * VKERNELS_HAS_RCCL  — a usable RCCL installation was found.
#   * VKERNELS_HAS_OFI   — a usable libfabric/OFI installation was found.
#   * VKERNELS_RCCL_INCLUDE_DIRS / VKERNELS_RCCL_LIBRARIES
#   * VKERNELS_OFI_INCLUDE_DIRS  / VKERNELS_OFI_LIBRARIES
#
# Nothing here is fatal when absent: the host reference (rccl.cpp) and the
# full test suite build and pass without RCCL or libfabric, exactly as the
# CI host job does.

# --- RCCL --------------------------------------------------------------------

set(VKERNELS_HAS_RCCL OFF CACHE BOOL "Whether a usable RCCL installation was found" FORCE)
set(VKERNELS_HAS_OFI OFF CACHE BOOL "Whether a usable libfabric/OFI installation was found" FORCE)

if(NOT VKERNELS_HAS_HIP)
  message(STATUS "vkernels: RCCL not searched (HIP not enabled)")
  return()
endif()

# RCCL ships in ROCm under <rocm_root>/include/rccl.h and
# <rocm_root>/lib/librccl.so. Prefer the ROCm root hinted by the HIP
# compiler, then a standalone RCCL install (RCCL_ROOT / RCCL_DIR).
foreach(_root
    "${ROCM_PATH}" "${ROCM_ROOT}" "${AMD_ROCM_PATH}"
    "$ENV{ROCM_PATH}" "$ENV{ROCM_ROOT}" "$ENV{AMD_ROCM_PATH}"
    "${_ROCM_PATH_FROM_HIP}")
  if(_root)
    list(APPEND _rccl_roots "${_root}")
  endif()
endforeach()

find_path(VKERNELS_RCCL_INCLUDE_DIR
  NAMES rccl.h
  HINTS ${_rccl_roots} ENV RCCL_ROOT ENV RCCL_DIR
  PATH_SUFFIXES include)
find_library(VKERNELS_RCCL_LIBRARY
  NAMES rccl
  HINTS ${_rccl_roots} ENV RCCL_ROOT ENV RCCL_DIR
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Rccl
  "RCCL (rccl.h / librccl) not found; building the comm layer without the HIP/RCCL transport (issue #19)."
  VKERNELS_RCCL_LIBRARY VKERNELS_RCCL_INCLUDE_DIR)

if(VKERNELS_RCCL_FOUND)
  set(VKERNELS_HAS_RCCL ON CACHE BOOL "Whether a usable RCCL installation was found" FORCE)
  set(VKERNELS_RCCL_INCLUDE_DIRS "${VKERNELS_RCCL_INCLUDE_DIR}")
  set(VKERNELS_RCCL_LIBRARIES "${VKERNELS_RCCL_LIBRARY}")
  if(NOT TARGET vkernels::rccl)
    add_library(vkernels::rccl UNKNOWN IMPORTED)
    set_target_properties(vkernels::rccl PROPERTIES
      IMPORTED_LOCATION "${VKERNELS_RCCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${VKERNELS_RCCL_INCLUDE_DIR}")
  endif()
  message(STATUS "vkernels: RCCL enabled (include=${VKERNELS_RCCL_INCLUDE_DIRS} lib=${VKERNELS_RCCL_LIBRARIES})")
else()
  message(STATUS "vkernels: RCCL not found; HIP/RCCL transport disabled (host reference still builds)")
endif()

# --- libfabric / OFI (independent of RCCL; needed by the net plugin) ---------

find_path(VKERNELS_OFI_INCLUDE_DIR
  NAMES rdma/fabric.h
  HINTS ${_rccl_roots} /usr ENV FI_PATH ENV OFI_ROOT ENV LIBFABRIC_ROOT
  PATH_SUFFIXES include)
find_library(VKERNELS_OFI_LIBRARY
  NAMES fabric
  HINTS ${_rccl_roots} /usr ENV FI_PATH ENV OFI_ROOT ENV LIBFABRIC_ROOT
  PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(Ofi
  "libfabric (rdma/fabric.h / libfabric) not found; the OFI/CXI net plugin will not be built (issue #19)."
  VKERNELS_OFI_LIBRARY VKERNELS_OFI_INCLUDE_DIR)

if(VKERNELS_OFI_FOUND)
  set(VKERNELS_HAS_OFI ON CACHE BOOL "Whether a usable libfabric/OFI installation was found" FORCE)
  set(VKERNELS_OFI_INCLUDE_DIRS "${VKERNELS_OFI_INCLUDE_DIR}")
  set(VKERNELS_OFI_LIBRARIES "${VKERNELS_OFI_LIBRARY}")
  if(NOT TARGET vkernels::ofi)
    add_library(vkernels::ofi UNKNOWN IMPORTED)
    set_target_properties(vkernels::ofi PROPERTIES
      IMPORTED_LOCATION "${VKERNELS_OFI_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${VKERNELS_OFI_INCLUDE_DIR}")
  endif()
  message(STATUS "vkernels: libfabric/OFI enabled (include=${VKERNELS_OFI_INCLUDE_DIRS} lib=${VKERNELS_OFI_LIBRARIES})")
else()
  message(STATUS "vkernels: libfabric/OFI not found; OFI/CXI net plugin disabled")
endif()
