# Optional HIP language support for vkernels (AMD CDNA / gfx942).
#
# HIP is enabled only when both:
#   * VKERNELS_BUILD_HIP is ON (default OFF), and
#   * a working hipcc / ROCm installation can be found.
#
# When HIP is unavailable we still build the full library using the CPU
# reference implementations. The result is exposed to the rest of the build
# via the boolean cache variable VKERNELS_HAS_HIP and the VKERNELS_HAS_HIP
# preprocessor macro (set on the vkernels target in src/c/).
#
# Parallel to CudaSupport.cmake.

set(VKERNELS_HAS_HIP OFF CACHE BOOL "Whether a usable ROCm/HIP installation was found" FORCE)

if(NOT VKERNELS_BUILD_HIP)
  message(STATUS "vkernels: HIP disabled by VKERNELS_BUILD_HIP=OFF (host-only build)")
  return()
endif()

# CMake 3.21+ provides first-class HIP language support. Try to enable it.
if(${CMAKE_VERSION} VERSION_LESS "3.21")
  message(STATUS "vkernels: CMake < 3.21 — HIP language not available; building host-only path")
  return()
endif()

# Check if the HIP language is available on this system.
include(CheckLanguage)
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
  find_program(VKERNELS_HIPCC_EXECUTABLE NAMES hipcc)
  if(NOT VKERNELS_HIPCC_EXECUTABLE)
    message(STATUS "vkernels: hipcc not found; building host-only (CPU reference) path")
    return()
  endif()
endif()

enable_language(HIP)
set(VKERNELS_HAS_HIP ON CACHE BOOL "Whether a usable ROCm/HIP installation was found" FORCE)

# Default target architecture to gfx942 (MI300X) — the primary target.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(VKERNELS_HIP_ARCHITECTURES "gfx942" CACHE STRING
    "Target AMD GPU architectures (e.g. gfx942;gfx90a)")
  set(CMAKE_HIP_ARCHITECTURES "${VKERNELS_HIP_ARCHITECTURES}")
endif()

message(STATUS "vkernels: HIP enabled (archs=${CMAKE_HIP_ARCHITECTURES})")
