# Optional CUDA language support for vkernels.
#
# CUDA is enabled only when both:
#   * VKERNELS_BUILD_CUDA is ON, and
#   * a working nvcc / CUDA toolkit can be found.
#
# When CUDA is unavailable we still build the full library using the CPU
# reference implementations, so the project is always host-buildable and
# testable. The result is exposed to the rest of the build via the boolean
# cache variable VKERNELS_HAS_CUDA and the VKERNELS_HAS_CUDA preprocessor
# macro (set on the vkernels target in src/c/).

set(VKERNELS_HAS_CUDA OFF CACHE BOOL "Whether a usable CUDA toolkit was found" FORCE)

if(NOT VKERNELS_BUILD_CUDA)
  message(STATUS "vkernels: CUDA disabled by VKERNELS_BUILD_CUDA=OFF (host-only build)")
  return()
endif()

find_program(VKERNELS_NVCC_EXECUTABLE NAMES nvcc)
if(NOT VKERNELS_NVCC_EXECUTABLE)
  message(STATUS "vkernels: nvcc not found; building host-only (CPU reference) path.")
  return()
endif()

# nvcc exists — try to enable the CUDA language. CMake will validate the
# toolkit and report a clear error if it is broken.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(VKERNELS_CUDA_ARCHITECTURES "native" CACHE STRING "Target CUDA architectures (native|70|80;86|...)")
  set(CMAKE_CUDA_ARCHITECTURES "${VKERNELS_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(VKERNELS_HAS_CUDA ON CACHE BOOL "Whether a usable CUDA toolkit was found" FORCE)

message(STATUS "vkernels: CUDA enabled (nvcc=${VKERNELS_NVCC_EXECUTABLE}, archs=${CMAKE_CUDA_ARCHITECTURES})")
