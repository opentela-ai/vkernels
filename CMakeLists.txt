cmake_minimum_required(VERSION 3.18)

project(
  vkernels
  VERSION 0.1.0
  DESCRIPTION "Low-level CUDA kernels and GPU communication primitives for opentela."
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(VKERNELS_BUILD_TESTS "Build unit tests" ON)
option(VKERNELS_BUILD_BENCHMARKS "Build micro-benchmarks" OFF)
option(VKERNELS_BUILD_CUDA "Compile CUDA kernel sources when a toolkit is found" ON)
option(VKERNELS_BUILD_PYTHON "Build Python bindings (opt-in)" OFF)
option(VKERNELS_ENABLE_COVERAGE "Enable gcov line-coverage instrumentation" OFF)
option(VKERNELS_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(VKERNELS_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(VKERNELS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

set(VKERNELS_TEST_FRAMEWORK "minitest"
  CACHE STRING "Test framework to use: minitest (bundled, no deps) or gtest (fetched)")
set_property(CACHE VKERNELS_TEST_FRAMEWORK PROPERTY STRINGS minitest gtest)

# Defaults that callers may override via -D or presets.
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for non-coverage, non-sanitizer builds.
if(NOT CMAKE_BUILD_TYPE AND NOT VKERNELS_ENABLE_COVERAGE
    AND NOT VKERNELS_ENABLE_ASAN AND NOT VKERNELS_ENABLE_UBSAN)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Warnings)
include(Sanitizers)
include(Coverage)
include(CudaSupport)
include(Testing)

if(VKERNELS_ENABLE_COVERAGE)
  message(STATUS "vkernels: line-coverage instrumentation enabled (gcov)")
endif()

add_subdirectory(csrc)

if(VKERNELS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VKERNELS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Configuration summary (printed at the end of configuration)
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==== vkernels ${PROJECT_VERSION} ====")
message(STATUS "  C++ standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA kernels  : ${VKERNELS_HAS_CUDA} (requested ${VKERNELS_BUILD_CUDA})")
if(VKERNELS_HAS_CUDA)
  message(STATUS "  CUDA archs    : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Tests         : ${VKERNELS_BUILD_TESTS} (${VKERNELS_TEST_FRAMEWORK})")
message(STATUS "  Benchmarks    : ${VKERNELS_BUILD_BENCHMARKS}")
message(STATUS "  Coverage      : ${VKERNELS_ENABLE_COVERAGE}")
message(STATUS "  ASan/UBSan    : ${VKERNELS_ENABLE_ASAN}/${VKERNELS_ENABLE_UBSAN}")
message(STATUS "=================================")
