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
option(VKERNELS_BUILD_HIP "Compile HIP kernel sources when ROCm is found (AMD gfx942)" OFF)
option(VKERNELS_BUILD_PYTHON "Build Python bindings (opt-in)" OFF)
option(VKERNELS_BUILD_RUST "Register the Rust binding test suite (opt-in, needs cargo)" OFF)
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
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/meta/cmake")
include(Warnings)
include(Sanitizers)
include(Coverage)
include(CudaSupport)
include(HipSupport)
include(RcclSupport)
include(Testing)

if(VKERNELS_ENABLE_COVERAGE)
  message(STATUS "vkernels: line-coverage instrumentation enabled (gcov)")
endif()

if(VKERNELS_BUILD_HIP)
  add_subdirectory(plugins/rccl-net-ofi)
endif()

add_subdirectory(src/c)

if(VKERNELS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VKERNELS_BUILD_BENCHMARKS)
  add_subdirectory(meta/benchmarks)
endif()

if(VKERNELS_BUILD_PYTHON)
  add_subdirectory(src/python)
endif()

if(VKERNELS_BUILD_RUST)
  # Rust bindings under src/rust/ (safe `vkernels` crate over the `vkernels-sys`
  # FFI layer). `cargo test` builds the C++ library itself via the cmake
  # build script, so this only registers the test; the crate builds on any
  # machine (host path by default, VKERNELS_RUST_CUDA=ON for CUDA).
  find_program(VKERNELS_CARGO_EXECUTABLE NAMES cargo)
  if(NOT VKERNELS_CARGO_EXECUTABLE)
    message(FATAL_ERROR "VKERNELS_BUILD_RUST=ON requires cargo on PATH")
  endif()
  if(VKERNELS_BUILD_TESTS)
    add_test(NAME rust_bindings
      COMMAND ${VKERNELS_CARGO_EXECUTABLE} test --manifest-path
              ${CMAKE_CURRENT_SOURCE_DIR}/src/rust/Cargo.toml)
    set_tests_properties(rust_bindings PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/rust")
  endif()
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
  message(STATUS "  HIP kernels   : ${VKERNELS_HAS_HIP} (requested ${VKERNELS_BUILD_HIP})")
  message(STATUS "  RCCL transport: ${VKERNELS_HAS_RCCL} (issue #19)")
  message(STATUS "  OFI/CXI plugin: ${VKERNELS_HAS_OFI} (issue #19)")
message(STATUS "  Tests         : ${VKERNELS_BUILD_TESTS} (${VKERNELS_TEST_FRAMEWORK})")
message(STATUS "  Benchmarks    : ${VKERNELS_BUILD_BENCHMARKS}")
message(STATUS "  Python bindings: ${VKERNELS_BUILD_PYTHON}")
message(STATUS "  Rust bindings  : ${VKERNELS_BUILD_RUST}")
message(STATUS "  Coverage      : ${VKERNELS_ENABLE_COVERAGE}")
message(STATUS "  ASan/UBSan    : ${VKERNELS_ENABLE_ASAN}/${VKERNELS_ENABLE_UBSAN}")
message(STATUS "=================================")
