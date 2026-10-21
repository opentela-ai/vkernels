# Benchmark / device-harness registration helpers (meta/benchmarks/).
#
#   vkernels_add_benchmark(NAME <target>
#                          SOURCES <src...>
#                          [NO_VKERNELS_LIB]
#                          [INCLUDE_DIRS <dir...>]
#                          [LIBS <lib...>]
#                          [TEST])
#     Builds an executable named `<NAME>` (no prefix: the bench binaries are
#     invoked by name from scripts and docs), links it against
#     vkernels::vkernels, and applies C++17 + warnings.
#
#     NO_VKERNELS_LIB skips the vkernels::vkernels link for harnesses that
#     link a separate ABI boundary (vkernels_c, nccl) via LIBS instead, the
#     same convention as vkernels_add_test.
#
#     TEST additionally registers the binary with CTest under `<NAME>` (when
#     VKERNELS_BUILD_TESTS is on) for self-checking device-vs-oracle
#     harnesses: exit 0 = pass, 77 = skipped (no GPU visible), anything
#     else = fail. Registered harnesses carry the `benchmark_correctness`
#     label so CI can select (-L) or exclude (-LE) them.

function(vkernels_add_benchmark)
  cmake_parse_arguments(ARG "NO_VKERNELS_LIB;TEST" "NAME" "SOURCES;INCLUDE_DIRS;LIBS" ${ARGN})

  add_executable(${ARG_NAME} ${ARG_SOURCES})
  if(ARG_INCLUDE_DIRS)
    target_include_directories(${ARG_NAME} PRIVATE ${ARG_INCLUDE_DIRS})
  endif()
  if(NOT ARG_NO_VKERNELS_LIB)
    target_link_libraries(${ARG_NAME} PRIVATE vkernels::vkernels)
  endif()
  if(ARG_LIBS)
    target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LIBS})
  endif()
  target_compile_features(${ARG_NAME} PRIVATE cxx_std_17)
  vkernels_set_warnings(${ARG_NAME})

  if(ARG_TEST AND VKERNELS_BUILD_TESTS)
    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
    set_tests_properties(${ARG_NAME} PROPERTIES
      SKIP_RETURN_CODE 77
      LABELS benchmark_correctness)
  endif()
endfunction()
