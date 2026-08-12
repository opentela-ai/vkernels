# Test registration helpers.
#
# The default test harness is the bundled, dependency-free `minitest`
# (tests/third_party/minitest.hpp). Set -DVKERNELS_TEST_FRAMEWORK=gtest to
# instead fetch GoogleTest via FetchContent for richer matchers; the harness
# functions keep CTest registration identical either way.
#
#   vkernels_add_test(NAME <target>
#                     SOURCES <src...>
#                     [INCLUDE_DIRS <dir...>]
#                     [LIBS <lib...>])
#     Builds an executable named `vkernels_test_<NAME>`, links it against the
#     vkernels library and the test-main object, registers it with CTest, and
#     applies coverage + warnings.

include(GoogleTest OPTIONAL RESULT_VARIABLE _have_gtest)

function(vkernels_add_test)
  cmake_parse_arguments(ARG "NO_VKERNELS_LIB" "NAME" "SOURCES;INCLUDE_DIRS;LIBS" ${ARGN})

  set(_exe "vkernels_test_${ARG_NAME}")
  add_executable(${_exe} ${ARG_SOURCES})
  target_compile_features(${_exe} PRIVATE cxx_std_17)
  # Most tests link the static C++ library directly. Tests that exercise a
  # separate ABI boundary (e.g. the CUDA-gated `vkernels_c` shared library)
  # pass NO_VKERNELS_LIB and provide the boundary library via LIBS instead,
  # so two copies of the CUDA kernels never link into the same binary.
  if(NOT ARG_NO_VKERNELS_LIB)
    target_link_libraries(${_exe} PRIVATE vkernels)
  endif()
  if(ARG_LIBS)
    target_link_libraries(${_exe} PRIVATE ${ARG_LIBS})
  endif()

  # The shared test-main provides int main() -> run() for the minitest harness.
  target_link_libraries(${_exe} PRIVATE vkernels_test_main)
  # Bundled, dependency-free test harness.
  target_include_directories(${_exe} PRIVATE ${CMAKE_SOURCE_DIR}/tests/third_party)

  if(ARG_INCLUDE_DIRS)
    target_include_directories(${_exe} PRIVATE ${ARG_INCLUDE_DIRS})
  endif()

  vkernels_set_warnings(${_exe})
  vkernels_apply_sanitizers(${_exe})
  if(VKERNELS_ENABLE_COVERAGE)
    vkernels_apply_coverage(${_exe})
  endif()

  add_test(NAME ${ARG_NAME} COMMAND ${_exe})
endfunction()
