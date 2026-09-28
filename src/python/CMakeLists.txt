# Python package: `vkernels` (discovery, CLI, and optional bindings). Managed
# by uv from the repository-root pyproject.toml (`uv sync` / `uv run vkl`).
#
# The discovery/CLI layer (`python3 -m vkernels.cli`) runs straight from the
# source tree and needs no build. When VKERNELS_BUILD_PYTHON=ON the compiled
# pybind11 backend (`vkernels._core`) is built from `vkernels/_core.cpp`,
# linked against the vkernels static library, and dropped into
# `build/<preset>/python/vkernels/` where the runtime loader
# (`vkernels/_backend.py`) finds it; the Python test suite is then
# registered with CTest.
find_package(Python3 COMPONENTS Interpreter QUIET)

# --- vkl smoke test (discovery + CLI, no build required) ------------------
if(Python3_FOUND AND VKERNELS_BUILD_TESTS)
  add_test(NAME vkl_smoke
    COMMAND ${Python3_EXECUTABLE} -m vkernels.cli list --json)
  set_tests_properties(vkl_smoke PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PASS_REGULAR_EXPRESSION "gemm")
endif()

# --- compiled bindings (opt-in, off by default) ---------------------------
if(VKERNELS_BUILD_PYTHON)
  if(NOT Python3_FOUND)
    message(FATAL_ERROR "VKERNELS_BUILD_PYTHON=ON requires a Python interpreter")
  endif()
  find_package(Python3 COMPONENTS Development.Module REQUIRED)

  # pybind11: use an installed copy if available, else fetch a pinned release.
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v3.1.0
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
  endif()

  # The compiled backend. Not subject to the repo's -Werror policy on
  # purpose: pybind11-generated code is third-party and not ours to police.
  pybind11_add_module(vkernels_python_core vkernels/_core.cpp)
  target_link_libraries(vkernels_python_core PRIVATE vkernels pybind11::module)
  set_target_properties(vkernels_python_core PROPERTIES
    OUTPUT_NAME "_core"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/vkernels")

  install(TARGETS vkernels_python_core LIBRARY DESTINATION vkernels COMPONENT python)
  foreach(_lib vkernels_c vkernels_hip vkernels_hostc)
    if(TARGET ${_lib})
      install(TARGETS ${_lib} LIBRARY DESTINATION vkernels COMPONENT python)
    endif()
  endforeach()
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/python/vkernels/build_info.json"
    CONTENT "{\"abi_version\":1,\"cuda\":$<IF:$<BOOL:${VKERNELS_HAS_CUDA}>,true,false>,\"hip\":$<IF:$<BOOL:${VKERNELS_HAS_HIP}>,true,false>}")
  install(FILES "${CMAKE_BINARY_DIR}/python/vkernels/build_info.json"
    DESTINATION vkernels COMPONENT python)

  # Python test suite (discovery + bindings), under tests/python/. The
  # interpreter can be overridden, e.g. when the extension was built against
  # the system Python but the tests should run in a venv that has numpy:
  #   cmake -DVKERNELS_PYTHON_EXECUTABLE=/path/to/venv/bin/python
  set(VKERNELS_PYTHON_EXECUTABLE "" CACHE FILEPATH
      "Python interpreter for the Python test suite (default: the interpreter found by CMake)")
  if(VKERNELS_PYTHON_EXECUTABLE)
    set(_vk_py_test_exe "${VKERNELS_PYTHON_EXECUTABLE}")
  else()
    set(_vk_py_test_exe "${Python3_EXECUTABLE}")
  endif()

  if(VKERNELS_BUILD_TESTS)
    add_test(NAME python_bindings
      COMMAND ${_vk_py_test_exe} -m pytest tests/python -v)
    set_tests_properties(python_bindings PROPERTIES
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      ENVIRONMENT
        "VKERNELS_EXTENSION=$<TARGET_FILE:vkernels_python_core>;VKERNELS_BACKEND=compiled;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_BINARY_DIR}/python;\
OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
  endif()
endif()
