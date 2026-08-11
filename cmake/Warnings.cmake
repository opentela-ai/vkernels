# Warning flags shared across vkernels targets. CUDA-specific warnings are
# added when the target's CUDA sources are present.
#
# The cross-execution-space-call diagnostic was a real nvcc flag for years
# but was removed in CUDA 13, so it is only promoted to an error when the
# active nvcc still recognises it (checked once at configure time). This keeps
# the host build unchanged and lets the CUDA build succeed on old and new
# toolkits alike.
include(CheckCompilerFlag)

function(vkernels_set_warnings target)
  set(_cxx -Wall -Wextra -Wpedantic)
  if(VKERNELS_WARNINGS_AS_ERRORS)
    list(APPEND _cxx -Werror)
  endif()

  # Host (GCC/Clang) flags are scoped to CXX only: a bare -Werror passed to
  # nvcc consumes the next token as its value (e.g. -Werror --expt-relaxed-constexpr
  # is read as -Werror=--expt-relaxed-constexpr), which is fatal. nvcc gets its
  # own, deliberately smaller, warning set below.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(_f IN LISTS _cxx)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_f}>)
    endforeach()
  endif()

  # nvcc is far noisier; allow it to be tamed without breaking host builds.
  if(TARGET ${target})
    get_target_property(_sources ${target} SOURCES)
    foreach(_src IN LISTS _sources)
      if(_src MATCHES "\\.cu$")
        set(_has_cuda TRUE)
      endif()
    endforeach()
    if(_has_cuda AND CMAKE_CUDA_COMPILER)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      # -Werror=cross-execution-space-call was removed in CUDA 13; promote it
      # to an error only when the active nvcc still recognises it.
      check_compiler_flag(CUDA "-Werror=cross-execution-space-call"
                           _nvcc_supports_cross_exec_space_call)
      if(_nvcc_supports_cross_exec_space_call)
        target_compile_options(${target} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>)
      endif()
    endif()
  endif()
endfunction()
