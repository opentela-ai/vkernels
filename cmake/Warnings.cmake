# Warning flags shared across vkernels targets. CUDA-specific warnings are
# added when the target's CUDA sources are present.

function(vkernels_set_warnings target)
  set(_cxx -Wall -Wextra -Wpedantic)
  if(VKERNELS_WARNINGS_AS_ERRORS)
    list(APPEND _cxx -Werror)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE ${_cxx})
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
        $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()
  endif()
endfunction()
