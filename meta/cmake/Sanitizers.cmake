# Optional sanitizers. Mutually compatible with most builds; turn coverage
# off when using ASan to avoid gcov link noise.

function(vkernels_apply_sanitizers target)
  set(_flags)
  if(VKERNELS_ENABLE_ASAN)
    list(APPEND _flags -fsanitize=address -fno-omit-frame-pointer)
  endif()
  if(VKERNELS_ENABLE_UBSAN)
    list(APPEND _flags -fsanitize=undefined)
  endif()
  if(_flags)
    target_compile_options(${target} PRIVATE ${_flags})
    target_link_options(${target} PRIVATE ${_flags})
  endif()
endfunction()
