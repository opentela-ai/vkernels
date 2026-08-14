# Coverage helpers (gcov). Applied per-target so third-party / test framework
# code is never instrumented, which keeps "100% of our code" a meaningful goal.
#
# Usage:
#   vkernels_enable_coverage()    # set global defaults (cheap, no-Werror, -O0 -g)
#   vkernels_apply_coverage(tgt)  # instrument a single target

function(vkernels_enable_coverage)
  # Coverage builds must be unoptimised and must keep frame pointers.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O0 -g -fno-omit-frame-pointer)
  endif()
endfunction()

function(vkernels_apply_coverage target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # `-fprofile-update=prefer-atomic`: thread-safe counter updates.  The
    # default (single) mode emits plain increments, so a test that runs
    # collectives on multiple threads (ring all-reduce, the distributed MoE
    # ranks) tears the shared counters: gcov then reports negative branch
    # counts and flips covered lines to uncovered, making the 100 % gate
    # nondeterministic.  Atomic updates cost a little but keep the gate
    # reliable.
    target_compile_options(${target} PRIVATE --coverage -fprofile-update=prefer-atomic)
    target_link_options(${target} PRIVATE --coverage)
    target_compile_definitions(${target} PRIVATE VKERNELS_COVERAGE_BUILD=1)
  endif()
endfunction()
