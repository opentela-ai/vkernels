# verifier/CMakeLists.txt
#
# Formal-verification harnesses.  OPT-IN and deliberately isolated from the
# production library and the test binaries:
#   * nothing here is compiled into `vkernels`, the C ABI, or the language
#     bindings;
#   * the sources live outside src/c/, so the 100%-line-coverage gate over
#     src/c/ neither sees nor requires them;
#   * every runner exits 77 (CTest SKIP_RETURN_CODE) when its prover
#     (esbmc / cbmc / z3 / gpuverify / gappa) is not installed, so enabling this
#     directory never breaks a build on a machine without the tool.
#
# Enable with:  cmake --preset host -DVKERNELS_BUILD_VERIFICATIONS=ON

enable_testing()

set(_verify_dir ${CMAKE_CURRENT_SOURCE_DIR})

# Register one verification family as a CTest test.  The script is expected
# to exit 77 to mean "prover unavailable -> skip".
function(vkernels_add_verification NAME SCRIPT)
  add_test(NAME verify_${NAME} COMMAND ${SCRIPT})
  set_tests_properties(verify_${NAME} PROPERTIES
    SKIP_RETURN_CODE 77
    WORKING_DIRECTORY ${_verify_dir})
endfunction()

vkernels_add_verification(bmc      ${_verify_dir}/bmc/run.sh)
vkernels_add_verification(smt      ${_verify_dir}/smt/run.sh)
vkernels_add_verification(gpu      ${_verify_dir}/gpu/run.sh)
vkernels_add_verification(numerics ${_verify_dir}/numerics/run.sh)
