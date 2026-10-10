#!/usr/bin/env bash
# verifier/bmc/run.sh
#
# Bounded model checking of the CPU oracles.  Each harness includes the real
# oracle header, calls the real oracle, and states its spec with VK_ASSERT;
# the runner compiles the harness together with the actual oracle source, so
# the proof is about the shipped code, not a copy of it.
#
# Two provers are supported behind the same harnesses.  ESBMC is preferred
# because its Clang frontend parses the C++14/17 the oracles use; CBMC's own
# frontend tops out at C++11 and cannot parse libstdc++-13 headers, so it is
# used only when explicitly requested.
#
#   ./run.sh                 # auto-select esbmc, else cbmc, else skip (exit 77)
#   PROVER=cbmc ./run.sh
#   ./run.sh --require       # fail (exit 1) instead of skipping
#   BMC_UNWIND=12 ./run.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
src="$root/src/c"

require=0
[[ "${1:-}" == "--require" ]] && require=1

skip() {
  echo "SKIP: $1"
  echo "      Install ESBMC (https://github.com/esbmc/esbmc/releases, prefer" \
       "'linux-armv8') or CBMC (https://diffblue.github.io/cbmc/), or pass" \
       "--require to fail instead of skipping."
  [[ "$require" == 1 ]] && exit 1
  exit 77
}

prover="${PROVER:-auto}"
if [[ "$prover" == auto ]]; then
  if command -v esbmc >/dev/null 2>&1; then prover=esbmc
  elif command -v cbmc >/dev/null 2>&1; then prover=cbmc
  else skip "no bounded model checker found (looked for esbmc, cbmc)."; fi
fi

unwind="${BMC_UNWIND:-8}"

case "$prover" in
  esbmc)
    bin="${ESBMC:-esbmc}"
    command -v "$bin" >/dev/null 2>&1 || skip "'$bin' not found."
    # Redefine the contract macros as ESBMC intrinsics: the proof then ranges
    # over exactly the contractual inputs and need not model exceptions.
    defines=(
      -DVK_VERIFY_CONTRACTS
      -DVK_PROVER_ESBMC
      "-DVK_EXPECTS(cond,msg)=__ESBMC_assume(cond)"
      "-DVK_ENSURES(cond,msg)=__ESBMC_assert(cond,msg)"
    )
    prover_flags=(--unwind "$unwind" --multi-property)
    run() {
      local name="$1"; shift
      echo "==> ESBMC proof: $name (unwind=$unwind)"
      "$bin" "${defines[@]}" "${prover_flags[@]}" \
        -I "$src" -I "$here" -I "$here/../common" \
        "$here/harness_${name}.cpp" "$@" --function main
    }
    ;;
  cbmc)
    bin="${CBMC:-cbmc}"
    command -v "$bin" >/dev/null 2>&1 || skip "'$bin' not found."
    defines=(
      -DVK_VERIFY_CONTRACTS
      -DVK_PROVER_CBMC
      "-DVK_EXPECTS(cond,msg)=__CPROVER_assume(cond)"
      "-DVK_ENSURES(cond,msg)=__CPROVER_assert(cond,msg)"
    )
    prover_flags=(
      --unwind "$unwind" --unwinding-assertions
      --bounds-check --pointer-check --conversion-check --div-by-zero-check
    )
    [[ "${CBMC_FLOATBV:-0}" == 1 ]] && prover_flags+=(--floatbv)
    run() {
      local name="$1"; shift
      echo "==> CBMC proof: $name (unwind=$unwind)"
      "$bin" "${defines[@]}" "${prover_flags[@]}" \
        -I "$src" -I "$here" -I "$here/../common" \
        "$here/harness_${name}.cpp" "$@"
    }
    ;;
  *) skip "unknown PROVER='$prover' (expected esbmc or cbmc)." ;;
esac

run elementwise "$src/vkernels/kernels/elementwise.cpp"
run reduce      "$src/vkernels/kernels/reduce.cpp"
run gemm        "$src/vkernels/kernels/gemm.cpp"

echo "All $prover proofs discharged."
