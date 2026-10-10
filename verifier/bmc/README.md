# bmc — bounded model checking of the CPU oracles

Each harness proves a property of the **real** oracle source (passed to the
prover on the command line together with the harness), not of a
re-implementation:

| Harness | Oracle under proof | Property |
|---|---|---|
| `harness_elementwise.cpp` | `kernels/elementwise.cpp` | `add`/`scale`/`relu` match their spec element-wise |
| `harness_reduce.cpp` | `kernels/reduce.cpp` | `sum` = sequential accumulation; `max` ∈ input and dominates it |
| `harness_gemm.cpp` | `kernels/gemm.cpp` | `C = α·A@B + β·C0` with the oracle's index map; A/B unmodified |

Postconditions use `VK_ASSERT` from `../common/verify_support.h`, which maps
to the selected prover's assertion intrinsic. The runner selects the prover
with `-DVK_PROVER_ESBMC` / `-DVK_PROVER_CBMC`.

## Why the contract macro is redefined

`vkernels/util/error.hpp` makes `VK_EXPECTS`/`VK_ENSURES` overridable. The
runner defines them as the prover's assume/assert intrinsics
(`-DVK_EXPECTS(cond,msg)=__ESBMC_assume(cond)`, …). Exceptions and
`std::string` are not usefully modeled, and the assumption is the
semantically correct thing: the proof ranges over exactly the inputs the
contract admits. An ordinary build leaves the macros undefined and keeps the
throwing behavior, so **production is unaffected and `src/c/` coverage is
unchanged**.

## Running

```bash
./run.sh                 # auto: esbmc, else cbmc, else skip (exit 77)
PROVER=esbmc ./run.sh
PROVER=cbmc ./run.sh
./run.sh --require       # exit 1 if no prover is available
BMC_UNWIND=12 ./run.sh   # widen the bound
```

The proof is bounded: CBMC/ESBMC checks the fixed sizes declared in each
harness (`kN = 4`, `M=N=K=2`). Memory-safety properties are size-independent
for these kernels; the value postconditions are proved for the stated bound.
Widen the constants to strengthen the value proof at extra cost.

## Prover notes

- **ESBMC** (preferred): its Clang frontend parses the C++14/17 the oracles
  use. Prebuilt `esbmc-linux-armv8.zip` /
  `esbmc-linux.zip` from <https://github.com/esbmc/esbmc/releases>.
- **CBMC**: faster for pure C, but CBMC 6.11's frontend tops out at C++11
  and cannot parse libstdc++-13's `<type_traits>`, so it fails on these
  sources. Kept for the day that limit lifts, and for any C-only harnesses.
