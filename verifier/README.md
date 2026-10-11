# verifier — formal verification of the kernel oracles

This directory holds **formal-verification harnesses** for the kernels in
`src/c/vkernels/`. They are deliberately *not* tests and *not* production
code:

- **Not production.** Nothing here is compiled into `libvkernels`, the C ABI,
  the Python/Rust bindings, or any benchmark. Production correctness is
  established by the two-implementation model (CPU oracle + device kernel)
  and does not pay for verification at runtime. Enabling this directory
  cannot change a single byte of a production build.
- **Not tests.** The `tests/` tree exercises both implementations against
  each other on sampled inputs. Verification here aims at the two things
  testing cannot reach: (a) the CPU **oracle** itself matches the
  mathematical spec for *all* inputs (bounded model checking), and (b) the
  device kernel's **index/tiling logic** is a refinement of the oracle's
  (translation validation). See `docs/ORIENTATION.md` §4 for why the oracle
  is the right thing to verify.

Because the sources live **outside `src/c/`**, the 100 %-line-coverage gate
(`meta/scripts/coverage.py --source-dir src/c`) neither sees nor requires
them, and ordinary `cmake`/`ctest` runs are unaffected.

## Layout

```
verifier/
├── common/verify_support.h   # VK_ASSUME / VK_ASSERT / VK_NONDET_FLOAT / vk_feq
├── bmc/                      # bounded model checking of the CPU oracles
│   ├── harness_elementwise.cpp
│   ├── harness_reduce.cpp
│   ├── harness_gemm.cpp
│   └── run.sh
├── smt/                      # SMT / exhaustive equivalence of tiling logic
│   ├── gemm_index_equivalence.py
│   └── run.sh
├── gpu/                      # race / barrier-divergence verification (.cu/.hip)
│   ├── Dockerfile.gpuverify  # amd64 userland for the x86-64 GPUVerify release
│   ├── gpuverify.sh          # runs that release under Docker/amd64 emulation
│   ├── harnesses/            # GPUVerify-dialect mirrors of the kernels
│   │   ├── elementwise.cu
│   │   ├── reduce.cu
│   │   ├── gemm.cu
│   │   ├── comm_allreduce.cu
│   │   ├── comm_peer_copy.cu
│   │   ├── comm_kv_gather.cu
│   │   └── comm_kv_scatter.cu
│   ├── README.md
│   └── run.sh
└── numerics/                 # certified floating-point error bounds
    ├── README.md
    ├── bounds/
    │   └── sequential_sum_fp32.gappa
    └── run.sh
```

## Running

Every family is registered with CTest only when the tree is opted in:

```bash
cmake --preset host -DVKERNELS_BUILD_VERIFICATIONS=ON
cmake --build --preset host
ctest --preset host -R '^verify_' --output-on-failure
```

Or run a family directly (or `make verify` for all four):

```bash
verifier/bmc/run.sh          # ESBMC (or CBMC), else exit 77 (skip)
verifier/smt/run.sh          # exhaustive always; symbolic when z3 present
verifier/gpu/run.sh
verifier/numerics/run.sh
```

Each runner exits **77** when its prover is not installed. CTest treats 77
as `Skipped` (`SKIP_RETURN_CODE`), so a machine without ESBMC/Z3/Gappa/GPUVerify
still gets a green, honest report instead of a failed build. There is no
silent pass: if a prover *is* present, a failed proof is a failed test.

## Toolchains

| Family    | Tool                                                                                  | Install |
|-----------|---------------------------------------------------------------------------------------|---------|
| `bmc`     | **ESBMC** (preferred, Clang frontend) or CBMC                                          | prebuilt `esbmc-linux-armv8.zip` / `cbmc` |
| `smt`     | Python 3; optional Z3 (`uv pip install z3-solver`) for the symbolic section           | stdlib else |
| `gpu`     | GPUVerify (x86-64; run under amd64 emulation), and/or `compute-sanitizer` (NVIDIA) / `rocprof` racecheck (AMD) | release zip + `docker build`; CUDA toolkit / ROCm |
| `numerics`| Gappa, FPTaylor or Rosa                                                               | source build |

On this checkout the toolchain is installed user-locally (there is no root;
`apt` needs a password here):

| Tool | Version | Location | Notes |
|---|---|---|---|
| ESBMC | 8.5.0 aarch64 | `~/.local/bin/esbmc` | prebuilt `esbmc-linux-armv8.zip`; the working prover |
| CBMC | 6.11.0 aarch64 | `~/.local/bin/cbmc` | frontend tops out at C++11; cannot parse libstdc++-13, so `bmc` auto-selects ESBMC |
| Z3 | 5.1.0 | repo `.venv` | `uv pip install z3-solver`, used by `smt` |
| compute-sanitizer | CUDA 13.0 | `~/.local/bin/compute-sanitizer` -> `/usr/local/cuda-13.0` | GB10 (sm_121); `racecheck` needs a **shared** libcudart test |
| GPUVerify | 2018-03-22 (x86-64) | `~/.local/opt/gpuverify` + `gpuverify-amd64` docker image | no aarch64 build; run under amd64 emulation via `gpu/gpuverify.sh`; needs `docker run --privileged --rm tonistiigi/binfmt --install amd64` |
| Gappa | 1.8.3 aarch64 | `~/.local/bin/gappa` | built from source against conda `gmp`/`mpfr`/Boost |

## What is proven where

| Harness | Target | Property | Status |
|---|---|---|---|
| `bmc/harness_elementwise.cpp` | `kernels/elementwise.cpp` | `add`/`scale`/`relu` match their spec element-wise; no OOB | **proved** |
| `bmc/harness_reduce.cpp` | `kernels/reduce.cpp` | `sum` = sequential accumulation; `max` ∈ input and dominates all elements | **proved** |
| `bmc/harness_gemm.cpp` | `kernels/gemm.cpp` | `C = α·A@B + β·C0` per element; A/B unmodified | **proved** |
| `smt/gemm_index_equivalence.py` | tiled GEMM index map | tiled traversal visits exactly the naive `(i,j,k)` set; every output written once; tile flattening injective (z3) | **proved** |
| `gpu/run.sh` → `harnesses/elementwise.cu` | `add_kernel`, `scale_kernel`, `relu_kernel` (mirror `kernels/elementwise.cu`) | no data races, intra- and inter-block, all schedules; `racecheck` 0 hazards (dynamic complement on GB10) | **proved** (GPUVerify, amd64-emulated) |
| `gpu/run.sh` → `harnesses/reduce.cu` | `sum_reduce`, `max_reduce` (the two instantiations of `kernels/reduce.cu`'s `reduce<Max>` template) | no data races, no barrier divergence, all schedules | **proved** (GPUVerify, amd64-emulated) |
| `gpu/run.sh` → `harnesses/gemm.cu` | `gemm_kernel` (mirror `kernels/gemm.cu`) | no data races for the launcher's bounded sizes (M,N,K ≤ 64), all schedules | **proved** (GPUVerify, amd64-emulated) |
| `gpu/run.sh` → `harnesses/comm_allreduce.cu` | `fused_reduce_stub` (mirror `comm/allreduce.cu`) | no data races, all schedules | **proved** (GPUVerify, amd64-emulated) |
| `gpu/run.sh` → `harnesses/comm_peer_copy.cu` | `peer_copy_kernel` (mirror `comm/pipeline_boundary.cu`) | no data races, all schedules | **proved** (GPUVerify, amd64-emulated) |
| `gpu/run.sh` → `harnesses/comm_kv_gather.cu` | `kv_gather_kernel` (`SlotT` = i32, i64; mirror `comm/kv_gather.cu`) | no data races on the pinned page/slot domain: grid-stride loops proven single-iteration in-domain and collapsed to guards; K/V gather index map injective across threads for any slot map | **proved** (GPUVerify, amd64-emulated) |
| `gpu/run.sh` → `harnesses/comm_kv_scatter.cu` | `kv_scatter_kernel` (`SlotT` = i32, i64; mirror `comm/kv_scatter.cu`) | same collapse argument; K/V scatter injective via pairwise slot-uniqueness contract | **proved** (GPUVerify, amd64-emulated) |
| `numerics/run.sh` | 4-term fp32 sequential sum | `|fl(Σx) − Σx| ≤ 5·2⁻²⁴` for all inputs in [−1,1] (Gappa) | **proved** |

Kernel coverage: 9 of the 20 CUDA `__global__` kernels in `src/c/vkernels`
are race-proved (12 harness entries — reduce and gather/scatter count once
per `SlotT`/op instantiation in production). Not yet covered: 7 kernels
(`gather_1d`, `gather_2d`, `convert_slots_i64_to_i32`, the four
`p2p_kv_donate`/`p2p_kv_restore` kernels and their plan kernels) that fit
the same dialect-mirror recipe, and 4 wmma kernels (`gemm_bf16` ×3,
`dsa_topk_logits_kernel_wmma`) whose fragment semantics need a sound
over-approximating model. The 52 HIP kernels (`moe`, `kda`, `dsa`, …)
would first need dialect translation. HIP kernel functionally is instead
covered by the CTest parity suite.

`vk_feq` (in `common/verify_support.h`) is NaN-tolerant IEEE equality: the
kernels promise no NaN semantics, and plain `==` would make a proof over
unconstrained nondeterministic floats fail on `NaN != NaN`.

## How the contract macro is wired

`vkernels/util/error.hpp` makes `VK_EXPECTS`/`VK_ENSURES` overridable with
`#ifndef`. The `bmc` runner predefines them as the prover's intrinsics
(`-DVK_VERIFY_CONTRACTS -D'VK_EXPECTS(cond,msg)=__ESBMC_assume(cond)'`), so
the proof ranges over exactly the contractual inputs without modeling
exceptions or `std::string`. An ordinary build leaves the macros undefined
and keeps the throwing behavior, so **production is unaffected and `src/c/`
coverage is unchanged**.

## Adding a new harness

1. Add a `harness_<kernel>.cpp` under `bmc/` that includes the real header,
   declares inputs with `VK_NONDET_FLOAT()`, calls the **real oracle
   function**, and states postconditions with `VK_ASSERT`.
2. Append the harness name and its source file to `bmc/run.sh`.
3. Confirm it type-checks under the host compiler:
   `g++ -std=c++17 -fsyntax-only -I src/c -I common bmc/harness_<kernel>.cpp`.

The harness must compile both under a prover (where the macros become proof
obligations) and under a normal compiler (where they become no-ops/asserts),
so the verification tree stays lint-clean in ordinary builds.
