# gpu — race and barrier-divergence verification

The device kernels (`src/c/vkernels/kernels/*.{cu,hip}`) are concurrent
shared-memory programs.  Testing shows they agree with the CPU oracle on the
inputs you ran; it does not show they are race-free for all schedules.  That
is the property this family targets, and it is the one the repo's history
cares about most — the gfx942 GPU-fault case files
(`docs/kernels/kda.md`, `docs/kernels/dsa.md`) are exactly shared-memory /
barrier issues.

## Static: GPUVerify

[GPUVerify](https://www.doc.ic.ac.uk/~afd/gpuverify/) applies Owicki–Gries
reasoning to CUDA/OpenCL and reports data races and barrier divergence,
independently of scheduling.

**Harnesses.** GPUVerify's 2018 frontend cannot parse the production
translation units — they pull in `<cuda_runtime.h>` and modern project headers
(C++17 nested namespaces, libstdc++-13) that the bundled clang rejects.  The
kernel bodies are therefore restated verbatim in GPUVerify's own dialect under
`harnesses/`, each with a `#include "cuda.h"` stub and a comment naming the
production source it mirrors.  Keep them in sync when a kernel changes.

```bash
verifier/gpu/gpuverify.sh --blockDim=32 --gridDim=4   verifier/gpu/harnesses/elementwise.cu
verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=1  verifier/gpu/harnesses/reduce.cu
verifier/gpu/gpuverify.sh --blockDim=16,16,1 --gridDim=2,2,1 verifier/gpu/harnesses/gemm.cu
verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=2  verifier/gpu/harnesses/comm_allreduce.cu
verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=2  verifier/gpu/harnesses/comm_peer_copy.cu
verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=1,2 --timeout=800 verifier/gpu/harnesses/comm_kv_gather.cu
verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=1,2 --timeout=800 verifier/gpu/harnesses/comm_kv_scatter.cu
```

`run.sh` runs exactly these and exits non-zero if a harness reports a race or
barrier divergence.  GPUVerify is per kernel entry; a file with several
`__global__` functions is verified entry-by-entry in one invocation.

The GEMM harness carries a guard the production kernel does not:
`if (M > 64 || N > 64 || K > 64) return;`.  GPUVerify explores the index
arithmetic for *all* sizes, and with 32-bit `int` indices a huge `N` makes
`row*N + col` overflow and alias, which it reports as a spurious race.  The
guard models the launcher's domain and makes the proof about the intended
sizes; the underlying concern (no overflow in the index map) is separately
covered by `smt/gemm_index_equivalence.py`.

The kv gather/scatter harnesses collapse the grid-stride loops to plain
guards.  In their pinned domain each of those loops runs at most one
iteration per thread, so `for (i = init; i < n; i += s)` becomes `if
(i < n)` without changing the checked semantics.  The reason is GPUVerify's
two-thread abstraction: loop-head variables are havoc'd, and for the
div/mod-derived `(t, chunk)` addressing the invariant generator cannot
re-derive the "accessBreak" facts that tie the watched offset back to
thread ids.  Loop-carried writes then surface as false write-write races
(a havoc'd instrumentation flag is never related to an actual access).
With the accesses straight-line the two-thread race check is exact, and the
`__requires` domain facts discharge it.  See the harness headers for the
soundness argument.

**Availability on aarch64.** GPUVerify is distributed for x86-64 only (last
release 2018-03-22) and is unmaintained, so there is no aarch64 build.  It runs
here under amd64 emulation:

```bash
# one-time: register amd64 binfmt (must be done on each boot / new machine)
docker run --privileged --rm tonistiigi/binfmt --install amd64

# one-time: fetch and unpack the release
curl -sL -o /tmp/gpuverify.zip \
  https://github.com/mc-imperial/gpuverify/releases/download/2018-03-22/GPUVerifyLinux64.zip
mkdir -p ~/.local/opt && unzip -q /tmp/gpuverify.zip -d /tmp/gv
mv /tmp/gv/2018-03-22 ~/.local/opt/gpuverify

# one-time: build the amd64 userland (Python 2, psutil, Mono, libtinfo5)
docker build --platform linux/amd64 -t gpuverify-amd64 verifier/gpu
```

`gpuverify.sh` then runs the bundle in that container, copying each harness to
a writable temp dir first (GPUVerify writes `.bc` files next to its inputs).
On a native x86-64 machine, plain `gpuverify` is used instead.  Override the
bundle location with `VK_GPUVERIFY_HOME` and the image name with
`VK_GPUVERIFY_IMAGE`.


## Dynamic: racecheck

The sampled complement on real hardware.  NVIDIA:

`compute-sanitizer` instruments the CUDA **runtime** through the dynamic
`libcudart`, so the test must (a) actually launch device kernels and (b)
link `libcudart` shared.  `nvcc` links it statically by default, which makes
`racecheck` print *"terminated before first instrumented API call"*; build
the test with a shared runtime instead:

```bash
PATH=/usr/local/cuda/bin:$PATH cmake --preset cuda \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -B build/cuda-racecheck -S .
cmake --build build/cuda-racecheck --target vkernels_test_cuda_contracts
compute-sanitizer --tool racecheck \
    ./build/cuda-racecheck/tests/vkernels_test_cuda_contracts
```

`vkernels_test_cuda_contracts` exercises `cuda::sum`, `cuda::max` and the
ragged `cuda::gemm`, so it drives real launches.  On this checkout the result
is `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`.  (The
host-side `vkernels_test_gemm` only exercises the CPU oracle and never
launches a kernel, so `racecheck` has nothing to instrument on it.)  Point
`run.sh` at another binary with `VK_CUDA_TEST_BIN=...`.

AMD (ROCm):

```bash
rocprof --tool racecheck -- ./build/hip/tests/vkernels_test_kda
```

These catch races that manifest on the schedule actually observed, so they
complement — never replace — GPUVerify.

## Scope note

Register-file / LDS layout and the `s_waitcnt` software pipeline have a
well-defined operational model (see the `hip-async-coordination` skill). If
GPUVerify cannot discharge a kernel, the fallback is to model the specific
pipeline (double-buffering, wait counters, phase) in a small transition
system and check its interleavings separately.
