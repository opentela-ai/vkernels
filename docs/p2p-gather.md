# P2P run-list gather

A single-launch primitive that gathers many disjoint byte-runs from peer UVA
memory into one local scratch buffer, replacing the per-run
`cudaMemcpyPeerAsync` / `cudaMemcpy2DAsync` loop that coalescers like KVAAS's
`PeerFetcher.fetch_pages_into_scratch` pay for a fragmented hit prefix.

```
csrc/vkernels/comm/p2p_gather.hpp      public API + contract
csrc/vkernels/comm/p2p_gather.cpp      host reference (the correctness oracle)
csrc/vkernels/comm/p2p_gather.cu       CUDA single-launch kernel
csrc/vkernels/comm/p2p_gather_cuda.hpp CUDA-only declarations (cudaStream_t)
csrc/vkernels/comm/p2p_gather_c.h      C ABI (extern "C", status codes)
csrc/vkernels/comm/p2p_gather_c.cu     C ABI implementation (CUDA-gated)
tests/comm/test_p2p_gather.cpp         host oracle tests
tests/comm/test_p2p_gather_c.cu       C ABI runtime tests (CUDA-gated)
benchmarks/p2p_gather_bench.cu         CUDA sweep vs the per-run loop
```

## API

### 1-D: parallel arrays

```cpp
void p2p_gather_runs(Span<std::uint8_t> dst,
                     const void* const* src_ptrs,
                     const std::size_t* dst_offsets,
                     const std::size_t* lengths,
                     std::size_t num_runs,
                     Stream* stream = nullptr);
```

For each `i` in `[0, num_runs)`, copies `lengths[i]` bytes from `src_ptrs[i]`
to `dst.data() + dst_offsets[i]`. This is the flat-array form matching the
Python `p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, *, stream=None)`.

### 2-D: strided tiles (layer-major pages)

```cpp
struct Gather2DRun {
  const void* src;       // peer row-major region
  std::size_t src_stride;// bytes between source rows (>= width)
  std::size_t dst_offset;// byte offset into dst of the tile's first row
  std::size_t dst_stride;// bytes between destination rows (>= width)
  std::size_t width;     // bytes per row copied
  std::size_t height;    // number of rows copied
};
void p2p_gather_runs_2d(Span<std::uint8_t> dst, Span<const Gather2DRun> runs,
                        Stream* stream = nullptr);
```

Each run copies a `height × width`-byte strided tile, honouring its own source
stride. `width` must not exceed either stride.

### Legacy seam (for benchmarking only)

```cpp
void memcpy_peer_batch_async(Span<std::uint8_t> dst, const void* const* src_ptrs,
                             const std::size_t* dst_offsets, const std::size_t* lengths,
                             std::size_t num_runs, Stream* stream = nullptr);
```

The predecessor: one copy per run. Same contract and lifetime; kept so
benchmarks can sweep run count against the per-run loop and so the
"single launch vs one-per-run" property is assertable on the host
(`Stream::submitted()` grows by `num_runs` here versus by `1` for
`p2p_gather_runs`).

## Contract

- `dst` is a local allocation on the executing device; `dst.size()` bounds
  every `dst_offsets[i] + lengths[i]` and every 2-D tile's last row.
- Each `src_ptrs[i]` is a peer-accessible UVA (or IPC-mapped) pointer.
- Output runs must be mutually disjoint, and each run's source range must not
  overlap its destination range. The host reference validates all of this and
  throws `std::invalid_argument` on violation; the CUDA entry point reuses the
  same `stage_runs_1d` / `stage_runs_2d` validation, so a malformed run list is
  rejected before any allocation or launch.
- `num_runs == 0` (and any run with zero length/width/height) is a valid no-op
  that enqueues nothing.

## Lifetime

- The `dst` allocation and the peer memory behind every `src_ptrs[i]` must
  outlive `stream` (the kernel/task reads them asynchronously).
- The run-metadata arrays (`src_ptrs`, `dst_offsets`, `lengths`, and the
  `Gather2DRun` list) are read and copied into owned storage before the call
  returns, so the caller may free or mutate the originals as soon as the call
  returns — only the IPC mappings and `dst`/`src` memory must persist.
- When `stream == nullptr` the work runs to completion before returning. A
  non-null stream is enqueued onto and the call returns without
  synchronising; the caller owns the ordering and completion (e.g.
  `stream->wait()`).

Peer access and any IPC mappings are the caller's responsibility: establish
them **before** the launch and hold them **until `stream` completes**. The
kernel issues direct peer reads over NVLink with no host staging of data.

## Host reference vs CUDA implementation

The CPU reference (`p2p_gather.cpp`) is the always-compiled correctness
oracle and is fully unit-tested. The CUDA path (`p2p_gather.cu`, in
`vkernels::comm::cuda`) reuses the oracle's `stage_runs_1d` / `stage_runs_2d`
for validation, then dispatches **adaptively** (issue #6):

- **Few runs** (below the crossover — 16-32 in the issue #6 table, ~3 on the
  H100 NVL box measured here): one `cudaMemcpyAsync` / `cudaMemcpy2DAsync`
  per run on the caller's stream — the copy engine wins there (no SM
  occupancy, cheap per-run driver calls), and this is byte-for-byte the
  baseline the primitive is measured against.
- **Many runs** (at/above the crossover): the run descriptors are staged to a
  per-launch device buffer (allocated with `cudaMallocAsync` on the caller's
  stream) and **exactly one kernel** is launched — there are no per-run CUDA
  API calls after the metadata is prepared.

The decision is the pure, host-tested function `prefer_gather_kernel(num_runs,
 total_bytes, strided)` with a cost model fitted to H100 NVL measurements
(sgs-gpu07, CUDA 13 / driver 580.82.07, real NVLink peer reads GPU1->GPU0):

- copy engine ≈ `max(20 µs, 4.20 µs/MiB) + 7.37 µs/run` (2-D:
  `max(10.75 µs, 4.20 µs/MiB) + 7.30 µs/run` — the engine's 2-D setup is
  cheaper);
- gather kernel ≈ `max(8.6 µs, 4.20 µs/MiB)` flat in run count (2-D:
  `14.0 µs + 0.13 µs/run + 4.20 µs/MiB` — one block per row);
- a 4-run floor, applied only at ≥1 MiB 1-D payloads, keeps the 1-2 run
  cases on the copy engine where the measured margins are ~1% (inside
  noise); below 1 MiB the engine never wins (its fixed floor dominates),
  so the model decides from one run (1.75x kernel win at 4 KiB).

`set_gather_dispatch(mode, min_runs)` overrides the model for testing
(`kForceKernel` / `kForceCopyEngine`) and A/B tuning on a target machine.

The default memory pool is tuned once (`cudaMemPoolSetAttribute` with
`cudaMemPoolAttrReleaseThreshold = UINT64_MAX`) so freed memory stays in the
pool and a subsequent `cudaMallocAsync` is a pool-internal pointer bump
rather than a driver memory-acquisition syscall. This keeps the per-launch
staging allocation cheap even for very large run lists, and — unlike a
mutable `__constant__` symbol — is safe across concurrent streams (a
second launch gets its own allocation that cannot be overwritten while the
first stream's kernel is still reading it).

The kernel copies 16 bytes per thread (`uint4` load/store) for runs whose
source, destination and (2-D) strides are 16-byte aligned, with a <16-byte
scalar tail handled by one thread per run/row; unaligned runs fall back to
the byte-per-thread path. The vectorizable flag is computed once on the host
at staging time, so the kernel does no per-thread alignment arithmetic.
Grid units are 16 bytes (vectorized) or 1 byte (scalar) per thread, so one
grid sized to the largest unit count covers mixed lists.

Kernel grid (current limits):
- 1-D: `grid.x` tiles the longest run's unit range (256 threads/block),
  `grid.y = num_runs`. Capped at 65535 runs (the grid-y limit).
- 2-D: `grid.x` tiles the row `width`, `grid.y = max_height` across runs,
  `grid.z = num_runs`. Capped at 65535 runs and 65535 rows independently.

## Prepared plan API (reuse across layer launches)

KVAAS resolves one run list and reuses it for all 40 layers, but the one-shot
functions repeat validation, host-vector construction, device metadata
allocation, H2D metadata copy and free for every layer. A plan moves all of
that to a single prepare step:

```cpp
// Host reference (always compiled) and CUDA implementation both expose:
//   P2PGatherPlan1D / P2PGatherPlan2D (host, vkernels::comm)
//   vkernels::comm::cuda::P2PGatherPlan1D / _2D (CUDA, cudaStream_t)
//
// Construct:  validate the run list ONCE (throws on violation), copy the
//             descriptors into owned storage, and — CUDA — allocate a
//             persistent per-device buffer and upload the descriptors with
//             a synchronous cudaMemcpy (one-time cost, no cross-stream
//             race because there is no stream association).
// execute():  enqueue only — no validation, no allocation, no H2D copy.
//             CUDA: adaptive dispatch (copy engine / kernel) exactly like
//             the one-shot functions. Safe concurrently on several streams
//             because the plan's metadata is read-only after construction.
```

The plan is bound to one destination allocation (the scratch buffer KVAAS
reuses across layers); a plan for a different destination is prepared
separately. Lifetime: destroy the plan only after every stream it was
executed on has been synchronised (the persistent buffer is freed in the
destructor).

**2-D layer-relative plans** (KVAAS pattern, issue #8): a prepared 2-D
plan stores the construction-time source base pointers (peer page bases,
without the per-layer offset). `execute(src_byte_offset, stream)` adds the
scalar offset to every run's source pointer before copying. Both dispatch
branches apply it — the kernel as a scalar parameter, the copy engine as a
pointer offset — so no per-layer H2D descriptor upload is needed.

```cpp
// CUDA plan, the KVAAS 40-layer pattern:
std::vector<std::uint8_t> scratch(48 * 1024 * 1024);  // bound destination
// 1-D:
P2PGatherPlan1D plan(scratch.data(), scratch.size(), srcs, offs, lens, runs);
for (int layer = 0; layer < 40; ++layer) plan.execute(stream);  // enqueue only

// 2-D layer-relative (one topology, shifting source for every layer):
P2PGatherPlan2D plan2d(scratch.data(), scratch.size(), page_runs, num_pages);
for (int layer = 0; layer < 40; ++layer)
  plan2d.execute(layer * layer_bytes, stream);  // offset applied, no H2D
```

When the offset is a multiple of 16 the vectorized kernel path stays
enabled (prepare-time alignment is preserved); an unaligned offset falls
back to the scalar path and grid.x is sized to max width to cover every
row. `cudaMemcpy2DAsync` handles any alignment natively.

## C ABI

Non-C++ consumers (e.g. a serving runtime that links the primitive without
the C++ headers) use the `extern "C"` entry points declared in
`p2p_gather_c.h` and exported by the CUDA-gated `libvkernels_c` shared
library. They take raw pointers and a `cudaStream_t` and return a
`vkernels_status_t` code (never throw across the boundary):

```c
vkernels_status_t vkernels_p2p_gather_runs(
    uint8_t* dst, size_t dst_capacity,
    const void* const* src_ptrs, const size_t* dst_offsets,
    const size_t* lengths, size_t num_runs, cudaStream_t stream);

vkernels_status_t vkernels_p2p_gather_runs_2d(
    uint8_t* dst, size_t dst_capacity,
    const vkernels_gather_2d_run_t* runs, size_t num_runs,
    cudaStream_t stream);
```

The plan API is exposed as opaque handles for the same non-C++ consumers:

```c
vkernels_p2p_plan_1d_t* vkernels_p2p_plan_1d_create(
    uint8_t* dst, size_t dst_capacity, const void* const* src_ptrs,
    const size_t* dst_offsets, const size_t* lengths, size_t num_runs,
    vkernels_status_t* status_out);   // NULL + status on validation failure
void     vkernels_p2p_plan_1d_destroy(vkernels_p2p_plan_1d_t* plan);
vkernels_status_t vkernels_p2p_plan_1d_execute(vkernels_p2p_plan_1d_t* plan,
                                              cudaStream_t stream);
// vkernels_p2p_plan_2d_create / _destroy / _execute mirror the 1-D trio.

// Layer-relative 2-D execute (KVAAS reuse pattern): adds src_byte_offset
// to every run's source pointer before copying.
vkernels_status_t vkernels_p2p_plan_2d_execute_offset(
    vkernels_p2p_plan_2d_t* plan, size_t src_byte_offset, cudaStream_t stream);
```

`vkernels_gather_2d_run_t` is layout-compatible with `Gather2DRun`. A C++
exception thrown by the staging validators (`std::invalid_argument` via
`VK_EXPECTS`) or the launch path (`std::runtime_error` via `VK_ENSURES`) is
caught inside the wrapper and mapped to `VKERNELS_ERR_INVALID_ARGUMENT` or
`VKERNELS_ERR_INTERNAL` respectively.

## Benchmark

```sh
cmake --preset cuda -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build --preset cuda
./build/cuda/benchmarks/p2p_gather_bench            # idle GPU
./build/cuda/benchmarks/p2p_gather_bench --concurrent  # persistent filler on a 2nd stream
VK_BENCH_FILL_BLOCKS=1024 ./build/cuda/benchmarks/p2p_gather_bench --concurrent  # full-occupancy load
```

The `--concurrent` filler is a persistent kernel occupying one block per SM
(`VK_BENCH_FILL_BLOCKS` to change) for the whole bench, released by a
volatile stop flag — a finite burst of fill kernels drains before the
measured sections and silently measures idle (see
`docs/performance/p2p-gather/h100-nvl.md` for the nsys evidence). The
bench allocates its scratch buffers once and warms up the pool tuning and
both kernels before the filler: synchronous `cudaMalloc`/`cudaFree` and
the first-use `cudaMemPoolSetAttribute` wait for device quiescence and
would block for the filler's whole lifetime.

The benchmark covers the issue #6 acceptance criteria: a fixed 48 MiB
payload swept across 1, 2, 4, 8, 16, 32, 64 and 192 runs (bracketing the
measured 16-32 crossover), the existing 4 MiB fragmentation and 4 KiB page
sweeps, and 2-D strided tiles. Each row reports the per-run copy-engine
baseline, the forced kernel, the adaptive path (and which branch it took),
and — separately — the host enqueue time (validation + descriptor
construction + metadata allocation + H2D enqueue + launch), so the
preparation/allocation contribution is visible next to kernel execution. A
final section prepares a plan ONCE and times 40 executes, the KVAAS
layer-reuse pattern.

Timing is raw `cudaEvent` (device) + `steady_clock` (host enqueue) with
warmup and a median over iterations. On a single-GPU box the peer source is
another device allocation and the baseline uses `cudaMemcpyDeviceToDevice`;
on a multi-GPU system point `src_ptrs[i]` at peer memory and enable peer
access — the kernel and measurement are unchanged.

Measured results so far live in `docs/performance/p2p-gather/` (per
architecture): `gb10.md` for NVIDIA GB10, `h100-nvl.md` for the H100 NVL
machine (issue #6 environment) with the baseline table and the acceptance
sweep to be re-run on the target.
