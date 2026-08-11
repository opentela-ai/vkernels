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
for validation, then stages the run descriptors to a per-launch device
buffer (allocated with `cudaMallocAsync` on the caller's stream) and launches
exactly one kernel — there are no per-run CUDA API calls after the metadata
is prepared.

The default memory pool is tuned once (`cudaMemPoolSetAttribute` with
`cudaMemPoolAttrReleaseThreshold = UINT64_MAX`) so freed memory stays in the
pool and a subsequent `cudaMallocAsync` is a pool-internal pointer bump
rather than a driver memory-acquisition syscall. This keeps the per-launch
staging allocation cheap even for very large run lists, and — unlike a
mutable `__constant__` symbol — is safe across concurrent streams (a
second launch gets its own allocation that cannot be overwritten while the
first stream's kernel is still reading it).

Kernel grid (current limits):
- 1-D: `grid.x` tiles the longest run's byte range (256 threads/block),
  `grid.y = num_runs`. Capped at 65535 runs (the grid-y limit).
- 2-D: `grid.x` tiles the row `width`, `grid.y = max_height` across runs,
  `grid.z = num_runs`. Capped at 65535 runs and 65535 rows independently.

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

`vkernels_gather_2d_run_t` is layout-compatible with `Gather2DRun`. A C++
exception thrown by the staging validators (`std::invalid_argument` via
`VK_EXPECTS`) or the launch path (`std::runtime_error` via `VK_ENSURES`) is
caught inside the wrapper and mapped to `VKERNELS_ERR_INVALID_ARGUMENT` or
`VKERNELS_ERR_INTERNAL` respectively.

## Benchmark

```sh
cmake --preset cuda -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build --preset cuda
./build/cuda/benchmarks/p2p_gather_bench
```

The benchmark sweeps run count and size (1-D, 4 MiB total — fragmentation —
and fixed 4 KiB pages — count) and 2-D strided tiles, timing the
single-launch kernel against a per-run `cudaMemcpyAsync` / `cudaMemcpy2DAsync`
loop with raw `cudaEvent` and a median over iterations. On a single-GPU box
the peer source is another device allocation and the baseline uses
`cudaMemcpyDeviceToDevice`; on a multi-GPU system point `src_ptrs[i]` at peer
memory and enable peer access — the kernel and measurement are unchanged.
