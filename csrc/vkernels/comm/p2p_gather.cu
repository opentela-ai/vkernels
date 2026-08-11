// vkernels/comm/p2p_gather.cu — CUDA single-launch P2P/UVA run-list gather.
//
// The host reference (p2p_gather.cpp) is the correctness oracle and carries
// the full contract checks. This CUDA path is the performance implementation:
// it reuses the oracle's `stage_runs_1d` / `stage_runs_2d` to validate and
// stage the run metadata ONCE on the host, then launches exactly one kernel
// that reads every `src` directly from peer memory over NVLink (peer access
// is enabled by the caller). There are no per-run CUDA API calls after the
// metadata is prepared — that is the overhead the per-run
// cudaMemcpyPeerAsync / cudaMemcpy2DAsync loop paid.
//
// Run descriptors are staged into a PER-LAUNCH device buffer allocated with
// cudaMallocAsync on the caller's stream and freed with cudaFreeAsync before
// the entry point returns (the kernel is already enqueued). This is the key
// design choice over an earlier __constant__ approach:
//
//   * Correctness on every arch. Taking the address of a __constant__ symbol
//     and passing it as a generic `const T*` kernel argument faults with
//     cudaErrorIllegalAddress on Hopper (sm_90); a plain device pointer does
//     not. (Every block still reads the SAME run, so the metadata lands in L2
//     after the first few blocks — the constant-cache broadcast property is
//     preserved in practice.)
//   * Concurrency. A single mutable __constant__ symbol is unsafe across
//     concurrent streams (a second launch can overwrite descriptors while the
//     first stream's kernel is still reading them). A per-launch allocation
//     is private to one launch and therefore safe to overlap.
//
// To keep the per-launch allocation cheap, the default memory pool is tuned
// ONCE (release threshold = UINT64_MAX) so freed memory stays in the pool and
// a subsequent cudaMallocAsync is a pool-internal pointer bump rather than a
// driver memory-acquisition syscall.
//
// The CUDA entry points live in `vkernels::comm::cuda` (mirroring the
// elementwise `cuda::` split) so they coexist with the host reference as
// separate symbols. The C ABI (p2p_gather_c.h / p2p_gather_c.cu) wraps these
// for non-C++ consumers, returning error codes instead of throwing.
#include "vkernels/comm/p2p_gather.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/p2p_gather_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <cstddef>
#  include <cstdint>
#  include <limits>
#  include <mutex>
#  include <vector>

namespace vkernels::comm {
namespace cuda {
namespace {

// Hardware grid-axis limits used to bound input ranges up front (the launch
// would fail anyway; checking gives a clear contract violation instead).
constexpr unsigned kMaxGridAxis = 65535u;

// Fixed-width device descriptor (8-byte fields keep host/device layout stable
// across the host->device copy). `dst` is the resolved absolute destination
// pointer (scratch base + dst_offset) so the kernel needs no per-run offset
// arithmetic.
struct RunDev1D {
  const unsigned char* src;
  unsigned char* dst;
  unsigned long long length;
};

struct RunDev2D {
  const unsigned char* src;
  unsigned char* dst;
  unsigned long long src_stride;
  unsigned long long dst_stride;
  unsigned long long width;
  unsigned long long height;
};

// Tune the default memory pool once so freed memory is retained: repeated
// per-launch cudaMallocAsync/cudaFreeAsync pairs become cheap pool-internal
// pointer bumps instead of driver syscalls. Idempotent; std::call_once makes
// the one-time guarantee explicit and data-race-free.
void tune_default_pool_once() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) return;
    std::uint64_t keep = UINT64_MAX;  // never release freed memory to the OS
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
  });
}

// 1-D gather. blockIdx.y selects the run; blockIdx.x tiles its byte range.
// Shorter runs' trailing threads return via the `idx < length` guard, so the
// grid can be sized to the longest run. `runs` is a plain device pointer
// (read from L2 by every block of the same run), not a __constant__ symbol,
// so it is correct on every architecture and safe across concurrent streams.
__global__ void gather_1d_kernel(const RunDev1D* __restrict__ runs, int num_runs) {
  int r = blockIdx.y;
  if (r >= num_runs) return;
  RunDev1D run = runs[r];
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < run.length) run.dst[idx] = run.src[idx];  // direct peer read over NVLink
}

// 2-D gather. blockIdx.z selects the run, blockIdx.y the row within it, and
// blockIdx.x tiles the row's `width` bytes. Using grid.z (runs) and grid.y
// (rows) independently avoids the combined (run*row) encoding, which would
// overflow the grid.y limit for tall tiles. Each block compares its row
// against the run's own `height` (read from the descriptor), so no separate
// max-height parameter is needed and there is no narrowing int cast.
__global__ void gather_2d_kernel(const RunDev2D* __restrict__ runs, int num_runs) {
  int r = blockIdx.z;
  unsigned long long row = static_cast<unsigned long long>(blockIdx.y);
  if (r >= num_runs) return;
  RunDev2D run = runs[r];
  if (row >= run.height) return;  // run shorter than the grid's row extent
  unsigned long long col =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col < run.width)
    run.dst[row * run.dst_stride + col] = run.src[row * run.src_stride + col];
}

// Number of 256-byte blocks covering `n` bytes (at least one). `n` is bounded
// by the caller to stay well under the grid.x limit (INT_MAX blocks).
inline int tiles(std::size_t n) {
  VK_EXPECTS(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()) * 256u,
             "byte range exceeds the grid.x block limit");
  const std::size_t t = (n + 255u) / 256u;
  return t > 0 ? static_cast<int>(t) : 1;
}

// RAII wrapper over a stream-ordered device allocation so the staging buffer
// is freed even when a subsequent VK_ENSURES throws. The destructor issues
// cudaFreeAsync (which returns an error code, never throws). Tuning the
// default pool here makes this allocation cheap after the first call.
struct ScopedDevBuf {
  void* ptr = nullptr;
  cudaStream_t stream = nullptr;
  explicit ScopedDevBuf(std::size_t bytes, cudaStream_t s) : stream(s) {
    tune_default_pool_once();
    cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for run metadata failed");
  }
  ~ScopedDevBuf() {
    if (ptr) cudaFreeAsync(ptr, stream);
  }
  ScopedDevBuf(const ScopedDevBuf&) = delete;
  ScopedDevBuf& operator=(const ScopedDevBuf&) = delete;
};

}  // namespace

void p2p_gather_runs(std::uint8_t* dst, std::size_t dst_capacity,
                     const void* const* src_ptrs, const std::size_t* dst_offsets,
                     const std::size_t* lengths, std::size_t num_runs,
                     cudaStream_t stream) {
  auto staged = stage_runs_1d(dst, dst_capacity, src_ptrs, dst_offsets, lengths, num_runs);
  if (staged.empty()) return;  // nothing to copy, no launch

  std::vector<RunDev1D> dev;
  dev.reserve(staged.size());
  std::size_t max_len = 0;
  for (const auto& r : staged) {
    dev.push_back({r.src, dst + r.dst_offset, static_cast<unsigned long long>(r.length)});
    if (max_len < r.length) max_len = r.length;
  }
  VK_EXPECTS(dev.size() <= kMaxGridAxis, "too many runs for the grid.y axis");

  const std::size_t bytes = dev.size() * sizeof(RunDev1D);
  ScopedDevBuf meta(bytes, stream);
  cudaError_t err = cudaMemcpyAsync(meta.ptr, dev.data(), bytes,
                                    cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for run metadata failed");

  dim3 block(256);
  dim3 grid(tiles(max_len), static_cast<unsigned>(dev.size()));
  gather_1d_kernel<<<grid, block, 0, stream>>>(
      static_cast<const RunDev1D*>(meta.ptr), static_cast<int>(dev.size()));
  err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda gather_1d launch failed");
}

void p2p_gather_runs_2d(std::uint8_t* dst, std::size_t dst_capacity, const Gather2DRun* runs,
                        std::size_t num_runs, cudaStream_t stream) {
  auto staged = stage_runs_2d(dst, dst_capacity, runs, num_runs);
  if (staged.empty()) return;

  std::vector<RunDev2D> dev;
  dev.reserve(staged.size());
  std::size_t max_w = 0, max_h = 0;
  for (const auto& r : staged) {
    dev.push_back({r.src, dst + r.dst_offset,
                   static_cast<unsigned long long>(r.src_stride),
                   static_cast<unsigned long long>(r.dst_stride),
                   static_cast<unsigned long long>(r.width),
                   static_cast<unsigned long long>(r.height)});
    if (max_w < r.width) max_w = r.width;
    if (max_h < r.height) max_h = r.height;
  }
  VK_EXPECTS(dev.size() <= kMaxGridAxis, "too many runs for the grid.z axis");
  VK_EXPECTS(max_h <= kMaxGridAxis, "tile height exceeds the grid.y axis");

  const std::size_t bytes = dev.size() * sizeof(RunDev2D);
  ScopedDevBuf meta(bytes, stream);
  cudaError_t err = cudaMemcpyAsync(meta.ptr, dev.data(), bytes,
                                    cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for 2-D run metadata failed");

  dim3 block(256);
  dim3 grid(tiles(max_w), static_cast<unsigned>(max_h), static_cast<unsigned>(dev.size()));
  gather_2d_kernel<<<grid, block, 0, stream>>>(
      static_cast<const RunDev2D*>(meta.ptr), static_cast<int>(dev.size()));
  err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda gather_2d launch failed");
}

}  // namespace cuda
}  // namespace vkernels::comm

#endif  // VKERNELS_HAS_CUDA
