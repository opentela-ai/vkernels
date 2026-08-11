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
// Run descriptors live in `__constant__` memory. Every thread in a block
// reads the SAME run (blockIdx.y/z selects it), so the read is a uniform
// broadcast — ideal for the constant cache — and `cudaMemcpyToSymbolAsync`
// is a single cheap driver call with NO per-launch allocation. Lists larger
// than the constant cap fall back to a stream-ordered `cudaMallocAsync`
// staging buffer; that path still launches one kernel, just with a metadata
// allocation that large (multi-thousand-run) lists justify.
//
// The CUDA entry points live in `vkernels::comm::cuda` (mirroring the
// elementwise `cuda::` split) so they coexist with the host reference as
// separate symbols; CUDA test and benchmark targets call them directly with a
// `cudaStream_t`.
#include "vkernels/comm/p2p_gather.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/p2p_gather_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <cstddef>
#  include <optional>
#  include <vector>

namespace vkernels::comm {
namespace cuda {
namespace {

// Fixed-width device descriptor (8-byte fields keep host/device layout stable
// across the host->constant copy). `dst` is the resolved absolute destination
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

// Run descriptors live in ONE __constant__ symbol shared by the 1-D and 2-D
// paths (they are never active in the same launch). A union of POD arrays is
// the standard CUDA idiom for bounded per-launch metadata that every thread
// in a block reads uniformly (constant-cache broadcast), and `cudaMemcpyToSymbol`
// is a single cheap driver call with no per-launch allocation. Lists larger
// than the cap fall back to a stream-ordered `cudaMallocAsync` buffer.
constexpr int kMaxConst1D = 2048;  // 48 KiB; 1-D is the primary use case
constexpr int kMaxConst2D = 1024;  // 48 KiB; both fit in one 64 KiB bank

union RunDescConst {
  RunDev1D d1[kMaxConst1D];  // 2048 * 24 B = 48 KiB
  RunDev2D d2[kMaxConst2D];  // 1024 * 48 B = 48 KiB
};
__constant__ RunDescConst c_runs;

// 1-D gather. blockIdx.y selects the run; blockIdx.x tiles its byte range.
// Shorter runs' trailing threads return via the `idx < length` guard, so the
// grid can be sized to the longest run.
__global__ void gather_1d_kernel(const RunDev1D* __restrict__ runs, int num_runs) {
  int r = blockIdx.y;
  if (r >= num_runs) return;
  RunDev1D run = runs[r];
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < run.length) run.dst[idx] = run.src[idx];  // direct peer read over NVLink
}

// 2-D gather. blockIdx.z selects the run, blockIdx.y the row within it, and
// blockIdx.x tiles the row's `width` bytes. Using grid.z (capped at 65535
// runs) and grid.y (capped at 65535 rows) independently avoids the combined
// (run*row) encoding, which would overflow the grid.y limit for tall tiles.
__global__ void gather_2d_kernel(const RunDev2D* __restrict__ runs, int num_runs,
                                 int max_height) {
  int r = blockIdx.z;
  int row = blockIdx.y;
  if (r >= num_runs) return;
  RunDev2D run = runs[r];
  if (row >= static_cast<int>(run.height)) return;  // run shorter than max_height
  unsigned long long col =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col < run.width)
    run.dst[row * run.dst_stride + col] = run.src[row * run.src_stride + col];
}

// Number of 256-byte blocks covering `n` bytes (at least one).
inline int tiles(std::size_t n) {
  return static_cast<int>((n + 255u) / 256u) > 0 ? static_cast<int>((n + 255u) / 256u) : 1;
}

// RAII wrapper over a stream-ordered device allocation so the staging buffer
// is freed even when a subsequent VK_ENSURES throws. The destructor issues
// cudaFreeAsync (which returns an error code, never throws).
struct ScopedDevBuf {
  void* ptr = nullptr;
  cudaStream_t stream = nullptr;
  explicit ScopedDevBuf(std::size_t bytes, cudaStream_t s) : stream(s) {
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
  const std::size_t bytes = dev.size() * sizeof(RunDev1D);

  const RunDev1D* runs_ptr = nullptr;
  std::optional<ScopedDevBuf> fallback;  // device buffer, only for > cap lists
  if (dev.size() <= static_cast<std::size_t>(kMaxConst1D)) {
    // Common case: stage into __constant__ memory (one cheap driver call,
    // no allocation; uniform block reads hit the constant cache).
    cudaError_t err = cudaMemcpyToSymbolAsync(c_runs, dev.data(), bytes, 0,
                                              cudaMemcpyHostToDevice, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyToSymbolAsync for run metadata failed");
    runs_ptr = c_runs.d1;
  } else {
    // Very large list: stage into a fresh device buffer. Still one kernel.
    fallback.emplace(bytes, stream);
    cudaError_t err = cudaMemcpyAsync(fallback->ptr, dev.data(), bytes,
                                      cudaMemcpyHostToDevice, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for run metadata failed");
    runs_ptr = static_cast<const RunDev1D*>(fallback->ptr);
  }

  dim3 block(256);
  dim3 grid(tiles(max_len), static_cast<unsigned>(dev.size()));
  gather_1d_kernel<<<grid, block, 0, stream>>>(runs_ptr, static_cast<int>(dev.size()));
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda gather_1d launch failed");
}

void p2p_gather_runs_2d(std::uint8_t* dst, std::size_t dst_capacity, const Gather2DRun* runs,
                        std::size_t num_runs, cudaStream_t stream) {
  auto staged = stage_runs_2d(dst_capacity, runs, num_runs);
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
  const std::size_t bytes = dev.size() * sizeof(RunDev2D);

  const RunDev2D* runs_ptr = nullptr;
  std::optional<ScopedDevBuf> fallback;
  if (dev.size() <= static_cast<std::size_t>(kMaxConst2D)) {
    cudaError_t err = cudaMemcpyToSymbolAsync(c_runs, dev.data(), bytes, 0,
                                              cudaMemcpyHostToDevice, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyToSymbolAsync for 2-D run metadata failed");
    runs_ptr = c_runs.d2;
  } else {
    fallback.emplace(bytes, stream);
    cudaError_t err = cudaMemcpyAsync(fallback->ptr, dev.data(), bytes,
                                      cudaMemcpyHostToDevice, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for 2-D run metadata failed");
    runs_ptr = static_cast<const RunDev2D*>(fallback->ptr);
  }

  const int mh = static_cast<int>(max_h > 0 ? max_h : 1);
  dim3 block(256);
  dim3 grid(tiles(max_w), static_cast<unsigned>(mh), static_cast<unsigned>(dev.size()));
  gather_2d_kernel<<<grid, block, 0, stream>>>(runs_ptr, static_cast<int>(dev.size()), mh);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda gather_2d launch failed");
}

}  // namespace cuda
}  // namespace vkernels::comm

#endif  // VKERNELS_HAS_CUDA
