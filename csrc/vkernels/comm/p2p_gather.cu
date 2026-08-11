// vkernels/comm/p2p_gather.cu — CUDA single-launch P2P/UVA run-list gather.
//
// The host reference (p2p_gather.cpp) is the correctness oracle and carries
// the full contract checks. This CUDA path is the performance implementation:
// it reuses the oracle's `stage_runs_1d` / `stage_runs_2d` to validate and
// stage the run metadata ONCE on the host, then dispatches ADAPTIVELY (see
// prefer_gather_kernel, issue #6):
//
//   * Few runs (below the measured 16-32 crossover on H100 NVL): one
//     cudaMemcpyAsync / cudaMemcpy2DAsync per run on the caller's stream.
//     The copy engine wins there — it pays no SM occupancy and the per-run
//     driver calls are cheap — and this is byte-for-byte the baseline the
//     primitive is measured against.
//   * Many runs (at or above the crossover): exactly one kernel launch that
//     reads every `src` directly from peer memory over NVLink. There are no
//     per-run CUDA API calls after the metadata is prepared.
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
// The P2PGatherPlan* classes extend the same metadata machinery into a
// prepared-plan API (issue #6): validation, descriptor construction and the
// device metadata upload happen ONCE at construction into a persistent
// per-device buffer (plain cudaMalloc + synchronous cudaMemcpy — no stream
// association, so concurrent execute() on arbitrary streams is race-free),
// and execute() only enqueues. This is the KVAAS pattern: one run list
// reused across 40 layer launches with zero per-layer allocation or H2D
// copies.
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
// arithmetic. `vec` marks runs whose src and dst are both 16-byte aligned
// and at least 16 bytes long; those take the vectorized uint4 path (one
// 16-byte chunk per thread plus a <16-byte scalar tail), the rest stay on
// the byte-per-thread path. The flag is computed once on the host, so the
// kernel does no per-thread alignment arithmetic.
struct RunDev1D {
  const unsigned char* src;
  unsigned char* dst;
  unsigned long long length;
  bool vec;
};

struct RunDev2D {
  const unsigned char* src;
  unsigned char* dst;
  unsigned long long src_stride;
  unsigned long long dst_stride;
  unsigned long long width;
  unsigned long long height;
  bool vec;
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

// 1-D gather. blockIdx.y selects the run; blockIdx.x tiles its unit range.
// Units are 16 bytes for vectorized runs (ceil(length/16)) and 1 byte for
// scalar runs, so one grid sized to the largest unit count covers mixed
// lists. Shorter runs' trailing units return via the unit guard, so the
// grid can be sized to the longest run. `runs` is a plain device pointer
// (read from L2 by every block of the same run), not a __constant__ symbol,
// so it is correct on every architecture and safe across concurrent streams.
//
// Vectorized path: thread `i` copies bytes [16i, 16i+16) with a uint4 load
// and store; the single thread at i == length/16 copies the <16-byte scalar
// tail at the end of the run. The tail thread always exists in the grid
// because ceil(length/16) units are covered by construction.
__global__ void gather_1d_kernel(const RunDev1D* __restrict__ runs, int num_runs) {
  int r = blockIdx.y;
  if (r >= num_runs) return;
  RunDev1D run = runs[r];
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (run.vec) {
    const unsigned long long chunks = run.length >> 4u;  // full 16-byte chunks
    if (i < chunks) {
      const unsigned long long off = i << 4u;
      *reinterpret_cast<uint4*>(run.dst + off) =
          *reinterpret_cast<const uint4*>(run.src + off);
    }
    const unsigned tail = static_cast<unsigned>(run.length & 15u);
    if (i == chunks && tail != 0u) {  // exactly one thread per run: the tail
      const unsigned long long off = chunks << 4u;
      for (unsigned k = 0; k < tail; ++k) run.dst[off + k] = run.src[off + k];
    }
  } else if (i < run.length) {
    run.dst[i] = run.src[i];  // direct peer read over NVLink
  }
}

// 2-D gather. blockIdx.z selects the run, blockIdx.y the row within it, and
// blockIdx.x tiles the row's `width` in the same unit scheme as the 1-D
// kernel. Using grid.z (runs) and grid.y (rows) independently avoids the
// combined (run*row) encoding, which would overflow the grid.y limit for
// tall tiles. Each block compares its row against the run's own `height`
// (read from the descriptor), so no separate max-height parameter is needed
// and there is no narrowing int cast.
__global__ void gather_2d_kernel(const RunDev2D* __restrict__ runs, int num_runs) {
  int r = blockIdx.z;
  unsigned long long row = static_cast<unsigned long long>(blockIdx.y);
  if (r >= num_runs) return;
  RunDev2D run = runs[r];
  if (row >= run.height) return;  // run shorter than the grid's row extent
  unsigned char* drow = run.dst + row * run.dst_stride;
  const unsigned char* srow = run.src + row * run.src_stride;
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (run.vec) {
    const unsigned long long chunks = run.width >> 4u;
    if (i < chunks) {
      const unsigned long long off = i << 4u;
      *reinterpret_cast<uint4*>(drow + off) =
          *reinterpret_cast<const uint4*>(srow + off);
    }
    const unsigned tail = static_cast<unsigned>(run.width & 15u);
    if (i == chunks && tail != 0u) {  // one thread per row: the row's tail
      const unsigned long long off = chunks << 4u;
      for (unsigned k = 0; k < tail; ++k) drow[off + k] = srow[off + k];
    }
  } else if (i < run.width) {
    drow[i] = srow[i];
  }
}

// Number of 256-thread blocks covering `units` units (at least one). `units`
// is bounded by the caller to stay well under the grid.x limit (INT_MAX
// blocks).
inline int tiles(std::size_t units) {
  VK_EXPECTS(units <= static_cast<std::size_t>(std::numeric_limits<int>::max()) * 256u,
             "unit range exceeds the grid.x block limit");
  const std::size_t t = (units + 255u) / 256u;
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

// 16-byte alignment of a byte pointer.
inline bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 0xF) == 0;
}

// Number of grid units a 1-D run occupies: ceil(length/16) on the vectorized
// path (one 16-byte chunk per thread), `length` on the scalar path (one byte
// per thread).
inline std::size_t units_1d(const StagedRun1D& r, bool vec) {
  return vec ? (r.length + 15u) / 16u : r.length;
}

// Number of grid units a 2-D run's row occupies: ceil(width/16) vectorized,
// `width` scalar.
inline std::size_t units_2d(const StagedRun2D& r, bool vec) {
  return vec ? (r.width + 15u) / 16u : r.width;
}

// Launch the 1-D gather kernel for an already-staged descriptor list.
// `d_runs` points at `num_runs` device descriptors (owned by the caller —
// per-launch ScopedDevBuf for the one-shot path, the plan's persistent
// buffer for the plan path). `max_units` is the largest run's grid-unit
// count, which sizes grid.x so every run's units are covered.
void launch_gather_1d(const RunDev1D* d_runs, std::size_t num_runs,
                      std::size_t max_units, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(tiles(max_units), static_cast<unsigned>(num_runs));
  gather_1d_kernel<<<grid, block, 0, stream>>>(d_runs, static_cast<int>(num_runs));
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda gather_1d launch failed");
}

// Launch the 2-D gather kernel for an already-staged descriptor list.
void launch_gather_2d(const RunDev2D* d_runs, std::size_t num_runs,
                      std::size_t max_units, std::size_t max_height,
                      cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(tiles(max_units), static_cast<unsigned>(max_height),
            static_cast<unsigned>(num_runs));
  gather_2d_kernel<<<grid, block, 0, stream>>>(d_runs, static_cast<int>(num_runs));
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda gather_2d launch failed");
}

// Adaptive dispatch target for one-shot launches: build the device
// descriptors, upload them with one stream-ordered H2D copy, and launch the
// kernel. Callers check prefer_gather_kernel() first; this is the
// many-runs (fragmented) path.
void kernel_1d_oneshot(std::uint8_t* dst, const std::vector<StagedRun1D>& staged,
                       cudaStream_t stream) {
  std::vector<RunDev1D> dev;
  dev.reserve(staged.size());
  std::size_t max_units = 0;
  for (const auto& r : staged) {
    const bool vec =
        aligned16(r.src) && aligned16(dst + r.dst_offset) && r.length >= 16;
    dev.push_back({r.src, dst + r.dst_offset,
                   static_cast<unsigned long long>(r.length), vec});
    const std::size_t u = units_1d(r, vec);
    if (max_units < u) max_units = u;
  }
  VK_EXPECTS(dev.size() <= kMaxGridAxis, "too many runs for the grid.y axis");

  const std::size_t bytes = dev.size() * sizeof(RunDev1D);
  ScopedDevBuf meta(bytes, stream);
  cudaError_t err = cudaMemcpyAsync(meta.ptr, dev.data(), bytes,
                                    cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for run metadata failed");
  launch_gather_1d(static_cast<const RunDev1D*>(meta.ptr), dev.size(), max_units,
                   stream);
}

void kernel_2d_oneshot(std::uint8_t* dst, const std::vector<StagedRun2D>& staged,
                       cudaStream_t stream) {
  std::vector<RunDev2D> dev;
  dev.reserve(staged.size());
  std::size_t max_units = 0, max_h = 0;
  for (const auto& r : staged) {
    const bool vec = aligned16(r.src) && aligned16(dst + r.dst_offset) &&
                     (r.src_stride % 16u == 0) && (r.dst_stride % 16u == 0) &&
                     r.width >= 16;
    dev.push_back({r.src, dst + r.dst_offset,
                   static_cast<unsigned long long>(r.src_stride),
                   static_cast<unsigned long long>(r.dst_stride),
                   static_cast<unsigned long long>(r.width),
                   static_cast<unsigned long long>(r.height), vec});
    const std::size_t u = units_2d(r, vec);
    if (max_units < u) max_units = u;
    if (max_h < r.height) max_h = r.height;
  }
  VK_EXPECTS(dev.size() <= kMaxGridAxis, "too many runs for the grid.z axis");
  VK_EXPECTS(max_h <= kMaxGridAxis, "tile height exceeds the grid.y axis");

  const std::size_t bytes = dev.size() * sizeof(RunDev2D);
  ScopedDevBuf meta(bytes, stream);
  cudaError_t err = cudaMemcpyAsync(meta.ptr, dev.data(), bytes,
                                    cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for 2-D run metadata failed");
  launch_gather_2d(static_cast<const RunDev2D*>(meta.ptr), dev.size(), max_units,
                   max_h, stream);
}

// Adaptive dispatch target for few runs: one cudaMemcpyAsync per run on the
// caller's stream — byte-for-byte the baseline the kernel is measured
// against (issue #6). cudaMemcpyDefault lets the runtime resolve peer vs
// local from the UVA pointers, exactly like the legacy cudaMemcpyPeerAsync
// loop it preserves.
void copy_engine_1d(const std::vector<StagedRun1D>& staged, std::uint8_t* dst,
                    cudaStream_t stream) {
  for (const auto& r : staged) {
    cudaError_t err =
        cudaMemcpyAsync(dst + r.dst_offset, r.src, r.length,
                        cudaMemcpyDefault, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync (copy-engine path) failed");
  }
}

void copy_engine_2d(const std::vector<StagedRun2D>& staged, std::uint8_t* dst,
                    cudaStream_t stream) {
  for (const auto& r : staged) {
    cudaError_t err =
        cudaMemcpy2DAsync(dst + r.dst_offset, r.dst_stride, r.src, r.src_stride,
                          r.width, r.height, cudaMemcpyDefault, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpy2DAsync (copy-engine path) failed");
  }
}

}  // namespace

void p2p_gather_runs(std::uint8_t* dst, std::size_t dst_capacity,
                     const void* const* src_ptrs, const std::size_t* dst_offsets,
                     const std::size_t* lengths, std::size_t num_runs,
                     cudaStream_t stream) {
  auto staged = stage_runs_1d(dst, dst_capacity, src_ptrs, dst_offsets, lengths, num_runs);
  if (staged.empty()) return;  // nothing to copy, no launch

  std::size_t total = 0;
  for (const auto& r : staged) total += r.length;
  if (prefer_gather_kernel(staged.size(), total)) {
    kernel_1d_oneshot(dst, staged, stream);
  } else {
    copy_engine_1d(staged, dst, stream);
  }
}

void p2p_gather_runs_2d(std::uint8_t* dst, std::size_t dst_capacity, const Gather2DRun* runs,
                        std::size_t num_runs, cudaStream_t stream) {
  auto staged = stage_runs_2d(dst, dst_capacity, runs, num_runs);
  if (staged.empty()) return;

  std::size_t total = 0;
  for (const auto& r : staged) total += r.width * r.height;
  if (prefer_gather_kernel(staged.size(), total, /*strided=*/true)) {
    kernel_2d_oneshot(dst, staged, stream);
  } else {
    copy_engine_2d(staged, dst, stream);
  }
}

// ---------------------------------------------------------------------------
// Prepared plans
// ---------------------------------------------------------------------------
//
// Impl definitions live at namespace scope (a nested-class member cannot be
// defined inside an anonymous namespace — nvcc rejects it). The types are
// private to the plan classes and only reachable through them, so external
// linkage is harmless.

// Shared metadata for a prepared 1-D plan. `dev` holds the device
// descriptors (absolute dst pointers) built once; `d_runs` is a persistent
// device allocation filled with a synchronous cudaMemcpy at construction, so
// every subsequent execute() — on any stream — reads fully-resident
// metadata. `host_runs` mirrors the staged runs for the copy-engine path.
struct P2PGatherPlan1D::Impl {
  std::uint8_t* dst = nullptr;
  std::size_t dst_capacity = 0;
  std::vector<StagedRun1D> host_runs;
  std::vector<RunDev1D> dev;
  std::size_t max_units = 0;
  std::size_t total = 0;
  RunDev1D* d_runs = nullptr;
  std::size_t d_bytes = 0;

  ~Impl() {
    if (d_runs) cudaFree(d_runs);
  }
};

struct P2PGatherPlan2D::Impl {
  std::uint8_t* dst = nullptr;
  std::size_t dst_capacity = 0;
  std::vector<StagedRun2D> host_runs;
  std::vector<RunDev2D> dev;
  std::size_t max_units = 0;
  std::size_t max_height = 0;
  std::size_t total = 0;
  RunDev2D* d_runs = nullptr;
  std::size_t d_bytes = 0;

  ~Impl() {
    if (d_runs) cudaFree(d_runs);
  }
};

P2PGatherPlan1D::P2PGatherPlan1D(std::uint8_t* dst, std::size_t dst_capacity,
                                 const void* const* src_ptrs,
                                 const std::size_t* dst_offsets,
                                 const std::size_t* lengths, std::size_t num_runs)
    : impl_(new Impl) {
  impl_->dst = dst;
  impl_->dst_capacity = dst_capacity;
  impl_->host_runs =
      stage_runs_1d(dst, dst_capacity, src_ptrs, dst_offsets, lengths, num_runs);

  // Build the device descriptors once (same layout as the one-shot path).
  impl_->dev.reserve(impl_->host_runs.size());
  for (const auto& r : impl_->host_runs) {
    const bool vec =
        aligned16(r.src) && aligned16(dst + r.dst_offset) && r.length >= 16;
    impl_->dev.push_back({r.src, dst + r.dst_offset,
                          static_cast<unsigned long long>(r.length), vec});
    const std::size_t u = units_1d(r, vec);
    if (impl_->max_units < u) impl_->max_units = u;
    impl_->total += r.length;
  }
  VK_EXPECTS(impl_->dev.size() <= kMaxGridAxis, "too many runs for the grid.y axis");
  if (impl_->dev.empty()) return;

  // Persistent metadata buffer, filled synchronously: one-time cost, and no
  // stream association means no cross-stream race when execute() runs on
  // several streams (issue #6 item 5). Free the buffer again if the upload
  // fails so a partially-built plan leaks nothing.
  impl_->d_bytes = impl_->dev.size() * sizeof(RunDev1D);
  cudaError_t err = cudaMalloc(&impl_->d_runs, impl_->d_bytes);
  VK_ENSURES(err == cudaSuccess, "cudaMalloc for plan metadata failed");
  err = cudaMemcpy(impl_->d_runs, impl_->dev.data(), impl_->d_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(impl_->d_runs);
    impl_->d_runs = nullptr;
  }
  VK_ENSURES(err == cudaSuccess, "cudaMemcpy for plan metadata failed");
}

P2PGatherPlan1D::~P2PGatherPlan1D() { delete impl_; }

std::size_t P2PGatherPlan1D::num_runs() const { return impl_->host_runs.size(); }

std::size_t P2PGatherPlan1D::total_bytes() const { return impl_->total; }

void P2PGatherPlan1D::execute(cudaStream_t stream) const {
  if (impl_->host_runs.empty()) return;  // valid no-op plan
  if (prefer_gather_kernel(impl_->host_runs.size(), impl_->total)) {
    // One launch; descriptors are already on the device (persistent buffer).
    launch_gather_1d(impl_->d_runs, impl_->host_runs.size(), impl_->max_units,
                     stream);
  } else {
    copy_engine_1d(impl_->host_runs, impl_->dst, stream);
  }
}

P2PGatherPlan2D::P2PGatherPlan2D(std::uint8_t* dst, std::size_t dst_capacity,
                                 const Gather2DRun* runs, std::size_t num_runs)
    : impl_(new Impl) {
  impl_->dst = dst;
  impl_->dst_capacity = dst_capacity;
  impl_->host_runs = stage_runs_2d(dst, dst_capacity, runs, num_runs);

  impl_->dev.reserve(impl_->host_runs.size());
  for (const auto& r : impl_->host_runs) {
    const bool vec = aligned16(r.src) && aligned16(dst + r.dst_offset) &&
                     (r.src_stride % 16u == 0) && (r.dst_stride % 16u == 0) &&
                     r.width >= 16;
    impl_->dev.push_back({r.src, dst + r.dst_offset,
                          static_cast<unsigned long long>(r.src_stride),
                          static_cast<unsigned long long>(r.dst_stride),
                          static_cast<unsigned long long>(r.width),
                          static_cast<unsigned long long>(r.height), vec});
    const std::size_t u = units_2d(r, vec);
    if (impl_->max_units < u) impl_->max_units = u;
    if (impl_->max_height < r.height) impl_->max_height = r.height;
    impl_->total += r.width * r.height;
  }
  VK_EXPECTS(impl_->dev.size() <= kMaxGridAxis, "too many runs for the grid.z axis");
  VK_EXPECTS(impl_->max_height <= kMaxGridAxis, "tile height exceeds the grid.y axis");
  if (impl_->dev.empty()) return;

  impl_->d_bytes = impl_->dev.size() * sizeof(RunDev2D);
  cudaError_t err = cudaMalloc(&impl_->d_runs, impl_->d_bytes);
  VK_ENSURES(err == cudaSuccess, "cudaMalloc for 2-D plan metadata failed");
  err = cudaMemcpy(impl_->d_runs, impl_->dev.data(), impl_->d_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(impl_->d_runs);
    impl_->d_runs = nullptr;
  }
  VK_ENSURES(err == cudaSuccess, "cudaMemcpy for 2-D plan metadata failed");
}

P2PGatherPlan2D::~P2PGatherPlan2D() { delete impl_; }

std::size_t P2PGatherPlan2D::num_runs() const { return impl_->host_runs.size(); }

std::size_t P2PGatherPlan2D::total_bytes() const { return impl_->total; }

void P2PGatherPlan2D::execute(cudaStream_t stream) const {
  if (impl_->host_runs.empty()) return;  // valid no-op plan
  if (prefer_gather_kernel(impl_->host_runs.size(), impl_->total, /*strided=*/true)) {
    launch_gather_2d(impl_->d_runs, impl_->host_runs.size(), impl_->max_units,
                     impl_->max_height, stream);
  } else {
    copy_engine_2d(impl_->host_runs, impl_->dst, stream);
  }
}

}  // namespace cuda
}  // namespace vkernels::comm

#endif  // VKERNELS_HAS_CUDA
