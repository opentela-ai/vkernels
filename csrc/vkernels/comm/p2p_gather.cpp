// vkernels/comm/p2p_gather.cpp — host reference (oracle) implementation.
//
// The CPU reference is the correctness oracle: it is always compiled, fully
// unit-tested, and carries the contract checks (capacity, non-null sources,
// source/destination non-overlap, mutually disjoint output runs). The CUDA
// path (p2p_gather.cu) launches a single kernel that performs the same copies
// directly from peer memory over NVLink and trusts the already-coalesced
// metadata so the hot path stays free of O(N log N) validation.
#include "vkernels/comm/p2p_gather.hpp"

#include "vkernels/util/error.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace vkernels::comm {
namespace {

// Copy every staged 1-D run. Runs are mutually disjoint so order does not
// affect the result; the CUDA kernel exploits the same property.
void copy_all_1d(const std::vector<StagedRun1D>& runs, std::uint8_t* base) {
  for (const auto& r : runs) std::memcpy(base + r.dst_offset, r.src, r.length);
}

// Copy every staged 2-D tile, row by row, honouring each run's strides.
void copy_all_2d(const std::vector<StagedRun2D>& runs, std::uint8_t* base) {
  for (const auto& r : runs)
    for (std::size_t y = 0; y < r.height; ++y)
      std::memcpy(base + r.dst_offset + y * r.dst_stride,
                  r.src + y * r.src_stride, r.width);
}

}  // namespace

// Validate and stage a 1-D run list. Empty runs (length 0) need no source and
// are dropped. Returns the staged runs in input order; output runs are checked
// to be mutually disjoint (sorted by destination offset).
std::vector<StagedRun1D> stage_runs_1d(const std::uint8_t* dst_base,
                                      std::size_t dst_capacity,
                                      const void* const* src_ptrs,
                                      const std::size_t* dst_offsets,
                                      const std::size_t* lengths,
                                      std::size_t num_runs) {
  VK_EXPECTS(num_runs == 0 || src_ptrs != nullptr, "src_ptrs must be non-null");
  VK_EXPECTS(num_runs == 0 || dst_offsets != nullptr, "dst_offsets must be non-null");
  VK_EXPECTS(num_runs == 0 || lengths != nullptr, "lengths must be non-null");

  std::vector<StagedRun1D> runs;
  runs.reserve(num_runs);
  for (std::size_t i = 0; i < num_runs; ++i) {
    const std::size_t len = lengths[i];
    const std::size_t off = dst_offsets[i];
    if (len == 0) continue;  // empty run: no source needed, nothing to copy
    VK_EXPECTS(off <= dst_capacity && (dst_capacity - off) >= len,
               "run exceeds dst capacity");
    const auto* s = static_cast<const std::uint8_t*>(src_ptrs[i]);
    VK_EXPECTS(s != nullptr, "src_ptrs[i] must be non-null for a non-empty run");
    // Source and destination byte ranges for this run must not overlap. The
    // CUDA path reuses this check; with UVA the peer and local pointers share
    // one address space, so a mistakenly-aliased source is still caught.
    // Comparing uintptr_t avoids the UB of ordering pointers from different
    // allocations.
    const std::uintptr_t su = reinterpret_cast<std::uintptr_t>(s);
    const std::uintptr_t du = reinterpret_cast<std::uintptr_t>(dst_base) + off;
    VK_EXPECTS(su + len <= du || su >= du + len, "src and dst ranges must not overlap");
    runs.push_back({s, off, len});
  }

  // Output runs must be mutually disjoint (checked in sorted order).
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  spans.reserve(runs.size());
  for (const auto& r : runs) spans.emplace_back(r.dst_offset, r.dst_offset + r.length);
  std::sort(spans.begin(), spans.end());
  for (std::size_t i = 1; i < spans.size(); ++i)
    VK_EXPECTS(spans[i - 1].second <= spans[i].first, "output runs must be disjoint");

  return runs;
}

// Validate and stage a 2-D strided run list. Empty tiles (zero width or
// height) are dropped. The destination extent is checked in steps so an
// oversized tile cannot overflow while computing it. Two further checks
// mirror `stage_runs_1d` so the CUDA kernel (which copies runs with no
// inter-run ordering) can rely on the host oracle:
//   * per-tile src/dst non-overlap, using the conservative bounding box that
//     spans every row (sufficient: a non-overlapping bounding box guarantees
//     no row overlaps; not necessary when strides carve gaps), and
//   * mutually-disjoint output bounding boxes (sorted, then checked pairwise).
//
// The bounding-box check is intentionally conservative: it may reject a
// pathological strided layout whose tiles' boxes overlap while their actual
// rows do not. For the coalesced, packed layouts this primitive targets such
// false positives do not arise, and the guarantee (no two tiles write the
// same destination byte) is what the concurrent CUDA kernel needs.
std::vector<StagedRun2D> stage_runs_2d(const std::uint8_t* dst_base,
                                      std::size_t dst_capacity, const Gather2DRun* runs,
                                      std::size_t num_runs) {
  VK_EXPECTS(num_runs == 0 || runs != nullptr, "runs must be non-null");

  std::vector<StagedRun2D> out;
  out.reserve(num_runs);
  std::vector<std::pair<std::uintptr_t, std::uintptr_t>> dst_boxes;
  dst_boxes.reserve(num_runs);
  for (std::size_t i = 0; i < num_runs; ++i) {
    const Gather2DRun& g = runs[i];
    if (g.width == 0 || g.height == 0) continue;  // empty tile: nothing to copy
    VK_EXPECTS(g.width <= g.src_stride, "width must not exceed src_stride");
    VK_EXPECTS(g.width <= g.dst_stride, "width must not exceed dst_stride");
    VK_EXPECTS(g.src != nullptr, "src must be non-null for a non-empty tile");
    const std::size_t off = g.dst_offset;
    VK_EXPECTS(off <= dst_capacity, "2-D run starts past dst capacity");
    const std::size_t rows = g.height - 1;  // height >= 1 here
    VK_EXPECTS(rows <= (dst_capacity - off) / g.dst_stride,
               "2-D row stride exceeds dst capacity");
    const std::size_t row_span = rows * g.dst_stride;
    VK_EXPECTS(g.width <= dst_capacity - off - row_span, "2-D run exceeds dst capacity");

    // Per-tile src/dst non-overlap via the conservative bounding box spanning
    // every row (min corner = first row start; the box also covers the inter-
    // row gaps when stride > width, which is what makes it conservative).
    const std::size_t src_span = rows * g.src_stride;
    const std::uintptr_t s_lo = reinterpret_cast<std::uintptr_t>(g.src);
    const std::uintptr_t s_hi = s_lo + src_span + g.width;
    const std::uintptr_t d_lo = reinterpret_cast<std::uintptr_t>(dst_base) + off;
    const std::uintptr_t d_hi = d_lo + row_span + g.width;
    VK_EXPECTS(s_hi <= d_lo || s_lo >= d_hi, "2-D src and dst ranges must not overlap");

    const auto* s = static_cast<const std::uint8_t*>(g.src);
    out.push_back({s, off, g.src_stride, g.dst_stride, g.width, g.height});
    dst_boxes.emplace_back(d_lo, d_hi);
  }

  // Output tiles must be mutually disjoint (checked in sorted box order).
  std::sort(dst_boxes.begin(), dst_boxes.end());
  for (std::size_t i = 1; i < dst_boxes.size(); ++i)
    VK_EXPECTS(dst_boxes[i - 1].second <= dst_boxes[i].first,
               "output tiles must be disjoint");

  return out;
}

void p2p_gather_runs(Span<std::uint8_t> dst, const void* const* src_ptrs,
                     const std::size_t* dst_offsets, const std::size_t* lengths,
                     std::size_t num_runs, Stream* stream) {
  auto runs = stage_runs_1d(dst.data(), dst.size(), src_ptrs, dst_offsets,
                            lengths, num_runs);
  if (runs.empty()) return;  // 0 runs, or every run empty: enqueue nothing

  std::uint8_t* base = dst.data();
  auto copy_all = [runs = std::move(runs), base]() { copy_all_1d(runs, base); };
  if (stream == nullptr) {
    copy_all();
    return;
  }
  stream->submit(std::move(copy_all));
}

void p2p_gather_runs_2d(Span<std::uint8_t> dst, Span<const Gather2DRun> runs, Stream* stream) {
  auto staged = stage_runs_2d(dst.data(), dst.size(), runs.data(), runs.size());
  if (staged.empty()) return;  // 0 tiles, or every tile empty: enqueue nothing

  std::uint8_t* base = dst.data();
  auto copy_all = [staged = std::move(staged), base]() { copy_all_2d(staged, base); };
  if (stream == nullptr) {
    copy_all();
    return;
  }
  stream->submit(std::move(copy_all));
}

void memcpy_peer_batch_async(Span<std::uint8_t> dst, const void* const* src_ptrs,
                             const std::size_t* dst_offsets, const std::size_t* lengths,
                             std::size_t num_runs, Stream* stream) {
  auto runs = stage_runs_1d(dst.data(), dst.size(), src_ptrs, dst_offsets,
                            lengths, num_runs);
  if (runs.empty()) return;

  std::uint8_t* base = dst.data();
  if (stream == nullptr) {
    copy_all_1d(runs, base);
    return;
  }
  // The legacy seam: one copy per run. Stream::submitted() grows by
  // runs.size() here, against 1 for p2p_gather_runs.
  for (const auto& r : runs)
    stream->submit([base, r]() { std::memcpy(base + r.dst_offset, r.src, r.length); });
}

// ---------------------------------------------------------------------------
// Adaptive dispatch
// ---------------------------------------------------------------------------
namespace {

// Runtime-tunable dispatch state. Atomically read by the CUDA path
// (p2p_gather.cu) and written by set_gather_dispatch; a pair of atomics is
// enough because the two values are independent (mode and min-runs floor).
std::atomic<unsigned> g_dispatch_mode(static_cast<unsigned>(GatherDispatchMode::kAdaptive));
std::atomic<std::size_t> g_dispatch_min_runs(4);

// Cost-model constants fitted to measurements on sgs-gpu07 (4x H100 NVL,
// CUDA 13.0 / driver 580.82.07, real NVLink peer reads GPU1->GPU0, 50-iter
// medians, idle):
//   * copy-engine loop: 201.7 us @ 1 run / 48 MiB, +~7.37 us per extra
//     run, with a ~20 us per-call floor that dominates small transfers
//     (~17-20 us for a single 4 KiB copy). The per-MiB term is the
//     bandwidth (48 MiB / 201.7 us ~= 4.20 us/MiB); the per-run term is
//     the driver/engine overhead the loop pays for fragmentation.
//   * gather kernel: ~210 us flat at 48 MiB (~4.20 us/MiB of SM-driven
//     peer reads; the vectorized 16-byte path measured 253 -> 210 us vs
//     the issue #6 non-vectorized kernel) with a ~8.6 us launch floor;
//     flat in run count (grid.y growth is negligible).
//   * strided (2-D) shapes differ because the engine's per-call setup and
//     the kernel's grid shape change: cudaMemcpy2DAsync has a ~10.8 us
//     floor and ~7.30 us/run (64-row tiles measured), and the 2-D kernel
//     pays one block per row, so its floor is ~14 us and grows ~0.13 us
//     per run.
constexpr double kCopyFixedUs = 20.0;      // 1-D per-call floor
constexpr double kCopy2DFixedUs = 10.75;   // 2-D per-call floor
constexpr double kCopyPerMiBUs = 4.20;
constexpr double kCopyPerRunUs = 7.37;     // 1-D per-extra-run
constexpr double kCopy2DPerRunUs = 7.30;   // 2-D per-extra-run
constexpr double kKernelFixedUs = 8.6;     // 1-D launch + tiny-grid floor
constexpr double kKernel2DFixedUs = 14.0;  // 2-D launch + one block per row
constexpr double kKernel2DPerRunUs = 0.13; // 2-D grid.z growth
constexpr double kKernelPerMiBUs = 4.20;

// The min-runs floor guards the large-payload, low-run region where the
// copy engine's bandwidth advantage lives (measured crossover ~3 runs at
// 48 MiB; the 1-2-run margins are ~1%, inside noise). Below this payload
// the copy engine never wins — its fixed floor dominates and the kernel
// takes even a single run (1.85x at 4 KiB) — so the model decides from 1
// run there. Strided 2-D dispatches never apply the floor: the acceptance
// region is 1-D, and the 2-D model's single-op margin (copy 1.3x at 32 KiB)
// is below the 1 MiB threshold anyway.
constexpr std::size_t kFloorMinBytes = 1024u * 1024u;  // 1 MiB

}  // namespace

void set_gather_dispatch(GatherDispatchMode mode, std::size_t min_runs_for_kernel) {
  g_dispatch_mode.store(static_cast<unsigned>(mode));
  g_dispatch_min_runs.store(min_runs_for_kernel);
}

std::pair<GatherDispatchMode, std::size_t> gather_dispatch_config() {
  return {static_cast<GatherDispatchMode>(g_dispatch_mode.load()),
          g_dispatch_min_runs.load()};
}

double est_copy_engine_us(std::size_t num_runs, std::size_t total_bytes, bool strided) {
  if (total_bytes == 0) return 0.0;  // nothing to copy
  const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  const double runs = static_cast<double>(num_runs);
  const double fixed = strided ? kCopy2DFixedUs : kCopyFixedUs;
  const double per_run = strided ? kCopy2DPerRunUs : kCopyPerRunUs;
  const double one_call = std::max(fixed, kCopyPerMiBUs * mib);
  return one_call + per_run * (runs > 1.0 ? runs - 1.0 : 0.0);
}

double est_gather_kernel_us(std::size_t num_runs, std::size_t total_bytes, bool strided) {
  if (total_bytes == 0) return 0.0;  // nothing to copy
  const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  if (strided) {
    // One block per row (grid.y = height) plus grid.z = num_runs: the floor
    // and a small per-run term cover the block overhead.
    return kKernel2DFixedUs + kKernel2DPerRunUs * static_cast<double>(num_runs > 0 ? num_runs - 1 : 0) +
           kKernelPerMiBUs * mib;
  }
  (void)num_runs;  // flat in run count below the 65535-run grid cap
  return std::max(kKernelFixedUs, kKernelPerMiBUs * mib);
}

bool prefer_gather_kernel(std::size_t num_runs, std::size_t total_bytes, bool strided) {
  if (total_bytes == 0) return false;  // nothing to copy: never pay a launch
  switch (static_cast<GatherDispatchMode>(g_dispatch_mode.load())) {
    case GatherDispatchMode::kForceKernel: return true;
    case GatherDispatchMode::kForceCopyEngine: return false;
    case GatherDispatchMode::kAdaptive: break;
  }
  // Floor: keep the low-run large-payload 1-D region on the engine. 2-D
  // dispatches use the model directly (see kFloorMinBytes note).
  if (!strided && total_bytes >= kFloorMinBytes && num_runs < g_dispatch_min_runs.load())
    return false;
  return est_gather_kernel_us(num_runs, total_bytes, strided) <
         est_copy_engine_us(num_runs, total_bytes, strided);
}

// ---------------------------------------------------------------------------
// Prepared plan API (host reference)
// ---------------------------------------------------------------------------
namespace {

// Sum of the staged runs' lengths. The kernel path sizes its grid by the
// longest run and the copy-engine path needs the total for dispatch.
std::size_t total_bytes_1d(const std::vector<StagedRun1D>& runs) {
  std::size_t t = 0;
  for (const auto& r : runs) t += r.length;
  return t;
}

std::size_t total_bytes_2d(const std::vector<StagedRun2D>& runs) {
  std::size_t t = 0;
  for (const auto& r : runs) t += r.width * r.height;
  return t;
}

}  // namespace

P2PGatherPlan1D::P2PGatherPlan1D(std::uint8_t* dst, std::size_t dst_capacity,
                                 const void* const* src_ptrs,
                                 const std::size_t* dst_offsets,
                                 const std::size_t* lengths, std::size_t num_runs)
    : dst_(dst),
      dst_capacity_(dst_capacity),
      runs_(stage_runs_1d(dst, dst_capacity, src_ptrs, dst_offsets, lengths, num_runs)),
      total_bytes_(0) {
  total_bytes_ = total_bytes_1d(runs_);
}

void P2PGatherPlan1D::execute(Stream* stream) const {
  if (runs_.empty()) return;  // 0 runs, or every run empty: enqueue nothing
  // Capture `this`, not a copy of the descriptors: a plan's whole point is
  // that execute() performs no per-call host-vector construction. The plan
  // (like the source pointers and destination it references) must therefore
  // outlive the stream; see the class contract.
  const P2PGatherPlan1D* self = this;
  auto copy_all = [self]() { copy_all_1d(self->runs_, self->dst_); };
  if (stream == nullptr) {
    copy_all();
    return;
  }
  stream->submit(std::move(copy_all));
}

P2PGatherPlan2D::P2PGatherPlan2D(std::uint8_t* dst, std::size_t dst_capacity,
                                 const Gather2DRun* runs, std::size_t num_runs)
    : dst_(dst),
      dst_capacity_(dst_capacity),
      runs_(stage_runs_2d(dst, dst_capacity, runs, num_runs)),
      total_bytes_(0) {
  total_bytes_ = total_bytes_2d(runs_);
}

void P2PGatherPlan2D::execute(Stream* stream) const {
  if (runs_.empty()) return;  // 0 tiles, or every tile empty: enqueue nothing
  const P2PGatherPlan2D* self = this;
  auto copy_all = [self]() { copy_all_2d(self->runs_, self->dst_); };
  if (stream == nullptr) {
    copy_all();
    return;
  }
  stream->submit(std::move(copy_all));
}

}  // namespace vkernels::comm
