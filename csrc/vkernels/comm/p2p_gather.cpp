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
    const std::uint8_t* d = dst_base + off;
    VK_EXPECTS(s + len <= d || s >= d + len, "src and dst ranges must not overlap");
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
// oversized tile cannot overflow while computing it.
std::vector<StagedRun2D> stage_runs_2d(std::size_t dst_capacity, const Gather2DRun* runs,
                                      std::size_t num_runs) {
  std::vector<StagedRun2D> out;
  out.reserve(num_runs);
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
    const auto* s = static_cast<const std::uint8_t*>(g.src);
    out.push_back({s, off, g.src_stride, g.dst_stride, g.width, g.height});
  }
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
  auto staged = stage_runs_2d(dst.size(), runs.data(), runs.size());
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

}  // namespace vkernels::comm
