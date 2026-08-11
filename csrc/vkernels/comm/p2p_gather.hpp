// vkernels/comm/p2p_gather.hpp
//
// Single-launch P2P/UVA run-list gather.
//
// KVAAS coalesces a peer-hit prefix into maximal contiguous donor runs, but
// the legacy transport still submits one cudaMemcpyPeerAsync (or
// cudaMemcpy2DAsync) per run. Fragmented prefixes therefore pay repeated
// driver and launch overhead even though every run targets one local scratch
// buffer. `p2p_gather_runs` (and its 2-D sibling) replaces that loop with a
// single operation: the run metadata is staged once, then one kernel launch
// (CUDA) / one stream task (host) copies every run. There are no per-run API
// calls after the metadata is prepared.
//
// Contract
// --------
//  * `dst` is a local allocation on the executing device; `dst.size()` is its
//    capacity in bytes and bounds every `dst_offsets[i] + lengths[i]`.
//  * Each `src_ptrs[i]` is a peer-accessible UVA (or IPC-mapped) pointer.
//    Peer access and the IPC mapping must be established by the caller BEFORE
//    the launch and held until `stream` completes — direct peer reads over
//    NVLink are issued from inside the kernel, with no host staging of data.
//  * `dst_offsets[i]` / `lengths[i]` describe disjoint output runs. Source and
//    destination byte ranges must not overlap. The host reference (the
//    correctness oracle) validates these; the CUDA entry point trusts the
//    caller (the metadata is the coalescer's output) so the hot path stays
//    free of O(N log N) checks.
//
// Lifetime
// --------
//  * The `dst` allocation and the peer memory behind every `src_ptrs[i]` must
//    outlive `stream` (the kernel/task reads them asynchronously).
//  * The run-metadata arrays (`src_ptrs`, `dst_offsets`, `lengths`) are read
//    and copied into owned storage before `p2p_gather_runs` returns, so the
//    caller may free or mutate the originals as soon as the call returns —
//    only the IPC mappings, not the metadata, must persist.
//  * When `stream == nullptr` the work runs to completion before returning
//    (the CUDA default-stream model). A non-null stream is enqueued onto and
//    the call returns without synchronising.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vkernels/core/stream.hpp"
#include "vkernels/util/span.hpp"

namespace vkernels::comm {

// 1-D run-list gather. For each i in [0, num_runs), copies `lengths[i]`
// bytes from `src_ptrs[i]` to `dst.data() + dst_offsets[i]`. `num_runs == 0`
// is a valid no-op.
//
// `src_ptrs`, `dst_offsets` and `lengths` are each `num_runs` long and may be
// freed as soon as the call returns (see Lifetime above).
void p2p_gather_runs(Span<std::uint8_t> dst, const void* const* src_ptrs,
                     const std::size_t* dst_offsets, const std::size_t* lengths,
                     std::size_t num_runs, Stream* stream = nullptr);

// One strided 2-D run: copy a `height` x `width`-byte tile from a row-major
// peer region (`src`, row stride `src_stride`) into `dst` at byte offset
// `dst_offset` with row stride `dst_stride`. `width` must not exceed either
// stride. Used for layer-major page layouts where a donated page is a strided
// sub-region of a peer allocation.
struct Gather2DRun {
  const void* src;
  std::size_t src_stride;
  std::size_t dst_offset;
  std::size_t dst_stride;
  std::size_t width;
  std::size_t height;
};

// Validated, owned run descriptors shared by the host oracle (p2p_gather.cpp)
// and the CUDA implementation (p2p_gather.cu). `stage_runs_1d` / `stage_runs_2d`
// perform the full contract checks — non-null sources, capacity, src/dst
// non-overlap, mutually disjoint output runs — and throw std::invalid_argument
// on violation. Exposing them here lets the CUDA path reuse the same tested
// validation instead of duplicating it (the hot kernel launch stays free of
// O(N log N) checks; validation runs once on the host before launch).
struct StagedRun1D {
  const std::uint8_t* src;
  std::size_t dst_offset;
  std::size_t length;
};

struct StagedRun2D {
  const std::uint8_t* src;
  std::size_t dst_offset;
  std::size_t src_stride;
  std::size_t dst_stride;
  std::size_t width;
  std::size_t height;
};

std::vector<StagedRun1D> stage_runs_1d(const std::uint8_t* dst_base,
                                       std::size_t dst_capacity,
                                       const void* const* src_ptrs,
                                       const std::size_t* dst_offsets,
                                       const std::size_t* lengths,
                                       std::size_t num_runs);

std::vector<StagedRun2D> stage_runs_2d(std::size_t dst_capacity,
                                       const Gather2DRun* runs, std::size_t num_runs);

// 2-D run-list gather: every run is a strided tile copied by a single
// operation. Rows within a run are copied individually so per-run source
// strides are honoured even when the whole list launches together. `runs`
// may be freed as soon as the call returns.
void p2p_gather_runs_2d(Span<std::uint8_t> dst, Span<const Gather2DRun> runs,
                        Stream* stream = nullptr);

// Legacy transport seam: one copy operation per run, the predecessor the
// single-launch `p2p_gather_runs` replaces. Retained so benchmarks can sweep
// run count / size / fragmentation against the per-run loop and so the
// "no per-run API calls" property can be asserted on the host
// (Stream::submitted() grows by `num_runs` here versus by 1 for
// `p2p_gather_runs`). Same contract and lifetime rules as `p2p_gather_runs`.
void memcpy_peer_batch_async(Span<std::uint8_t> dst, const void* const* src_ptrs,
                             const std::size_t* dst_offsets, const std::size_t* lengths,
                             std::size_t num_runs, Stream* stream = nullptr);

}  // namespace vkernels::comm
