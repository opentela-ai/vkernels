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
//    correctness oracle) validates these for both the 1-D and 2-D paths —
//    per-run src/dst non-overlap and mutually-disjoint outputs — so the CUDA
//    entry point (which has no inter-run ordering guarantee) can trust the
//    already-coalesced metadata and the hot path stays free of O(N log N)
//    checks.
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
#include <utility>
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

std::vector<StagedRun2D> stage_runs_2d(const std::uint8_t* dst_base,
                                       std::size_t dst_capacity,
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

// ---------------------------------------------------------------------------
// Adaptive dispatch (CUDA path only; host-testable pure functions)
// ---------------------------------------------------------------------------
//
// On H100 NVL the single-launch SM gather kernel is NOT always faster than
// the per-run cudaMemcpy*Async loop it replaces: the copy engine wins for
// very few runs (per-run driver calls are cheap, the engine runs without
// occupying SMs), while the kernel wins for fragmented lists. Issue #6
// measured a 48 MiB payload and found the PR #5 kernel crossing the copy
// loop between 16 and 32 runs; the vectorized uint4 kernel is ~43 us
// faster at 48 MiB and crosses at ~3 runs. The CUDA entry points therefore
// dispatch adaptively: below the threshold they issue one cudaMemcpyAsync /
// cudaMemcpy2DAsync per run (the exact baseline), at or above it they
// launch the single gather kernel. The decision is a pure function of run
// count and total bytes so it is unit-testable on the host and tunable per
// deployment.

// Forced-dispatch modes, for testing and A/B tuning on a target machine.
// kAdaptive applies the cost model below; the force modes bypass it.
enum class GatherDispatchMode { kAdaptive = 0, kForceKernel = 1, kForceCopyEngine = 2 };

// Set the dispatch mode and the minimum run count at which the gather
// kernel is eligible. Defaults: kAdaptive with min_runs = 4 (measured on
// H100 NVL: crossover ~3 runs at 48 MiB; 1-2-run margins are ~1%, so the
// floor keeps the low-run large-payload region on the copy engine, and the
// acceptance range 1-16 runs stays within 5% of the copy-engine baseline
// while 8/16-run wins are preserved). Below a 1 MiB payload the floor does
// not apply — the copy engine never wins there — so the model decides from
// one run. Thread-safe (atomics); typically set once before launching.
void set_gather_dispatch(GatherDispatchMode mode = GatherDispatchMode::kAdaptive,
                         std::size_t min_runs_for_kernel = 4);
std::pair<GatherDispatchMode, std::size_t> gather_dispatch_config();

// Estimated device time (microseconds) of the per-run copy-engine loop for
// `total_bytes` split across `num_runs` cudaMemcpy*Async calls. Constants
// fitted to H100 NVL measurements (sgs-gpu07, CUDA 13 / driver 580.82.07,
// real NVLink peer reads): ~201.7 us for 48 MiB in one run (~4.20 us/MiB)
// plus ~7.37 us per extra run, with a ~20 us per-call floor that dominates
// transfers below a few MiB. `strided` (2-D cudaMemcpy2DAsync) has a lower
// per-call floor (~10.8 us) and a slightly lower per-run term (~7.30 us).
// `num_runs` counts only non-empty runs (empty runs are dropped before
// dispatch); zero bytes estimates 0.
double est_copy_engine_us(std::size_t num_runs, std::size_t total_bytes,
                          bool strided = false);

// Estimated device time (microseconds) of the single-launch gather kernel:
// ~210 us for 48 MiB (~4.20 us/MiB of SM-driven peer reads, no per-run
// driver cost), plus a ~8.6 us launch floor; flat in run count below the
// 65535-run cap. Strided (2-D) kernels pay one block per row, so the floor
// is ~14 us and grows ~0.13 us per extra run.
double est_gather_kernel_us(std::size_t num_runs, std::size_t total_bytes,
                            bool strided = false);

// Pure dispatch decision: true -> single-launch SM gather kernel, false ->
// per-run copy-engine loop. Honours the configured mode and min-runs floor
// (large 1-D payloads only; strided 2-D uses the model directly, its
// low-run margin is not the acceptance region); zero bytes (nothing to
// copy) never takes the kernel.
bool prefer_gather_kernel(std::size_t num_runs, std::size_t total_bytes,
                          bool strided = false);

// ---------------------------------------------------------------------------
// Prepared plan API
// ---------------------------------------------------------------------------
//
// KVAAS resolves one run list and reuses it for all 40 layers, but the
// one-shot functions repeat validation, host-vector construction, device
// metadata allocation, H2D metadata copy and free for every layer. A plan
// moves all of that to a single prepare step: the constructor validates the
// run list once (via stage_runs_*), copies the descriptors into owned
// storage, and — on the CUDA path — uploads them to a persistent per-device
// buffer. execute() then only enqueues: no validation, no allocation, no
// H2D copy. The plan is bound to one destination allocation (the scratch
// buffer KVAAS reuses across layers); a plan for a different destination is
// prepared separately.
//
// Concurrency: after prepare the run metadata is immutable, so one plan may
// be executed concurrently on several streams without cross-stream races
// (the CUDA device descriptors are read-only to every kernel).

// A prepared 1-D run list, bound to one destination allocation.
class P2PGatherPlan1D {
 public:
  // Validate `num_runs` runs against (dst, dst_capacity) once — same
  // contract checks as p2p_gather_runs, throwing std::invalid_argument on
  // violation — and copy the descriptors into owned storage. num_runs == 0
  // is a valid no-op plan. The source pointers and the destination must
  // outlive every execute() that uses the plan (the metadata is copied, the
  // memory is not).
  P2PGatherPlan1D(std::uint8_t* dst, std::size_t dst_capacity,
                  const void* const* src_ptrs, const std::size_t* dst_offsets,
                  const std::size_t* lengths, std::size_t num_runs);

  P2PGatherPlan1D(const P2PGatherPlan1D&) = delete;
  P2PGatherPlan1D& operator=(const P2PGatherPlan1D&) = delete;

  std::size_t num_runs() const { return runs_.size(); }
  std::size_t total_bytes() const { return total_bytes_; }
  std::uint8_t* dst() const { return dst_; }
  std::size_t dst_capacity() const { return dst_capacity_; }

  // Copy every run to the bound destination: exactly ONE stream task
  // regardless of run count (host reference), or one kernel launch / one
  // copy-engine sequence (CUDA). No validation, allocation or metadata work
  // happens here — all of it was done in the constructor (execute captures
  // the plan itself, not a copy of the descriptors, so there is no per-call
  // host-vector construction). A null stream runs to completion before
  // returning.
  //
  // Lifetime: the plan must outlive the stream it is executed on (destroy
  // it only after stream->wait()).
  void execute(Stream* stream = nullptr) const;

 private:
  std::uint8_t* dst_;
  std::size_t dst_capacity_;
  std::vector<StagedRun1D> runs_;
  std::size_t total_bytes_ = 0;
};

// A prepared 2-D strided run list, bound to one destination allocation.
// The plan stores the source base pointers from the input runs — the
// caller's construction-time pointers. After construction the plan can be
// executed with a layer-relative `src_byte_offset` so KVAAS reuses one
// run topology across all 40 layers without re-creating the plan or
// performing per-layer metadata work.
class P2PGatherPlan2D {
 public:
  P2PGatherPlan2D(std::uint8_t* dst, std::size_t dst_capacity,
                  const Gather2DRun* runs, std::size_t num_runs);

  P2PGatherPlan2D(const P2PGatherPlan2D&) = delete;
  P2PGatherPlan2D& operator=(const P2PGatherPlan2D&) = delete;

  std::size_t num_runs() const { return runs_.size(); }
  std::size_t total_bytes() const { return total_bytes_; }
  std::uint8_t* dst() const { return dst_; }
  std::size_t dst_capacity() const { return dst_capacity_; }

  // Same contract as P2PGatherPlan1D::execute: one stream task, no per-call
  // metadata work, plan must outlive the stream.
  void execute(Stream* stream = nullptr) const;

  // Layer-relative execute: adds `src_byte_offset` to every run's source
  // pointer before copying. This is the KVAAS pattern — one run topology
  // (page bases, strides, widths, heights) reused across layers, with only
  // the per-layer byte offset changing. Every other aspect of execute() —
  // single stream task, no validation, no allocation — is preserved.
  void execute(std::size_t src_byte_offset, Stream* stream = nullptr) const;

 private:
  std::uint8_t* dst_;
  std::size_t dst_capacity_;
  std::vector<StagedRun2D> runs_;
  std::size_t total_bytes_ = 0;
};

}  // namespace vkernels::comm
