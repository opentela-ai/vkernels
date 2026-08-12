// vkernels/comm/p2p_gather_cuda.hpp
//
// CUDA-only declarations for the single-launch P2P run-list gather. Kept
// separate from p2p_gather.hpp because the CUDA entry points take
// `cudaStream_t`, which must not be exposed to host-only translation units.
// Included only when VKERNELS_HAS_CUDA; the definitions live in
// p2p_gather.cu.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/comm/p2p_gather.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;
typedef CUstream_st* cudaStream_t_p2p;  // avoid pulling cuda_runtime.h into this header

namespace vkernels::comm::cuda {

// 1-D run-list gather, single kernel launch. `dst` is a local device
// pointer of `dst_capacity` bytes; `src_ptrs` are peer UVA pointers.
// Enqueued on `stream`; returns without synchronising.
void p2p_gather_runs(std::uint8_t* dst, std::size_t dst_capacity,
                     const void* const* src_ptrs, const std::size_t* dst_offsets,
                     const std::size_t* lengths, std::size_t num_runs,
                     cudaStream_t_p2p stream);

// 2-D strided run-list gather, single kernel launch. See Gather2DRun.
void p2p_gather_runs_2d(std::uint8_t* dst, std::size_t dst_capacity, const Gather2DRun* runs,
                        std::size_t num_runs, cudaStream_t_p2p stream);

// ---------------------------------------------------------------------------
// Prepared plan API (CUDA implementation)
// ---------------------------------------------------------------------------
//
// Same semantics as vkernels::comm::P2PGatherPlan1D / _2D (validate once at
// construction, execute() only enqueues) with the CUDA specifics: the
// constructor also uploads the run descriptors to a persistent per-device
// buffer with a synchronous cudaMemcpy (one-time cost, no cross-stream
// race), and execute() adaptively dispatches between the per-run copy
// engine and the single-launch kernel using prefer_gather_kernel.
//
// Lifetime: the plan owns the device descriptor buffer; destroy the plan
// only after every stream it was executed on has been synchronised.
// Read-only after construction, so concurrent execute() on several streams
// is safe.
class P2PGatherPlan1D {
 public:
  P2PGatherPlan1D(std::uint8_t* dst, std::size_t dst_capacity,
                  const void* const* src_ptrs, const std::size_t* dst_offsets,
                  const std::size_t* lengths, std::size_t num_runs);
  ~P2PGatherPlan1D();
  P2PGatherPlan1D(const P2PGatherPlan1D&) = delete;
  P2PGatherPlan1D& operator=(const P2PGatherPlan1D&) = delete;

  std::size_t num_runs() const;
  std::size_t total_bytes() const;

  // Enqueue the gather on `stream`. Adaptive: per-run cudaMemcpyAsync below
  // the dispatch crossover, one kernel launch at or above it. No
  // validation, no allocation, no H2D metadata copy.
  void execute(cudaStream_t_p2p stream) const;

 private:
  struct Impl;
  Impl* impl_;
};

class P2PGatherPlan2D {
 public:
  P2PGatherPlan2D(std::uint8_t* dst, std::size_t dst_capacity,
                  const Gather2DRun* runs, std::size_t num_runs);
  ~P2PGatherPlan2D();
  P2PGatherPlan2D(const P2PGatherPlan2D&) = delete;
  P2PGatherPlan2D& operator=(const P2PGatherPlan2D&) = delete;

  std::size_t num_runs() const;
  std::size_t total_bytes() const;

  // Zero-offset execute (existing behaviour).
  void execute(cudaStream_t_p2p stream) const;

  // Layer-relative execute: adds `src_byte_offset` to every run's source
  // pointer. The kernel reads `srow = run.src + offset + row*src_stride`;
  // the copy-engine branch passes `r.src + offset` to cudaMemcpy2DAsync.
  // The plan stores the construction-time source base pointers; both
  // branches apply the scalar offset with no per-layer H2D descriptor
  // upload.
  void execute(std::size_t src_byte_offset, cudaStream_t_p2p stream) const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
