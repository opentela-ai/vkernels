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

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
