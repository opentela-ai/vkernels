// vkernels/comm/p2p_kv_restore_cuda.hpp
//
// CUDA-only declarations for the fused peer-to-indexed-KV restore kernel.
// Kept separate from p2p_kv_restore.hpp because the CUDA entry points take
// `cudaStream_t`, which must not be exposed to host-only translation units.
// Included only when VKERNELS_HAS_CUDA; the definitions live in
// p2p_kv_restore.cu.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;
typedef CUstream_st* cudaStream_t_kv;  // avoid pulling cuda_runtime.h into this header

namespace vkernels::comm::cuda {

// Fused peer-to-indexed-KV restore for one layer. See p2p_kv_restore.hpp
// for the full contract. `k_dst` / `v_dst` are local device pointers;
// `peer_src_ptrs` are peer UVA pointers (raw `void*` from the caller).
// Enqueued on `stream`; returns without synchronising.
//
// The CUDA implementation resolves effective page pointers
// (`peer_src_ptrs[p] + src_page_offsets[p]`) on the host, stages them into
// a device-side `page_ptrs` array with a streaming H2D copy, and launches
// one kernel that reads peer memory over NVLink and writes directly into
// the indexed K/V slots. No scratch buffer is allocated.
void p2p_kv_restore_layer(void* k_dst, void* v_dst,
                          const int* slot_ids,
                          const void* const* peer_src_ptrs,
                          const std::size_t* src_page_offsets,
                          std::size_t num_pages, std::size_t page_size,
                          std::size_t num_kv_heads, std::size_t head_dim,
                          std::size_t elem_size,
                          cudaStream_t_kv stream);

// Two-stage reference on the GPU:
//   1. cudaMemcpyAsync per page (peer → scratch), then
//   2. indexed KV scatter kernel (scratch → slots).
// Useful as a correctness baseline and for measuring the scratch cost.
void p2p_kv_restore_layer_twostage(void* k_dst, void* v_dst,
                                   const int* slot_ids,
                                   const void* const* peer_src_ptrs,
                                   const std::size_t* src_page_offsets,
                                   std::size_t num_pages, std::size_t page_size,
                                   std::size_t num_kv_heads, std::size_t head_dim,
                                   std::size_t elem_size,
                                   cudaStream_t_kv stream);

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
