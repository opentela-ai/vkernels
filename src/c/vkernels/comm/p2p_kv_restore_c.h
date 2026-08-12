// vkernels/comm/p2p_kv_restore_c.h
//
// C ABI for the fused peer-to-indexed-KV restore kernel. Non-C++ consumers
// call these `extern "C"` entry points. Errors are RETURNED as codes — no
// C++ exceptions cross the ABI boundary.
//
// CUDA only: the functions take device pointers and a raw `cudaStream_t`.
// The header is includable from both C and C++; when the CUDA runtime headers
// are present the entry points are declared, otherwise only the status codes
// (which are CUDA-independent) are visible.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    define VKERNELS_C_HAS_CUDA 1
#    include <cuda_runtime.h>
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Status codes mirroring vkernels::Code.
typedef enum {
  VKERNELS_OK = 0,
  VKERNELS_ERR_INVALID_ARGUMENT = 1,
  VKERNELS_ERR_OUT_OF_RANGE = 2,
  VKERNELS_ERR_UNSUPPORTED = 3,
  VKERNELS_ERR_INTERNAL = 4,
} vkernels_status_t;

#ifdef VKERNELS_C_HAS_CUDA

// Fused peer-to-indexed-KV restore for one layer. See p2p_kv_restore.hpp
// for the full contract.
//
// Returns VKERNELS_OK on success. On contract violation (null pointers,
// zero dimensions, non-unique slots) returns VKERNELS_ERR_INVALID_ARGUMENT.
// On device allocation or launch failure returns VKERNELS_ERR_INTERNAL.
vkernels_status_t vkernels_p2p_kv_restore_layer(
    void* k_dst, void* v_dst,
    const int* slot_ids,
    const void* const* peer_src_ptrs,
    const size_t* src_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream);

// Two-stage reference (peer→scratch→scatter) for correctness baselines.
vkernels_status_t vkernels_p2p_kv_restore_layer_twostage(
    void* k_dst, void* v_dst,
    const int* slot_ids,
    const void* const* peer_src_ptrs,
    const size_t* src_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
