// vkernels/comm/kv_gather_c.h
//
// C ABI for the fused indexed K/V layer gather kernel (issue #2).
// Non-C++ consumers (e.g. a serving runtime) call these `extern "C"` entry
// points. Errors are RETURNED as codes -- no C++ exceptions cross the ABI
// boundary.
//
// CUDA only: the host-input entry point validates on the host and uploads
// `slot_ids` (int32 or int64); the device-slot entry point takes a
// caller-owned device pointer and is check-free. Both take a raw
// `cudaStream_t`. The header is includable from both C and C++; when the
// CUDA runtime headers are present the entry points are declared, otherwise
// only the status codes (which are CUDA-independent) are visible.
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

// Fused indexed K/V gather for one layer. See kv_gather.hpp for the full
// contract. `slot_ids` is a HOST array (shape [num_pages * page_size],
// int32 when `slot_ids_int64 == 0`, int64 when `!= 0`); the entry point
// validates non-negativity and bounds (slot < num_slots) once and uploads it
// to a per-launch device buffer, then launches ONE kernel. Enqueued on
// `stream`; returns without synchronising.
//
// Returns VKERNELS_OK on success. On contract violation (null pointers,
// zero dimensions, negative/out-of-range slot, non-BF16/FP16 elem_size)
// returns VKERNELS_ERR_INVALID_ARGUMENT. On device allocation or launch
// failure returns VKERNELS_ERR_INTERNAL.
vkernels_status_t vkernels_kv_gather_layer(
    void* dst, const void* k_src, const void* v_src,
    const void* slot_ids, int slot_ids_int64,
    size_t num_slots,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream);

// Fused indexed K/V gather for one layer with a DEVICE slot map. `slot_ids`
// is a caller-owned CUDA device pointer (shape [num_pages * page_size],
// int32 or int64 per `slot_ids_int64`). Slot contents are NOT validated
// (reading device memory would force a D2H sync): the caller MUST guarantee
// non-negative, in-range slots (repeats allowed) and keep `slot_ids` and the
// sources alive until the kernel completes on `stream`. One kernel launch,
// no upload. Returns VKERNELS_OK or VKERNELS_ERR_INVALID_ARGUMENT (shape) /
// VKERNELS_ERR_INTERNAL (launch).
vkernels_status_t vkernels_kv_gather_layer_device_slots(
    void* dst, const void* k_src, const void* v_src,
    const void* slot_ids, int slot_ids_int64,
    size_t num_slots,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
