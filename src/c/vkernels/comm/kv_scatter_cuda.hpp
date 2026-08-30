// vkernels/comm/kv_scatter_cuda.hpp
//
// CUDA-only declarations for the fused indexed K/V layer scatter kernel
// (issue #1). Kept separate from kv_scatter.hpp because the CUDA entry
// points take `cudaStream_t`, which must not be exposed to host-only
// translation units. Included only when VKERNELS_HAS_CUDA; the definitions
// live in kv_scatter.cu.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/comm/kv_scatter.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;
typedef CUstream_st* cudaStream_t_kv;  // avoid pulling cuda_runtime.h into this header

namespace vkernels::comm::cuda {

// Fused indexed K/V scatter for one layer (host-input slot map). Validates
// the contract on the host (null pointers, positive dimensions, BF16/FP16
// element size, non-negative, in-range AND unique destination slots),
// uploads `slot_ids` to a per-launch device buffer (int32 or int64,
// honouring `slot_ids_int64`) and launches ONE kernel that reads every
// page's tokens from the contiguous `src` and writes K and V into the
// indexed destination slots. Enqueued on `stream`; returns without
// synchronising. `num_pages == 0` is a valid no-op. `num_slots` is the
// destination capacity (every slot_id must be unique and in [0, num_slots)).
void kv_scatter_layer(void* k_dst, void* v_dst,
                      const void* slot_ids, bool slot_ids_int64,
                      std::size_t num_slots,
                      const void* src,
                      std::size_t num_pages, std::size_t page_size,
                      std::size_t num_kv_heads, std::size_t head_dim,
                      std::size_t elem_size,
                      cudaStream_t_kv stream);

// Fused indexed K/V scatter for one layer (device-slot map). `slot_ids` is a
// caller-owned DEVICE pointer (shape [num_pages * page_size], int32 when
// `slot_ids_int64 == false`, int64 when true -- e.g. SGLang's torch.int64
// radix-tree indices). The device path is check-free (reading device memory
// to validate uniqueness/range would force a D2H sync): the caller MUST
// guarantee non-negative, in-range AND UNIQUE slots and keep `slot_ids`,
// `k_dst`, `v_dst` and `src` alive until the kernel completes on `stream`.
// One kernel launch, no upload, no validation. `num_pages == 0` is a valid
// no-op.
void kv_scatter_layer_device_slots(void* k_dst, void* v_dst,
                                   const void* slot_ids, bool slot_ids_int64,
                                   std::size_t num_slots,
                                   const void* src,
                                   std::size_t num_pages, std::size_t page_size,
                                   std::size_t num_kv_heads, std::size_t head_dim,
                                   std::size_t elem_size,
                                   cudaStream_t_kv stream);

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
