// vkernels/comm/p2p_gather_c.h
//
// C ABI for the single-launch P2P/UVA run-list gather. Non-C++ consumers
// (e.g. a serving runtime that links the primitive without the C++ headers)
// call these `extern "C"` entry points, which stage run metadata once and
// launch exactly one CUDA kernel (see p2p_gather.cu). Errors are RETURNED as
// codes — no C++ exceptions cross the ABI boundary.
//
// CUDA only: the functions take device pointers and a raw `cudaStream_t`.
// The header is includable from both C and C++; when the CUDA runtime headers
// are present the entry points are declared, otherwise only the POD structs
// and status codes (which are CUDA-independent) are visible.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Pull in cudaStream_t whenever a CUDA toolkit is on the include path. We do
// NOT rely on the project's VKERNELS_HAS_CUDA macro here so a standalone C
// consumer with only the CUDA toolkit installed can use the header.
#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    define VKERNELS_C_HAS_CUDA 1
#    include <cuda_runtime.h>
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Status codes mirroring vkernels::Code. Functions return VKERNELS_OK on
// success; a non-zero code indicates the run list was rejected (invalid
// arguments, capacity violation, overlapping runs) or the launch failed.
typedef enum {
  VKERNELS_OK = 0,
  VKERNELS_ERR_INVALID_ARGUMENT = 1,
  VKERNELS_ERR_OUT_OF_RANGE = 2,
  VKERNELS_ERR_UNSUPPORTED = 3,
  VKERNELS_ERR_INTERNAL = 4,
} vkernels_status_t;

// One strided 2-D run, mirroring vkernels::comm::Gather2DRun: copy a
// `height` x `width`-byte tile from a row-major peer region (`src`, row
// stride `src_stride`) into `dst` at byte offset `dst_offset` with row
// stride `dst_stride`. `width` must not exceed either stride.
typedef struct {
  const void* src;
  size_t src_stride;
  size_t dst_offset;
  size_t dst_stride;
  size_t width;
  size_t height;
} vkernels_gather_2d_run_t;

// 1-D run-list gather. For each i in [0, num_runs), copies `lengths[i]`
// bytes from `src_ptrs[i]` to `dst + dst_offsets[i]`. `num_runs == 0` is a
// valid no-op (returns VKERNELS_OK).
//
// `dst` is a local device allocation of `dst_capacity` bytes; `src_ptrs[i]`
// are peer-accessible UVA pointers. Peer access must be established before
// the call and held until `stream` completes. `stream == NULL` runs to
// completion before returning; a non-null stream is enqueued onto.
#ifdef VKERNELS_C_HAS_CUDA
vkernels_status_t vkernels_p2p_gather_runs(uint8_t* dst, size_t dst_capacity,
                                           const void* const* src_ptrs,
                                           const size_t* dst_offsets,
                                           const size_t* lengths, size_t num_runs,
                                           cudaStream_t stream);

// 2-D strided run-list gather. `runs` is an array of `num_runs`
// descriptors; `num_runs == 0` is a valid no-op.
vkernels_status_t vkernels_p2p_gather_runs_2d(uint8_t* dst, size_t dst_capacity,
                                              const vkernels_gather_2d_run_t* runs,
                                              size_t num_runs, cudaStream_t stream);
#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
