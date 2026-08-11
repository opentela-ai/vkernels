// vkernels/capi/capi.hpp — C ABI for the vkernels library.
//
// The C++ API under csrc/vkernels/ is consumed directly by C++ (tests,
// benchmarks) and through a thin pybind11 layer (src/vkernels/_core.cpp).
// This header exposes the same surface as a plain C ABI so that bindings in
// languages without a C++ toolchain — first of all the Rust crate under
// rust/ — can link the library without a C++ compiler. The functions are
// implemented in capi.cpp and are compiled into the vkernels static library
// itself, so there is nothing extra to link.
//
// Conventions
// -----------
//  * Buffers are raw pointers + lengths. Every function that can fail
//    (contract checks, allocations) returns an int32_t status code: VK_OK on
//    success, one of the VK_ERROR_* codes otherwise. Pure getters that cannot
//    throw return their value directly.
//  * The C++ contract checks (VK_EXPECTS / VK_ENSURES) throw exceptions,
//    which cannot cross the ABI. capi.cpp catches every exception and stores
//    the message in a thread-local buffer; call vk_last_error() to read it.
//    The message is valid until the next failing call on the same thread.
//  * Functions that allocate results (vk_queue_pop, vk_channel_recv,
//    vk_stage_runs_1d/2d, vk_make_ring_channels) return memory owned by the
//    caller; release it with vk_free() (and, for the channel arrays,
//    vk_channel_delete() on each element first).
//  * Opaque handle types (vk_device, vk_stream, vk_queue, vk_channel,
//    vk_overlap) are heap-allocated wrappers around the C++ objects; create
//    with vk_*_new and destroy with vk_*_delete. Handles are not thread-safe
//    to destroy while another thread uses them.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes (mirror vkernels::Code)                                */
/* ------------------------------------------------------------------ */
enum {
  VK_OK = 0,
  VK_ERROR_INVALID_ARGUMENT = 1,
  VK_ERROR_OUT_OF_RANGE = 2,
  VK_ERROR_UNSUPPORTED = 3,
  VK_ERROR_INTERNAL = 4
};

/* ------------------------------------------------------------------ */
/* Version / config                                                    */
/* ------------------------------------------------------------------ */

/* Static version string of the C++ library ("0.1.0"). */
const char* vk_version(void);

/* 1 when the library was compiled with CUDA enabled, else 0. */
int vk_has_cuda(void);

/* Thread-local message of the most recent failing call ("" if none). */
const char* vk_last_error(void);

/* Status code of the most recent failing call (VK_OK if none). */
int vk_last_error_code(void);

/* Free memory returned by vk_* functions that allocate results. */
void vk_free(void* p);

/* ------------------------------------------------------------------ */
/* kernels: element-wise, reduce, gemm (csrc/vkernels/kernels)         */
/* ------------------------------------------------------------------ */

/* out = a + b. Lengths must match; raises VK_ERROR_INVALID_ARGUMENT. */
int32_t vk_add(const float* a, size_t a_len, const float* b, size_t b_len,
               float* out, size_t out_len);

/* out = alpha * x. Lengths must match. */
int32_t vk_scale(const float* x, size_t x_len, float alpha, float* out,
                 size_t out_len);

/* out = max(x, 0). Lengths must match. */
int32_t vk_relu(const float* x, size_t x_len, float* out, size_t out_len);

/* *out = sum(x). x must be non-empty. */
int32_t vk_sum(const float* x, size_t x_len, float* out);

/* *out = max(x). x must be non-empty. */
int32_t vk_max(const float* x, size_t x_len, float* out);

/* C = alpha * A @ B + beta * C. A is M*K, B is K*N, C is M*N elements. */
int32_t vk_gemm(size_t M, size_t N, size_t K, float alpha, const float* A,
                size_t A_len, const float* B, size_t B_len, float beta,
                float* C, size_t C_len);

/* ------------------------------------------------------------------ */
/* core: device + stream (csrc/vkernels/core)                          */
/* ------------------------------------------------------------------ */

typedef struct vk_device vk_device;

vk_device* vk_device_new(int index); /* index == -1 selects the default. */
void vk_device_delete(vk_device* d);

int vk_device_index(const vk_device* d);
/* 1 when d can access `other` directly (always 0 on a host build). */
int vk_device_supports_peer(const vk_device* d, const vk_device* other);
int vk_device_eq(const vk_device* a, const vk_device* b);

/* Make this device current / block until its work finishes. On the host
 * build both are no-ops; under CUDA they can raise VK_ERROR_INTERNAL. */
int32_t vk_device_set_current(vk_device* d);
int32_t vk_device_sync(vk_device* d);

typedef struct vk_stream vk_stream;

vk_stream* vk_stream_new(void);
void vk_stream_delete(vk_stream* s);

/* Enqueue `fn(ctx)` to run on the stream, in submission order. The
 * callback runs on the stream's worker thread; it must not call back into
 * the stream. */
int32_t vk_stream_submit(vk_stream* s, void (*fn)(void*), void* ctx);
/* Block the calling thread until every submitted task has run. */
void vk_stream_wait(vk_stream* s);
/* Number of tasks submitted so far (completed + queued). */
size_t vk_stream_submitted(const vk_stream* s);

/* ------------------------------------------------------------------ */
/* comm: topology (csrc/vkernels/comm/topology.hpp)                    */
/* ------------------------------------------------------------------ */

typedef struct vk_topology {
  int32_t rank;
  int32_t world;
  int32_t next; /* (rank + 1) % world */
  int32_t prev; /* (rank - 1 + world) % world */
} vk_topology;

/* *out = ring slot for `rank` of a ring of `world` ranks. */
int32_t vk_ring_rank(int32_t rank, int32_t world, vk_topology* out);

/* Fills `out` with `world` entries, one per rank; the array is malloc'd and
 * owned by the caller (release with vk_free). */
int32_t vk_build_ring_topology(int32_t world, vk_topology** out,
                               size_t* out_count);

/* ------------------------------------------------------------------ */
/* comm: channels (csrc/vkernels/comm/channel.hpp)                     */
/* ------------------------------------------------------------------ */

typedef struct vk_queue vk_queue;   /* BlockingQueue of float32 chunks */
typedef struct vk_channel vk_channel; /* MockChannel (send/recv) */

vk_queue* vk_queue_new(void);
void vk_queue_delete(vk_queue* q);

/* Append one float32 chunk (copied). */
int32_t vk_queue_push(vk_queue* q, const float* data, size_t len);
/* Blocking pop; returns a malloc'd copy of the chunk (vk_free it). */
int32_t vk_queue_pop(vk_queue* q, float** out_data, size_t* out_len);
void vk_queue_close(vk_queue* q);
int vk_queue_closed(const vk_queue* q);

/* Channel that sends into `out` and receives from `in`. */
vk_channel* vk_channel_new(vk_queue* out, vk_queue* in);
void vk_channel_delete(vk_channel* c);

/* Blocking send of one float32 chunk to the peer (copied). */
int32_t vk_channel_send(vk_channel* c, const float* data, size_t len);
/* Blocking receive; returns a malloc'd copy (vk_free it). */
int32_t vk_channel_recv(vk_channel* c, float** out_data, size_t* out_len);
int vk_channel_closed(const vk_channel* c);

/* Build `world` mock channels in a ring: channel[r].send() reaches
 * channel[(r+1) % world].recv(). On success *out is a malloc'd array of
 * `world` channel handles; delete each with vk_channel_delete, then
 * vk_free the array. */
int32_t vk_make_ring_channels(int32_t world, vk_channel*** out,
                              size_t* out_count);

/* ------------------------------------------------------------------ */
/* comm: ring all-reduce (csrc/vkernels/comm/allreduce.hpp)            */
/* ------------------------------------------------------------------ */

/* Run rank `rank` of a ring all-reduce; on success `local` holds the
 * element-wise sum across every rank. `local` is unchanged on error. */
int32_t vk_ring_allreduce_rank(float* local, size_t local_len, int32_t rank,
                               int32_t world, vk_channel* next,
                               vk_channel* prev);

/* ------------------------------------------------------------------ */
/* comm: overlap (csrc/vkernels/comm/overlap.hpp)                      */
/* ------------------------------------------------------------------ */

typedef struct vk_overlap vk_overlap;

vk_overlap* vk_overlap_new(void);
void vk_overlap_delete(vk_overlap* ex);
int vk_overlap_uses_two_streams(const vk_overlap* ex);

/* Run `iters` iterations: compute(i) -> int on stream A, comm(i, value) on
 * stream B; the data dependency is honoured via a per-iteration future. The
 * callbacks run on the executor's worker threads; both must be non-null.
 * On success *out_compute_count and *out_comm_count are both `iters`. */
int32_t vk_overlap_run(vk_overlap* ex, size_t iters,
                       int (*compute)(size_t i, void* ctx), void* compute_ctx,
                       void (*comm)(size_t i, int value, void* ctx),
                       void* comm_ctx, size_t* out_compute_count,
                       size_t* out_comm_count);

/* ------------------------------------------------------------------ */
/* comm: p2p run-list gather (csrc/vkernels/comm/p2p_gather.hpp)       */
/* ------------------------------------------------------------------ */

/* 1-D copy run (mirrors comm::StagedRun1D). */
typedef struct vk_staged_run_1d {
  const void* src;
  size_t dst_offset;
  size_t length;
} vk_staged_run_1d;

/* Strided 2-D tile input (mirrors comm::Gather2DRun). */
typedef struct vk_gather_2d {
  const void* src;
  size_t src_stride;
  size_t dst_offset;
  size_t dst_stride;
  size_t width;
  size_t height;
} vk_gather_2d;

/* Staged 2-D tile (mirrors comm::StagedRun2D). */
typedef struct vk_staged_run_2d {
  const void* src;
  size_t dst_offset;
  size_t src_stride;
  size_t dst_stride;
  size_t width;
  size_t height;
} vk_staged_run_2d;

/* Validate a 1-D run list against `dst` and return the staged runs as a
 * malloc'd array of `*out_count` vk_staged_run_1d (vk_free it). Raises
 * VK_ERROR_INVALID_ARGUMENT on contract violations. */
int32_t vk_stage_runs_1d(const uint8_t* dst, size_t dst_capacity,
                         const void* const* src_ptrs,
                         const size_t* dst_offsets, const size_t* lengths,
                         size_t num_runs, vk_staged_run_1d** out,
                         size_t* out_count);

/* Validate a 2-D run list against `dst` and return the staged tiles as a
 * malloc'd array of `*out_count` vk_staged_run_2d (vk_free it). */
int32_t vk_stage_runs_2d(const uint8_t* dst, size_t dst_capacity,
                         const vk_gather_2d* runs, size_t num_runs,
                         vk_staged_run_2d** out, size_t* out_count);

/* Copy every 1-D run into dst in a single operation (one stream task when
 * `stream` is non-null; synchronous when null). `dst` must outlive the
 * stream. */
int32_t vk_p2p_gather_runs(uint8_t* dst, size_t dst_capacity,
                           const void* const* src_ptrs,
                           const size_t* dst_offsets, const size_t* lengths,
                           size_t num_runs, vk_stream* stream);

/* Copy every 2-D tile into dst in a single operation. */
int32_t vk_p2p_gather_runs_2d(uint8_t* dst, size_t dst_capacity,
                              const vk_gather_2d* runs, size_t num_runs,
                              vk_stream* stream);

/* Legacy seam: one copy per run (one stream task per run, so
 * vk_stream_submitted grows by the run count instead of 1). */
int32_t vk_memcpy_peer_batch_async(uint8_t* dst, size_t dst_capacity,
                                   const void* const* src_ptrs,
                                   const size_t* dst_offsets,
                                   const size_t* lengths, size_t num_runs,
                                   vk_stream* stream);

#ifdef __cplusplus
}  // extern "C"
#endif
