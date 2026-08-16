// vkernels/comm/pipeline_boundary_c.h
//
// C ABI for the graph-capturable PP-boundary transfer (issue #10).
// Non-C++ consumers call these `extern "C"` entry points. Errors are
// RETURNED as codes — no C++ exception crosses the ABI boundary.
//
// Two layers, mirroring rccl_c (always-compiled host planning surface) and
// p2p_kv_restore_c (CUDA-only device primitive):
//   * The transport classification + eager-break decision are ALWAYS
//     visible and callable without a GPU (pipeline_boundary_c.cpp).
//   * The prepared device plan (create / execute / destroy over a
//     cudaStream_t) is visible only when the CUDA runtime headers are
//     present (pipeline_boundary_c.cu).
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
  VKERNELS_PP_OK = 0,
  VKERNELS_PP_ERR_INVALID_ARGUMENT = 1,
  VKERNELS_PP_ERR_OUT_OF_RANGE = 2,
  VKERNELS_PP_ERR_UNSUPPORTED = 3,
  VKERNELS_PP_ERR_INTERNAL = 4,
} vkernels_pp_status_t;

// Transport a PP-boundary transfer takes (mirrors
// vkernels::comm::PipelineTransport; integer values are stable across the
// ABI).
typedef enum {
  VKERNELS_PP_TRANSPORT_SAME_NODE_PEER = 0,
  VKERNELS_PP_TRANSPORT_CROSS_NODE_NCCL = 1,
  VKERNELS_PP_TRANSPORT_HOST_STAGED = 2,
} vkernels_pp_transport_t;

// Direction of one rank's participation in a directed boundary (mirrors
// vkernels::comm::BoundaryDirection).
typedef enum {
  VKERNELS_PP_DIR_SEND = 0,
  VKERNELS_PP_DIR_RECV = 1,
} vkernels_pp_dir_t;

// Deployment facts a PP boundary is classified from (mirrors
// vkernels::comm::PipelineBoundaryConfig; 0/1 booleans for C portability).
typedef struct {
  int same_node;             // peer stage on the same node
  int nccl_graph_supported;  // NCCL/RCCL graph-capture API available
  int gloo_fallback;         // host-side gloo wired for the boundary
} vkernels_pp_config_t;

// Classify the boundary transport. On success returns one of
// VKERNELS_PP_TRANSPORT_* and, when status_out is non-null, sets it to
// VKERNELS_PP_OK. On a null config sets *status_out to
// VKERNELS_PP_ERR_INVALID_ARGUMENT and returns
// VKERNELS_PP_TRANSPORT_HOST_STAGED.
int vkernels_pp_classify(const vkernels_pp_config_t* cfg,
                         vkernels_pp_status_t* status_out);

// Eager-break decision mirroring vLLM `eager_break_during_capture`: 1 when
// the boundary is NOT graph-capturable (the host send/recv must be excluded
// from the captured segment and run between launches), 0 otherwise. Same
// status contract as vkernels_pp_classify.
int vkernels_pp_eager_break(const vkernels_pp_config_t* cfg,
                            vkernels_pp_status_t* status_out);

// Human-readable transport name for VKERNELS_PP_TRANSPORT_*, or "?" on an
// unknown value. Never returns null.
const char* vkernels_pp_transport_name(int transport);

#ifdef VKERNELS_C_HAS_CUDA

// A prepared directed device transfer over one PP boundary. Validate once
// at create; execute() enqueues ONE `cudaMemcpyAsync` (peer) or
// `ncclSend` / `ncclRecv` (cross-node, when NCCL is linked) on `stream`, so
// one plan moves one boundary's hidden state every decode iteration with no
// per-iteration validation or allocation. `kHostStaged` is refused at
// create (a host-staged boundary is never device-captured — use the host
// reference's eager-break path instead).
//
// On success the create function returns a non-NULL handle and sets
// *status_out to VKERNELS_PP_OK; on a contract violation or device failure
// it returns NULL and sets *status_out. Destroy the plan only after every
// stream it was executed on has been synchronised.
typedef struct vkernels_pp_boundary_plan vkernels_pp_boundary_plan_t;

vkernels_pp_boundary_plan_t* vkernels_pp_boundary_plan_create(
    int world, int rank, size_t payload_bytes,
    int transport, int dir, void* peer_buf,
    vkernels_pp_status_t* status_out);

void vkernels_pp_boundary_plan_destroy(vkernels_pp_boundary_plan_t* plan);

// Enqueue one directed transfer for `my_buf` (payload_bytes from create) on
// `stream`. Returns VKERNELS_PP_OK on success, VKERNELS_PP_ERR_INVALID_ARGUMENT
// on a null plan / my_buf, VKERNELS_PP_ERR_UNSUPPORTED when the cross-node
// transport was requested but NCCL is not linked, or VKERNELS_PP_ERR_INTERNAL
// on a launch failure.
int vkernels_pp_boundary_plan_execute(vkernels_pp_boundary_plan_t* plan,
                                      void* my_buf, cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
