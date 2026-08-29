// Stable C ABI for the equal-shard NCCL KV all-gather communicator and plan.
#pragma once

#include <stddef.h>

#include "vkernels/comm/fabric_import_c.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runtime/build capability probe. Returns 1 only when libvkernels_c was built
// and linked with NCCL; otherwise every communicator/plan create returns
// VKERNELS_FI_ERR_UNSUPPORTED.
int vkernels_nccl_is_available(void);
int vkernels_nccl_graph_capture_supported(void);

// NCCL bootstrap id helpers. Rank 0 creates one id, distributes exactly this
// many opaque bytes out of band, and every rank passes the same bytes to
// communicator_create.
size_t vkernels_nccl_unique_id_bytes(void);
vkernels_fi_status_t vkernels_nccl_get_unique_id(
    void* out, size_t capacity);

typedef struct vkernels_nccl_communicator vkernels_nccl_communicator_t;

// Collective host call. The caller must select the intended CUDA device before
// calling. Every rank calls with the same id/world and a distinct rank.
vkernels_nccl_communicator_t* vkernels_nccl_communicator_create(
    int world, int rank, const void* unique_id, size_t unique_id_size,
    vkernels_fi_status_t* status_out);

int vkernels_nccl_communicator_world(
    const vkernels_nccl_communicator_t* comm);
int vkernels_nccl_communicator_rank(
    const vkernels_nccl_communicator_t* comm);
int vkernels_nccl_communicator_device(
    const vkernels_nccl_communicator_t* comm);

typedef enum {
  VKERNELS_NCCL_ASYNC_HEALTHY = 0,
  VKERNELS_NCCL_ASYNC_IN_PROGRESS = 1,
  VKERNELS_NCCL_ASYNC_ERROR = 2,
} vkernels_nccl_async_state_t;

// Poll network/communicator failures while waiting for stream completion.
// A caller observing ASYNC_ERROR must abort and recreate the communicator.
vkernels_fi_status_t vkernels_nccl_communicator_poll_async_error(
    const vkernels_nccl_communicator_t* comm, int* state_out);

// Normal teardown requires all streams using the communicator to be complete.
// Abort is the failure-path teardown when a rank or collective has failed.
vkernels_fi_status_t vkernels_nccl_communicator_destroy_synchronized(
    vkernels_nccl_communicator_t* comm);
vkernels_fi_status_t vkernels_nccl_communicator_abort(
    vkernels_nccl_communicator_t* comm);

typedef struct vkernels_cross_node_kv_allgather_plan
    vkernels_cross_node_kv_allgather_plan_t;

#ifdef VKERNELS_C_HAS_CUDA

// `global_slot_ids` is HOST int32 [num_pages * page_size], unique and in
// range, ordered as rank 0 shard, rank 1 shard, ... . `num_pages` must divide
// evenly by communicator world size. The communicator is borrowed and must
// outlive the plan and every execution.
vkernels_cross_node_kv_allgather_plan_t*
vkernels_cross_node_kv_allgather_plan_create(
    vkernels_nccl_communicator_t* comm,
    size_t num_slots, size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* global_slot_ids, size_t num_pages, size_t page_size,
    vkernels_fi_status_t* status_out);

// Destroy only after every execution stream has completed.
void vkernels_cross_node_kv_allgather_plan_destroy(
    vkernels_cross_node_kv_allgather_plan_t* plan);

size_t vkernels_cross_node_kv_allgather_plan_total_bytes(
    const vkernels_cross_node_kv_allgather_plan_t* plan);
size_t vkernels_cross_node_kv_allgather_plan_local_shard_bytes(
    const vkernels_cross_node_kv_allgather_plan_t* plan);
size_t vkernels_cross_node_kv_allgather_plan_local_num_pages(
    const vkernels_cross_node_kv_allgather_plan_t* plan);

// Enqueues gather -> ncclAllGather -> scatter on `stream`. Source and
// destination K/V pointers may alias. Returns before stream completion.
vkernels_fi_status_t vkernels_cross_node_kv_allgather_plan_execute(
    vkernels_cross_node_kv_allgather_plan_t* plan,
    const void* k_src, const void* v_src, void* k_dst, void* v_dst,
    cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
