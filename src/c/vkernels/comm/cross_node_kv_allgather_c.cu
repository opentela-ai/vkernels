#include "vkernels/comm/cross_node_kv_allgather_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#include <cstring>
#include <exception>
#include <stdexcept>

#include "vkernels/comm/cross_node_kv_allgather_cuda.hpp"

namespace {

using Communicator = vkernels::comm::cuda::NcclCommunicator;
using AllGatherPlan = vkernels::comm::cuda::CrossNodeKvAllGatherPlan;

vkernels_fi_status_t exception_status(const std::exception& error) {
  return dynamic_cast<const std::invalid_argument*>(&error) != nullptr
             ? VKERNELS_FI_ERR_INVALID_ARGUMENT
             : VKERNELS_FI_ERR_INTERNAL;
}

Communicator* as_communicator(vkernels_nccl_communicator_t* comm) {
  return reinterpret_cast<Communicator*>(comm);
}

const Communicator* as_communicator(
    const vkernels_nccl_communicator_t* comm) {
  return reinterpret_cast<const Communicator*>(comm);
}

AllGatherPlan* as_plan(vkernels_cross_node_kv_allgather_plan_t* plan) {
  return reinterpret_cast<AllGatherPlan*>(plan);
}

const AllGatherPlan* as_plan(
    const vkernels_cross_node_kv_allgather_plan_t* plan) {
  return reinterpret_cast<const AllGatherPlan*>(plan);
}

}  // namespace

extern "C" int vkernels_nccl_is_available(void) {
  return Communicator::is_available() ? 1 : 0;
}

extern "C" int vkernels_nccl_graph_capture_supported(void) {
  return Communicator::graph_capture_supported() ? 1 : 0;
}

extern "C" size_t vkernels_nccl_unique_id_bytes(void) {
  return Communicator::unique_id_bytes();
}

extern "C" vkernels_fi_status_t vkernels_nccl_get_unique_id(
    void* out, size_t capacity) {
  if (!Communicator::is_available()) return VKERNELS_FI_ERR_UNSUPPORTED;
  if (out == nullptr || capacity < Communicator::unique_id_bytes())
    return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  try {
    const auto id = Communicator::make_unique_id();
    std::memcpy(out, id.data(), id.size());
    return VKERNELS_FI_OK;
  } catch (const std::exception& error) {
    return exception_status(error);
  }
}

extern "C" vkernels_nccl_communicator_t*
vkernels_nccl_communicator_create(
    int world, int rank, const void* unique_id, size_t unique_id_size,
    vkernels_fi_status_t* status_out) {
  if (status_out != nullptr) *status_out = VKERNELS_FI_OK;
  if (!Communicator::is_available()) {
    if (status_out != nullptr) *status_out = VKERNELS_FI_ERR_UNSUPPORTED;
    return nullptr;
  }
  try {
    return reinterpret_cast<vkernels_nccl_communicator_t*>(
        new Communicator(world, rank, unique_id, unique_id_size));
  } catch (const std::exception& error) {
    if (status_out != nullptr) *status_out = exception_status(error);
    return nullptr;
  }
}

extern "C" int vkernels_nccl_communicator_world(
    const vkernels_nccl_communicator_t* comm) {
  return comm == nullptr ? -1 : as_communicator(comm)->world();
}

extern "C" int vkernels_nccl_communicator_rank(
    const vkernels_nccl_communicator_t* comm) {
  return comm == nullptr ? -1 : as_communicator(comm)->rank();
}

extern "C" int vkernels_nccl_communicator_device(
    const vkernels_nccl_communicator_t* comm) {
  return comm == nullptr ? -1 : as_communicator(comm)->device();
}

extern "C" vkernels_fi_status_t
vkernels_nccl_communicator_poll_async_error(
    const vkernels_nccl_communicator_t* comm, int* state_out) {
  if (comm == nullptr || state_out == nullptr)
    return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  if (!Communicator::is_available()) return VKERNELS_FI_ERR_UNSUPPORTED;
  try {
    *state_out = as_communicator(comm)->poll_async_error();
    return VKERNELS_FI_OK;
  } catch (const std::exception& error) {
    return exception_status(error);
  }
}

extern "C" vkernels_fi_status_t
vkernels_nccl_communicator_destroy_synchronized(
    vkernels_nccl_communicator_t* comm) {
  if (comm == nullptr) return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  Communicator* cpp = as_communicator(comm);
  try {
    cpp->destroy_synchronized();
    delete cpp;
    return VKERNELS_FI_OK;
  } catch (const std::exception& error) {
    if (!cpp->is_destroyed()) {
      try {
        cpp->abort();
      } catch (...) {
        // Preserve the original finalize failure; delete below sees a
        // best-effort cleanup state and must not throw across the C ABI.
        (void)0;
      }
    }
    delete cpp;
    return exception_status(error);
  }
}

extern "C" vkernels_fi_status_t vkernels_nccl_communicator_abort(
    vkernels_nccl_communicator_t* comm) {
  if (comm == nullptr) return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  Communicator* cpp = as_communicator(comm);
  try {
    cpp->abort();
    delete cpp;
    return VKERNELS_FI_OK;
  } catch (const std::exception& error) {
    delete cpp;
    return exception_status(error);
  }
}

extern "C" vkernels_cross_node_kv_allgather_plan_t*
vkernels_cross_node_kv_allgather_plan_create(
    vkernels_nccl_communicator_t* comm,
    size_t num_slots, size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* global_slot_ids, size_t num_pages, size_t page_size,
    vkernels_fi_status_t* status_out) {
  if (status_out != nullptr) *status_out = VKERNELS_FI_OK;
  if (!Communicator::is_available()) {
    if (status_out != nullptr) *status_out = VKERNELS_FI_ERR_UNSUPPORTED;
    return nullptr;
  }
  try {
    return reinterpret_cast<vkernels_cross_node_kv_allgather_plan_t*>(
        new AllGatherPlan(as_communicator(comm), num_slots, num_kv_heads,
                          head_dim, elem_size, global_slot_ids, num_pages,
                          page_size));
  } catch (const std::exception& error) {
    if (status_out != nullptr) *status_out = exception_status(error);
    return nullptr;
  }
}

extern "C" void vkernels_cross_node_kv_allgather_plan_destroy(
    vkernels_cross_node_kv_allgather_plan_t* plan) {
  delete as_plan(plan);
}

extern "C" size_t vkernels_cross_node_kv_allgather_plan_total_bytes(
    const vkernels_cross_node_kv_allgather_plan_t* plan) {
  return plan == nullptr ? 0 : as_plan(plan)->total_bytes();
}

extern "C" size_t
vkernels_cross_node_kv_allgather_plan_local_shard_bytes(
    const vkernels_cross_node_kv_allgather_plan_t* plan) {
  return plan == nullptr ? 0 : as_plan(plan)->local_shard_bytes();
}

extern "C" size_t
vkernels_cross_node_kv_allgather_plan_local_num_pages(
    const vkernels_cross_node_kv_allgather_plan_t* plan) {
  return plan == nullptr ? 0 : as_plan(plan)->local_num_pages();
}

extern "C" vkernels_fi_status_t
vkernels_cross_node_kv_allgather_plan_execute(
    vkernels_cross_node_kv_allgather_plan_t* plan,
    const void* k_src, const void* v_src, void* k_dst, void* v_dst,
    cudaStream_t stream) {
  if (plan == nullptr) return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  if (!Communicator::is_available()) return VKERNELS_FI_ERR_UNSUPPORTED;
  try {
    as_plan(plan)->execute(k_src, v_src, k_dst, v_dst, stream);
    return VKERNELS_FI_OK;
  } catch (const std::exception& error) {
    return exception_status(error);
  }
}

#endif  // VKERNELS_C_HAS_CUDA && !__CUDA_ARCH__
