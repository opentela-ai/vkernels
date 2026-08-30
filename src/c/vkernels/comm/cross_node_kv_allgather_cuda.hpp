// vkernels/comm/cross_node_kv_allgather_cuda.hpp
//
// Equal-shard cross-node KV all-gather over an owned NCCL communicator.
// The plan owns device slot maps and packed buffers; execute() enqueues
// gather -> ncclAllGather -> scatter on one CUDA stream.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vkernels/comm/kv_gather_cuda.hpp"
#include "vkernels/comm/kv_scatter_cuda.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA

namespace vkernels::comm::cuda {

// RAII owner for one rank of an NCCL communicator. Construction is a
// collective host operation: every rank must call it with the same unique id,
// world size, and a distinct rank. The current CUDA device is captured and
// must remain the device on which the communicator is used.
class NcclCommunicator {
 public:
  static bool is_available();
  static bool graph_capture_supported();
  static std::size_t unique_id_bytes();
  static std::vector<std::uint8_t> make_unique_id();

  NcclCommunicator(int world, int rank, const void* unique_id,
                   std::size_t unique_id_size);
  ~NcclCommunicator();

  NcclCommunicator(const NcclCommunicator&) = delete;
  NcclCommunicator& operator=(const NcclCommunicator&) = delete;

  int world() const { return world_; }
  int rank() const { return rank_; }
  int device() const { return device_; }
  void* native_handle() const { return comm_; }

  // Call only after every stream using the communicator has completed.
  void destroy_synchronized();
  // Emergency teardown for a failed rank/collective. Safe to call once.
  void abort();
  // 0 = healthy/complete, 1 = NCCL still in progress, 2 = fatal async error.
  int poll_async_error() const;
  bool is_destroyed() const { return comm_ == nullptr; }

 private:
  int world_ = 0;
  int rank_ = 0;
  int device_ = -1;
  void* comm_ = nullptr;  // ncclComm_t without leaking nccl.h into the header
};

// A reusable all-gather for one globally described paged KV set.
//
// `global_slot_ids` is a HOST int32 array of
// [num_pages * page_size] UNIQUE destination slots, in rank-major shard
// order. `num_pages` must divide evenly by comm.world(); each rank packs the
// contiguous page slice assigned to its rank. Every rank must construct the
// same geometry and global slot ordering.
//
// The communicator is borrowed and must outlive the plan and all executions.
// The plan owns device copies of both slot maps plus its send/receive buffers.
class CrossNodeKvAllGatherPlan {
 public:
  CrossNodeKvAllGatherPlan(NcclCommunicator* comm,
                           std::size_t num_slots,
                           std::size_t num_kv_heads,
                           std::size_t head_dim,
                           std::size_t elem_size,
                           const int* global_slot_ids,
                           std::size_t num_pages,
                           std::size_t page_size);
  ~CrossNodeKvAllGatherPlan();

  CrossNodeKvAllGatherPlan(const CrossNodeKvAllGatherPlan&) = delete;
  CrossNodeKvAllGatherPlan& operator=(const CrossNodeKvAllGatherPlan&) = delete;

  std::size_t num_pages() const { return num_pages_; }
  std::size_t local_num_pages() const { return local_num_pages_; }
  std::size_t page_size() const { return page_size_; }
  std::size_t total_bytes() const { return total_bytes_; }
  std::size_t local_shard_bytes() const { return local_shard_bytes_; }
  bool is_graph_capturable() const {
    return NcclCommunicator::graph_capture_supported();
  }

  // Enqueue one layer. `k_src`/`v_src` may alias `k_dst`/`v_dst`: the local
  // gather completes before the collective, and scatter begins only after the
  // collective completes, because all three operations share `stream`. CUDA
  // graph capture/launch is collective too: every rank must capture or launch
  // the matching operation uniformly.
  void execute(const void* k_src, const void* v_src,
               void* k_dst, void* v_dst,
               cudaStream_t_kv stream) const;

 private:
  void release_device_state() noexcept;

  NcclCommunicator* comm_ = nullptr;  // borrowed
  std::size_t num_slots_ = 0;
  std::size_t num_kv_heads_ = 0;
  std::size_t head_dim_ = 0;
  std::size_t elem_size_ = 0;
  std::size_t num_pages_ = 0;
  std::size_t local_num_pages_ = 0;
  std::size_t page_size_ = 0;
  std::size_t total_bytes_ = 0;
  std::size_t local_shard_bytes_ = 0;
  int* dev_local_slots_ = nullptr;
  int* dev_global_slots_ = nullptr;
  void* send_buffer_ = nullptr;
  void* receive_buffer_ = nullptr;
};

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
