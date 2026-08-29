// vkernels/comm/cross_node_kv_allgather.cu
#include "vkernels/comm/cross_node_kv_allgather_cuda.hpp"

#if VKERNELS_HAS_CUDA

#include <cuda_runtime.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
#include <nccl.h>
#endif

#include "vkernels/util/error.hpp"

namespace vkernels::comm::cuda {

namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* name) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument(std::string(name) + " overflows size_t");
  return a * b;
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
void check_nccl(ncclResult_t status, const char* operation) {
  if (status != ncclSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             ncclGetErrorString(status));
}

ncclComm_t as_nccl(void* comm) {
  return reinterpret_cast<ncclComm_t>(comm);
}
#endif

}  // namespace

bool NcclCommunicator::is_available() {
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  return true;
#else
  return false;
#endif
}

bool NcclCommunicator::graph_capture_supported() {
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL && \
    ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2 && NCCL_MINOR >= 9)) && \
    CUDART_VERSION >= 11030
  return true;
#else
  return false;
#endif
}

std::size_t NcclCommunicator::unique_id_bytes() {
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  return NCCL_UNIQUE_ID_BYTES;
#else
  return 0;
#endif
}

std::vector<std::uint8_t> NcclCommunicator::make_unique_id() {
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  ncclUniqueId id{};
  check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&id);
  return std::vector<std::uint8_t>(begin, begin + sizeof(id));
#else
  throw std::runtime_error("vkernels was built without NCCL support");
#endif
}

NcclCommunicator::NcclCommunicator(int world, int rank,
                                   const void* unique_id,
                                   std::size_t unique_id_size)
    : world_(world), rank_(rank) {
  VK_EXPECTS(world_ > 0, "NCCL world must be positive");
  VK_EXPECTS(rank_ >= 0 && rank_ < world_, "NCCL rank must be in [0, world)");
  VK_EXPECTS(unique_id != nullptr, "NCCL unique id must be non-null");
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  VK_EXPECTS(unique_id_size == sizeof(ncclUniqueId),
             "NCCL unique id has the wrong size");
  check_cuda(cudaGetDevice(&device_), "cudaGetDevice");
  ncclUniqueId id{};
  std::memcpy(&id, unique_id, sizeof(id));
  ncclComm_t raw = nullptr;
  check_nccl(ncclCommInitRank(&raw, world_, id, rank_), "ncclCommInitRank");
  comm_ = reinterpret_cast<void*>(raw);
#else
  (void)unique_id_size;
  throw std::runtime_error("vkernels was built without NCCL support");
#endif
}

NcclCommunicator::~NcclCommunicator() {
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  if (comm_ != nullptr) {
    (void)ncclCommDestroy(as_nccl(comm_));
    comm_ = nullptr;
  }
#endif
}

void NcclCommunicator::destroy_synchronized() {
  VK_EXPECTS(comm_ != nullptr, "NCCL communicator is already destroyed");
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  ncclComm_t raw = as_nccl(comm_);
#if NCCL_VERSION_CODE >= 21500
  check_nccl(ncclCommFinalize(raw), "ncclCommFinalize");
#endif
  check_nccl(ncclCommDestroy(raw), "ncclCommDestroy");
  comm_ = nullptr;
#else
  throw std::runtime_error("vkernels was built without NCCL support");
#endif
}

int NcclCommunicator::poll_async_error() const {
  VK_EXPECTS(comm_ != nullptr, "NCCL communicator is destroyed");
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  ncclResult_t async_error = ncclSuccess;
  check_nccl(ncclCommGetAsyncError(as_nccl(comm_), &async_error),
             "ncclCommGetAsyncError");
  if (async_error == ncclSuccess) return 0;
  if (async_error == ncclInProgress) return 1;
  return 2;
#else
  throw std::runtime_error("vkernels was built without NCCL support");
#endif
}

void NcclCommunicator::abort() {
  VK_EXPECTS(comm_ != nullptr, "NCCL communicator is already destroyed");
#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  ncclComm_t raw = as_nccl(comm_);
  check_nccl(ncclCommAbort(raw), "ncclCommAbort");
  comm_ = nullptr;
#else
  throw std::runtime_error("vkernels was built without NCCL support");
#endif
}

CrossNodeKvAllGatherPlan::CrossNodeKvAllGatherPlan(
    NcclCommunicator* comm, std::size_t num_slots,
    std::size_t num_kv_heads, std::size_t head_dim, std::size_t elem_size,
    const int* global_slot_ids, std::size_t num_pages, std::size_t page_size)
    : comm_(comm),
      num_slots_(num_slots),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      elem_size_(elem_size),
      num_pages_(num_pages),
      page_size_(page_size) {
  VK_EXPECTS(comm_ != nullptr && !comm_->is_destroyed(),
             "all-gather plan needs a live NCCL communicator");
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
  VK_EXPECTS(current_device == comm_->device(),
             "all-gather plan must be created on its communicator device");
  world_ = comm_->world();
  rank_ = comm_->rank();
  VK_EXPECTS(world_ > 1, "all-gather plan needs at least two ranks");
  VK_EXPECTS(num_pages_ % static_cast<std::size_t>(world_) == 0,
             "all-gather num_pages must divide evenly by world");
  VK_EXPECTS(elem_size_ == 2, "elem_size must be 2 for BF16/FP16");
  if (num_pages_ == 0) return;
  VK_EXPECTS(num_slots_ > 0, "num_slots must be positive");
  VK_EXPECTS(num_kv_heads_ > 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim_ > 0, "head_dim must be positive");
  VK_EXPECTS(page_size_ > 0, "page_size must be positive");
  VK_EXPECTS(global_slot_ids != nullptr, "global_slot_ids must be non-null");

  const std::size_t total_tokens =
      checked_mul(num_pages_, page_size_, "all-gather token count");
  std::unordered_set<int> seen;
  seen.reserve(total_tokens);
  for (std::size_t i = 0; i < total_tokens; ++i) {
    const int slot = global_slot_ids[i];
    VK_EXPECTS(slot >= 0, "global_slot_ids must be non-negative");
    VK_EXPECTS(static_cast<std::size_t>(slot) < num_slots_,
               "global_slot_ids must be < num_slots");
    VK_EXPECTS(seen.insert(slot).second, "global_slot_ids must be unique");
  }

  local_num_pages_ = num_pages_ / static_cast<std::size_t>(world_);
  const std::size_t local_tokens =
      checked_mul(local_num_pages_, page_size_, "all-gather local token count");
  std::size_t token_bytes = checked_mul(2, num_kv_heads_, "KV token bytes");
  token_bytes = checked_mul(token_bytes, head_dim_, "KV token bytes");
  token_bytes = checked_mul(token_bytes, elem_size_, "KV token bytes");
  total_bytes_ = checked_mul(total_tokens, token_bytes, "all-gather bytes");
  local_shard_bytes_ =
      checked_mul(local_tokens, token_bytes, "all-gather local bytes");

  const std::size_t global_slot_bytes =
      checked_mul(total_tokens, sizeof(int), "global slot bytes");
  const std::size_t local_slot_bytes =
      checked_mul(local_tokens, sizeof(int), "local slot bytes");
  const int* local_slots =
      global_slot_ids + static_cast<std::size_t>(rank_) * local_tokens;

  try {
    check_cuda(cudaMalloc(&dev_global_slots_, global_slot_bytes),
               "cudaMalloc global all-gather slots");
    check_cuda(cudaMalloc(&dev_local_slots_, local_slot_bytes),
               "cudaMalloc local all-gather slots");
    check_cuda(cudaMalloc(&send_buffer_, local_shard_bytes_),
               "cudaMalloc all-gather send buffer");
    check_cuda(cudaMalloc(&receive_buffer_, total_bytes_),
               "cudaMalloc all-gather receive buffer");
    check_cuda(cudaMemcpy(dev_global_slots_, global_slot_ids, global_slot_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy global all-gather slots");
    check_cuda(cudaMemcpy(dev_local_slots_, local_slots, local_slot_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy local all-gather slots");
  } catch (...) {
    release_device_state();
    throw;
  }
}

CrossNodeKvAllGatherPlan::~CrossNodeKvAllGatherPlan() {
  release_device_state();
}

void CrossNodeKvAllGatherPlan::release_device_state() noexcept {
  if (receive_buffer_ != nullptr) cudaFree(receive_buffer_);
  if (send_buffer_ != nullptr) cudaFree(send_buffer_);
  if (dev_local_slots_ != nullptr) cudaFree(dev_local_slots_);
  if (dev_global_slots_ != nullptr) cudaFree(dev_global_slots_);
  receive_buffer_ = nullptr;
  send_buffer_ = nullptr;
  dev_local_slots_ = nullptr;
  dev_global_slots_ = nullptr;
}

void CrossNodeKvAllGatherPlan::execute(
    const void* k_src, const void* v_src, void* k_dst, void* v_dst,
    cudaStream_t_kv stream) const {
  VK_EXPECTS(k_src != nullptr || num_pages_ == 0, "k_src must be non-null");
  VK_EXPECTS(v_src != nullptr || num_pages_ == 0, "v_src must be non-null");
  VK_EXPECTS(k_dst != nullptr || num_pages_ == 0, "k_dst must be non-null");
  VK_EXPECTS(v_dst != nullptr || num_pages_ == 0, "v_dst must be non-null");
  VK_EXPECTS(comm_ != nullptr && !comm_->is_destroyed(),
             "all-gather communicator is destroyed");
  if (num_pages_ == 0) return;
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
  VK_EXPECTS(current_device == comm_->device(),
             "all-gather execute must use its communicator device");

  kv_gather_layer_device_slots(
      send_buffer_, k_src, v_src, dev_local_slots_, false, num_slots_,
      local_num_pages_, page_size_, num_kv_heads_, head_dim_, elem_size_, stream);

#if defined(VKERNELS_HAS_NCCL) && VKERNELS_HAS_NCCL
  check_nccl(ncclAllGather(send_buffer_, receive_buffer_, local_shard_bytes_,
                           ncclChar, as_nccl(comm_->native_handle()), stream),
             "ncclAllGather");
#else
  throw std::runtime_error("vkernels was built without NCCL support");
#endif

  kv_scatter_layer_device_slots(
      k_dst, v_dst, dev_global_slots_, false, num_slots_, receive_buffer_,
      num_pages_, page_size_, num_kv_heads_, head_dim_, elem_size_, stream);
}

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
