// vkernels/comm/cross_node_kv.cu — CUDA device path for the cross-node KV
// restore / donate plans (issue #49).
//
// The host reference (cross_node_kv.cpp) is the always-compiled,
// 100%-line-covered correctness oracle and carries the full contract
// (transport classification, the ByteChannel eager-break, the GraphCapture
// integration, the byte-identical acceptance). This CUDA path is the
// device realization of issue #49 scope #1: cuda::fabric_import_device_ptr
// (fabric_import.cu) yields a directly device-addressable pointer to the
// remote VRAM; this plan lays the per-page peer bases out as
// `imported_device_ptr + p * page_layer_bytes`, and the EXISTING fused
// P2PKvRestorePlan / P2PKvDonatePlan (issues #27, #36) run over it on a
// real cudaStream — ONE kernel launch, the peer loads/stores now naming
// cross-node memory instead of a same-node peer. The kernels run
// UNCHANGED; only the peer base they dereference is cross-node.
//
// kHostBounce (no fabric-mapped device pointer): the plan gathers / scatters
// through a DEVICE scratch (cudaMalloc, owned by the plan) with the existing
// kv_gather / kv_scatter, mirroring the pinned host network buffer to / from
// that scratch with fabric_bounce_device_to_pinned / fabric_bounce_pinned_to_device
// (stream-ordered cudaMemcpyAsync). The device kernels never dereference the
// non-mapped cudaMallocHost pinned buffer directly; the path still produces the
// SAME bytes the direct-store kernel would (the host oracle proves it). The
// caller owns the pinned scratch and ships it over the host transport.
//
// Lifetime: `imported_device_ptr` and (for the bounce) the pinned scratch
// must outlive every stream the plan is executed on. Read-only after
// construction, so concurrent execute() on several streams is safe (each
// call supplies its own destination).
#include "vkernels/comm/cross_node_kv.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/cross_node_kv_cuda.hpp"
#  include "vkernels/comm/fabric_import_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <cstddef>
#  include <cstdint>
#  include <memory>
#  include <utility>
#  include <unordered_set>
#  include <vector>

namespace vkernels::comm::cuda {
namespace {

// One page's worth of ONE layer: [page_size, 2, num_kv_heads, head_dim].
inline std::size_t per_slot_bytes(std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size) {
  return num_kv_heads * head_dim * elem_size;
}

inline std::size_t token_stride_bytes(std::size_t num_kv_heads,
                                      std::size_t head_dim,
                                      std::size_t elem_size) {
  return 2 * per_slot_bytes(num_kv_heads, head_dim, elem_size);
}

void validate_plan_shape(std::size_t num_slots, std::size_t num_kv_heads,
                         std::size_t head_dim, std::size_t elem_size,
                         std::size_t num_pages, std::size_t page_size) {
  VK_EXPECTS(num_slots > 0 || num_pages == 0, "num_slots must be positive");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");
}

void validate_restore_slots(const int* slot_ids, std::size_t total_tokens,
                            std::size_t num_slots) {
  std::unordered_set<int> seen;
  seen.reserve(total_tokens);
  for (std::size_t i = 0; i < total_tokens; ++i) {
    int slot = slot_ids[i];
    VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
    VK_EXPECTS(static_cast<std::size_t>(slot) < num_slots,
               "slot_ids must be < num_slots");
    VK_EXPECTS(seen.insert(slot).second, "slot_ids must be unique");
  }
}

void validate_donate_slots(const int* slot_ids, std::size_t total_tokens,
                           std::size_t num_slots) {
  for (std::size_t i = 0; i < total_tokens; ++i) {
    int slot = slot_ids[i];
    VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
    VK_EXPECTS(static_cast<std::size_t>(slot) < num_slots,
               "slot_ids must be < num_slots");
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// CrossNodeKvRestorePlan
// ---------------------------------------------------------------------------

CrossNodeKvRestorePlan::CrossNodeKvRestorePlan(
    std::size_t num_slots, std::size_t num_kv_heads, std::size_t head_dim,
    std::size_t elem_size, const int* slot_ids, std::size_t num_pages,
    std::size_t page_size, FabricImportTransport transport,
    void* imported_device_ptr)
    : transport_(transport),
      num_slots_(num_slots),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      elem_size_(elem_size),
      page_size_(page_size),
      num_pages_(num_pages),
      total_bytes_(0),
      imported_device_ptr_(imported_device_ptr) {
  VK_EXPECTS(transport == FabricImportTransport::kFabricMapped ||
                 transport == FabricImportTransport::kSameNodePeer ||
                 transport == FabricImportTransport::kHostBounce,
             "unknown fabric import transport");
  validate_plan_shape(num_slots, num_kv_heads, head_dim, elem_size,
                      num_pages, page_size);
  if (num_pages_ == 0) return;  // valid no-op plan

  VK_EXPECTS(slot_ids != nullptr, "slot_ids must be non-null");
  if (is_import_graph_capturable(transport_))
    VK_EXPECTS(imported_device_ptr_ != nullptr,
               "graph-capturable cross-node restore needs an imported "
               "device pointer (fabric_import_device_ptr)");

  const std::size_t total_tokens = num_pages_ * page_size_;
  validate_restore_slots(slot_ids, total_tokens, num_slots_);
  owned_slots_.assign(slot_ids, slot_ids + total_tokens);

  const std::size_t page_layer_bytes =
      page_size_ * token_stride_bytes(num_kv_heads_, head_dim_, elem_size_);
  total_bytes_ = num_pages_ * page_layer_bytes;  // one layer per execute()

  if (is_import_graph_capturable(transport_)) {
    // Per-page peer bases, exactly as the same-node path: the existing
    // fused P2PKvRestorePlan validates + uploads the host slot map and
    // peer ptrs ONCE, then execute() adds source_layer_offset_bytes to
    // every base before reading cross-node memory over NVLink/fabric.
    std::vector<const void*> peer_bases(num_pages_);
    const auto* base = static_cast<const std::uint8_t*>(imported_device_ptr_);
    for (std::size_t p = 0; p < num_pages_; ++p)
      peer_bases[p] = base + p * page_layer_bytes;
    direct_plan_ = std::make_unique<P2PKvRestorePlan>(
        num_slots_, num_kv_heads_, head_dim_, elem_size_,
        owned_slots_.data(), peer_bases.data(), num_pages_, page_size_);
  } else {
    // kHostBounce: upload the validated slot map to a device buffer ONCE
    // so the kv_scatter the bounce execute() enqueues can index it on the
    // GPU (kv_scatter takes a DEVICE slot_ids). Also mirror one layer's
    // worth of the pinned host network buffer into a device scratch ONCE:
    // the device kv_scatter reads the scratch (a real device pointer), not
    // the non-mapped pinned host pointer directly. Both owned + freed in
    // ~dtor. total_bytes_ > 0 here (num_pages_ > 0).
    const std::size_t slot_bytes = total_tokens * sizeof(int);
    cudaError_t err = cudaMalloc(&dev_slots_, slot_bytes);
    VK_ENSURES(err == cudaSuccess, "cudaMalloc for cross-node slot map failed");
    err = cudaMemcpy(dev_slots_, owned_slots_.data(), slot_bytes,
                     cudaMemcpyHostToDevice);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpy for cross-node slot map failed");
    err = cudaMalloc(&d_scratch_, total_bytes_);
    VK_ENSURES(err == cudaSuccess, "cudaMalloc for cross-node bounce scratch failed");
  }
}

CrossNodeKvRestorePlan::~CrossNodeKvRestorePlan() {
  if (d_scratch_ != nullptr) cudaFree(d_scratch_);
  if (dev_slots_ != nullptr) cudaFree(dev_slots_);
}

void CrossNodeKvRestorePlan::execute(void* k_dst, void* v_dst,
                                     std::size_t source_layer_offset_bytes,
                                     const void* pinned,
                                     cudaStream_t stream) const {
  VK_EXPECTS(k_dst != nullptr || num_pages_ == 0, "k_dst must be non-null");
  VK_EXPECTS(v_dst != nullptr || num_pages_ == 0, "v_dst must be non-null");
  if (num_pages_ == 0) return;  // valid no-op plan

  if (is_import_graph_capturable(transport_)) {
    // ONE real fused kernel over the imported pointer. The existing
    // P2PKvRestorePlan::execute() adds source_layer_offset_bytes to every
    // peer base, reads cross-node memory, and scatters into local slots --
    // the same operation the same-node path issues, now cross-node.
    direct_plan_->execute(k_dst, v_dst, source_layer_offset_bytes,
                          stream);
    return;
  }

  // ---- Host-bounce path: scatter the pinned layer into local slots -----
  // A fabric-mapped device pointer is unavailable. The remote donor shipped
  // the layer's bytes over the host transport; this side recv'd them into
  // `pinned` (cudaMallocHost, caller-owned). The device kv_scatter cannot
  // dereference `pinned` directly (a non-mapped cudaMallocHost pointer is
  // not device-accessible), so first mirror the layer into the device
  // scratch (stream-ordered), then scatter the scratch into the indexed
  // local K/V slots -- the SAME bytes the direct-store kernel would
  // produce, proved by the two-stage oracle on the host reference.
  VK_EXPECTS(pinned != nullptr, "host-bounce cross-node restore needs pinned");
  fabric_bounce_pinned_to_device(d_scratch_, pinned, total_bytes_, stream);
  kv_scatter(k_dst, v_dst, d_scratch_, dev_slots_ /*DEVICE slot map*/,
             num_pages_, page_size_, num_kv_heads_, head_dim_, elem_size_,
             stream);
}

// ---------------------------------------------------------------------------
// CrossNodeKvDonatePlan
// ---------------------------------------------------------------------------

CrossNodeKvDonatePlan::CrossNodeKvDonatePlan(
    std::size_t num_slots, std::size_t num_kv_heads, std::size_t head_dim,
    std::size_t elem_size, const int* slot_ids, std::size_t num_pages,
    std::size_t page_size, FabricImportTransport transport,
    void* imported_device_ptr)
    : transport_(transport),
      num_slots_(num_slots),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      elem_size_(elem_size),
      page_size_(page_size),
      num_pages_(num_pages),
      total_bytes_(0),
      imported_device_ptr_(imported_device_ptr) {
  VK_EXPECTS(transport == FabricImportTransport::kFabricMapped ||
                 transport == FabricImportTransport::kSameNodePeer ||
                 transport == FabricImportTransport::kHostBounce,
             "unknown fabric import transport");
  validate_plan_shape(num_slots, num_kv_heads, head_dim, elem_size,
                      num_pages, page_size);
  if (num_pages_ == 0) return;  // valid no-op plan

  VK_EXPECTS(slot_ids != nullptr, "slot_ids must be non-null");
  if (is_import_graph_capturable(transport_))
    VK_EXPECTS(imported_device_ptr_ != nullptr,
               "graph-capturable cross-node donate needs an imported "
               "device pointer (fabric_import_device_ptr)");

  const std::size_t total_tokens = num_pages_ * page_size_;
  validate_donate_slots(slot_ids, total_tokens, num_slots_);
  owned_slots_.assign(slot_ids, slot_ids + total_tokens);

  const std::size_t page_layer_bytes =
      page_size_ * token_stride_bytes(num_kv_heads_, head_dim_, elem_size_);
  total_bytes_ = num_pages_ * page_layer_bytes;  // one layer per execute()

  if (is_import_graph_capturable(transport_)) {
    // Per-page peer DESTINATION bases, exactly as the same-node path: the
    // existing fused P2PKvDonatePlan validates + uploads the host slot map
    // and peer ptrs ONCE, then execute() adds destination_layer_offset_bytes
    // to every base before writing cross-node memory over NVLink/fabric.
    std::vector<void*> peer_bases(num_pages_);
    auto* base = static_cast<std::uint8_t*>(imported_device_ptr_);
    for (std::size_t p = 0; p < num_pages_; ++p)
      peer_bases[p] = base + p * page_layer_bytes;
    direct_plan_ = std::make_unique<P2PKvDonatePlan>(
        num_slots_, num_kv_heads_, head_dim_, elem_size_,
        owned_slots_.data(), peer_bases.data(), num_pages_, page_size_);
  } else {
    // kHostBounce: upload the validated slot map to a device buffer ONCE
    // so the kv_gather the bounce execute() enqueues can index it on the
    // GPU (kv_gather takes a DEVICE slot_ids). Also own a device scratch
    // the gather writes (a real device pointer, not the non-mapped pinned
    // host buffer execute() returns). total_bytes_ > 0 here. Both freed in
    // ~dtor.
    const std::size_t slot_bytes = total_tokens * sizeof(int);
    cudaError_t err = cudaMalloc(&dev_slots_, slot_bytes);
    VK_ENSURES(err == cudaSuccess, "cudaMalloc for cross-node slot map failed");
    err = cudaMemcpy(dev_slots_, owned_slots_.data(), slot_bytes,
                     cudaMemcpyHostToDevice);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpy for cross-node slot map failed");
    err = cudaMalloc(&d_scratch_, total_bytes_);
    VK_ENSURES(err == cudaSuccess, "cudaMalloc for cross-node bounce scratch failed");
  }
}

CrossNodeKvDonatePlan::~CrossNodeKvDonatePlan() {
  if (d_scratch_ != nullptr) cudaFree(d_scratch_);
  if (dev_slots_ != nullptr) cudaFree(dev_slots_);
}

void CrossNodeKvDonatePlan::execute(const void* k_src, const void* v_src,
                                    std::size_t destination_layer_offset_bytes,
                                    void** out_pinned,
                                    cudaStream_t stream) const {
  VK_EXPECTS(k_src != nullptr || num_pages_ == 0, "k_src must be non-null");
  VK_EXPECTS(v_src != nullptr || num_pages_ == 0, "v_src must be non-null");
  if (num_pages_ == 0) return;  // valid no-op plan

  if (is_import_graph_capturable(transport_)) {
    // ONE real fused kernel over the imported pointer. The existing
    // P2PKvDonatePlan::execute() adds destination_layer_offset_bytes to
    // every peer base, reads local slots, and writes cross-node memory --
    // the same operation the same-node path issues, now cross-node.
    direct_plan_->execute(k_src, v_src, destination_layer_offset_bytes,
                          stream);
    return;
  }

  // ---- Host-bounce path: gather local slots into a pinned layer  ---------
  // A fabric-mapped device pointer is unavailable. The plan gathers the
  // indexed local K/V slots into the device scratch (a real device pointer)
  // with the existing kv_gather -- the SAME bytes the direct-store kernel
  // would produce -- then copies the scratch into a freshly allocated
  // (cudaMallocHost) pinned layer (fabric_bounce_device_to_pinned, the
  // device kernel never dereferences the non-mapped pinned host buffer)
  // and returns it to the caller, who sends it over the host transport and
  // frees it with fabric_bounce_scratch_free.
  VK_EXPECTS(out_pinned != nullptr, "host-bounce cross-node donate needs out_pinned");
  int status = VKERNELS_FI_OK;
  void* pinned = fabric_bounce_scratch_alloc(total_bytes_, &status);
  VK_ENSURES(status == VKERNELS_FI_OK && pinned != nullptr,
             "host-bounce cross-node donate: pinned alloc failed");
  // Gather into the device scratch (a real device pointer), then mirror it
  // into the pinned host network buffer (stream-ordered). The device
  // kv_gather never dereferences `pinned` directly (non-mapped cudaMallocHost
  // is not device-accessible); the whole path is still one async enqueue.
  kv_gather(d_scratch_, k_src, v_src, dev_slots_ /*DEVICE slot map*/,
            num_pages_, page_size_, num_kv_heads_, head_dim_, elem_size_,
            stream);
  fabric_bounce_device_to_pinned(pinned, d_scratch_, total_bytes_, stream);
  *out_pinned = pinned;
}

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
