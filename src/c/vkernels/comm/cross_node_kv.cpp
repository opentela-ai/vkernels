// vkernels/comm/cross_node_kv.cpp -- host reference (oracle) implementation.
//
// The CPU reference is the byte-exact oracle for the cross-node KV transfer
// (issue #49): the ByteChannel transport, the cross-node restore / donate
// plans that wrap the existing prepared fused kernels (issues #27, #36), and
// the GraphCapture integration (ties into #10). It is always compiled and
// fully unit-tested on a machine with no GPU; the CUDA path
// (cross_node_kv.cu, guarded by VKERNELS_HAS_CUDA) performs the real
// CU_MEM_HANDLE_TYPE_FABRIC import and the pinned-host (cudaMallocHost)
// bounce, and mirrors this API.
//
// The acceptance (issue #49) is proved here on the host oracle:
//   * A cross-node prepared fused RESTORE round-trip (peer pages on B ->
//     local K/V on A) and a cross-node DONATE round-trip (local K/V on A ->
//     peer pages on B) produce byte-identical results to the same-node
//     NVLink path. The direct path runs the existing *_execute_offset
//     kernels UNCHANGED over a FabricImport that mirrors the remote bytes,
//     so the host reference's memcpy IS the same memcpy the same-node path
//     issues. The host-bounce path runs the existing kv_gather_layer /
//     kv_scatter_layer primitives, which the two-stage oracles already prove
//     byte-identical to the fused kernels.
//   * The host-bounce fallback validates correctly where a fabric-mapped
//     device pointer is unavailable (kHostBounce: device_ptr() == nullptr,
//     execute() recvs over a ByteChannel and scatters with kv_scatter_layer).
//   * Per-hop throughput is reported vs the same-node roofline and the
//     synchronous bulk-copy fallback (fabric_import.cpp), with the GH200
//     DRAM-only / host-bounce caveat called out.
#include "vkernels/comm/cross_node_kv.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <unordered_set>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::comm {

namespace {

// One page's worth of ONE layer: [page_size, 2, num_kv_heads, head_dim].
inline std::size_t per_slot_bytes(std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size) {
  return num_kv_heads * head_dim * elem_size;
}

inline std::size_t token_stride_bytes(std::size_t num_kv_heads, std::size_t head_dim,
                                      std::size_t elem_size) {
  return 2 * per_slot_bytes(num_kv_heads, head_dim, elem_size);
}

// Shared geometry validation, mirroring P2PKvRestorePlan::validate_shape and
// P2PKvDonatePlan::validate_shape: `num_pages == 0` is a valid no-op that
// relaxes the "positive dimensions" checks (otherwise every dimension must
// be positive and elem_size must be 2 for BF16/FP16).
void validate_plan_shape(std::size_t num_slots, std::size_t num_kv_heads,
                         std::size_t head_dim, std::size_t elem_size,
                         std::size_t num_pages, std::size_t page_size) {
  VK_EXPECTS(num_slots > 0 || num_pages == 0, "num_slots must be positive");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");
}

// Validate a slot map for the RESTORE (scatter -> UNIQUE destination slots,
// non-negative, in [0, num_slots)). Mirrors P2PKvRestorePlan's host-input
// validation so the cross-node plan validates once, at construction.
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

// Validate a slot map for the DONATE (gather -> repeats allowed, only
// non-negativity and bounds). Mirrors P2PKvDonatePlan's host-input
// validation.
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
// ByteChannel link -- mirrors make_ring_channels(2) but for arbitrary bytes
// and non-ring (a dedicated in each direction).
// ---------------------------------------------------------------------------

std::pair<std::unique_ptr<ByteChannel>, std::unique_ptr<ByteChannel>>
make_byte_link() {
  auto q_ab = std::make_shared<ByteBlockingQueue>();  // a -> b
  auto q_ba = std::make_shared<ByteBlockingQueue>();  // b -> a
  auto a = std::make_unique<MockByteChannel>(q_ab, q_ba);
  auto b = std::make_unique<MockByteChannel>(q_ba, q_ab);
  return {std::move(a), std::move(b)};
}

// ---------------------------------------------------------------------------
// CrossNodeKvRestorePlan
// ---------------------------------------------------------------------------

CrossNodeKvRestorePlan::CrossNodeKvRestorePlan(std::size_t num_slots,
                                               std::size_t num_kv_heads,
                                               std::size_t head_dim,
                                               std::size_t elem_size,
                                               const int* slot_ids,
                                               std::size_t num_pages,
                                               std::size_t page_size,
                                               FabricImportTransport transport,
                                               FabricImport* import)
    : transport_(transport),
      num_slots_(num_slots),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      elem_size_(elem_size),
      page_size_(page_size),
      num_pages_(num_pages),
      total_bytes_(0),
      import_(import) {
  VK_EXPECTS(transport == FabricImportTransport::kFabricMapped ||
                 transport == FabricImportTransport::kSameNodePeer ||
                 transport == FabricImportTransport::kHostBounce,
             "unknown fabric import transport");
  validate_plan_shape(num_slots, num_kv_heads, head_dim, elem_size,
                      num_pages, page_size);
  if (num_pages_ == 0) return;  // valid no-op plan

  VK_EXPECTS(slot_ids != nullptr, "slot_ids must be non-null");
  // For the graph-capturable transports a FabricImport yields the
  // directly device-addressable pointer the existing kernel dereferences,
  // so it MUST be non-null (and must outlive this plan + every stream).
  if (is_import_graph_capturable(transport_))
    VK_EXPECTS(import_ != nullptr && import_->has_device_ptr(),
               "graph-capturable cross-node restore needs a FabricImport "
               "with a device pointer");

  const std::size_t total_tokens = num_pages_ * page_size_;
  validate_restore_slots(slot_ids, total_tokens, num_slots_);
  owned_slots_.assign(slot_ids, slot_ids + total_tokens);

  const std::size_t page_layer_bytes = page_size_ * token_stride_bytes(
      num_kv_heads_, head_dim_, elem_size_);
  total_bytes_ = num_pages_ * page_layer_bytes;  // one layer per execute()

  if (is_import_graph_capturable(transport_)) {
    // Lay the per-page peer bases out exactly as the same-node path:
    // peer_bases[p] = import->device_ptr() + p * page_layer_bytes. The
    // existing P2PKvRestorePlan::execute() adds source_layer_offset_bytes
    // to every base, so one plan fans the same remote run list out into a
    // distinct local K/V layer buffer per layer.
    std::vector<const void*> peer_bases(num_pages_);
    const auto* base = static_cast<const std::uint8_t*>(import_->device_ptr());
    for (std::size_t p = 0; p < num_pages_; ++p)
      peer_bases[p] = base + p * page_layer_bytes;
    // Borrow the already-validated slots (no re-validation, no copy); the
    // plan is destroyed before owned_slots_ (reverse member order).
    direct_plan_ = std::make_unique<P2PKvRestorePlan>(
        from_device_slots, num_slots_, num_kv_heads_, head_dim_, elem_size_,
        owned_slots_.data(), peer_bases.data(), num_pages_, page_size_);
  }
}

void CrossNodeKvRestorePlan::execute(void* k_dst, void* v_dst,
                                     std::size_t source_layer_offset_bytes,
                                     Stream* stream,
                                     GraphCapture* graph,
                                     ByteChannel* channel) const {
  VK_EXPECTS(k_dst != nullptr || num_pages_ == 0, "k_dst must be non-null");
  VK_EXPECTS(v_dst != nullptr || num_pages_ == 0, "v_dst must be non-null");
  if (num_pages_ == 0) return;  // valid no-op plan

  if (is_import_graph_capturable(transport_)) {
    // ---- Direct path: the existing kernel runs UNCHANGED --------------
    // ONE device op (the existing P2PKvRestorePlan::execute with no
    // stream -- sync memcpy in the host model, the fused peer read +
    // indexed scatter on the GPU). Capturing records it for a no-host-
    // progress replay; outside capture it runs through `stream` (one
    // task) or synchronously when stream is null.
    if (graph != nullptr && graph->in_capture()) {
      const CrossNodeKvRestorePlan* self = this;
      graph->submit([self, k_dst, v_dst, source_layer_offset_bytes] {
        self->direct_plan_->execute(k_dst, v_dst, source_layer_offset_bytes,
                                    /*stream=*/nullptr);
      });
      return;
    }
    direct_plan_->execute(k_dst, v_dst, source_layer_offset_bytes, stream);
    return;
  }

  // ---- Host-bounce path: recv the contiguous layer, scatter locally ----
  // A fabric-mapped device pointer is unavailable (the GH200 DRAM-only /
  // no-GPUDirect-RDMA case). The remote donor shipped the layer's bytes
  // over the network transport; this side recvs them and scatters into the
  // indexed local K/V slots with the existing kv_scatter (the SAME bytes
  // as the direct-store kernel, proved by the two-stage oracle).
  VK_EXPECTS(channel != nullptr, "host-bounce cross-node restore needs a channel");
  const CrossNodeKvRestorePlan* self = this;
  auto run = [self, k_dst, v_dst, channel] {
    std::vector<std::uint8_t> buf = channel->recv();
    VK_EXPECTS(buf.size() >= self->total_bytes_,
               "cross-node restore recv: channel chunk smaller than one layer");
    kv_scatter(k_dst, v_dst, /*scratch=*/buf.data(),
               self->owned_slots_.data(), self->num_pages_, self->page_size_,
               self->num_kv_heads_, self->head_dim_, self->elem_size_,
               /*stream=*/nullptr);
  };
  if (graph != nullptr && graph->in_capture()) {
    // Eager-break (ties into #10): the host recv cannot be serviced
    // during a graph replay, so END the current segment, run the transfer
    // eagerly (host progress), then BEGIN the next -- exactly the
    // PipelineBoundaryPlan contract so a host-staged cross-node transfer
    // is excluded from the captured segment instead of freezing inside it.
    graph->end();
    run();
    graph->begin();
    return;
  }
  if (stream != nullptr)
    stream->submit(std::move(run));
  else
    run();
}

// ---------------------------------------------------------------------------
// CrossNodeKvDonatePlan
// ---------------------------------------------------------------------------

CrossNodeKvDonatePlan::CrossNodeKvDonatePlan(std::size_t num_slots,
                                             std::size_t num_kv_heads,
                                             std::size_t head_dim,
                                             std::size_t elem_size,
                                             const int* slot_ids,
                                             std::size_t num_pages,
                                             std::size_t page_size,
                                             FabricImportTransport transport,
                                             FabricImport* import)
    : transport_(transport),
      num_slots_(num_slots),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      elem_size_(elem_size),
      page_size_(page_size),
      num_pages_(num_pages),
      total_bytes_(0),
      import_(import) {
  VK_EXPECTS(transport == FabricImportTransport::kFabricMapped ||
                 transport == FabricImportTransport::kSameNodePeer ||
                 transport == FabricImportTransport::kHostBounce,
             "unknown fabric import transport");
  validate_plan_shape(num_slots, num_kv_heads, head_dim, elem_size,
                      num_pages, page_size);
  if (num_pages_ == 0) return;  // valid no-op plan

  VK_EXPECTS(slot_ids != nullptr, "slot_ids must be non-null");
  if (is_import_graph_capturable(transport_))
    VK_EXPECTS(import_ != nullptr && import_->has_device_ptr(),
               "graph-capturable cross-node donate needs a FabricImport "
               "with a device pointer");

  const std::size_t total_tokens = num_pages_ * page_size_;
  validate_donate_slots(slot_ids, total_tokens, num_slots_);
  owned_slots_.assign(slot_ids, slot_ids + total_tokens);

  const std::size_t page_layer_bytes = page_size_ * token_stride_bytes(
      num_kv_heads_, head_dim_, elem_size_);
  total_bytes_ = num_pages_ * page_layer_bytes;  // one layer per execute()

  if (is_import_graph_capturable(transport_)) {
    // Per-page peer destination bases, exactly as the same-node path:
    // peer_bases[p] = import->device_ptr() + p * page_layer_bytes. The
    // existing P2PKvDonatePlan::execute() adds destination_layer_offset_bytes
    // to every base before writing.
    std::vector<void*> peer_bases(num_pages_);
    auto* base = static_cast<std::uint8_t*>(import_->device_ptr());
    for (std::size_t p = 0; p < num_pages_; ++p)
      peer_bases[p] = base + p * page_layer_bytes;
    direct_plan_ = std::make_unique<P2PKvDonatePlan>(
        from_device_slots, num_slots_, num_kv_heads_, head_dim_, elem_size_,
        owned_slots_.data(), peer_bases.data(), num_pages_, page_size_);
  }
}

void CrossNodeKvDonatePlan::execute(const void* k_src, const void* v_src,
                                    FabricHandle* remote,
                                    std::size_t destination_layer_offset_bytes,
                                    Stream* stream, GraphCapture* graph,
                                    ByteChannel* channel) const {
  VK_EXPECTS(k_src != nullptr || num_pages_ == 0, "k_src must be non-null");
  VK_EXPECTS(v_src != nullptr || num_pages_ == 0, "v_src must be non-null");
  if (num_pages_ == 0) return;  // valid no-op plan

  if (is_import_graph_capturable(transport_)) {
    // ---- Direct path: the existing kernel runs UNCHANGED --------------
    // The existing P2PKvDonatePlan::execute() writes K/V into the imported
    // mirror (the local address space that now names the remote memory) at
    // the destination layer offset, exactly one stream task. Then
    // FabricImport::write_back(*remote) ships the mirror to the remote
    // FabricHandle (the host model of the fabric write-back; a no-op for
    // kSameNodePeer where the mirror already aliases the remote bytes).
    const CrossNodeKvDonatePlan* self = this;
    FabricHandle* remote_h = remote;
    auto direct = [self, k_src, v_src, remote_h,
                   destination_layer_offset_bytes] {
      self->direct_plan_->execute(k_src, v_src,
                                  destination_layer_offset_bytes,
                                  /*stream=*/nullptr);
      self->import_->write_back(*remote_h);
    };
    if (graph != nullptr && graph->in_capture()) {
      graph->submit(std::move(direct));
      return;
    }
    // Outside capture: run the device write through `stream` (ordering),
    // then the host write_back. A null stream runs to completion.
    if (stream != nullptr) {
      stream->submit([self, k_src, v_src,
                      destination_layer_offset_bytes] {
        self->direct_plan_->execute(k_src, v_src,
                                    destination_layer_offset_bytes,
                                    /*stream=*/nullptr);
      });
      stream->wait();
      self->import_->write_back(*remote_h);
    } else {
      direct();
    }
    return;
  }

  // ---- Host-bounce path: gather locally, ship the contiguous layer ----
  // A fabric-mapped device pointer is unavailable. The plan gathers the
  // indexed local K/V slots into a contiguous [num_pages, page_size, 2,
  // heads, head_dim] scratch with the existing kv_gather (the SAME bytes
  // as the direct-store kernel, proved by the two-stage oracle) and sends
  // it over the network transport; the far-side restore recvs it.
  VK_EXPECTS(channel != nullptr, "host-bounce cross-node donate needs a channel");
  const CrossNodeKvDonatePlan* self = this;
  auto run = [self, k_src, v_src, channel] {
    std::vector<std::uint8_t> scratch(self->total_bytes_);
    kv_gather(scratch.data(), k_src, v_src, self->owned_slots_.data(),
              self->num_pages_, self->page_size_, self->num_kv_heads_,
              self->head_dim_, self->elem_size_, /*stream=*/nullptr);
    channel->send(std::move(scratch));
  };
  if (graph != nullptr && graph->in_capture()) {
    // Eager-break (ties into #10): the host gather + send cannot be
    // serviced during a replay, so exclude it from the captured segment.
    graph->end();
    run();
    graph->begin();
    return;
  }
  if (stream != nullptr)
    stream->submit(std::move(run));
  else
    run();
}

}  // namespace vkernels::comm
