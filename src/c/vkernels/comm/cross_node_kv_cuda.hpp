// vkernels/comm/cross_node_kv_cuda.hpp
//
// CUDA-only declarations for the cross-node KV restore / donate plans
// (issue #49). Kept separate from cross_node_kv.hpp because the CUDA entry
// points take `cudaStream_t`, which must not be exposed to host-only
// translation units (the host reference in cross_node_kv.cpp is the
// always-compiled correctness oracle). Included only when
// VKERNELS_HAS_CUDA; the definitions live in cross_node_kv.cu.
//
// The device path is the realization of issue #49 scope #1: a caller
// imports the remote VRAM once with cuda::fabric_import_device_ptr
// (fabric_import.cu, CU_MEM_HANDLE_TYPE_FABRIC) and passes the resulting
// device pointer here. This plan lays the per-page peer bases out as
// `imported_device_ptr + p * page_layer_bytes` and calls the EXISTING
// fused P2PKvRestorePlan / P2PKvDonatePlan (issues #27, #36) over it on a
// real cudaStream -- one kernel launch, the peer loads/stores now naming
// cross-node memory instead of a same-node peer. The kernels run
// UNCHANGED; only the peer base they dereference is cross-node. The import
// (and its lifetime) stays the caller's, exactly as fabric_import_cuda.hpp
// already models -- the host FabricImport class is the host oracle of that
// wrapper; on the device a deployment calls fabric_import_device_ptr +
// fabric_import_release directly.
//
// The host-bounce path (no fabric-mapped device pointer) gathers / scatters
// through a PINNED scratch the caller provides (fabric_bounce_scratch_alloc)
// with the existing kv_gather / kv_scatter, producing the SAME bytes the
// direct-store kernel would (the host oracle proves it).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vkernels/comm/cross_node_kv.hpp"
#include "vkernels/comm/p2p_kv_donate_cuda.hpp"
#include "vkernels/comm/p2p_kv_restore_cuda.hpp"
#include "vkernels/util/config.hpp"

// cudaStream_t_kv is forward-declared (struct CUstream_st*) by the
// p2p_kv_*_cuda.hpp includes above, so we reuse it here instead of
// pulling <cuda_runtime.h> into this header.
#if VKERNELS_HAS_CUDA

namespace vkernels::comm::cuda {

// ---------------------------------------------------------------------------
// CrossNodeKvRestorePlan -- remote peer pages -> local K/V layers (CUDA)
// ---------------------------------------------------------------------------
//
// Same semantics as vkernels::comm::CrossNodeKvRestorePlan (validate once
// at construction, execute() only enqueues) with the CUDA specifics: the
// constructor validates the (geometry, slot map, transport) shape once,
// lays the per-page peer bases out as
// `imported_device_ptr + p * page_layer_bytes`, and constructs the EXISTING
// fused P2PKvRestorePlan over them (it uploads the page descriptors + slot
// map to a persistent per-device buffer once). execute() enqueues ONE real
// fused kernel over the imported pointer per layer -- the cross-node
// realization of the existing same-node path.
//
// `imported_device_ptr` is the real device pointer the caller obtained from
// cuda::fabric_import_device_ptr (kFabricMapped / kSameNodePeer); it MUST
// be non-null and outlive this plan + every stream (the caller releases it
// with fabric_import_release / cudaIpcCloseMemHandle). For kHostBounce it
// is null (no fabric-mapped pointer); execute() scatters a PINNED scratch
// the caller recv'd over the host transport into local slots instead.
//
// Lifetime: `imported_device_ptr` (and, for the bounce, the pinned scratch)
// must outlive every stream the plan is executed on. Read-only after
// construction, so concurrent execute() on several streams is safe (each
// call supplies its own destination).
class CrossNodeKvRestorePlan {
 public:
  CrossNodeKvRestorePlan(std::size_t num_slots, std::size_t num_kv_heads,
                         std::size_t head_dim, std::size_t elem_size,
                         const int* slot_ids, std::size_t num_pages,
                         std::size_t page_size, FabricImportTransport transport,
                         void* imported_device_ptr = nullptr);
  ~CrossNodeKvRestorePlan();
  CrossNodeKvRestorePlan(const CrossNodeKvRestorePlan&) = delete;
  CrossNodeKvRestorePlan& operator=(const CrossNodeKvRestorePlan&) = delete;

  std::size_t num_pages() const { return num_pages_; }
  std::size_t page_size() const { return page_size_; }
  std::size_t num_slots() const { return num_slots_; }
  std::size_t num_kv_heads() const { return num_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t elem_size() const { return elem_size_; }
  std::size_t total_bytes() const { return total_bytes_; }
  std::size_t bounce_bytes() const { return total_bytes_; }
  FabricImportTransport transport() const { return transport_; }
  bool is_graph_capturable() const { return is_import_graph_capturable(transport_); }

  // One fused cross-node restore for `layer` into (k_dst, v_dst), adding
  // `source_layer_offset_bytes` to every peer page base. kFabricMapped /
  // kSameNodePeer: ONE real kernel over `imported_device_ptr` on `stream`.
  // kHostBounce: copy `pinned` (the layer the caller recv'd over the host
  // transport) into the device scratch with fabric_bounce_pinned_to_device,
  // then scatter the scratch into local slots with the existing kv_scatter
  // on `stream` -- the SAME bytes the direct-store kernel would produce.
  // The device kernel never dereferences `pinned` directly (a non-mapped
  // cudaMallocHost pointer is not device-accessible); the bounce copy is
  // stream-ordered, so the whole path is still one async enqueue.
  void execute(void* k_dst, void* v_dst,
               std::size_t source_layer_offset_bytes,
               const void* pinned = nullptr,
               cudaStream_t_kv stream = nullptr) const;

 private:
  FabricImportTransport transport_;
  std::size_t num_slots_, num_kv_heads_, head_dim_, elem_size_;
  std::size_t page_size_, num_pages_, total_bytes_;
  std::vector<int> owned_slots_;
  void* imported_device_ptr_;
  std::unique_ptr<P2PKvRestorePlan> direct_plan_;  // existing fused kernel
  int* dev_slots_ = nullptr;   // device slot map for the kHostBounce scatter
  void* d_scratch_ = nullptr;  // device mirror of the pinned layer (kHostBounce)
};

// ---------------------------------------------------------------------------
// CrossNodeKvDonatePlan -- local K/V layers -> remote peer pages (CUDA)
// ---------------------------------------------------------------------------
//
// The mirror: `imported_device_ptr` names the remote peer pages (caller-
// imported via cuda::fabric_import_device_ptr). This plan lays the per-page
// peer DESTINATION bases out as `imported_device_ptr + p * page_layer_bytes`
// and calls the EXISTING fused P2PKvDonatePlan over it on a real cudaStream
// -- one kernel launch writing cross-node memory over NVLink/fabric. The
// kernel runs UNCHANGED; only the peer destination it writes is cross-node.
//
// kHostBounce: gather local slots into a per-call pinned scratch (allocated
// here, returned via *out_pinned) with the existing kv_gather, then the
// caller sends it over the host transport and frees it.
class CrossNodeKvDonatePlan {
 public:
  CrossNodeKvDonatePlan(std::size_t num_slots, std::size_t num_kv_heads,
                        std::size_t head_dim, std::size_t elem_size,
                        const int* slot_ids, std::size_t num_pages,
                        std::size_t page_size, FabricImportTransport transport,
                        void* imported_device_ptr = nullptr);
  ~CrossNodeKvDonatePlan();
  CrossNodeKvDonatePlan(const CrossNodeKvDonatePlan&) = delete;
  CrossNodeKvDonatePlan& operator=(const CrossNodeKvDonatePlan&) = delete;

  std::size_t num_pages() const { return num_pages_; }
  std::size_t page_size() const { return page_size_; }
  std::size_t num_slots() const { return num_slots_; }
  std::size_t num_kv_heads() const { return num_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t elem_size() const { return elem_size_; }
  std::size_t total_bytes() const { return total_bytes_; }
  std::size_t scratch_bytes() const { return total_bytes_; }
  std::size_t bounce_bytes() const { return total_bytes_; }
  FabricImportTransport transport() const { return transport_; }
  bool is_graph_capturable() const { return is_import_graph_capturable(transport_); }

  // One fused cross-node donate from (k_src, v_src), adding
  // `destination_layer_offset_bytes` to every peer page base.
  // kFabricMapped / kSameNodePeer: ONE real kernel over
  // `imported_device_ptr` on `stream`. kHostBounce: gather local slots into
  // the device scratch with the existing kv_gather, then copy the scratch
  // into a freshly allocated (cudaMallocHost) pinned layer with
  // fabric_bounce_device_to_pinned and return it in *out_pinned for the
  // caller to send + free (fabric_bounce_scratch_free) -- the SAME bytes the
  // direct-store kernel would produce. The device kernel never dereferences
  // *out_pinned directly (a non-mapped cudaMallocHost pointer is not
  // device-accessible); the bounce copy is stream-ordered.
  void execute(const void* k_src, const void* v_src,
               std::size_t destination_layer_offset_bytes,
               void** out_pinned = nullptr,
               cudaStream_t_kv stream = nullptr) const;

 private:
  FabricImportTransport transport_;
  std::size_t num_slots_, num_kv_heads_, head_dim_, elem_size_;
  std::size_t page_size_, num_pages_, total_bytes_;
  std::vector<int> owned_slots_;
  void* imported_device_ptr_;
  std::unique_ptr<P2PKvDonatePlan> direct_plan_;  // existing fused kernel
  int* dev_slots_ = nullptr;   // device slot map for the kHostBounce gather
  void* d_scratch_ = nullptr;  // device mirror of the pinned layer (kHostBounce)
};

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
