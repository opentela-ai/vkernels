// vkernels/comm/cross_node_kv.hpp
//
// Cross-node transfer for the prepared fused KV restore / donate kernels
// (issue #49). The prepared kernels (issues #27, #36) are transport-agnostic
// at the pointer level but dereference peer_src_ptrs / peer_dst_ptrs that
// must ALREADY be device-addressable on the local fabric (CUDA-IPC,
// same-node). This module -- the remaining slice -- makes the existing
// prepared plans operate ACROSS NODES instead of being replaced by a
// host-staged bulk copy (kvaas NIXL/libfabric `transfer()`).
//
// Three paths, classified by fabric_import.hpp (issue #49 scope #1):
//
// * kFabricMapped / kSameNodePeer -- a FabricImport yields a directly
//   device-addressable pointer to the remote VRAM. The existing
//   P2PKvRestorePlan / P2PKvDonatePlan execute()/execute_offset() run
//   UNCHANGED over it: their peer page bases are laid out as
//   import.device_ptr() + p * page_layer_bytes, and execute_offset() adds
//   the per-layer offset exactly as on the same-node path. No bulk copy.
//
// * kHostBounce -- a fabric-mapped device pointer is unavailable (no
//   GPUDirect-RDMA, or the GH200 DRAM-only libfabric constraint). The plan
//   gathers into a pinned device-accessible scratch, ships the contiguous
//   pages over a ByteChannel (the host transport), and scatters on the far
//   side -- producing the SAME bytes as the direct-store kernel, analogous
//   to the existing execute_via_scratch() (issue #36). This is the path
//   that "validates correctly where a fabric-mapped device pointer is
//   unavailable".
//
// Stream-ordered / graph-capturable variants (issue #49 scope #3, ties into
// #10): execute() takes the same GraphCapture pipeline_boundary.hpp uses.
// kFabricMapped / kSameNodePeer record ONE device op into the graph and
// replay with no host progress (the cross-node KV transfer sits inside a
// captured decode segment). kHostBounce eager-breaks -- it ends the current
// segment, runs the host send/recv over the ByteChannel, and begins the
// next -- exactly the PipelineBoundaryPlan eager-break contract (#10), so a
// host-staged cross-node transfer is excluded from the captured segment
// instead of freezing inside it.
//
// Two-implementation model (mirrors p2p_kv_restore / p2p_kv_donate /
// pipeline_boundary):
//   * Host reference (cross_node_kv.cpp, always compiled) -- the byte
//     channel, the cross-node restore / donate plans, and the graph
//     integration. Fully unit-tested with 100% line coverage on a machine
//     with no GPU; the host reference IS the byte-exact oracle for the
//     direct path (the existing kernels run unchanged) and the host-bounce
//     path (the existing gather/scatter primitives run unchanged).
//   * CUDA implementation (cross_node_kv_cuda.hpp + cross_node_kv.cu,
//     guarded by VKERNELS_HAS_CUDA) -- the real device path over the
//     imported pointer and the pinned-host (cudaMallocHost) bounce.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <utility>
#include <vector>

#include "vkernels/comm/fabric_import.hpp"
#include "vkernels/comm/p2p_kv_donate.hpp"
#include "vkernels/comm/p2p_kv_restore.hpp"
#include "vkernels/comm/pipeline_boundary.hpp"
#include "vkernels/core/stream.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels::comm {

// ---------------------------------------------------------------------------
// ByteChannel -- a byte-capable blocking transport for KV payloads
// ---------------------------------------------------------------------------
//
// The existing Channel (channel.hpp) carries std::vector<float> -- the unit
// a ring all-reduce circulates. KV restore / donate moves arbitrary BF16 /
// FP16 bytes, so this module adds a byte-capable twin. The MockByteChannel
// is the in-process implementation (backed by a thread-safe queue) that
// makes the host-bounce path and its graph eager-break testable without a
// network or a GPU; a real deployment adds an NcclByteChannel /
// LibfabricByteChannel (NIXL `transfer()`) behind the same interface.
class ByteBlockingQueue {
 public:
  void push(std::vector<std::uint8_t> v) {
    {
      std::lock_guard<std::mutex> lk(m_);
      q_.push(std::move(v));
    }
    cv_.notify_one();
  }
  std::vector<std::uint8_t> pop() {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [this] { return !q_.empty(); });
    std::vector<std::uint8_t> v = std::move(q_.front());
    q_.pop();
    return v;
  }
  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    cv_.notify_all();
  }
  bool closed() const {
    std::lock_guard<std::mutex> lk(m_);
    return closed_;
  }

 private:
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::queue<std::vector<std::uint8_t>> q_;
  bool closed_ = false;
};

class ByteChannel {
 public:
  virtual ~ByteChannel() = default;
  virtual void send(std::vector<std::uint8_t> chunk) = 0;
  virtual std::vector<std::uint8_t> recv() = 0;
  virtual bool closed() const = 0;
};

// A MockByteChannel sends into `out` and receives from `in` (two
// ByteBlockingQueues), exactly mirroring MockChannel.
class MockByteChannel : public ByteChannel {
 public:
  MockByteChannel(std::shared_ptr<ByteBlockingQueue> out,
                  std::shared_ptr<ByteBlockingQueue> in)
      : out_(std::move(out)), in_(std::move(in)) {
    VK_EXPECTS(out_ != nullptr && in_ != nullptr, "MockByteChannel needs both queues");
  }
  void send(std::vector<std::uint8_t> chunk) override { out_->push(std::move(chunk)); }
  std::vector<std::uint8_t> recv() override { return in_->pop(); }
  bool closed() const override { return in_->closed(); }

 private:
  std::shared_ptr<ByteBlockingQueue> out_;
  std::shared_ptr<ByteBlockingQueue> in_;
};

// Build a directed, in-process byte link: `a` sends into the queue `b`
// receives from, and `b` sends into the queue `a` receives from (two
// independent directions, like make_ring_channels(2) but for bytes and
// non-ring). Returns (a, b).
std::pair<std::unique_ptr<ByteChannel>, std::unique_ptr<ByteChannel>>
make_byte_link();

// ---------------------------------------------------------------------------
// Direction tag (a cross-node plan is bound to one direction, mirroring the
// existing P2PKvRestorePlan / P2PKvDonatePlan split).
// ---------------------------------------------------------------------------
enum class CrossNodeKvDirection { kRestore = 0, kDonate = 1 };

// ---------------------------------------------------------------------------
// Access-pattern routing -- choose P2P vs a collective before building a plan
// ---------------------------------------------------------------------------
//
// This decision is deliberately separate from FabricImportTransport.  The
// latter answers *how one peer edge is reached*; this enum answers *whether
// the workload is one peer edge at all*.  A remote cache miss in KVAAS has one
// consumer and must remain point-to-point.  All-gather is selected only when
// every rank needs the complete rank-sharded value, the shards are even (the
// ncclAllGather contract), and the caller has an executable collective.
enum class CrossNodeKvTransferKind {
  kPointToPoint = 0,
  kAllGather = 1,
};

struct CrossNodeKvAccess {
  std::size_t world_size = 1;
  std::size_t receiver_count = 1;
  bool evenly_sharded = false;
  bool collective_available = false;
  bool collective_graph_supported = false;
};

struct CrossNodeKvRoute {
  CrossNodeKvTransferKind kind = CrossNodeKvTransferKind::kPointToPoint;
  // The edge transport used directly for kPointToPoint and the safe fallback
  // when a collective is not eligible.  It remains useful to callers that
  // want to log or pre-create the fallback while using kAllGather.
  FabricImportTransport point_to_point_transport =
      FabricImportTransport::kHostBounce;
  bool graph_capturable = false;
};

// Select the communication primitive from the access pattern, then classify
// the P2P edge from `fabric`.  Throws std::invalid_argument when world_size is
// zero, receiver_count is zero, or receiver_count exceeds world_size.
//
// All-gather eligibility is intentionally strict:
//   receiver_count == world_size > 1 && evenly_sharded && collective_available
// Any other shape uses the byte-correct point-to-point restore/donate plans.
CrossNodeKvRoute select_cross_node_kv_route(const CrossNodeKvAccess& access,
                                             const FabricImportConfig& fabric);

// ---------------------------------------------------------------------------
// CrossNodeKvRestorePlan -- remote peer pages -> local K/V layers, cross-node
// ---------------------------------------------------------------------------
//
// Wraps the existing P2PKvRestorePlan (issue #27) so it runs across nodes:
// a FabricImport yields a directly device-addressable pointer to the remote
// peer pages, the existing restore kernel runs UNCHANGED over it (its peer
// bases laid out as import.device_ptr() + p * page_layer_bytes), or -- when
// no fabric-mapped pointer is available -- the plan recvs the contiguous
// pages over a ByteChannel and scatters into local slots with the existing
// kv_scatter, producing the SAME bytes.
//
// Created once over a (geometry, slot map, remote handle, transport); reused
// across all model layers (40 for Qwen3-14B) by varying the per-layer
// (k_dst, v_dst, source_layer_offset_bytes) at every execute(). Exactly the
// KVAAS restore pattern, now cross-node.
class CrossNodeKvRestorePlan {
 public:
  // Host-input plan. `transport` is the resolved fabric import transport
  // (use classify_fabric_import(cfg)); `import` is the FabricImport the
  // caller built for kFabricMapped / kSameNodePeer (borrowed -- must outlive
  // the plan and every stream it runs on, since the existing restore
  // kernel dereferences import->device_ptr()). For kHostBounce `import`
  // is ignored (may be null) and execute() recvs over a ByteChannel.
  //
  // `slot_ids` is a HOST int32 array [num_pages * page_size]; the plan
  // validates uniqueness, non-negativity and bounds (slot < num_slots)
  // once and owns a copy. Throws std::invalid_argument on a contract
  // violation (zero dimensions, non-BF16/FP16 elem_size, duplicate/
  // negative/out-of-range slot, capturable transport with a null import).
  CrossNodeKvRestorePlan(std::size_t num_slots, std::size_t num_kv_heads,
                         std::size_t head_dim, std::size_t elem_size,
                         const int* slot_ids, std::size_t num_pages,
                         std::size_t page_size, FabricImportTransport transport,
                         FabricImport* import = nullptr);

  CrossNodeKvRestorePlan(const CrossNodeKvRestorePlan&) = delete;
  CrossNodeKvRestorePlan& operator=(const CrossNodeKvRestorePlan&) = delete;

  std::size_t num_pages() const { return num_pages_; }
  std::size_t page_size() const { return page_size_; }
  std::size_t num_slots() const { return num_slots_; }
  std::size_t num_kv_heads() const { return num_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t elem_size() const { return elem_size_; }
  // Bytes per execute (num_pages * page_size * token_stride).
  std::size_t total_bytes() const { return total_bytes_; }
  FabricImportTransport transport() const { return transport_; }
  bool is_graph_capturable() const { return is_import_graph_capturable(transport_); }
  // Bytes the host-bounce recv needs (== total_bytes(): one contiguous
  // [num_pages, page_size, 2, heads, head_dim] page region).
  std::size_t bounce_bytes() const { return total_bytes_; }

  // Cross-node prepared fused restore for one layer into (k_dst, v_dst),
  // adding `source_layer_offset_bytes` to every remote page base.
  //
  // kFabricMapped / kSameNodePeer: the existing P2PKvRestorePlan::execute()
  // runs UNCHANGED over the imported pointer -- exactly one stream task,
  // no scratch, no bulk copy. kHostBounce: `channel` MUST be non-null; the
  // plan recvs the contiguous pages and scatters into local slots with the
  // existing kv_scatter (the SAME bytes).
  //
  // `graph`, when non-null and capturing, records ONE device op for the
  // graph-capturable transports (replayed with no host progress) or
  // eager-breaks for kHostBounce (end, recv over `channel`, begin) --
  // exactly the PipelineBoundaryPlan contract (#10). A null stream runs to
  // completion; a non-null stream submits one task.
  void execute(void* k_dst, void* v_dst,
               std::size_t source_layer_offset_bytes,
               Stream* stream = nullptr, GraphCapture* graph = nullptr,
               ByteChannel* channel = nullptr) const;

 private:
  FabricImportTransport transport_;
  std::size_t num_slots_, num_kv_heads_, head_dim_, elem_size_;
  std::size_t page_size_, num_pages_, total_bytes_;
  std::vector<int> owned_slots_;  // validated slot map, used by the bounce
  FabricImport* import_;    // borrowed; null for kHostBounce
  std::unique_ptr<P2PKvRestorePlan> direct_plan_;
};

// ---------------------------------------------------------------------------
// CrossNodeKvDonatePlan -- local K/V layers -> remote peer pages, cross-node
// ---------------------------------------------------------------------------
//
// The mirror of CrossNodeKvRestorePlan: wraps the existing P2PKvDonatePlan
// (issue #36) so it runs across nodes. A FabricImport yields a directly
// device-addressable pointer to the remote peer pages; the existing donate
// kernel runs UNCHANGED over it and write_back() ships the imported mirror
// to the remote FabricHandle (the host model of the fabric write-back). Or
// -- when no fabric-mapped pointer is available -- the plan gathers local
// slots into a contiguous scratch with the existing execute_via_scratch()
// and sends it over a ByteChannel, producing the SAME bytes.
class CrossNodeKvDonatePlan {
 public:
  // Host-input plan. `transport` is the resolved fabric import transport;
  // `import` is the FabricImport for kFabricMapped / kSameNodePeer
  // (borrowed -- must outlive the plan and every stream, since the existing
  // donate kernel dereferences import->device_ptr()). For kHostBounce
  // `import` is ignored (may be null) and execute() gathers into a per-call
  // scratch and sends over a ByteChannel.
  //
  // `slot_ids` is a HOST int32 array [num_pages * page_size]; the plan
  // validates non-negativity and bounds (uniqueness NOT required -- gather
  // semantics) once and owns a copy.
  CrossNodeKvDonatePlan(std::size_t num_slots, std::size_t num_kv_heads,
                        std::size_t head_dim, std::size_t elem_size,
                        const int* slot_ids, std::size_t num_pages,
                        std::size_t page_size, FabricImportTransport transport,
                        FabricImport* import = nullptr);

  CrossNodeKvDonatePlan(const CrossNodeKvDonatePlan&) = delete;
  CrossNodeKvDonatePlan& operator=(const CrossNodeKvDonatePlan&) = delete;

  std::size_t num_pages() const { return num_pages_; }
  std::size_t page_size() const { return page_size_; }
  std::size_t num_slots() const { return num_slots_; }
  std::size_t num_kv_heads() const { return num_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t elem_size() const { return elem_size_; }
  std::size_t total_bytes() const { return total_bytes_; }
  // Bytes needed for the copy-engine / host-bounce fallback scratch
  // (== total_bytes()). Same as the existing P2PKvDonatePlan::scratch_bytes.
  std::size_t scratch_bytes() const { return total_bytes_; }
  FabricImportTransport transport() const { return transport_; }
  bool is_graph_capturable() const { return is_import_graph_capturable(transport_); }
  // Bytes the host-bounce send ships (== total_bytes()).
  std::size_t bounce_bytes() const { return total_bytes_; }

  // Cross-node prepared fused donate for one layer from (k_src, v_src),
  // adding `destination_layer_offset_bytes` to every remote page base, then
  // (for the direct path) shipping the imported mirror to `remote` with
  // FabricImport::write_back(). `remote` is unused on the host-bounce path
  // (the bytes go straight over `channel` to the far-side restore).
  //
  // kFabricMapped / kSameNodePeer: the existing P2PKvDonatePlan::execute()
  // runs UNCHANGED over the imported pointer, then write_back(remote) -- one
  // stream task, no scratch, no bulk copy. kHostBounce: `channel` MUST be
  // non-null; the plan gathers local slots with execute_via_scratch() into a
  // per-call scratch and sends it over `channel`.
  //
  // `graph`, when non-null and capturing, records ONE device op (execute +
  // write_back) for the graph-capturable transports or eager-breaks for
  // kHostBounce (end, send over `channel`, begin). A null stream runs to
  // completion.
  void execute(const void* k_src, const void* v_src,
               FabricHandle* remote,
               std::size_t destination_layer_offset_bytes,
               Stream* stream = nullptr, GraphCapture* graph = nullptr,
               ByteChannel* channel = nullptr) const;

 private:
  FabricImportTransport transport_;
  std::size_t num_slots_, num_kv_heads_, head_dim_, elem_size_;
  std::size_t page_size_, num_pages_, total_bytes_;
  std::vector<int> owned_slots_;  // validated slot map, used by the bounce
  FabricImport* import_;    // borrowed; null for kHostBounce
  std::unique_ptr<P2PKvDonatePlan> direct_plan_;
};

}  // namespace vkernels::comm
