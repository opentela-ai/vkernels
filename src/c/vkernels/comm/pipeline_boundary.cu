// vkernels/comm/pipeline_boundary.cu — CUDA device path for the
// graph-capturable PP-boundary transfer (issue #10).
//
// The host reference (pipeline_boundary.cpp) is the correctness oracle and
// carries the full contract (transport classification, the eager-break
// decision, a host model of graph capture / replay, and the boundary plan
// over the ring Channel). This CUDA path is the device realization of the
// graph-capturable boundary:
//
//   * Same-node peer (NVLink / HBM): ONE `cudaMemcpyAsync(...,
//     cudaMemcpyDeviceToDevice, stream)` between `my_buf` and the shared
//     `peer_buf`. The caller captures it between
//     cudaStreamBeginCapture / cudaStreamEndCapture; the launched graph
//     replays it every decode iteration with no host participation — the
//     property the host tests assert (pipeline_boundary.hpp models the same
//     memcpy and asserts the ring Channel is never touched on replay).
//
//   * Cross-node NCCL / RCCL: when NCCL is linked (`__has_include(<nccl.h>)`)
//     the same path issues `ncclSend` / `ncclRecv` on `stream`, captured
//     identically. When NCCL is absent the free function refuses with a
//     std::runtime_error so a cross-node deployment never silently falls
//     back to a host copy that would deadlock the graph.
//
// A `kHostStaged` boundary is never constructed on the device path — the
// CUDA plan rejects it at construction (host-staged boundaries are not
// graph-capturable; use the host reference's eager-break path instead).
#include "vkernels/comm/pipeline_boundary.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/pipeline_boundary_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <cstddef>
#  include <stdexcept>

#  if defined(__has_include)
#    if __has_include(<nccl.h>)
#      define VKERNELS_PP_HAS_NCCL 1
#      include <nccl.h>
#    endif
#  endif

namespace vkernels::comm::cuda {

namespace {

// Validate `transport`/`dir` are device-capturable at construction. The
// host reference's is_graph_capturable() is reused for the transport check
// (kSameNodePeer / kCrossNodeNccl only — kHostStaged is refused here).
bool device_capturable(PipelineTransport t) {
  return is_graph_capturable(t);
}

}  // namespace

void pipeline_boundary_layer(void* my_buf, void* peer_buf,
                             std::size_t payload_bytes,
                             PipelineTransport transport,
                             BoundaryDirection dir,
                             cudaStream_t stream) {
  VK_EXPECTS(my_buf != nullptr, "my_buf must be non-null");
  VK_EXPECTS(peer_buf != nullptr, "peer_buf must be non-null");
  VK_EXPECTS(payload_bytes > 0, "payload_bytes must be positive");
  VK_EXPECTS(device_capturable(transport),
             "transport is not device-capturable");

  void* const dst = (dir == BoundaryDirection::kSend) ? peer_buf : my_buf;
  const void* const src = (dir == BoundaryDirection::kSend) ? my_buf : peer_buf;

  if (transport == PipelineTransport::kSameNodePeer) {
    // Peer copy over NVLink / HBM — pure device, graph-capturable.
    cudaError_t err = cudaMemcpyAsync(dst, src, payload_bytes,
                                       cudaMemcpyDeviceToDevice, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync D2D failed");
    return;
  }

  // Cross-node NCCL. When NCCL is linked, issue ncclSend / ncclRecv on the
  // captured stream (the caller passes the comm via the shared `peer_buf`,
  // which for this path carries the `ncclComm_t` of the receiving peer —
  // documented in pipeline_boundary.hpp). When NCCL is not linked, refuse:
  // a host-bounce fallback would deadlock the captured graph exactly as the
  // gloo path does, so we never silently take it.
#  if defined(VKERNELS_PP_HAS_NCCL)
  {
    ncclComm_t comm = static_cast<ncclComm_t>(peer_buf);
    ncclResult_t nccl_err =
        (dir == BoundaryDirection::kSend)
            ? ncclSend(src, payload_bytes, ncclChar, 0, comm, stream)
            : ncclRecv(dst, payload_bytes, ncclChar, 0, comm, stream);
    VK_ENSURES(nccl_err == ncclSuccess, "ncclSend/ncclRecv failed");
    return;
  }
#  else
  (void)dst;
  (void)src;
  throw std::runtime_error(
      "pipeline_boundary cross-node NCCL path is not linked "
      "(build vkernels_c against NCCL to enable it)");
#  endif
}

// ---------------------------------------------------------------------------
// Prepared directed boundary-transfer plan
// ---------------------------------------------------------------------------

PipelineBoundaryPlan::PipelineBoundaryPlan(int world, int rank,
                                           std::size_t payload_bytes,
                                           PipelineTransport transport,
                                           BoundaryDirection dir,
                                           void* peer_buf)
    : world_(world),
      rank_(rank),
      payload_bytes_(payload_bytes),
      transport_(transport),
      dir_(dir),
      peer_buf_(peer_buf) {
  VK_EXPECTS(world > 0, "world must be positive");
  VK_EXPECTS(rank >= 0 && rank < world, "rank out of range");
  VK_EXPECTS(payload_bytes > 0, "payload_bytes must be positive");
  // Only the graph-capturable transports reach the device path; a
  // host-staged boundary is never captured (use the host reference's
  // eager-break path instead).
  VK_EXPECTS(device_capturable(transport),
             "transport is not device-capturable");
  VK_EXPECTS(dir == BoundaryDirection::kSend || dir == BoundaryDirection::kRecv,
             "unknown boundary direction");
  VK_EXPECTS(peer_buf != nullptr, "device-path boundary needs a peer buffer");
}

PipelineBoundaryPlan::~PipelineBoundaryPlan() = default;

int PipelineBoundaryPlan::world() const { return world_; }
int PipelineBoundaryPlan::rank() const { return rank_; }
std::size_t PipelineBoundaryPlan::payload_bytes() const { return payload_bytes_; }
PipelineTransport PipelineBoundaryPlan::transport() const { return transport_; }
BoundaryDirection PipelineBoundaryPlan::direction() const { return dir_; }
bool PipelineBoundaryPlan::is_send() const {
  return dir_ == BoundaryDirection::kSend;
}

void PipelineBoundaryPlan::execute(void* my_buf, cudaStream_t stream) const {
  pipeline_boundary_layer(my_buf, peer_buf_, payload_bytes_, transport_, dir_,
                          stream);
}

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
