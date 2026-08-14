// vkernels/comm/pipeline_boundary_cuda.hpp
//
// CUDA-only declarations for the graph-capturable PP-boundary transfer
// (issue #10). Kept separate from pipeline_boundary.hpp because the CUDA
// entry points take `cudaStream_t`, which must not be exposed to host-only
// translation units (the host reference in pipeline_boundary.cpp is the
// always-compiled correctness oracle). Included only when
// VKERNELS_HAS_CUDA; the definitions live in pipeline_boundary.cu.
//
// Device path (same-node peer): one `cudaMemcpyAsync(...,
// cudaMemcpyDeviceToDevice, stream)`, captured into a graph segment by the
// caller between cudaStreamBeginCapture / cudaStreamEndCapture and replayed
// with no host progress — exactly the property pipeline_boundary.hpp models
// and the host tests assert.
//
// Cross-node path: the production build replaces the peer copy with
// `ncclSend` / `ncclRecv` (forward-declared and conditionally compiled when
// NCCL is available). When NCCL is absent the device plan refuses the
// cross-node transport so a host-staged deployment is never silently
// captured.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/comm/pipeline_boundary.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;
typedef CUstream_st* cudaStream_t_vk;  // avoid pulling cuda_runtime.h here

namespace vkernels::comm::cuda {

// Free function: one directed peer-to-peer boundary transfer for one
// layer — `kSend` copies `my_buf` -> `peer_buf`, `kRecv` copies
// `peer_buf` -> `my_buf`, via `cudaMemcpyAsync(..., D2D, stream)`. Same
// contract as the host reference's device path: enqueued on `stream`,
// returns without synchronising, no host ring I/O.
//
// For `kCrossNodeNccl` the production build issues `ncclSend` / `ncclRecv`
// instead; when NCCL is unavailable this entry point returns
// VKERNELS_ERR_UNSUPPORTED (see pipeline_boundary.cpp).
void pipeline_boundary_layer(void* my_buf, void* peer_buf,
                             std::size_t payload_bytes,
                             PipelineTransport transport,
                             BoundaryDirection dir,
                             cudaStream_t_vk stream);

// ---------------------------------------------------------------------------
// Prepared directed boundary-transfer plan (issue #10)
// ---------------------------------------------------------------------------
//
// Same semantics as vkernels::comm::PipelineBoundaryPlan (validate once at
// construction, execute() only enqueues) with the CUDA specifics: the
// constructor validates the (world, rank, payload_bytes, transport,
// direction) shape once; execute(my_buf, stream) enqueues ONE
// `cudaMemcpyAsync` (peer) or `ncclSend` / `ncclRecv` (cross-node, when
// NCCL is linked) on the supplied stream, so a single plan moves one
// boundary's hidden state every decode iteration with no per-iteration
// validation or allocation. `my_buf` (and, for the device path, `peer_buf`)
// must outlive every stream the plan is executed on; the plan is read-only
// after construction, so concurrent execute() on several streams is safe.
class PipelineBoundaryPlan {
 public:
  // `payload_bytes` is the byte width of the hidden-state tensor (e.g.
  // hidden_dim * sizeof(dtype)); `peer_buf` is the shared peer staging
  // buffer for the device path and MUST be non-null when
  // is_graph_capturable(transport). For `kHostStaged` the CUDA plan
  // refuses construction (a host-staged boundary is never captured — use
  // the host reference's eager-break path instead).
  PipelineBoundaryPlan(int world, int rank, std::size_t payload_bytes,
                       PipelineTransport transport, BoundaryDirection dir,
                       void* peer_buf);
  ~PipelineBoundaryPlan();
  PipelineBoundaryPlan(const PipelineBoundaryPlan&) = delete;
  PipelineBoundaryPlan& operator=(const PipelineBoundaryPlan&) = delete;

  int world() const;
  int rank() const;
  std::size_t payload_bytes() const;
  PipelineTransport transport() const;
  BoundaryDirection direction() const;
  bool is_send() const;

  // Enqueue one directed transfer for `my_buf` (payload_bytes() bytes) on
  // `stream`. Throws std::invalid_argument on a null `my_buf`, a
  // std::runtime_error when the cross-node transport was requested but
  // NCCL is not linked, or on a launch failure — the C ABI
  // (pipeline_boundary_c.h) catches and folds these into a status code.
  void execute(void* my_buf, cudaStream_t_vk stream) const;

 private:
  int world_;
  int rank_;
  std::size_t payload_bytes_;
  PipelineTransport transport_;
  BoundaryDirection dir_;
  void* peer_buf_;
};

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
