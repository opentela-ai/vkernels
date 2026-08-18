// vkernels/comm/pipeline_boundary_c.cu — CUDA C ABI for the prepared
// device-boundary plan (issue #10). Wraps the C++
// `vkernels::comm::cuda::PipelineBoundaryPlan`, catching every C++
// exception and converting it to a `vkernels_pp_status_t` so nothing is
// thrown across the language boundary.
#include "vkernels/comm/pipeline_boundary_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#  include "vkernels/comm/pipeline_boundary_cuda.hpp"

#  include <exception>
#  include <stdexcept>
#  include <type_traits>

namespace {

vkernels::comm::PipelineTransport from_c_transport(int t) {
  switch (t) {
    case VKERNELS_PP_TRANSPORT_SAME_NODE_PEER:
      return vkernels::comm::PipelineTransport::kSameNodePeer;
    case VKERNELS_PP_TRANSPORT_CROSS_NODE_NCCL:
      return vkernels::comm::PipelineTransport::kCrossNodeNccl;
    case VKERNELS_PP_TRANSPORT_HOST_STAGED:
      return vkernels::comm::PipelineTransport::kHostStaged;
    default:
      throw std::invalid_argument("unknown transport");
  }
}

vkernels::comm::BoundaryDirection from_c_dir(int d) {
  switch (d) {
    case VKERNELS_PP_DIR_SEND:
      return vkernels::comm::BoundaryDirection::kSend;
    case VKERNELS_PP_DIR_RECV:
      return vkernels::comm::BoundaryDirection::kRecv;
    default:
      throw std::invalid_argument("unknown direction");
  }
}

vkernels_pp_status_t to_status(const std::exception& e) {
  if (dynamic_cast<const std::invalid_argument*>(&e))
    return VKERNELS_PP_ERR_INVALID_ARGUMENT;
  // pipeline_boundary_layer throws std::runtime_error when the cross-node
  // NCCL path is requested but NCCL is not linked — surface that as
  // VKERNELS_PP_ERR_UNSUPPORTED so the caller can fall back to the
  // host-staged eager-break path.
  if (dynamic_cast<const std::runtime_error*>(&e))
    return VKERNELS_PP_ERR_UNSUPPORTED;
  (void)e;
  return VKERNELS_PP_ERR_INTERNAL;
}

}  // namespace

extern "C" vkernels_pp_boundary_plan_t* vkernels_pp_boundary_plan_create(
    int world, int rank, size_t payload_bytes,
    int transport, int dir, void* peer_buf,
    vkernels_pp_status_t* status_out) {
  if (status_out != nullptr)
    *status_out = VKERNELS_PP_OK;
  try {
    auto* plan = new vkernels::comm::cuda::PipelineBoundaryPlan(
        world, rank, payload_bytes, from_c_transport(transport),
        from_c_dir(dir), peer_buf);
    return reinterpret_cast<vkernels_pp_boundary_plan_t*>(plan);
  } catch (const std::exception& e) {
    if (status_out != nullptr)
      *status_out = to_status(e);
    return nullptr;
  }
}

extern "C" void vkernels_pp_boundary_plan_destroy(
    vkernels_pp_boundary_plan_t* plan) {
  delete reinterpret_cast<vkernels::comm::cuda::PipelineBoundaryPlan*>(plan);
}

extern "C" int vkernels_pp_boundary_plan_execute(
    vkernels_pp_boundary_plan_t* plan, void* my_buf, cudaStream_t stream) {
  if (plan == nullptr || my_buf == nullptr)
    return VKERNELS_PP_ERR_INVALID_ARGUMENT;
  try {
    reinterpret_cast<vkernels::comm::cuda::PipelineBoundaryPlan*>(plan)
        ->execute(my_buf, stream);
    return VKERNELS_PP_OK;
  } catch (const std::exception& e) {
    return to_status(e);
  }
}

#endif  // defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)
