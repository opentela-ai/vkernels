// vkernels/comm/pipeline_boundary_c.cpp — thin `extern "C"` wrapper over
// the PP-boundary transport classification + eager-break decision (issue
// #10).
//
// Always compiled (no GPU, no NCCL), so the host CI job and its 100%
// line-coverage gate exercise the same planning surface a non-C++ consumer
// reaches through pipeline_boundary_c.h. The wrapped host functions
// (classify_boundary, eager_break_during_capture, pipeline_transport_name)
// are pure, so nothing is ever thrown across the ABI boundary.
#include "vkernels/comm/pipeline_boundary_c.h"

#include "vkernels/comm/pipeline_boundary.hpp"

namespace {

int to_c_transport(vkernels::comm::PipelineTransport t) {
  switch (t) {
    case vkernels::comm::PipelineTransport::kSameNodePeer:
      return VKERNELS_PP_TRANSPORT_SAME_NODE_PEER;
    case vkernels::comm::PipelineTransport::kCrossNodeNccl:
      return VKERNELS_PP_TRANSPORT_CROSS_NODE_NCCL;
    case vkernels::comm::PipelineTransport::kHostStaged:
      return VKERNELS_PP_TRANSPORT_HOST_STAGED;
  }
  return VKERNELS_PP_TRANSPORT_HOST_STAGED;  // LCOV_EXCL_LINE (exhaustive)
}

}  // namespace

extern "C" int vkernels_pp_classify(const vkernels_pp_config_t* cfg,
                                    vkernels_pp_status_t* status_out) {
  if (cfg == nullptr) {
    if (status_out != nullptr)
      *status_out = VKERNELS_PP_ERR_INVALID_ARGUMENT;
    return VKERNELS_PP_TRANSPORT_HOST_STAGED;
  }
  vkernels::comm::PipelineBoundaryConfig cpp;
  cpp.same_node = cfg->same_node != 0;
  cpp.nccl_graph_supported = cfg->nccl_graph_supported != 0;
  cpp.gloo_fallback = cfg->gloo_fallback != 0;
  const int t = to_c_transport(vkernels::comm::classify_boundary(cpp));
  if (status_out != nullptr)
    *status_out = VKERNELS_PP_OK;
  return t;
}

extern "C" int vkernels_pp_eager_break(const vkernels_pp_config_t* cfg,
                                       vkernels_pp_status_t* status_out) {
  if (cfg == nullptr) {
    if (status_out != nullptr)
      *status_out = VKERNELS_PP_ERR_INVALID_ARGUMENT;
    return 0;
  }
  vkernels::comm::PipelineBoundaryConfig cpp;
  cpp.same_node = cfg->same_node != 0;
  cpp.nccl_graph_supported = cfg->nccl_graph_supported != 0;
  cpp.gloo_fallback = cfg->gloo_fallback != 0;
  const int eager = vkernels::comm::eager_break_during_capture(cpp) ? 1 : 0;
  if (status_out != nullptr)
    *status_out = VKERNELS_PP_OK;
  return eager;
}

extern "C" const char* vkernels_pp_transport_name(int transport) {
  switch (transport) {
    case VKERNELS_PP_TRANSPORT_SAME_NODE_PEER:
      return vkernels::comm::pipeline_transport_name(
          vkernels::comm::PipelineTransport::kSameNodePeer);
    case VKERNELS_PP_TRANSPORT_CROSS_NODE_NCCL:
      return vkernels::comm::pipeline_transport_name(
          vkernels::comm::PipelineTransport::kCrossNodeNccl);
    case VKERNELS_PP_TRANSPORT_HOST_STAGED:
      return vkernels::comm::pipeline_transport_name(
          vkernels::comm::PipelineTransport::kHostStaged);
    default:
      return "?";
  }
}
