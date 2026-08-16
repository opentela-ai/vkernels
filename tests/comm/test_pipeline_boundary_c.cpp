// tests/comm/test_pipeline_boundary_c.cpp
//
// Host tests for the always-compiled `extern "C"` transport-classification
// surface (issue #10). The ABI (pipeline_boundary_c.h / pipeline_boundary_c.cpp)
// wraps the host reference with no GPU dependency, so every entry point is
// exercised on its happy path and the null-config contract on its error
// path — the same 100% line-coverage gate the rest of the host CI job runs.
#include "vkernels/comm/pipeline_boundary_c.h"

#include "minitest.hpp"

#include <string>

namespace {

// vkernels_pp_config_t is an aggregate; brace-init keeps the address stable
// for the duration of the call.
#define PP_CFG(same, nccl, gloo) \
  vkernels_pp_config_t { same, nccl, gloo }

}  // namespace

TEST(PipelineBoundaryCABI, ClassifySameNodePeer) {
  vkernels_pp_config_t c = PP_CFG(1, 0, 0);
  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_classify(&c, &status),
            VKERNELS_PP_TRANSPORT_SAME_NODE_PEER);
  EXPECT_EQ(status, VKERNELS_PP_OK);
}

TEST(PipelineBoundaryCABI, ClassifyCrossNodeNccl) {
  vkernels_pp_config_t c = PP_CFG(0, 1, 0);
  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_classify(&c, &status),
            VKERNELS_PP_TRANSPORT_CROSS_NODE_NCCL);
  EXPECT_EQ(status, VKERNELS_PP_OK);
}

TEST(PipelineBoundaryCABI, ClassifyHostStagedViaGloo) {
  vkernels_pp_config_t c = PP_CFG(0, 0, 1);
  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_classify(&c, &status),
            VKERNELS_PP_TRANSPORT_HOST_STAGED);
  EXPECT_EQ(status, VKERNELS_PP_OK);
}

TEST(PipelineBoundaryCABI, ClassifyHostStagedByDefault) {
  vkernels_pp_config_t c = PP_CFG(0, 0, 0);
  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_classify(&c, &status),
            VKERNELS_PP_TRANSPORT_HOST_STAGED);
  EXPECT_EQ(status, VKERNELS_PP_OK);
}

TEST(PipelineBoundaryCABI, ClassifyNullConfig) {
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_EQ(vkernels_pp_classify(nullptr, &status),
            VKERNELS_PP_TRANSPORT_HOST_STAGED);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
}

TEST(PipelineBoundaryCABI, ClassifyNullStatusOut) {
  vkernels_pp_config_t c = PP_CFG(1, 0, 0);
  EXPECT_EQ(vkernels_pp_classify(&c, nullptr),
            VKERNELS_PP_TRANSPORT_SAME_NODE_PEER);
}

TEST(PipelineBoundaryCABI, EagerBreakFalseForCapturable) {
  vkernels_pp_config_t a = PP_CFG(1, 0, 0);
  vkernels_pp_status_t sa = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_eager_break(&a, &sa), 0);
  EXPECT_EQ(sa, VKERNELS_PP_OK);
  vkernels_pp_config_t b = PP_CFG(0, 1, 0);
  vkernels_pp_status_t sb = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_eager_break(&b, &sb), 0);
  EXPECT_EQ(sb, VKERNELS_PP_OK);
}

TEST(PipelineBoundaryCABI, EagerBreakTrueForHostStaged) {
  vkernels_pp_config_t a = PP_CFG(0, 0, 1);
  vkernels_pp_status_t sa = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_eager_break(&a, &sa), 1);
  EXPECT_EQ(sa, VKERNELS_PP_OK);
  vkernels_pp_config_t b = PP_CFG(0, 0, 0);
  vkernels_pp_status_t sb = VKERNELS_PP_ERR_INTERNAL;
  EXPECT_EQ(vkernels_pp_eager_break(&b, &sb), 1);
  EXPECT_EQ(sb, VKERNELS_PP_OK);
}

TEST(PipelineBoundaryCABI, EagerBreakNullConfig) {
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_EQ(vkernels_pp_eager_break(nullptr, &status), 0);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
}

TEST(PipelineBoundaryCABI, EagerBreakNullStatusOut) {
  vkernels_pp_config_t c = PP_CFG(0, 0, 1);
  EXPECT_EQ(vkernels_pp_eager_break(&c, nullptr), 1);
}

TEST(PipelineBoundaryCABI, TransportNameKnown) {
  EXPECT_EQ(std::string(vkernels_pp_transport_name(
                VKERNELS_PP_TRANSPORT_SAME_NODE_PEER)),
            "same-node-peer");
  EXPECT_EQ(std::string(vkernels_pp_transport_name(
                VKERNELS_PP_TRANSPORT_CROSS_NODE_NCCL)),
            "cross-node-nccl");
  EXPECT_EQ(std::string(
                vkernels_pp_transport_name(VKERNELS_PP_TRANSPORT_HOST_STAGED)),
            "host-staged");
}

TEST(PipelineBoundaryCABI, TransportNameUnknown) {
  EXPECT_EQ(std::string(vkernels_pp_transport_name(99)), "?");
}
