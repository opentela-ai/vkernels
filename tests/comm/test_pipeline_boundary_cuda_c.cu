// tests/comm/test_pipeline_boundary_cuda_c.cu
//
// Runtime tests for the CUDA-only `extern "C"` PP-boundary plan (issue #10).
// Single-GPU, using same-device buffers as a stand-in for peer memory —
// sufficient to exercise the validators, the enqueued `cudaMemcpyAsync`
// peer copy, the round-trip round-trip graph capture/replay (the acceptance
// criterion: a rank-pair round trip captured once and replayed N times with
// no host progress), and the status-code return paths.
//
// The cross-node NCCL happy path requires a real multi-node + NCCL harness
// and is exercised by the host reference's acceptance test
// (test_pipeline_boundary.cpp::CapturedCrossNodeNcclRoundTrip); only the
// cross-node *create* path is asserted here.
#include "vkernels/comm/pipeline_boundary_c.h"

#include "minitest.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

namespace {

constexpr size_t kPayload = 256;

std::vector<uint8_t> patterned(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = static_cast<uint8_t>(seed + (i % 251));
  return v;
}

bool device_equal(const uint8_t* d_a, const uint8_t* d_b, size_t n) {
  std::vector<uint8_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_a, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, n, cudaMemcpyDeviceToHost);
  return std::memcmp(ha.data(), hb.data(), n) == 0;
}

bool device_equals_host(const uint8_t* d, const std::vector<uint8_t>& h) {
  std::vector<uint8_t> hh(h.size());
  cudaMemcpy(hh.data(), d, h.size(), cudaMemcpyDeviceToHost);
  return std::memcmp(hh.data(), h.data(), h.size()) == 0;
}

}  // namespace

TEST(PipelineBoundaryCUDA, PeerSendCopiesMyBufToPeer) {
  uint8_t *my = nullptr, *peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&my, kPayload) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  const auto h = patterned(kPayload, 7);
  ASSERT_TRUE(cudaMemcpy(my, h.data(), kPayload, cudaMemcpyHostToDevice) ==
              cudaSuccess);

  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  vkernels_pp_boundary_plan_t* plan = vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_SEND, peer, &status);
  ASSERT_TRUE(plan != nullptr);
  EXPECT_EQ(status, VKERNELS_PP_OK);

  cudaStream_t stream;
  ASSERT_TRUE(cudaStreamCreate(&stream) == cudaSuccess);
  EXPECT_EQ(vkernels_pp_boundary_plan_execute(plan, my, stream),
            VKERNELS_PP_OK);
  ASSERT_TRUE(cudaStreamSynchronize(stream) == cudaSuccess);
  EXPECT_TRUE(device_equals_host(peer, h));

  vkernels_pp_boundary_plan_destroy(plan);
  cudaStreamDestroy(stream);
  cudaFree(my);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, PeerRecvCopiesPeerToMyBuf) {
  uint8_t *my = nullptr, *peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&my, kPayload) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  const auto h = patterned(kPayload, 11);
  ASSERT_TRUE(cudaMemcpy(peer, h.data(), kPayload, cudaMemcpyHostToDevice) ==
              cudaSuccess);

  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  vkernels_pp_boundary_plan_t* plan = vkernels_pp_boundary_plan_create(
      2, 1, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_RECV, peer, &status);
  ASSERT_TRUE(plan != nullptr);
  EXPECT_EQ(status, VKERNELS_PP_OK);

  cudaStream_t stream;
  ASSERT_TRUE(cudaStreamCreate(&stream) == cudaSuccess);
  EXPECT_EQ(vkernels_pp_boundary_plan_execute(plan, my, stream),
            VKERNELS_PP_OK);
  ASSERT_TRUE(cudaStreamSynchronize(stream) == cudaSuccess);
  EXPECT_TRUE(device_equals_host(my, h));

  vkernels_pp_boundary_plan_destroy(plan);
  cudaStreamDestroy(stream);
  cudaFree(my);
  cudaFree(peer);
}

// Acceptance: a rank-pair round trip (send A->B, recv B->A) captured once
// and replayed N times on a graph with no host progress. After each replay
// B holds A's bytes and A holds its original bytes back.
TEST(PipelineBoundaryCUDA, CapturedReplayRoundTrip) {
  constexpr int kReplays = 16;
  uint8_t *my = nullptr, *peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&my, kPayload) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  const auto h = patterned(kPayload, 23);
  ASSERT_TRUE(cudaMemcpy(my, h.data(), kPayload, cudaMemcpyHostToDevice) ==
              cudaSuccess);

  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  vkernels_pp_boundary_plan_t* send = vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_SEND, peer, &status);
  ASSERT_TRUE(send != nullptr);
  EXPECT_EQ(status, VKERNELS_PP_OK);
  vkernels_pp_boundary_plan_t* recv = vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_RECV, peer, &status);
  ASSERT_TRUE(recv != nullptr);
  EXPECT_EQ(status, VKERNELS_PP_OK);

  cudaStream_t stream;
  ASSERT_TRUE(cudaStreamCreate(&stream) == cudaSuccess);
  ASSERT_TRUE(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) ==
              cudaSuccess);
  EXPECT_EQ(vkernels_pp_boundary_plan_execute(send, my, stream),
            VKERNELS_PP_OK);
  EXPECT_EQ(vkernels_pp_boundary_plan_execute(recv, my, stream),
            VKERNELS_PP_OK);
  cudaGraph_t graph;
  ASSERT_TRUE(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  cudaGraphExec_t exec;
  ASSERT_TRUE(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
  for (int i = 0; i < kReplays; ++i) {
    ASSERT_TRUE(cudaGraphLaunch(exec, stream) == cudaSuccess);
    ASSERT_TRUE(cudaStreamSynchronize(stream) == cudaSuccess);
    EXPECT_TRUE(device_equals_host(peer, h));  // B has A's bytes
    EXPECT_TRUE(device_equals_host(my, h));    // A got them back
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  vkernels_pp_boundary_plan_destroy(send);
  vkernels_pp_boundary_plan_destroy(recv);
  cudaStreamDestroy(stream);
  cudaFree(my);
  cudaFree(peer);
}

// The cross-node transport is device-capturable, so create succeeds even
// when NCCL is not linked; execute then refuses with VKERNELS_PP_ERR_UNSUPPORTED
// (no silent host-bounce that would deadlock a captured graph). A multi-node
// + NCCL harness exercises the real ncclSend/ncclRecv happy path.
TEST(PipelineBoundaryCUDA, CrossNodeCreateSucceeds) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  vkernels_pp_boundary_plan_t* plan = vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_CROSS_NODE_NCCL,
      VKERNELS_PP_DIR_SEND, peer, &status);
  EXPECT_TRUE(plan != nullptr);
  EXPECT_EQ(status, VKERNELS_PP_OK);
  vkernels_pp_boundary_plan_destroy(plan);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, CreateRejectsWorldZero) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_TRUE(vkernels_pp_boundary_plan_create(
      0, 0, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_SEND, peer, &status) == nullptr);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, CreateRejectsRankOutOfRange) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_TRUE(vkernels_pp_boundary_plan_create(
      2, 2, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_SEND, peer, &status) == nullptr);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, CreateRejectsPayloadZero) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_TRUE(vkernels_pp_boundary_plan_create(
      2, 0, 0, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_SEND, peer, &status) == nullptr);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, CreateRejectsHostStaged) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_TRUE(vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_HOST_STAGED,
      VKERNELS_PP_DIR_SEND, peer, &status) == nullptr);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, CreateRejectsNullPeer) {
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_TRUE(vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_SEND, nullptr, &status) == nullptr);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
}

TEST(PipelineBoundaryCUDA, CreateRejectsBadTransport) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_TRUE(vkernels_pp_boundary_plan_create(
      2, 0, kPayload, 99, VKERNELS_PP_DIR_SEND, peer, &status) == nullptr);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, CreateRejectsBadDir) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_OK;
  EXPECT_TRUE(vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      99, peer, &status) == nullptr);
  EXPECT_EQ(status, VKERNELS_PP_ERR_INVALID_ARGUMENT);
  cudaFree(peer);
}

TEST(PipelineBoundaryCUDA, ExecuteRejectsNullPlan) {
  uint8_t* my = nullptr;
  ASSERT_TRUE(cudaMalloc(&my, kPayload) == cudaSuccess);
  cudaStream_t stream;
  ASSERT_TRUE(cudaStreamCreate(&stream) == cudaSuccess);
  EXPECT_EQ(vkernels_pp_boundary_plan_execute(nullptr, my, stream),
            VKERNELS_PP_ERR_INVALID_ARGUMENT);
  cudaStreamDestroy(stream);
  cudaFree(my);
}

TEST(PipelineBoundaryCUDA, ExecuteRejectsNullMyBuf) {
  uint8_t* peer = nullptr;
  ASSERT_TRUE(cudaMalloc(&peer, kPayload) == cudaSuccess);
  vkernels_pp_status_t status = VKERNELS_PP_ERR_INTERNAL;
  vkernels_pp_boundary_plan_t* plan = vkernels_pp_boundary_plan_create(
      2, 0, kPayload, VKERNELS_PP_TRANSPORT_SAME_NODE_PEER,
      VKERNELS_PP_DIR_SEND, peer, &status);
  ASSERT_TRUE(plan != nullptr);
  cudaStream_t stream;
  ASSERT_TRUE(cudaStreamCreate(&stream) == cudaSuccess);
  EXPECT_EQ(vkernels_pp_boundary_plan_execute(plan, nullptr, stream),
            VKERNELS_PP_ERR_INVALID_ARGUMENT);
  vkernels_pp_boundary_plan_destroy(plan);
  cudaStreamDestroy(stream);
  cudaFree(peer);
}

#endif  // defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)
