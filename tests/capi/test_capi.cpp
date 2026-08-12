// tests/capi/test_capi.cpp
//
// Coverage + correctness tests for the C ABI layer (src/c/vkernels/capi).
// Every exported `vk_*` function is exercised on its happy path, and the
// functions whose C++ backends can throw a contract violation
// (`std::invalid_argument` from `VK_EXPECTS`) are also exercised on their
// error path so the catch-and-translate machinery (status code +
// `vk_last_error`) is covered end to end.
#include "minitest.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "vkernels/capi/capi.hpp"

namespace {

// Callback trampolines — plain function pointers so they can cross the C ABI.
void stream_fn(void* ctx) {
  int* counter = static_cast<int*>(ctx);
  ++*counter;
}

int overlap_compute(size_t i, void* ctx) {
  int* sum = static_cast<int*>(ctx);
  *sum += static_cast<int>(i + 1);
  return static_cast<int>(i + 1);
}

void overlap_comm(size_t, int value, void* ctx) {
  int* total = static_cast<int*>(ctx);
  *total += value;
}

}  // namespace

// ---------------------------------------------------------------------------
// Version / config / error state
// ---------------------------------------------------------------------------
TEST(Capi, VersionAndConfig) {
  EXPECT_NE(vk_version(), nullptr);
  EXPECT_GT(std::strlen(vk_version()), 0u);
  int cuda = vk_has_cuda();
  EXPECT_TRUE(cuda == 0 || cuda == 1);
}

TEST(Capi, LastErrorDefaultsToOk) {
  EXPECT_EQ(vk_last_error_code(), VK_OK);
  EXPECT_NE(vk_last_error(), nullptr);
}

TEST(Capi, FreeNullIsSafe) {
  vk_free(nullptr);
}

// ---------------------------------------------------------------------------
// Kernels: element-wise, reduce, gemm
// ---------------------------------------------------------------------------
TEST(Capi, AddHappyPath) {
  const float a[3] = {1.0f, 2.0f, 3.0f};
  const float b[3] = {10.0f, 20.0f, 30.0f};
  float out[3] = {};
  EXPECT_EQ(vk_add(a, 3, b, 3, out, 3), VK_OK);
  EXPECT_NEAR(out[0], 11.0f, 1e-6f);
  EXPECT_NEAR(out[1], 22.0f, 1e-6f);
  EXPECT_NEAR(out[2], 33.0f, 1e-6f);
}

TEST(Capi, AddLengthMismatchSetsError) {
  const float a[2] = {1.0f, 2.0f};
  const float b[3] = {1.0f, 2.0f, 3.0f};
  float out[3] = {};
  EXPECT_EQ(vk_add(a, 2, b, 3, out, 3), VK_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
  EXPECT_GT(std::strlen(vk_last_error()), 0u);
}

TEST(Capi, ScaleHappyPath) {
  const float x[3] = {1.0f, -2.0f, 3.0f};
  float out[3] = {};
  EXPECT_EQ(vk_scale(x, 3, 2.5f, out, 3), VK_OK);
  EXPECT_NEAR(out[0], 2.5f, 1e-6f);
  EXPECT_NEAR(out[1], -5.0f, 1e-6f);
  EXPECT_NEAR(out[2], 7.5f, 1e-6f);
}

TEST(Capi, ScaleLengthMismatchSetsError) {
  const float x[2] = {1.0f, 2.0f};
  float out[3] = {};
  EXPECT_EQ(vk_scale(x, 2, 1.0f, out, 3), VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, ReluHappyPath) {
  const float x[4] = {-1.0f, 0.0f, 2.0f, -3.0f};
  float out[4] = {};
  EXPECT_EQ(vk_relu(x, 4, out, 4), VK_OK);
  EXPECT_NEAR(out[0], 0.0f, 1e-6f);
  EXPECT_NEAR(out[1], 0.0f, 1e-6f);
  EXPECT_NEAR(out[2], 2.0f, 1e-6f);
  EXPECT_NEAR(out[3], 0.0f, 1e-6f);
}

TEST(Capi, ReluLengthMismatchSetsError) {
  const float x[2] = {1.0f, 2.0f};
  float out[3] = {};
  EXPECT_EQ(vk_relu(x, 2, out, 3), VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, SumHappyPath) {
  const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out = 0.0f;
  EXPECT_EQ(vk_sum(x, 4, &out), VK_OK);
  EXPECT_NEAR(out, 10.0f, 1e-6f);
}

TEST(Capi, SumEmptyThrows) {
  float out = 0.0f;
  EXPECT_EQ(vk_sum(nullptr, 0, &out), VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, MaxHappyPath) {
  const float x[4] = {1.0f, -5.0f, 7.0f, 3.0f};
  float out = 0.0f;
  EXPECT_EQ(vk_max(x, 4, &out), VK_OK);
  EXPECT_NEAR(out, 7.0f, 1e-6f);
}

TEST(Capi, MaxEmptyThrows) {
  float out = 0.0f;
  EXPECT_EQ(vk_max(nullptr, 0, &out), VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, GemmHappyPath) {
  // C[2x2] = A[2x3] @ B[3x2]
  const float A[6] = {1, 2, 3, 4, 5, 6};
  const float B[6] = {7, 8, 9, 10, 11, 12};
  float C[4] = {};
  EXPECT_EQ(vk_gemm(2, 2, 3, 1.0f, A, 6, B, 6, 0.0f, C, 4), VK_OK);
  EXPECT_NEAR(C[0], 58.0f, 1e-4f);   // 1*7 + 2*9 + 3*11
  EXPECT_NEAR(C[1], 64.0f, 1e-4f);   // 1*8 + 2*10 + 3*12
  EXPECT_NEAR(C[2], 139.0f, 1e-4f);  // 4*7 + 5*9 + 6*11
  EXPECT_NEAR(C[3], 154.0f, 1e-4f);  // 4*8 + 5*10 + 6*12
}

TEST(Capi, GemmSizeMismatchSetsError) {
  const float A[6] = {1, 2, 3, 4, 5, 6};
  const float B[6] = {7, 8, 9, 10, 11, 12};
  float C[4] = {};
  EXPECT_EQ(vk_gemm(2, 2, 3, 1.0f, A, 5, B, 6, 0.0f, C, 4),
            VK_ERROR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// core: device + stream
// ---------------------------------------------------------------------------
TEST(Capi, DeviceLifecycle) {
  vk_device* d = vk_device_new(-1);
  ASSERT_NE(d, nullptr);
  EXPECT_EQ(vk_device_index(d), -1);  // -1 selects the default device
  vk_device* d2 = vk_device_new(-1);
  ASSERT_NE(d2, nullptr);
  EXPECT_EQ(vk_device_eq(d, d2), 1);
  EXPECT_EQ(vk_device_supports_peer(d, d2), 0);  // single CPU device
  EXPECT_EQ(vk_device_set_current(d), VK_OK);
  EXPECT_EQ(vk_device_sync(d), VK_OK);

  vk_device* d3 = vk_device_new(2);
  ASSERT_NE(d3, nullptr);
  EXPECT_EQ(vk_device_index(d3), 2);
  EXPECT_EQ(vk_device_eq(d, d3), 0);  // index -1 != 2

  vk_device_delete(d3);
  vk_device_delete(d2);
  vk_device_delete(d);
}

TEST(Capi, StreamLifecycleAndSubmit) {
  vk_stream* s = vk_stream_new();
  ASSERT_NE(s, nullptr);
  int counter = 0;
  EXPECT_EQ(vk_stream_submitted(s), 0u);
  EXPECT_EQ(vk_stream_submit(s, stream_fn, &counter), VK_OK);
  EXPECT_EQ(vk_stream_submit(s, stream_fn, &counter), VK_OK);
  vk_stream_wait(s);
  EXPECT_EQ(counter, 2);
  EXPECT_EQ(vk_stream_submitted(s), 2u);
  vk_stream_delete(s);
}

// ---------------------------------------------------------------------------
// comm: topology
// ---------------------------------------------------------------------------
TEST(Capi, RingRank) {
  vk_topology t;
  EXPECT_EQ(vk_ring_rank(2, 3, &t), VK_OK);
  EXPECT_EQ(t.rank, 2);
  EXPECT_EQ(t.world, 3);
  EXPECT_EQ(t.next, 0);
  EXPECT_EQ(t.prev, 1);
}

TEST(Capi, RingRankInvalidThrows) {
  vk_topology t;
  EXPECT_EQ(vk_ring_rank(3, 3, &t), VK_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(vk_ring_rank(0, 0, &t), VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, BuildRingTopology) {
  vk_topology* arr = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_build_ring_topology(3, &arr, &count), VK_OK);
  ASSERT_NE(arr, nullptr);
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(arr[0].rank, 0);
  EXPECT_EQ(arr[1].next, 2);
  EXPECT_EQ(arr[2].next, 0);
  vk_free(arr);
}

TEST(Capi, BuildRingTopologyInvalidThrows) {
  vk_topology* arr = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_build_ring_topology(0, &arr, &count), VK_ERROR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// comm: channels
// ---------------------------------------------------------------------------
TEST(Capi, QueuePushPopAndClose) {
  vk_queue* q = vk_queue_new();
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(vk_queue_closed(q), 0);

  const float data[3] = {1.0f, 2.0f, 3.0f};
  EXPECT_EQ(vk_queue_push(q, data, 3), VK_OK);

  float* out = nullptr;
  size_t out_len = 0;
  EXPECT_EQ(vk_queue_pop(q, &out, &out_len), VK_OK);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out_len, 3u);
  EXPECT_NEAR(out[0], 1.0f, 1e-6f);
  EXPECT_NEAR(out[1], 2.0f, 1e-6f);
  EXPECT_NEAR(out[2], 3.0f, 1e-6f);
  vk_free(out);

  vk_queue_close(q);
  EXPECT_EQ(vk_queue_closed(q), 1);
  vk_queue_delete(q);
}

TEST(Capi, ChannelSendRecv) {
  vk_queue* q = vk_queue_new();
  ASSERT_NE(q, nullptr);
  vk_channel* c = vk_channel_new(q, q);  // self-loop: send into q, recv from q
  ASSERT_NE(c, nullptr);

  const float msg[2] = {7.0f, 8.0f};
  EXPECT_EQ(vk_channel_send(c, msg, 2), VK_OK);
  EXPECT_EQ(vk_channel_closed(c), 0);

  float* out = nullptr;
  size_t out_len = 0;
  EXPECT_EQ(vk_channel_recv(c, &out, &out_len), VK_OK);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out_len, 2u);
  EXPECT_NEAR(out[0], 7.0f, 1e-6f);
  EXPECT_NEAR(out[1], 8.0f, 1e-6f);
  vk_free(out);

  vk_channel_delete(c);
  vk_queue_delete(q);
}

TEST(Capi, ChannelNewWithNullQueuesThrows) {
  EXPECT_EQ(vk_channel_new(nullptr, nullptr), nullptr);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, MakeRingChannels) {
  vk_channel** arr = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_make_ring_channels(3, &arr, &count), VK_OK);
  ASSERT_NE(arr, nullptr);
  EXPECT_EQ(count, 3u);
  // channel[0] sends to channel[1].
  const float msg[1] = {42.0f};
  EXPECT_EQ(vk_channel_send(arr[0], msg, 1), VK_OK);
  float* out = nullptr;
  size_t out_len = 0;
  EXPECT_EQ(vk_channel_recv(arr[1], &out, &out_len), VK_OK);
  ASSERT_NE(out, nullptr);
  EXPECT_NEAR(out[0], 42.0f, 1e-6f);
  vk_free(out);

  for (size_t i = 0; i < count; ++i) vk_channel_delete(arr[i]);
  vk_free(arr);
}

TEST(Capi, MakeRingChannelsInvalidThrows) {
  vk_channel** arr = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_make_ring_channels(0, &arr, &count), VK_ERROR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// comm: ring all-reduce
// ---------------------------------------------------------------------------
TEST(Capi, RingAllreduceRankWorldOne) {
  vk_queue* q = vk_queue_new();
  ASSERT_NE(q, nullptr);
  vk_channel* c = vk_channel_new(q, q);
  ASSERT_NE(c, nullptr);

  float local[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(vk_ring_allreduce_rank(local, 4, 0, 1, c, c), VK_OK);
  EXPECT_NEAR(local[0], 1.0f, 1e-6f);  // world == 1: no-op
  EXPECT_NEAR(local[3], 4.0f, 1e-6f);

  vk_channel_delete(c);
  vk_queue_delete(q);
}

TEST(Capi, RingAllreduceRankInvalidThrows) {
  vk_queue* q = vk_queue_new();
  ASSERT_NE(q, nullptr);
  vk_channel* c = vk_channel_new(q, q);
  ASSERT_NE(c, nullptr);

  float local[3] = {1.0f, 2.0f, 3.0f};
  EXPECT_EQ(vk_ring_allreduce_rank(local, 3, 0, 0, c, c),
            VK_ERROR_INVALID_ARGUMENT);  // world must be positive
  EXPECT_EQ(vk_ring_allreduce_rank(local, 3, 3, 3, c, c),
            VK_ERROR_INVALID_ARGUMENT);  // rank out of range

  vk_channel_delete(c);
  vk_queue_delete(q);
}

// ---------------------------------------------------------------------------
// comm: overlap
// ---------------------------------------------------------------------------
TEST(Capi, OverlapRun) {
  vk_overlap* ex = vk_overlap_new();
  ASSERT_NE(ex, nullptr);
  EXPECT_TRUE(vk_overlap_uses_two_streams(ex));

  int sum = 0, total = 0;
  size_t compute_count = 0, comm_count = 0;
  EXPECT_EQ(vk_overlap_run(ex, 5, overlap_compute, &sum, overlap_comm, &total,
                           &compute_count, &comm_count),
            VK_OK);
  EXPECT_EQ(compute_count, 5u);
  EXPECT_EQ(comm_count, 5u);
  EXPECT_EQ(sum, 15);    // 1+2+3+4+5
  EXPECT_EQ(total, 15);  // comm received each compute value exactly once
  vk_overlap_delete(ex);
}

TEST(Capi, OverlapZeroIterations) {
  vk_overlap* ex = vk_overlap_new();
  ASSERT_NE(ex, nullptr);
  int sum = 0, total = 0;
  size_t compute_count = 0, comm_count = 0;
  EXPECT_EQ(vk_overlap_run(ex, 0, overlap_compute, &sum, overlap_comm, &total,
                           &compute_count, &comm_count),
            VK_OK);
  EXPECT_EQ(compute_count, 0u);
  EXPECT_EQ(comm_count, 0u);
  vk_overlap_delete(ex);
}

// ---------------------------------------------------------------------------
// comm: p2p run-list gather
// ---------------------------------------------------------------------------
TEST(Capi, StageRuns1d) {
  uint8_t dst[32] = {};
  const uint8_t a[2] = {1, 2};
  const uint8_t b[4] = {3, 4, 5, 6};
  const void* srcs[2] = {a, b};
  const size_t offs[2] = {0, 8};
  const size_t lens[2] = {2, 4};

  vk_staged_run_1d* runs = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_stage_runs_1d(dst, 32, srcs, offs, lens, 2, &runs, &count),
            VK_OK);
  ASSERT_NE(runs, nullptr);
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(runs[0].dst_offset, 0u);
  EXPECT_EQ(runs[0].length, 2u);
  EXPECT_EQ(runs[1].dst_offset, 8u);
  EXPECT_EQ(runs[1].length, 4u);
  vk_free(runs);
}

TEST(Capi, StageRuns1dCapacityViolationThrows) {
  uint8_t dst[4] = {};
  const uint8_t a[8] = {};
  const void* srcs[1] = {a};
  const size_t offs[1] = {0};
  const size_t lens[1] = {8};
  vk_staged_run_1d* runs = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_stage_runs_1d(dst, 4, srcs, offs, lens, 1, &runs, &count),
            VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, StageRuns2d) {
  uint8_t dst[64] = {};
  const uint8_t src[16] = {};
  vk_gather_2d runs_in[1] = {{src, 4, 0, 4, 4, 4}};
  vk_staged_run_2d* runs = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_stage_runs_2d(dst, 64, runs_in, 1, &runs, &count), VK_OK);
  ASSERT_NE(runs, nullptr);
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(runs[0].width, 4u);
  EXPECT_EQ(runs[0].height, 4u);
  vk_free(runs);
}

TEST(Capi, StageRuns2dNullSrcThrows) {
  uint8_t dst[64] = {};
  vk_gather_2d runs_in[1] = {{nullptr, 4, 0, 4, 4, 4}};
  vk_staged_run_2d* runs = nullptr;
  size_t count = 0;
  EXPECT_EQ(vk_stage_runs_2d(dst, 64, runs_in, 1, &runs, &count),
            VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, P2pGatherRunsSync) {
  const uint8_t a[2] = {1, 2};
  const uint8_t b[4] = {3, 4, 5, 6};
  uint8_t dst[32] = {};
  const void* srcs[2] = {a, b};
  const size_t offs[2] = {0, 8};
  const size_t lens[2] = {2, 4};
  EXPECT_EQ(vk_p2p_gather_runs(dst, 32, srcs, offs, lens, 2, nullptr), VK_OK);
  EXPECT_EQ(dst[0], 1);
  EXPECT_EQ(dst[1], 2);
  EXPECT_EQ(dst[8], 3);
  EXPECT_EQ(dst[11], 6);
}

TEST(Capi, P2pGatherRunsAsync) {
  const uint8_t a[3] = {10, 11, 12};
  uint8_t dst[16] = {};
  const void* srcs[1] = {a};
  const size_t offs[1] = {0};
  const size_t lens[1] = {3};
  vk_stream* s = vk_stream_new();
  ASSERT_NE(s, nullptr);
  EXPECT_EQ(vk_p2p_gather_runs(dst, 16, srcs, offs, lens, 1, s), VK_OK);
  vk_stream_wait(s);
  EXPECT_EQ(dst[0], 10);
  EXPECT_EQ(dst[2], 12);
  vk_stream_delete(s);
}

TEST(Capi, P2pGatherRunsNullSrcsThrows) {
  uint8_t dst[8] = {};
  const size_t offs[1] = {0};
  const size_t lens[1] = {1};
  EXPECT_EQ(vk_p2p_gather_runs(dst, 8, nullptr, offs, lens, 1, nullptr),
            VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, P2pGatherRuns2d) {
  const uint8_t src[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint8_t dst[64] = {};
  vk_gather_2d runs[1] = {{src, 4, 0, 4, 4, 4}};
  EXPECT_EQ(vk_p2p_gather_runs_2d(dst, 64, runs, 1, nullptr), VK_OK);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(dst[i], src[i]);
}

TEST(Capi, P2pGatherRuns2dWidthExceedsStrideThrows) {
  const uint8_t src[8] = {};
  uint8_t dst[32] = {};
  vk_gather_2d runs[1] = {{src, 2, 0, 2, 4, 1}};  // width 4 > src_stride 2
  EXPECT_EQ(vk_p2p_gather_runs_2d(dst, 32, runs, 1, nullptr),
            VK_ERROR_INVALID_ARGUMENT);
}

TEST(Capi, MemcpyPeerBatchAsync) {
  const uint8_t a[2] = {1, 2};
  const uint8_t b[2] = {3, 4};
  uint8_t dst[16] = {};
  const void* srcs[2] = {a, b};
  const size_t offs[2] = {0, 4};
  const size_t lens[2] = {2, 2};
  vk_stream* s = vk_stream_new();
  ASSERT_NE(s, nullptr);
  size_t before = vk_stream_submitted(s);
  EXPECT_EQ(vk_memcpy_peer_batch_async(dst, 16, srcs, offs, lens, 2, s), VK_OK);
  EXPECT_EQ(vk_stream_submitted(s) - before, 2u);  // one task per run
  vk_stream_wait(s);
  EXPECT_EQ(dst[0], 1);
  EXPECT_EQ(dst[4], 3);
  EXPECT_EQ(dst[5], 4);
  vk_stream_delete(s);
}
