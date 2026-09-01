// tests/capi/test_capi.cpp
//
// Coverage + correctness tests for the C ABI layer (src/c/vkernels/capi).
// Every exported `vk_*` function is exercised on its happy path, and the
// functions whose C++ backends can throw a contract violation
// (`std::invalid_argument` from `VK_EXPECTS`) are also exercised on their
// error path so the catch-and-translate machinery (status code +
// `vk_last_error`) is covered end to end.
#include "minitest.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
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

// --- K3-family helpers (mirror tests/kernels/* independent refs) ------

float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(float));
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// E2M1 nibble → float (matches moe.cpp fp4_nibble_to_float).
// Used by Fp4ToBf16DequantRoundTrip to validate the C ABI dequant
// against the same nibble table.
float e2m1_nibble_to_f32(uint8_t n) {
  int s = (n >> 3) & 1, e = (n >> 1) & 3, m = n & 1;
  if (e == 0) return m ? (s ? -0.25f : 0.25f) : 0.0f;
  if (e == 3) return m ? std::nanf("") : (s ? -INFINITY : INFINITY);
  float v = (1.0f + static_cast<float>(m) * 0.5f) *
            static_cast<float>(1 << (e - 1));
  return s ? -v : v;
}

// Nearest E2M1 nibble for a float value (matches test_moe_fused).
uint8_t float_to_e2m1_nibble(float f) {
  bool neg = f < 0;
  float af = std::fabs(f);
  if (af == 0.0f || std::isnan(af)) return neg ? 0x8 : 0x0;
  if (std::isinf(af)) return neg ? 0xE : 0x6;
  static const float vals[5] = {0.25f, 1.0f, 1.5f, 2.0f, 3.0f};
  static const uint8_t nibs[5] = {1, 2, 3, 4, 5};
  float best_d = std::fabs(af - vals[0]);
  uint8_t best_n = nibs[0];
  for (int i = 1; i < 5; ++i) {
    float d = std::fabs(af - vals[i]);
    if (d < best_d) { best_d = d; best_n = nibs[i]; }
  }
  return neg ? static_cast<uint8_t>(best_n | 0x8) : best_n;
}

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

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

// ===========================================================================
// K3-family C ABI wrappers (added by the Rust-bindings commit).
// These exercise every new vk_* entry point through the C ABI boundary,
// mirroring the independent refs in tests/kernels/* so the coverage gate
// (100% on the host build) is satisfied.
// ===========================================================================

// --- gfx942 primitives (moe.hpp) ------------------------------------------

TEST(CapiGfx942, DirectLdsFillBf16) {
  const uint16_t src[4] = {0x3C00, 0x4000, 0x4040, 0x4080};  // 1,2,3,4
  uint16_t dst[4] = {0, 0, 0, 0};
  EXPECT_EQ(vk_direct_lds_fill_bf16(dst, src, 4), VK_OK);
  EXPECT_EQ(dst[0], 0x3C00);
  EXPECT_EQ(dst[3], 0x4080);
}

TEST(CapiGfx942, DirectLdsFillBf16EmptyIsNoop) {
  uint16_t dst[1] = {0xFFFF};
  EXPECT_EQ(vk_direct_lds_fill_bf16(dst, nullptr, 0), VK_OK);
  EXPECT_EQ(dst[0], 0xFFFF);  // untouched
}

TEST(CapiGfx942, DirectLdsFillBf16NullThrows) {
  EXPECT_NE(vk_direct_lds_fill_bf16(nullptr, (const uint16_t*)"x", 1), VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

TEST(CapiGfx942, Fp4ToBf16Dequant) {
  // Byte 0x12: low nibble 2 (1.0), high nibble 1 (0.25), scale=1
  uint8_t packed[1] = {0x12};
  uint16_t out[2];
  EXPECT_EQ(vk_fp4_to_bf16_dequant(packed, 1, out, 2, 1.0f), VK_OK);
  EXPECT_NEAR(bf16_to_f32(out[0]), e2m1_nibble_to_f32(2), 1e-2f);
  EXPECT_NEAR(bf16_to_f32(out[1]), e2m1_nibble_to_f32(1), 1e-2f);
}

TEST(CapiGfx942, Fp4ToBf16DequantScale) {
  // Byte 0x22: low nibble 2 (1.0), high nibble 2 (1.0), scale=2.0
  uint8_t packed[1] = {0x22};
  uint16_t out[2];
  EXPECT_EQ(vk_fp4_to_bf16_dequant(packed, 1, out, 2, 2.0f), VK_OK);
  EXPECT_NEAR(bf16_to_f32(out[0]), e2m1_nibble_to_f32(2) * 2.0f, 1e-1f);
  EXPECT_NEAR(bf16_to_f32(out[1]), e2m1_nibble_to_f32(2) * 2.0f, 1e-1f);
}

TEST(CapiGfx942, Fp4ToBf16DequantRoundTrip) {
  // Quantize known floats to E2M1 nibbles, pack, dequant via C ABI,
  // and check the values match e2m1_nibble_to_f32.
  float vals[4] = {0.25f, 1.0f, 1.5f, 3.0f};
  uint8_t packed[2];
  packed[0] = float_to_e2m1_nibble(vals[0]) |
              (float_to_e2m1_nibble(vals[1]) << 4);
  packed[1] = float_to_e2m1_nibble(vals[2]) |
              (float_to_e2m1_nibble(vals[3]) << 4);
  uint16_t out[4];
  EXPECT_EQ(vk_fp4_to_bf16_dequant(packed, 2, out, 4, 1.0f), VK_OK);
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(bf16_to_f32(out[i]), vals[i], 1e-2f);
}

TEST(CapiGfx942, Fp4ToBf16DequantLengthMismatch) {
  uint8_t packed[1] = {0};
  uint16_t out[3];  // should be 2
  EXPECT_NE(vk_fp4_to_bf16_dequant(packed, 1, out, 3, 1.0f), VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

TEST(CapiGfx942, UseAsyncCopyDefault) {
  int r = vk_use_async_copy_default();
  EXPECT_TRUE(r == 0 || r == 1);
}

TEST(CapiGfx942, MfmaF32_16x16x16bf16) {
  // Pack 1.0 and 2.0 into each uint32 (lo=1.0, hi=2.0)
  uint32_t a[2] = {
      static_cast<uint32_t>(f32_to_bf16(1.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(2.0f)) << 16),
      static_cast<uint32_t>(f32_to_bf16(3.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(4.0f)) << 16)};
  uint32_t b[2] = {
      static_cast<uint32_t>(f32_to_bf16(1.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(1.0f)) << 16),
      static_cast<uint32_t>(f32_to_bf16(1.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(1.0f)) << 16)};
  float c[4] = {0, 0, 0, 0};
  EXPECT_EQ(vk_mfma_f32_16x16x16bf16(c, a, b, 0, 0, 0), VK_OK);
  // c[i] += a_f32[i] * b_f32[i] = a_f32[i] * 1.0
  EXPECT_NEAR(c[0], 1.0f, 1e-5f);
  EXPECT_NEAR(c[1], 2.0f, 1e-5f);
  EXPECT_NEAR(c[2], 3.0f, 1e-5f);
  EXPECT_NEAR(c[3], 4.0f, 1e-5f);
}

TEST(CapiGfx942, MfmaF32Accumulates) {
  uint32_t a[2] = {
      static_cast<uint32_t>(f32_to_bf16(1.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(1.0f)) << 16),
      static_cast<uint32_t>(f32_to_bf16(1.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(1.0f)) << 16)};
  uint32_t b[2] = {
      static_cast<uint32_t>(f32_to_bf16(2.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(2.0f)) << 16),
      static_cast<uint32_t>(f32_to_bf16(2.0f)) |
          (static_cast<uint32_t>(f32_to_bf16(2.0f)) << 16)};
  float c[4] = {10, 20, 30, 40};  // pre-existing, beta-style accumulate
  EXPECT_EQ(vk_mfma_f32_16x16x16bf16(c, a, b, 0, 0, 0), VK_OK);
  EXPECT_NEAR(c[0], 12.0f, 1e-5f);  // 10 + 1*2
  EXPECT_NEAR(c[1], 22.0f, 1e-5f);
  EXPECT_NEAR(c[2], 32.0f, 1e-5f);
  EXPECT_NEAR(c[3], 42.0f, 1e-5f);
}

TEST(CapiGfx942, MfmaF32NullThrows) {
  uint32_t a[2] = {0, 0}, b[2] = {0, 0};
  EXPECT_NE(vk_mfma_f32_16x16x16bf16(nullptr, a, b, 0, 0, 0), VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// --- bf16 GEMM (gemm_bf16.hpp, issue #29) ---------------------------------

TEST(CapiGemmBf16, IdentityAlphaOneBetaZero) {
  // A = [[1,2],[3,4]], B = identity, C should = A
  uint16_t A[4] = {f32_to_bf16(1), f32_to_bf16(2), f32_to_bf16(3),
                   f32_to_bf16(4)};
  uint16_t B[4] = {f32_to_bf16(1), f32_to_bf16(0), f32_to_bf16(0),
                   f32_to_bf16(1)};
  uint16_t C[4] = {0, 0, 0, 0};
  EXPECT_EQ(vk_gemm_bf16(2, 2, 2, 1.0f, A, B, 0.0f, C), VK_OK);
  EXPECT_NEAR(bf16_to_f32(C[0]), 1.0f, 1e-2f);
  EXPECT_NEAR(bf16_to_f32(C[1]), 2.0f, 1e-2f);
  EXPECT_NEAR(bf16_to_f32(C[2]), 3.0f, 1e-2f);
  EXPECT_NEAR(bf16_to_f32(C[3]), 4.0f, 1e-2f);
}

TEST(CapiGemmBf16, BetaAccumulation) {
  // A=[[1]], B=[[1]], C starts at 10, alpha=2 beta=1 -> 2*1+10=12
  uint16_t A[1] = {f32_to_bf16(1)};
  uint16_t B[1] = {f32_to_bf16(1)};
  uint16_t C[1] = {f32_to_bf16(10)};
  EXPECT_EQ(vk_gemm_bf16(1, 1, 1, 2.0f, A, B, 1.0f, C), VK_OK);
  EXPECT_NEAR(bf16_to_f32(C[0]), 12.0f, 1e-1f);
}

TEST(CapiGemmBf16, Config) {
  int bm = 0, bn = 0, bk = 0, threads = 0;
  vk_gemm_bf16_config(256, 7168, 33792, &bm, &bn, &bk, &threads);
  EXPECT_GT(bm, 0);
  EXPECT_GT(bn, 0);
  EXPECT_GT(bk, 0);
  EXPECT_GT(threads, 0);
}

// --- MLA forward (mla.hpp, issue #21) -------------------------------------

TEST(CapiMla, HandCheckedTwoQuery) {
  // B=1 H=1 S_q=2 S_kv=2, lr=2 rhd=2, scale=0.5
  // Matches test_mla.cpp::MlaFwd::HandChecked
  const float scale = 0.5f;
  std::vector<float> q = {1, 0, 0, 0, 0, 1, 1, 0};
  std::vector<float> k_c = {1, 0, 0, 1};
  std::vector<float> k_pe = {0, 1, 1, 0};
  std::vector<float> v_c = {1, 2, 3, 4};
  std::vector<float> out(2 * 2, -1.0f);

  EXPECT_EQ(vk_mla_fwd(1, 1, 2, 2, 0, 0, 2, 2, scale, q.data(),
                        k_c.data(), k_pe.data(), v_c.data(), out.data()),
            VK_OK);
  // q0 attends only to kv0 (causal): score=0.5*(1*1+0*0+0*0+0*1)=0.5,
  // softmax=1.0, out0 = 1*v_c[0..1] = [1,2]
  EXPECT_NEAR(out[0], 1.0f, 1e-5f);
  EXPECT_NEAR(out[1], 2.0f, 1e-5f);
  // q1 attends to both kv0 and kv1; this is a known-good case from
  // test_mla.cpp — just check no NaN and positive.
  EXPECT_FALSE(std::isnan(out[2]));
  EXPECT_FALSE(std::isnan(out[3]));
}

TEST(CapiMla, ConfigDecode) {
  int bq = 0, bn = 0, threads = 0;
  vk_mla_config(1, 512, 64, &bq, &bn, &threads);  // decode
  EXPECT_GT(bq, 0);
  EXPECT_GT(bn, 0);
  EXPECT_GT(threads, 0);
}

TEST(CapiMla, ConfigPrefill) {
  int bq = 0, bn = 0, threads = 0;
  vk_mla_config(64, 512, 64, &bq, &bn, &threads);  // prefill
  EXPECT_GT(bq, 0);
  EXPECT_GT(bn, 0);
  EXPECT_GT(threads, 0);
}

TEST(CapiMla, NullArgsThrow) {
  EXPECT_NE(vk_mla_fwd(1, 1, 1, 1, 0, 0, 2, 2, 0.5f, nullptr, nullptr,
                        nullptr, nullptr, nullptr),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// --- DSA sparse-MLA forward (dsa.hpp, issue #51) -------------------------

TEST(CapiDsa, HandCheckedTailDimZero) {
  // GLM-5.3-Flash layout (tail_dim == 0). q=[1,1], 2 keys, sm_scale=1.
  // Matches test_dsa.cpp::DsaFwd::HandCheckedTailDimZero: scores {1,1}
  // -> base-2 weights {1,1}/2 -> out {0.5,0.5}, lse=2.
  const float sm_scale = 1.0f;
  std::vector<float> q = {1, 1};
  std::vector<float> kv = {1, 0, 0, 1};
  std::vector<int32_t> idx = {0, 1};
  std::vector<float> out(2, -1.0f), lse(1, 7.0f);
  EXPECT_EQ(vk_dsa_sparse_fwd(1, 2, 1, 2, 0, 2, 1, 2, 1, sm_scale, /*lse=*/1,
                              q.data(), kv.data(), idx.data(), out.data(),
                              lse.data()),
            VK_OK);
  EXPECT_NEAR(out[0], 0.5f, 1e-6f);
  EXPECT_NEAR(out[1], 0.5f, 1e-6f);
  EXPECT_NEAR(lse[0], 2.0f, 1e-6f);
}

TEST(CapiDsa, HandCheckedTailDimPositive) {
  // DeepSeek-V3 layout (tail_dim > 0): dim=2 tail=1 d_v=1.
  // Matches test_dsa.cpp::DsaFwd::HandCheckedTailDimPositive: scores {2,3}
  // -> out = 4/1.5, lse = 3 + log2(1.5).
  const float sm_scale = 1.0f;
  std::vector<float> q = {1, 0, 0};            // main=[1,0] tail=[0]
  std::vector<float> kv = {2, 1, 1, 3, 0, 1}; // 2 keys, dim+tail=3
  std::vector<int32_t> idx = {0, 1};
  std::vector<float> out(1, -1.0f), lse(1, 7.0f);
  EXPECT_EQ(vk_dsa_sparse_fwd(1, 2, 1, 2, 1, 2, 1, 2, 1, sm_scale, 1,
                              q.data(), kv.data(), idx.data(), out.data(),
                              lse.data()),
            VK_OK);
  EXPECT_NEAR(out[0], 4.0f / 1.5f, 1e-6f);
  EXPECT_NEAR(lse[0], 3.0f + std::log2(1.5f), 1e-6f);
}

TEST(CapiDsa, TopkGroupSupportHelper) {
  EXPECT_EQ(vk_dsa_topk_group_topk_supported(128), 1);
  EXPECT_EQ(vk_dsa_topk_group_topk_supported(64), 0);
}

TEST(CapiDsa, TopkTransformPageTableAndTail) {
  std::vector<float> score(16);
  for (int i = 0; i < 16; ++i) score[i] = static_cast<float>(i);
  std::vector<int32_t> lengths = {2};
  std::vector<int32_t> page_table(64);
  for (int32_t i = 0; i < 64; ++i) page_table[i] = 300 + i;
  std::vector<int32_t> page_row = {0};
  std::vector<int32_t> seq_lens = {10};
  std::vector<int32_t> out(515, -1);
  EXPECT_EQ(vk_dsa_topk_transform(1, score.data(), lengths.data(), out.data(),
                                  16, 4, 512, 515, page_table.data(), 64,
                                  page_row.data(), nullptr, nullptr,
                                  seq_lens.data()),
            VK_OK);
  for (int32_t i = 0; i < 10; ++i) EXPECT_EQ(out[i], 300 + i);
  for (int32_t i = 10; i < 515; ++i) EXPECT_EQ(out[i], -1);
}

TEST(CapiDsa, TopkTransformConflictingMappingsSetError) {
  std::vector<float> score(16, 0.0f);
  std::vector<int32_t> lengths = {2};
  std::vector<int32_t> page_table(64, 0);
  std::vector<int32_t> offsets = {1};
  std::vector<int32_t> out(512, -1);
  EXPECT_EQ(vk_dsa_topk_transform(1, score.data(), lengths.data(), out.data(),
                                  16, 4, 512, 512, page_table.data(), 64,
                                  nullptr, offsets.data(), nullptr, nullptr),
            VK_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

TEST(CapiDsa, ConfigDecode) {
  int bq = 0, threads = 0, bi = 0, ii = 0;
  vk_dsa_config(1, 64, 256, 128, &bq, &threads, &bi, &ii);  // decode
  EXPECT_GT(bq, 0);
  EXPECT_GT(threads, 0);
  EXPECT_GT(bi, 0);
  EXPECT_GT(ii, 0);
}

TEST(CapiDsa, ConfigPrefill) {
  int bq = 0, threads = 0, bi = 0, ii = 0;
  vk_dsa_config(16, 64, 256, 128, &bq, &threads, &bi, &ii);  // prefill
  EXPECT_GT(bq, 0);
  EXPECT_GT(threads, 0);
  EXPECT_GT(bi, 0);
  EXPECT_GT(ii, 0);
}

TEST(CapiDsa, NullArgsThrow) {
  std::vector<float> q(4, 1), kv(4, 1), out(2, 1);
  std::vector<int32_t> idx(2, 0);
  EXPECT_NE(vk_dsa_sparse_fwd(1, 1, 1, 2, 0, 2, 1, 2, 1, 1.0f, 0, nullptr,
                              kv.data(), idx.data(), out.data(), nullptr),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// topk not divisible by block_I*inner_iter is rejected (issue #57).
TEST(CapiDsa, DivisibilityCheckThrows) {
  std::vector<float> q(2, 1), kv(4, 1), out(2, 0);
  std::vector<int32_t> idx(2, 0);
  // topk=2, block_I=4, inner_iter=1: 2 % (4*1) = 2 != 0.
  EXPECT_NE(vk_dsa_sparse_fwd(1, 2, 1, 2, 0, 2, 1, 4, 1, 1.0f, 0, q.data(),
                              kv.data(), idx.data(), out.data(), nullptr),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// `out` aliasing an input is rejected (issue #57): the two-pass softmax
// writes a row into `out` then reads it back.
TEST(CapiDsa, AliasThrows) {
  std::vector<float> q(2, 1), kv(4, 1), out(2, 0);
  std::vector<int32_t> idx(2, 0);
  // out == q (same buffer).
  EXPECT_NE(vk_dsa_sparse_fwd(1, 2, 1, 2, 0, 2, 1, 2, 1, 1.0f, 0, out.data(),
                              kv.data(), idx.data(), out.data(), nullptr),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// --- DSA paged-MQA gated top-k logits (dsa.hpp, issue #51, the kpool>1 ---
//    indexer path that FEEDS vk_dsa_sparse_fwd) -------------------------

// Mirrors test_dsa.cpp::DsaTopk::HandCheckedSingleHead: 1 head, 1 block,
// 2 keys [1,0][0,1], scales [2,3], gate [1], seq_len 2.
//   t=0: dot=1, acc=1*1=1, out=2*1=2 ;  t=1: dot=1, acc=1, out=3*1=3
TEST(CapiDsaTopk, HandCheckedSingleHead) {
  const float q[2] = {1, 1};
  const float kv[4] = {1, 0, 0, 1};        // [1 block][2 tokens][2]
  const float k_scale[2] = {2, 3};
  const float gate[1] = {1};
  const int32_t sl[1] = {2};
  const int32_t pt[1] = {0};
  std::vector<float> out(2, -1.0f);
  EXPECT_EQ(vk_dsa_topk_logits(1, 1, 2, 2, 1, 1, q, kv, k_scale, gate, sl,
                              pt, out.data()),
            VK_OK);
  EXPECT_NEAR(out[0], 2.0f, 1e-6f);
  EXPECT_NEAR(out[1], 3.0f, 1e-6f);
}

// seq_len truncates WITHIN a page (block=4, seq_len=2): tokens 2,3 of the
// page are past seq_len and LEFT UNWRITTEN. Mirrors test_dsa.cpp::DsaTopk.
TEST(CapiDsaTopk, SeqLenTruncatesWithinPage) {
  const float q[2] = {1, 1};
  const float kv[8] = {1, 1, 2, 2, 3, 3, 4, 4};  // [1][4 tokens][2]
  const float k_scale[4] = {1, 1, 1, 1};
  const float gate[1] = {1};
  const int32_t sl[1] = {2};
  const int32_t pt[1] = {0};
  std::vector<float> out(4, 0.0f);                 // max_seq_len = 1*4
  EXPECT_EQ(vk_dsa_topk_logits(1, 1, 2, 4, 1, 1, q, kv, k_scale, gate, sl,
                              pt, out.data()),
            VK_OK);
  EXPECT_NEAR(out[0], 2.0f, 1e-6f);
  EXPECT_NEAR(out[1], 4.0f, 1e-6f);
  EXPECT_EQ(out[2], 0.0f);   // past seq_len, left unwritten
  EXPECT_EQ(out[3], 0.0f);
}

// Empty (batch_size==0 or max_table_len==0) is a no-op: output untouched
// (mirrors CapiDsa::EmptyIsNoOp + test_dsa.cpp::DsaTopk::EmptyIsNoOp).
TEST(CapiDsaTopk, EmptyIsNoOp) {
  std::vector<float> out = {3.0f, 4.0f};
  const float q[2] = {1, 1}, kv[4] = {1, 1, 1, 1}, k_scale[2] = {1, 1},
      gate[1] = {1};
  const int32_t sl[1] = {2}, pt[1] = {0};
  EXPECT_EQ(vk_dsa_topk_logits(0, 1, 2, 2, 1, 1, q, kv, k_scale, gate, sl,
                              pt, out.data()),
            VK_OK);
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[1], 4.0f);
  EXPECT_EQ(vk_dsa_topk_logits(1, 1, 2, 2, 0, 1, q, kv, k_scale, gate, sl,
                              pt, out.data()),
            VK_OK);
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[1], 4.0f);
}

// Null pointer (batch_size>0, max_table_len>0) -> VK_ERROR_INVALID_ARGUMENT.
TEST(CapiDsaTopk, NullArgsThrow) {
  const float kv[4] = {1, 1, 1, 1}, k_scale[2] = {1, 1}, gate[1] = {1};
  const int32_t sl[1] = {2}, pt[1] = {0};
  std::vector<float> out(2, 1.0f);
  EXPECT_NE(vk_dsa_topk_logits(1, 1, 2, 2, 1, 1, nullptr, kv, k_scale, gate,
                              sl, pt, out.data()),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// dsa_topk_logits_split_for: optimal split_kv = max(1, min(ceildiv(msl,B),
// 228/bs)). bs=1 msl=4096 B=64 -> 64 (one page/split, 64 of 228 CUs);
// bs=64 msl=4096 B=64 -> min(64, 3) = 3 (192 of 228 CUs); empty -> 1.
TEST(CapiDsaTopk, SplitFor) {
  int sp = 0;
  EXPECT_EQ(vk_dsa_topk_logits_split_for(1, 4096, 64, &sp), VK_OK);
  EXPECT_EQ(sp, 64);
  EXPECT_EQ(vk_dsa_topk_logits_split_for(1, 1024, 64, &sp), VK_OK);
  EXPECT_EQ(sp, 16);
  EXPECT_EQ(vk_dsa_topk_logits_split_for(64, 4096, 64, &sp), VK_OK);
  EXPECT_EQ(sp, 3);
  EXPECT_EQ(vk_dsa_topk_logits_split_for(0, 4096, 64, &sp), VK_OK);
  EXPECT_EQ(sp, 1);
}

TEST(CapiDsaTopk, SplitForNullArg) {
  EXPECT_NE(vk_dsa_topk_logits_split_for(1, 4096, 64, nullptr), VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// --- MHC — multi-head hybrid-attention pre-norm (mhc.hpp, issue #51) ----

TEST(CapiMhc, PreGemmSqrsumHandChecked) {
  // hc_mult=2, hidden=2 -> hc_hidden_size=4, hc_mult3=8. fn is identity in
  // the first 4 rows, zero after (hc_mult3 > hc_hidden_size). x=[1,2,3,4]
  // -> out=[1,2,3,4,0,0,0,0], sqrsum=30. Mirrors test_mhc.cpp.
  std::vector<float> x = {1, 2, 3, 4};
  std::vector<float> fn(8 * 4, 0.0f);
  for (int o = 0; o < 4; ++o) fn[o * 4 + o] = 1.0f;
  std::vector<float> out(8, -1.0f), sqrsum(1, -1.0f);
  EXPECT_EQ(vk_mhc_pre_gemm_sqrsum(1, 2, 2, x.data(), fn.data(), out.data(),
                                   sqrsum.data()),
            VK_OK);
  EXPECT_NEAR(out[0], 1.0f, 1e-6f); EXPECT_NEAR(out[1], 2.0f, 1e-6f);
  EXPECT_NEAR(out[2], 3.0f, 1e-6f); EXPECT_NEAR(out[3], 4.0f, 1e-6f);
  EXPECT_NEAR(out[4], 0.0f, 1e-6f); EXPECT_NEAR(out[5], 0.0f, 1e-6f);
  EXPECT_NEAR(out[6], 0.0f, 1e-6f); EXPECT_NEAR(out[7], 0.0f, 1e-6f);
  EXPECT_NEAR(sqrsum[0], 30.0f, 1e-6f);
}

TEST(CapiMhc, PostHandChecked) {
  // hc=2, hidden=2. a = identity (2x2), b = [[1,2],[3,4]], c=[10,20],
  // d=[1,1] -> out = [11,12,23,24]. Mirrors test_mhc.cpp.
  std::vector<float> a = {1, 0, 0, 1};
  std::vector<float> b = {1, 2, 3, 4};
  std::vector<float> c = {10, 20};
  std::vector<float> d = {1, 1};
  std::vector<float> out(4, -1.0f);
  EXPECT_EQ(vk_mhc_post(1, 2, 2, a.data(), b.data(), c.data(), d.data(),
                        out.data()),
            VK_OK);
  EXPECT_NEAR(out[0], 11.0f, 1e-6f); EXPECT_NEAR(out[1], 12.0f, 1e-6f);
  EXPECT_NEAR(out[2], 23.0f, 1e-6f); EXPECT_NEAR(out[3], 24.0f, 1e-6f);
}

TEST(CapiMhc, NullArgsThrow) {
  std::vector<float> x(4, 1), fn(8, 1), out(8, 1), sq(1, 1);
  EXPECT_NE(vk_mhc_pre_gemm_sqrsum(1, 2, 2, nullptr, fn.data(), out.data(),
                                   sq.data()),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);

  std::vector<float> a(4, 1), b(4, 1), c(2, 1), dd(2, 1), oo(4, 1);
  EXPECT_NE(vk_mhc_post(1, 2, 2, a.data(), b.data(), nullptr, dd.data(),
                        oo.data()),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

// --- KDA — Kimi Delta Attention (kda.hpp, issue #21) ----------------------

TEST(CapiKda, LayerNormGatedIdentityWeightUnitGate) {
  // weight=1, gate=5 -> silu(5)≈4.966, x all ones, rms=1 -> out≈4.966
  constexpr int N = 1, D = 4;
  std::vector<float> x(D, 1.0f), w(D, 1.0f), gate(D, 5.0f), out(D, -1);
  EXPECT_EQ(vk_kda_layer_norm_gated(x.data(), w.data(), gate.data(),
                                     out.data(), N, D, 1e-6f),
            VK_OK);
  const float silu5 = 5.0f * sigmoid_f(5.0f);
  for (int d = 0; d < D; ++d) EXPECT_NEAR(out[d], silu5, 1e-4f);
}

TEST(CapiKda, LayerNormGatedZeroGate) {
  // gate=0 -> silu(0)=0 -> output is zero
  constexpr int N = 2, D = 3;
  std::vector<float> x(N * D, 7.0f), w(D, 2.0f), gate(N * D, 0.0f),
      out(N * D, -1);
  EXPECT_EQ(vk_kda_layer_norm_gated(x.data(), w.data(), gate.data(),
                                     out.data(), N, D, 1e-6f),
            VK_OK);
  for (float v : out) EXPECT_NEAR(v, 0.0f, 1e-6f);
}

TEST(CapiKda, GateChunkCumsum) {
  constexpr int B = 1, H = 1, nc = 2, cs = 3;
  std::vector<float> g(nc * cs, 0.5f);  // all 0.5
  std::vector<float> intra(nc * cs), inter(nc);
  EXPECT_EQ(vk_kda_gate_chunk_cumsum(g.data(), intra.data(), inter.data(),
                                      B, H, nc, cs),
            VK_OK);
  // intra[c,t] = sum_{l<=t} log(0.5) = (t+1)*log(0.5)
  float lg = std::log(0.5f);
  for (int c = 0; c < nc; ++c)
    for (int t = 0; t < cs; ++t)
      EXPECT_NEAR(intra[c * cs + t], lg * (t + 1), 1e-5f);
  // inter[0] = 0, inter[1] = intra[0, cs-1] = cs*log(0.5)
  EXPECT_NEAR(inter[0], 0.0f, 1e-6f);
  EXPECT_NEAR(inter[1], lg * cs, 1e-5f);
}

TEST(CapiKda, NaiveDeltaRuleFwd) {
  // K3 per-key-dim gated delta rule (issue #45). B=1 H=1 S=2 D=2,
  // q=k=v=ones, g=[0.5,0.5] per token (per-key-dim [B,H,S,D]), beta=0.5.
  //   t=0: S'=0 -> a=0 -> S0=0.5*v0*k0^T=0.5*[[1,1],[1,1]], o0=[1,1]
  //   t=1: S'=g1.*S0=0.25*[[1,1],[1,1]], a=S'*.k1=[0.5,0.5],
  //        S1=S'+0.5*(1-0.5)*k1^T=[[0.5,0.5],[0.5,0.5]], o1=[1,1]
  constexpr int B = 1, H = 1, S = 2, D = 2;
  std::vector<float> q(B * H * S * D, 1.0f), k(B * H * S * D, 1.0f),
      v(B * H * S * D, 1.0f), g(B * H * S * D, 0.5f), beta(B * H * S, 0.5f),
      out(B * H * S * D, 0.0f);
  EXPECT_EQ(vk_kda_naive_delta_rule_fwd(q.data(), k.data(), v.data(),
                                          g.data(), beta.data(), out.data(),
                                          B, H, S, D),
            VK_OK);
  for (int i = 0; i < B * H * S * D; ++i) EXPECT_NEAR(out[i], 1.0f, 1e-5f);
}

TEST(CapiKda, DeltaRuleFwdMatchesNaive) {
  // Cross-check the chunked forward (vk_kda_delta_rule_fwd, the STANDARD
  // gated delta rule with a SCALAR forget gate [B,H,S]) against the naive
  // oracle (vk_kda_naive_delta_rule_fwd, the K3 PER-KEY-DIM gated delta
  // rule [B,H,S,D]). Issue #45 split the two recurrences: the naive
  // predicts from the POST-gate state, the chunked from the PRE-gate
  // state, so they differ at arbitrary gates. They agree only at
  // g == 1 -- full history, no forgetting -- where the gate is the
  // identity in both and post-gate == pre-gate. (Before #45 the naive
  // oracle also implemented the standard rule and was cross-checked at
  // random gates; that now requires the per-key-dim g to be sized
  // [B,H,S,D], else the naive read runs out of bounds.)
  constexpr int B = 1, H = 1, S = 8, D = 4, chunk = 4;
  std::mt19937 rng(7);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  std::vector<float> q(B * H * S * D), k(q.size()), v(q.size());
  std::vector<float> g_naive(B * H * S * D, 1.0f);  // per-key-dim, no forget
  std::vector<float> g_chunk(B * H * S, 1.0f);      // scalar, no forget
  std::vector<float> beta(B * H * S);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  std::vector<float> out_naive(B * H * S * D, 0), out_chunk(B * H * S * D, 0);
  EXPECT_EQ(vk_kda_naive_delta_rule_fwd(q.data(), k.data(), v.data(),
                                          g_naive.data(), beta.data(),
                                          out_naive.data(), B, H, S, D),
            VK_OK);
  EXPECT_EQ(vk_kda_delta_rule_fwd(q.data(), k.data(), v.data(),
                                   g_chunk.data(), beta.data(),
                                   out_chunk.data(), B, H, S, D, chunk),
            VK_OK);
  float max_abs = 0;
  for (int i = 0; i < B * H * S * D; ++i)
    max_abs = std::max(max_abs, std::fabs(out_naive[i] - out_chunk[i]));
  EXPECT_LT(max_abs, 1e-3f);
}

TEST(CapiKda, DeltaRuleIntraInterOutput) {
  // Exercise the chunked intra/inter/output combine pipeline
  constexpr int B = 1, H = 1, S = 8, D = 4, chunk = 4;
  constexpr int nc = S / chunk;
  std::vector<float> q(B * H * S * D, 0.5f), k(B * H * S * D, 0.5f),
      v(B * H * S * D, 0.5f), g(B * H * S, 0.8f), beta(B * H * S, 0.5f);
  std::vector<float> intra(B * H * nc * chunk), inter(B * H * (nc + 1) * D * D,
                                                       0.0f),
      u(B * H * nc * chunk * D), out(B * H * S * D, 0.0f);

  // gate cumsum
  EXPECT_EQ(vk_kda_gate_chunk_cumsum(g.data(), intra.data(), inter.data(),
                                      B, H, nc, chunk),
            VK_OK);

  // intra for each chunk
  for (int c = 0; c < nc; ++c) {
    EXPECT_EQ(vk_kda_delta_rule_intra(q.data(), k.data(), v.data(),
                                       g.data(), beta.data(), intra.data(),
                                       inter.data(), u.data(), B, H, S, D,
                                       chunk, c),
              VK_OK);
    EXPECT_EQ(vk_kda_delta_rule_inter(k.data(), v.data(), g.data(),
                                       beta.data(), intra.data(), u.data(),
                                       inter.data(), B, H, S, D, chunk, c),
              VK_OK);
  }

  // output combine
  EXPECT_EQ(vk_kda_gla_fwd_o(q.data(), k.data(), g.data(), beta.data(),
                              intra.data(), inter.data(), u.data(),
                              out.data(), B, H, S, D, chunk),
            VK_OK);

  // Check no NaN and some non-zero
  bool any_nonzero = false;
  for (float v : out) {
    EXPECT_FALSE(std::isnan(v));
    if (v != 0.0f) any_nonzero = true;
  }
  EXPECT_TRUE(any_nonzero);
}

TEST(CapiKda, PackBitmatrix) {
  // 8 bits -> 1 byte, MSB first: bit 0 -> bit 7, bit 7 -> bit 0
  uint8_t bits[8] = {1, 0, 0, 0, 0, 0, 0, 0};  // only bit 0 set
  uint8_t packed[1] = {0};
  EXPECT_EQ(vk_kda_pack_bitmatrix(bits, packed, 8), VK_OK);
  // bit 0 -> byte 0, bit 7-0%8 = bit 7
  EXPECT_EQ(packed[0], 0x80);
}

TEST(CapiKda, PackBitmatrixMultipleBytes) {
  // 12 bits: bits[0]=1, bits[8]=1 -> byte 0 = 0x80, byte 1 = 0x80
  uint8_t bits[12] = {};
  bits[0] = 1;
  bits[8] = 1;
  uint8_t packed[2] = {0, 0};
  EXPECT_EQ(vk_kda_pack_bitmatrix(bits, packed, 12), VK_OK);
  EXPECT_EQ(packed[0], 0x80);
  EXPECT_EQ(packed[1], 0x80);
}

// --- MoE orchestration (moe_aux.hpp, issue #22) ---------------------------

TEST(CapiMoeAux, Mxfp4QuantRoundTrip) {
  // Mirror test_moe_aux::QuantRoundTrip: quantize a representative set,
  // dequant via the same nibble table, and check each element is within
  // one fp4 step of the input (the quantizer is nearest-value).
  constexpr int M = 2, hidden = 32, gs = 32;
  const float repr[] = {0.f, 0.25f, -0.25f, 1.f, -1.f, 1.5f,
                        -1.5f, 2.f, -2.f, 3.f, -3.f, 0.7f,
                        -0.7f, 2.3f, -2.3f, 0.125f};
  std::vector<uint16_t> A(M * hidden);
  for (size_t i = 0; i < A.size(); ++i)
    A[i] = f32_to_bf16(repr[i % (sizeof(repr) / sizeof(repr[0]))]);
  std::vector<uint8_t> packed(M * hidden / 2), scales(M);
  EXPECT_EQ(vk_mxfp4_moe_quant(A.data(), packed.data(), scales.data(),
                                M, hidden, gs),
            VK_OK);
  // Dequant each group and check proximity + sign preservation.
  for (int m = 0; m < M; ++m) {
    // ue8m0: s == 0xFF -> 0.0, else 2^(s - 127).
    float sc = 0.0f;
    uint32_t sb = static_cast<uint32_t>(scales[m]) << 23;
    std::memcpy(&sc, &sb, sizeof(float));
    for (int i = 0; i < hidden; ++i) {
      uint8_t byte = packed[m * (hidden / 2) + i / 2];
      uint8_t nib = (i & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
      float d = e2m1_nibble_to_f32(nib) * sc;
      float a = bf16_to_f32(A[m * hidden + i]);
      EXPECT_TRUE(std::isfinite(d));
      float step = (std::fabs(a) <= 0.25f) ? 0.25f : 0.5f;
      EXPECT_NEAR(std::fabs(a), std::fabs(d), step + 1e-6f);
      if (a != 0.0f) EXPECT_TRUE((a < 0) == (d < 0));
    }
  }
}

TEST(CapiMoeAux, Mxfp4QuantZeroGroup) {
  constexpr int M = 2, hidden = 32, gs = 32;
  std::vector<uint16_t> A(M * hidden, 0);  // all zero
  std::vector<uint8_t> packed(M * hidden / 2), scales(M);
  EXPECT_EQ(vk_mxfp4_moe_quant(A.data(), packed.data(), scales.data(),
                                M, hidden, gs),
            VK_OK);
  for (uint8_t s : scales) EXPECT_EQ(s, 0xFF);
  for (uint8_t b : packed) EXPECT_EQ(b, 0);
}

TEST(CapiMoeAux, Mxfp4QuantBadGroupSizeThrows) {
  constexpr int M = 1, hidden = 30;  // not divisible by 32
  std::vector<uint16_t> A(M * hidden, 0);
  std::vector<uint8_t> p(M * hidden / 2), s(M);
  EXPECT_NE(vk_mxfp4_moe_quant(A.data(), p.data(), s.data(), M, hidden,
                                32),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

TEST(CapiMoeAux, Mxfp4Sort) {
  // sorted_ids[r] is a FLAT index in [0, M*top_k); token = flat / top_k.
  // Row r of A_sorted = A[token * hidden .. (token+1)*hidden] (or zero).
  constexpr int M = 3, hidden = 4, top_k = 2;
  std::vector<int32_t> sids = {2, 6, 1};  // flat0=tok1, flat1=pad, flat2=tok0
  constexpr int EM = 3;
  std::vector<uint16_t> A(M * hidden);
  for (int i = 0; i < M; ++i)
  for (int j = 0; j < hidden; ++j) A[i * hidden + j] = f32_to_bf16(i + 1);
  std::vector<uint16_t> As(EM * hidden);
  EXPECT_EQ(vk_mxfp4_moe_sort(A.data(), sids.data(), As.data(), M, hidden,
                               top_k, EM),
            VK_OK);
  // r=0: flat=2, token=1 -> As[0..3] = A[4..7] = 2.0
  EXPECT_NEAR(bf16_to_f32(As[0]), 2.0f, 1e-5f);
  EXPECT_NEAR(bf16_to_f32(As[3]), 2.0f, 1e-5f);
  // r=1: flat=6 >= 6 (M*top_k) -> padding, all zero
  for (int j = 0; j < hidden; ++j)
  EXPECT_EQ(As[hidden + j], static_cast<uint16_t>(0));
  // r=2: flat=1, token=0 -> As[8..11] = A[0..3] = 1.0
  EXPECT_NEAR(bf16_to_f32(As[2 * hidden + 0]), 1.0f, 1e-5f);
  EXPECT_NEAR(bf16_to_f32(As[2 * hidden + 3]), 1.0f, 1e-5f);
}

TEST(CapiMoeAux, Mxfp4SortScales) {
  // sorted_ids[r] flat in [0, M*top_k); token = flat / top_k.
  constexpr int M = 2, n_groups = 3, top_k = 1, EM = 3;
  std::vector<int32_t> sids = {1, 1, 2};  // row2 = padding (>= 2)
  std::vector<uint8_t> scales = {100, 101, 102,   // token 0
                                 110, 111, 112};  // token 1
  std::vector<uint8_t> ss(EM * n_groups);
  EXPECT_EQ(vk_mxfp4_moe_sort_scales(scales.data(), sids.data(),
                                      ss.data(), M, n_groups, top_k, EM),
            VK_OK);
  // r=0,1: token=1 -> [110,111,112] both rows
  EXPECT_EQ(ss[0], 110); EXPECT_EQ(ss[1], 111); EXPECT_EQ(ss[2], 112);
  EXPECT_EQ(ss[3], 110); EXPECT_EQ(ss[4], 111); EXPECT_EQ(ss[5], 112);
  // r=2: flat=2 >= 2 -> padding, all zero
  for (int j = 0; j < n_groups; ++j) EXPECT_EQ(ss[2 * n_groups + j], 0);
}

TEST(CapiMoeAux, ScatterReduce) {
  // Mirror test_moe_aux::ScatterReduce: out[token] += partial[r] * topk_w[r],
  // token = sorted_ids[r] / top_k, padding rows (flat >= M*top_k) skipped.
  constexpr int M = 3, width = 5, top_k = 2, EM = 4;
  std::vector<int32_t> sids = {0, 1, 2, 6};  // r0,r1->tok0; r2->tok1; r3->pad
  std::vector<float> partial = {1,2,3,4,5,   10,20,30,40,50,
                                100,200,300,400,500,  9,9,9,9,9};
  std::vector<float> w = {0.5f, 0.5f, 1.0f, 0.0f};
  std::vector<float> out(M * width, 0.0f);
  EXPECT_EQ(vk_mxfp4_moe_scatter_reduce(partial.data(), w.data(),
                                         sids.data(), out.data(), M, width,
                                         top_k, EM),
            VK_OK);
  // token0 = 0.5*[1..5] + 0.5*[10..50] = [5.5, 11, 16.5, 22, 27.5]
  EXPECT_NEAR(out[0], 5.5f, 1e-5f);
  EXPECT_NEAR(out[4], 27.5f, 1e-5f);
  // token1 = 1.0*[100..500]
  EXPECT_NEAR(out[5], 100.0f, 1e-5f);
  EXPECT_NEAR(out[9], 500.0f, 1e-5f);
  // token2 untouched (padding row skipped) -> 0
  EXPECT_EQ(out[10], 0.0f);
}

TEST(CapiMoeAux, ScatterReduceQ) {
  // Mirror test_moe_aux::ScatterReduceQ: quantize a fp4-representable
  // partial, then scatter-reduce both the quantized (C ABI q path) and
  // dequantized (float path) forms — they must agree bit-for-bit.
  constexpr int M = 2, width = 32, top_k = 1, EM = 2, gs = 32;
  std::vector<int32_t> ids = {0, 1};
  const float repr[] = {0.f, 0.25f, -0.25f, 1.f, -1.f, 1.5f,
                        2.f, 3.f};
  std::vector<float> partial_f(EM * width);
  for (size_t i = 0; i < partial_f.size(); ++i)
    partial_f[i] = repr[i % (sizeof(repr) / sizeof(repr[0]))];
  std::vector<float> w = {2.0f, 0.5f};

  // Quantize the partial (one group per row).
  std::vector<uint8_t> pq(EM * width / 2), ps(EM);
  std::vector<uint16_t> bf16(EM * width);
  for (size_t i = 0; i < bf16.size(); ++i) bf16[i] = f32_to_bf16(partial_f[i]);
  EXPECT_EQ(vk_mxfp4_moe_quant(bf16.data(), pq.data(), ps.data(), EM,
                                width, gs),
            VK_OK);

  std::vector<float> out_q(M * width, 0.0f);
  EXPECT_EQ(vk_mxfp4_moe_scatter_reduce_q(pq.data(), ps.data(), w.data(),
                                           ids.data(), out_q.data(), M,
                                           width, top_k, EM, gs),
            VK_OK);

  // Dequant the quantized partial back to float and scatter-reduce it.
  std::vector<float> out_f(M * width, 0.0f);
  std::vector<uint16_t> dq(EM * width);
  EXPECT_EQ(vk_fp4_to_bf16_dequant(pq.data(), EM * width / 2, dq.data(),
                                    EM * width, 1.0f),
            VK_OK);  // raw dequant (scale 1); re-scale per row below
  // Re-implement dequant_act with the per-row scale (single group/row).
  std::vector<float> dqf(EM * width);
  for (int r = 0; r < EM; ++r) {
    uint32_t sb = static_cast<uint32_t>(ps[r]) << 23;
    float sc;
    std::memcpy(&sc, &sb, sizeof(float));
    for (int i = 0; i < width; ++i)
      dqf[r * width + i] = bf16_to_f32(dq[r * width + i]) * sc;
  }
  // fp4 values are exactly representable so the two paths agree.
  EXPECT_EQ(vk_mxfp4_moe_scatter_reduce(dqf.data(), w.data(), ids.data(),
                                         out_f.data(), M, width, top_k, EM),
            VK_OK);
  for (size_t i = 0; i < out_q.size(); ++i) EXPECT_EQ(out_q[i], out_f[i]);
}

TEST(CapiMoeAux, ScatterReduceQNanInf) {
  // fp4 nibble codes 6=+inf, 7=NaN, 14=-inf, 15=NaN propagate untouched.
  constexpr int M = 1, width = 4, top_k = 1, EM = 1, gs = 4;
  std::vector<uint8_t> pq = {0x76, 0xFE};
  std::vector<uint8_t> ps = {127};  // scale 2^0 = 1.0
  std::vector<int32_t> ids = {0};
  std::vector<float> w = {1.0f};
  std::vector<float> out(M * width, 0.0f);
  EXPECT_EQ(vk_mxfp4_moe_scatter_reduce_q(pq.data(), ps.data(), w.data(),
                                           ids.data(), out.data(), M, width,
                                           top_k, EM, gs),
            VK_OK);
  EXPECT_EQ(out[0], std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(out[1]));
  EXPECT_EQ(out[2], -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(out[3]));
}

// --- fused MXFP4 MoE (moe_fused.hpp) + moe_align_block_size ---------------

TEST(CapiMoeFused, AlignBlockSize) {
  // M=4, top_k=2, block_size=16, E=4
  constexpr int M = 4, top_k = 2, BLOCK = 16, E = 4;
  int32_t max_EM = static_cast<int32_t>(
      vk_moe_align_block_size_max_em(M, top_k, BLOCK, E));
  ASSERT_TRUE(max_EM >= M * top_k);
  std::vector<int32_t> topk_ids = {0, 1, 1, 2, 2, 3, 3, 0};
  std::vector<int32_t> sids(max_EM), eids(max_EM / BLOCK), EM_out(1);
  EXPECT_EQ(vk_moe_align_block_size(topk_ids.data(), M, top_k, BLOCK, E,
                                     sids.data(), eids.data(), EM_out.data()),
            VK_OK);
  // EM should be M*top_k = 8 (no padding needed if < BLOCK)
  // Actually EM is padded to multiple of block_size
  EXPECT_GE(EM_out[0], M * top_k);
  EXPECT_EQ(EM_out[0] % BLOCK, 0);
}

TEST(CapiMoeFused, AlignBlockSizeMaxEm) {
  // Just verify it returns a positive value
  size_t em = vk_moe_align_block_size_max_em(8, 2, 16, 4);
  EXPECT_GT(em, 0u);
}

TEST(CapiMoeFused, AlignBlockSizeNullThrows) {
  int32_t EM_out;
  EXPECT_NE(vk_moe_align_block_size(nullptr, 1, 1, 16, 4, nullptr, nullptr,
                                     &EM_out),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

TEST(CapiMoeFused, TinySanity) {
  // All weights = 1.0, M=16, hidden=128, ispp=64, top_k=1, E=1
  // gate = up = 128, act = silu(128)*128 ≈ 16384,
  // out = act * 64 ≈ 1,048,576
  constexpr int M = 16, hidden = 128, ispp = 64, top_k = 1, E = 1;
  constexpr int BLOCK = 16, gs = 32;

  std::vector<uint16_t> A(M * hidden, f32_to_bf16(1.0f));
  // w13: [E, 2*ispp, hidden/2] = [1, 128, 64] uint8 (all 1.0 -> nibble 2)
  std::vector<uint8_t> w13(E * 2 * ispp * (hidden / 2), 0x22);
  // w13_scale: [E, 2*ispp, hidden/32] = [1, 128, 4] uint8 (all 0x7F = 1.0)
  std::vector<uint8_t> w13s(E * 2 * ispp * (hidden / 32), 0x7F);
  // w2: [E, hidden, ispp/2] = [1, 128, 32] uint8 (all 1.0 -> nibble 2)
  std::vector<uint8_t> w2(E * hidden * (ispp / 2), 0x22);
  // w2_scale: [E, hidden, ispp/32] = [1, 128, 4] uint8 (all 0x7F)
  std::vector<uint8_t> w2s(E * hidden * (ispp / 32), 0x7F);

  // routing: all tokens go to expert 0
  std::vector<int32_t> topk_ids(M * top_k, 0);
  std::vector<float> topk_w(M * top_k, 1.0f);

  int32_t max_EM = static_cast<int32_t>(
      vk_moe_align_block_size_max_em(M, top_k, BLOCK, E));
  std::vector<int32_t> sids(max_EM), eids(max_EM / BLOCK), EM_out(1);
  vk_moe_align_block_size(topk_ids.data(), M, top_k, BLOCK, E, sids.data(),
                          eids.data(), EM_out.data());
  int EM = EM_out[0];

  std::vector<uint16_t> act_scratch(EM * ispp, 0);
  std::vector<float> out(M * hidden, 0.0f);

  EXPECT_EQ(vk_fused_moe_mxfp4(A.data(), w13.data(), w13s.data(), w2.data(),
                                w2s.data(), sids.data(), topk_w.data(),
                                eids.data(), act_scratch.data(), out.data(),
                                M, hidden, ispp, top_k, EM, gs, 0.0f,
                                0 /*SwiGLU*/, 0.0f, 0.0f, nullptr, nullptr),
            VK_OK);

  // All outputs should be the same (all weights=1, all inputs=1)
  // and should be large positive (silu(128)*128*64 ≈ 1M)
  EXPECT_FALSE(std::isnan(out[0]));
  EXPECT_GT(out[0], 100000.0f);
  for (int i = 1; i < M * hidden; ++i) {
    EXPECT_FALSE(std::isnan(out[i]));
    EXPECT_NEAR(out[i], out[0], std::fabs(out[0]) * 0.01f);
  }
}

TEST(CapiMoeFused, SituActivation) {
  // Mirror test_moe_fused::SiTU: A=1, w=1 (E2M1 byte 0x22), scale=127,
  // activation=kSiTU(1), beta=4, linear_beta=25. gate=up=128, unclamped.
  constexpr int M = 16, hidden = 128, ispp = 64, top_k = 1, E = 1;
  constexpr int BLOCK = 16, gs = 32, EM = M;

  std::vector<uint16_t> A(M * hidden, f32_to_bf16(1.0f));
  uint8_t one_byte = static_cast<uint8_t>(
      float_to_e2m1_nibble(1.0f) | (float_to_e2m1_nibble(1.0f) << 4));
  std::vector<uint8_t> w13(E * 2 * ispp * (hidden / 2), one_byte);
  std::vector<uint8_t> w13s(E * 2 * ispp * (hidden / gs), 127);
  std::vector<uint8_t> w2(E * hidden * (ispp / 2), one_byte);
  std::vector<uint8_t> w2s(E * hidden * (ispp / gs), 127);

  std::vector<int32_t> sids(EM), eids(EM / BLOCK, 0);
  for (int i = 0; i < EM; ++i) sids[i] = i;
  std::vector<float> topk_w(EM, 1.0f);

  std::vector<uint16_t> act_scratch(EM * ispp, 0);
  std::vector<float> out(M * hidden, 0.0f);

  EXPECT_EQ(vk_fused_moe_mxfp4(A.data(), w13.data(), w13s.data(), w2.data(),
                                w2s.data(), sids.data(), topk_w.data(),
                                eids.data(), act_scratch.data(), out.data(),
                                M, hidden, ispp, top_k, EM, gs,
                                1.0f /* clamp ignored */, 1 /*kSiTU*/,
                                4.0f /*beta*/, 25.0f /*linear_beta*/,
                                nullptr, nullptr),
            VK_OK);

  // situ_and_mul reference (vLLM): gate=up=128, unclamped.
  const float gate = 128.0f, up = 128.0f;
  const float sig = 1.0f / (1.0f + std::exp(-gate));
  const float expected_act = (4.0f * std::tanh(gate / 4.0f) * sig) *
                             (25.0f * std::tanh(up / 25.0f));
  const float expected_out = expected_act * static_cast<float>(ispp);

  float max_rel = 0.0f;
  for (int i = 0; i < M * hidden; ++i) {
    EXPECT_FALSE(std::isnan(out[i]));
    float err = std::fabs(out[i] - expected_out);
    if (expected_out != 0.0f) max_rel = std::max(max_rel, err / expected_out);
  }
  EXPECT_LT(max_rel, 1e-2f);
}

TEST(CapiMoeFused, NullBuffersThrow) {
  // The vk_fused_moe_mxfp4 null-guard (M>0, EM>0) must raise
  // VK_ERROR_INVALID_ARGUMENT when any primary buffer is null.
  constexpr int M = 1, hidden = 8, ispp = 4, top_k = 1, EM = 1, gs = 8;
  std::vector<uint8_t> w(1), ws(1);
  std::vector<int32_t> sids(1, 0), eids(1, 0);
  std::vector<float> tw(1, 1.0f);
  std::vector<uint16_t> act(1, 0);
  std::vector<float> out(M * hidden, 0.0f);
  // A is null -> guard fires.
  EXPECT_NE(vk_fused_moe_mxfp4(nullptr, w.data(), ws.data(), w.data(),
                                ws.data(), sids.data(), tw.data(),
                                eids.data(), act.data(), out.data(),
                                M, hidden, ispp, top_k, EM, gs, 0.0f, 0,
                                0.0f, 0.0f, nullptr, nullptr),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
  // w13 null -> guard fires.
  std::vector<uint16_t> A(M * hidden, f32_to_bf16(1.0f));
  EXPECT_NE(vk_fused_moe_mxfp4(A.data(), nullptr, ws.data(), w.data(),
                                ws.data(), sids.data(), tw.data(),
                                eids.data(), act.data(), out.data(),
                                M, hidden, ispp, top_k, EM, gs, 0.0f, 0,
                                0.0f, 0.0f, nullptr, nullptr),
            VK_OK);
  EXPECT_EQ(vk_last_error_code(), VK_ERROR_INVALID_ARGUMENT);
}

TEST(CapiMoeFused, EmptyIsNoop) {
  // M=0 or EM=0 skips the null-guard (a no-op), returns VK_OK even with
  // null buffers — matching the BLAS-style contract.
  std::vector<float> out(1, 99.0f);
  EXPECT_EQ(vk_fused_moe_mxfp4(nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, nullptr, out.data(),
                                0, 8, 4, 1, 0, 8, 0.0f, 0, 0.0f, 0.0f,
                                nullptr, nullptr),
            VK_OK);
  EXPECT_EQ(out[0], 99.0f);  // untouched
}
