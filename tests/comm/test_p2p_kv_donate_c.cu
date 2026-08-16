// tests/comm/test_p2p_kv_donate_c.cu
//
// Runtime tests for the `extern "C"` P2P KV donate ABI (issue #36). These
// are CUDA-only and run on a single GPU using same-device pointers as a
// stand-in for cross-GPU peer memory — the same stand-in the p2p_gather_c
// and p2p_kv_restore_c tests use, and sufficient to exercise the
// validators, the kernel launch, the adaptive dispatch, and the
// status-code return path.
//
// Unlike the restore (which scatters and therefore requires UNIQUE
// destination slots), the donate GATHERS: a repeated SOURCE slot simply
// re-reads the same memory, so duplicate slots are accepted (issue #36).
//
// Peer destination layout: [layers, page_size, 2, num_kv_heads, head_dim]
// row-major; within a layer, token t's K is at offset t*token_stride and V
// at t*token_stride + slot_bytes. `dst_page_offsets[p]` is a byte offset
// added to `peer_dst_ptrs[p]`; the prepared plan replaces it with a single
// `destination_layer_offset_bytes` added to every page base.
#include "vkernels/comm/p2p_kv_donate_c.h"

#include "minitest.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

namespace {

// Fill a host vector with a deterministic byte pattern.
std::vector<uint8_t> patterned(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + (i % 251));
  return v;
}

// Per-slot bytes (one K or one V slot): num_kv_heads * head_dim * elem.
inline size_t slot_bytes(size_t heads, size_t head_dim, size_t elem) {
  return heads * head_dim * elem;
}

// Per-token destination stride: [K, V] = 2 * slot_bytes.
inline size_t token_stride(size_t heads, size_t head_dim, size_t elem) {
  return 2 * slot_bytes(heads, head_dim, elem);
}

// Byte size of one peer page: [page_size, 2, num_kv_heads, head_dim].
inline size_t page_bytes(size_t page_size, size_t heads, size_t head_dim,
                         size_t elem) {
  return page_size * token_stride(heads, head_dim, elem);
}

// Compare two device buffers byte-for-byte.
bool device_equal(const uint8_t* d_a, const uint8_t* d_b, size_t n) {
  std::vector<uint8_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_a, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, n, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i)
    if (ha[i] != hb[i]) return false;
  return true;
}

// Device-allocate and H2D-copy a host vector.
uint8_t* to_device(const std::vector<uint8_t>& h) {
  uint8_t* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, h.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice) ==
              cudaSuccess);
  return d;
}
int* ints_to_device(const int* h, size_t n) {
  int* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, n * sizeof(int)) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice) ==
              cudaSuccess);
  return d;
}
int64_t* int64s_to_device(const int64_t* h, size_t n) {
  int64_t* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, n * sizeof(int64_t)) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d, h, n * sizeof(int64_t), cudaMemcpyHostToDevice) ==
              cudaSuccess);
  return d;
}

}  // namespace

// ---------------------------------------------------------------------------
// Fused equals two-stage (byte-exact)
// ---------------------------------------------------------------------------
TEST(P2pKvDonateCAbi, FusedEqualsTwoStageSinglePage) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16;

  auto h_page = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x42);
  auto h_k = patterned(kSlots * kSlotBytes, 0x11);
  auto h_v = patterned(kSlots * kSlotBytes, 0x22);
  const int h_slots[4] = {3, 7, 1, 12};

  uint8_t *d_page_f = nullptr, *d_page_t = nullptr;
  uint8_t *d_k = nullptr, *d_v = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_page_f, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_page_t, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_k, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_v, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, 4 * sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(d_k, h_k.data(), kSlots * kSlotBytes,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_v, h_v.data(), kSlots * kSlotBytes,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, 4 * sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  const void* ptrs_f[1] = {d_page_f};
  const void* ptrs_t[1] = {d_page_t};
  const size_t offs[1] = {0};

  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs_f, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  st = vkernels_p2p_kv_donate_layer_twostage(
      d_k, d_v, h_slots, ptrs_t, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_page_f, d_page_t, h_page.size()));

  cudaFree(d_page_f); cudaFree(d_page_t); cudaFree(d_k); cudaFree(d_v);
  cudaFree(d_slots);
}

TEST(P2pKvDonateCAbi, FusedEqualsTwoStageMultiPage) {
  constexpr size_t kPageSize = 2, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 128
  constexpr size_t kNumPages = 3, kSlots = 16;

  auto h0 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);
  auto h1 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x80);
  auto h2 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0xF0);
  auto h_k = patterned(kSlots * kSlotBytes, 0x33);
  auto h_v = patterned(kSlots * kSlotBytes, 0x55);
  const int h_slots[6] = {0, 5, 9, 2, 14, 7};

  uint8_t *df0 = nullptr, *df1 = nullptr, *df2 = nullptr;
  uint8_t *dt0 = nullptr, *dt1 = nullptr, *dt2 = nullptr;
  uint8_t *d_k = nullptr, *d_v = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&df0, h0.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&df1, h1.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&df2, h2.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dt0, h0.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dt1, h1.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dt2, h2.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_k, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_v, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, 6 * sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(d_k, h_k.data(), kSlots * kSlotBytes,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_v, h_v.data(), kSlots * kSlotBytes,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, 6 * sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  const void* ptrs_f[3] = {df0, df1, df2};
  const void* ptrs_t[3] = {dt0, dt1, dt2};
  const size_t offs[3] = {0, 0, 0};

  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs_f, offs, kNumPages, kPageSize, kHeads, kHeadDim,
      kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  st = vkernels_p2p_kv_donate_layer_twostage(
      d_k, d_v, h_slots, ptrs_t, offs, kNumPages, kPageSize, kHeads, kHeadDim,
      kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(df0, dt0, h0.size()));
  ASSERT_TRUE(device_equal(df1, dt1, h1.size()));
  ASSERT_TRUE(device_equal(df2, dt2, h2.size()));

  cudaFree(df0); cudaFree(df1); cudaFree(df2);
  cudaFree(dt0); cudaFree(dt1); cudaFree(dt2);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_slots);
}

// Scalar tail path must work on unaligned slot_bytes.
TEST(P2pKvDonateCAbi, UnalignedSlotBytesFusedEqualsTwoStage) {
  constexpr size_t kPageSize = 1, kHeads = 3, kHeadDim = 5, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 30 bytes
  constexpr size_t kSlots = 4;

  auto h_page = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x37);
  auto h_k = patterned(kSlots * kSlotBytes, 0x12);
  auto h_v = patterned(kSlots * kSlotBytes, 0x34);
  const int h_slots[1] = {2};

  uint8_t *d_page_f = nullptr, *d_page_t = nullptr;
  uint8_t *d_k = nullptr, *d_v = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_page_f, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_page_t, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_k, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_v, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(d_k, h_k.data(), kSlots * kSlotBytes,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_v, h_v.data(), kSlots * kSlotBytes,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  const void* ptrs_f[1] = {d_page_f};
  const void* ptrs_t[1] = {d_page_t};
  const size_t offs[1] = {0};

  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs_f, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  st = vkernels_p2p_kv_donate_layer_twostage(
      d_k, d_v, h_slots, ptrs_t, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_page_f, d_page_t, h_page.size()));

  cudaFree(d_page_f); cudaFree(d_page_t); cudaFree(d_k); cudaFree(d_v);
  cudaFree(d_slots);
}

// Non-zero dst_page_offset selects which token within the peer page is written.
TEST(P2pKvDonateCAbi, PageOffsetIsHonoured) {
  constexpr size_t kPageSize = 1, kHeads = 2, kHeadDim = 4, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16
  constexpr size_t kTokenStr = 2 * kSlotBytes;               // 32
  constexpr size_t kSlots = 4;

  // Peer buffer holds two tokens back-to-back; offset selects token 1.
  auto h_buf = patterned(page_bytes(2, kHeads, kHeadDim, kElem), 0x21);
  auto h_k = patterned(kSlots * kSlotBytes, 0x40);
  auto h_v = patterned(kSlots * kSlotBytes, 0x60);
  const int h_slots[1] = {1};

  uint8_t* d_buf = to_device(h_buf);
  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);

  const void* ptrs[1] = {d_buf};
  const size_t offs[1] = {kTokenStr};  // write into token 1

  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // Token 1 holds slot 1's K and V; token 0 stays at the 0x21 sentinel.
  std::vector<uint8_t> h_out(h_buf.size());
  cudaMemcpy(h_out.data(), d_buf, h_buf.size(), cudaMemcpyDeviceToHost);
  const uint8_t* got_k = h_out.data() + kTokenStr;
  const uint8_t* got_v = got_k + kSlotBytes;
  for (size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(got_k[i], h_k[1 * kSlotBytes + i]);
    ASSERT_EQ(got_v[i], h_v[1 * kSlotBytes + i]);
    ASSERT_EQ(h_out[i], 0x21);                   // token 0 K untouched
    ASSERT_EQ(h_out[kSlotBytes + i], 0x21);      // token 0 V untouched
  }

  cudaFree(d_buf); cudaFree(d_k); cudaFree(d_v);
}

// ---------------------------------------------------------------------------
// Gather semantics: repeated SOURCE slots are accepted (issue #36). Unlike
// the restore (which scatters and requires unique destinations), the donate
// gathers, so a repeated slot re-reads the same memory and must succeed.
// ---------------------------------------------------------------------------
TEST(P2pKvDonateCAbi, RepeatedSlotsAreAccepted) {
  constexpr size_t kPageSize = 4, kHeads = 1, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16
  constexpr size_t kTokenStr = 2 * kSlotBytes;
  constexpr size_t kSlots = 4;

  auto h_page = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x99);
  auto h_k = patterned(kSlots * kSlotBytes, 0x77);
  auto h_v = patterned(kSlots * kSlotBytes, 0xBB);
  const int h_slots[4] = {1, 1, 3, 3};  // repeated source slots (gather)

  uint8_t* d_page = to_device(h_page);
  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);

  const void* ptrs[1] = {d_page};
  const size_t offs[1] = {0};
  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // Tokens 0 and 1 both read slot 1; tokens 2 and 3 both read slot 3.
  std::vector<uint8_t> h_out(h_page.size());
  cudaMemcpy(h_out.data(), d_page, h_page.size(), cudaMemcpyDeviceToHost);
  for (size_t t = 0; t < kPageSize; ++t) {
    int slot = h_slots[t];
    const uint8_t* got_k = h_out.data() + t * kTokenStr;
    const uint8_t* got_v = got_k + kSlotBytes;
    for (size_t i = 0; i < kSlotBytes; ++i) {
      ASSERT_EQ(got_k[i], h_k[static_cast<size_t>(slot) * kSlotBytes + i]);
      ASSERT_EQ(got_v[i], h_v[static_cast<size_t>(slot) * kSlotBytes + i]);
    }
  }

  cudaFree(d_page); cudaFree(d_k); cudaFree(d_v);
}

// ---------------------------------------------------------------------------
// C ABI: error-code returns (contract checks fire before any copy)
// ---------------------------------------------------------------------------
TEST(P2pKvDonateCAbi, NullKSrcReturnsInvalidArgument) {
  const size_t offs[1] = {0};
  int slot = 0;
  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      nullptr, (void*)0x1000, &slot, (const void**)0x2000, offs,
      1, 16, 8, 128, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(P2pKvDonateCAbi, NullVSrcReturnsInvalidArgument) {
  const size_t offs[1] = {0};
  int slot = 0;
  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      (void*)0x1000, nullptr, &slot, (const void**)0x2000, offs,
      1, 16, 8, 128, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(P2pKvDonateCAbi, NullPeerPtrReturnsInvalidArgument) {
  const size_t offs[1] = {0};
  int slot = 0;
  const void* ptrs[1] = {nullptr};
  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      (void*)0x1000, (void*)0x2000, &slot, ptrs, offs,
      1, 16, 8, 128, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(P2pKvDonateCAbi, NonBF16ReturnsInvalidArgument) {
  const size_t offs[1] = {0};
  int slot = 0;
  const void* ptrs[1] = {(const void*)0x1000};
  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      (void*)0x2000, (void*)0x3000, &slot, ptrs, offs,
      1, 16, 8, 128, 4, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

// Zero pages is a valid no-op: nothing is written, no allocation needed.
TEST(P2pKvDonateCAbi, ZeroPagesIsNoOp) {
  constexpr size_t kHeads = 2, kHeadDim = 8, kElem = 2;
  auto h_page = patterned(page_bytes(4, kHeads, kHeadDim, kElem), 0xAB);
  uint8_t* d_page = to_device(h_page);
  uint8_t* d_k = to_device(patterned(64, 0x11));
  uint8_t* d_v = to_device(patterned(64, 0x22));
  int slot = 0;

  const void* ptrs[1] = {d_page};
  const size_t offs[1] = {0};
  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      d_k, d_v, &slot, ptrs, offs, 0, 4, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // Page buffer must be untouched.
  ASSERT_TRUE(device_equal(d_page, h_page.data(), h_page.size()));

  cudaFree(d_page); cudaFree(d_k); cudaFree(d_v);
}

// ---------------------------------------------------------------------------
// Async: two concurrent streams on the fused path
// ---------------------------------------------------------------------------
TEST(P2pKvDonateCAbi, ConcurrentStreams) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 8;

  // Two streams donate the SAME source into separate peer pages (same slot
  // map). The results must be byte-identical, verifying concurrent
  // execution on two streams is safe (no cross-stream corruption).
  auto h_p0 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);
  auto h_p1 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);
  auto h_k = patterned(kSlots * kSlotBytes, 0x20);
  auto h_v = patterned(kSlots * kSlotBytes, 0x40);
  const int h_slots[4] = {0, 4, 1, 5};

  uint8_t *dp0 = to_device(h_p0), *dp1 = to_device(h_p1);
  uint8_t *d_k0 = to_device(h_k), *d_k1 = to_device(h_k);
  uint8_t *d_v0 = to_device(h_v), *d_v1 = to_device(h_v);

  cudaStream_t s0, s1;
  ASSERT_TRUE(cudaStreamCreate(&s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamCreate(&s1) == cudaSuccess);

  const void* ptrs0[1] = {dp0};
  const void* ptrs1[1] = {dp1};
  const size_t offs[1] = {0};

  vkernels_status_t st = vkernels_p2p_kv_donate_layer(
      d_k0, d_v0, h_slots, ptrs0, offs, 1, kPageSize, kHeads, kHeadDim, kElem, s0);
  ASSERT_EQ(st, VKERNELS_OK);
  st = vkernels_p2p_kv_donate_layer(
      d_k1, d_v1, h_slots, ptrs1, offs, 1, kPageSize, kHeads, kHeadDim, kElem, s1);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_TRUE(cudaStreamSynchronize(s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamSynchronize(s1) == cudaSuccess);

  ASSERT_TRUE(device_equal(dp0, dp1, h_p0.size()));

  cudaStreamDestroy(s0); cudaStreamDestroy(s1);
  cudaFree(dp0); cudaFree(dp1); cudaFree(d_k0); cudaFree(d_v0);
  cudaFree(d_k1); cudaFree(d_v1);
}

// ---------------------------------------------------------------------------
// Prepared plan (issue #36)
// ---------------------------------------------------------------------------
//
// A plan validates the slot map and peer geometry ONCE at create and
// uploads the page descriptors (and, for the host-input variant, the slot
// map) to a persistent per-device buffer; execute_offset() only launches
// ONE page-by-token-group kernel that adds `destination_layer_offset_bytes`
// to every peer page base before writing. The (k_src, v_src) source is
// supplied at every execute_offset(), so one plan reads a distinct K/V
// layer buffer per layer (the KVAAS donate pattern) and writes into the
// peer pages. Reuse one plan across all model layers (40 for Qwen3-14B)
// with no per-layer allocation, H2D copy, or local packed-KV scratch.
//
// The one-shot C ABI takes a HOST slot_ids pointer (it does its own H2D
// copy); the plan's host-input create also takes a host pointer.

TEST(P2pKvDonateCAbi, PlanFusedEqualsOneShot) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16;

  auto h_k = patterned(kSlots * kSlotBytes, 0x11);
  auto h_v = patterned(kSlots * kSlotBytes, 0x22);
  auto h_p = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x44);
  const int h_slots[4] = {3, 7, 1, 12};

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_p_plan = to_device(h_p);
  uint8_t* d_p_shot = to_device(h_p);

  const void* ptrs_plan[1] = {d_p_plan};
  const void* ptrs_shot[1] = {d_p_shot};
  const size_t offs[1] = {0};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs_plan, 1, kPageSize, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan, d_k, d_v, 0, 0) == VKERNELS_OK);
  ASSERT_TRUE(vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs_shot, offs, 1, kPageSize, kHeads, kHeadDim,
      kElem, 0) == VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_p_plan, d_p_shot, h_p.size()));

  vkernels_p2p_kv_donate_plan_destroy(plan);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_p_plan); cudaFree(d_p_shot);
}

// The plan's scalar destination_layer_offset_bytes must equal the one-shot's
// per-page dst_page_offsets when every page uses the same offset.
TEST(P2pKvDonateCAbi, PlanOffsetMatchesPerPageOffset) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kTokenStr = 2 * kSlotBytes;               // 64
  constexpr size_t kSlots = 8;

  auto h_k = patterned(kSlots * kSlotBytes, 0x11);
  auto h_v = patterned(kSlots * kSlotBytes, 0x22);
  // Two-token buffer; the plan writes token 1 via destination_layer_offset.
  auto h_p_plan = patterned(page_bytes(2, kHeads, kHeadDim, kElem), 0x44);
  auto h_p_shot = patterned(page_bytes(2, kHeads, kHeadDim, kElem), 0x44);
  const int h_slots[2] = {5, 2};

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_p_plan = to_device(h_p_plan);
  uint8_t* d_p_shot = to_device(h_p_shot);

  const void* ptrs_plan[1] = {d_p_plan};
  const void* ptrs_shot[1] = {d_p_shot};
  const size_t off_one[1] = {kTokenStr};  // both target token 1

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs_plan, 1, kPageSize, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan, d_k, d_v, kTokenStr, 0) == VKERNELS_OK);
  ASSERT_TRUE(vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs_shot, off_one, 1, kPageSize, kHeads, kHeadDim,
      kElem, 0) == VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_p_plan, d_p_shot, h_p_plan.size()));

  vkernels_p2p_kv_donate_plan_destroy(plan);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_p_plan); cudaFree(d_p_shot);
}

// One plan on two streams (the KVAAS "one run list, many layers" reuse).
// Both executes donate the SAME source into the SAME peer page, so the
// final state is deterministic regardless of stream ordering — this tests
// concurrent plan sharing (no deadlock, no metadata mutation).
TEST(P2pKvDonateCAbi, PlanTwoStreams) {
  constexpr size_t kPageSize = 2, kHeads = 1, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 8;

  auto h_k = patterned(kSlots * kSlotBytes, 0x33);
  auto h_v = patterned(kSlots * kSlotBytes, 0x55);
  auto h_pa = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);
  auto h_pb = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_pa = to_device(h_pa);
  uint8_t* d_pb = to_device(h_pb);

  cudaStream_t s0, s1;
  ASSERT_TRUE(cudaStreamCreate(&s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamCreate(&s1) == cudaSuccess);

  // Two plans, each bound to its own peer page, sharing the same slot map
  // and source. Submitting on two streams must be safe and the two output
  // pages must agree (both donate the same source).
  const int h_slots[2] = {1, 6};
  const void* ptrs0[1] = {d_pa};
  const void* ptrs1[1] = {d_pb};
  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan0 = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs0, 1, kPageSize, &st);
  ASSERT_TRUE(plan0 != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);
  vkernels_p2p_kv_donate_plan_t* plan1 = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs1, 1, kPageSize, &st);
  ASSERT_TRUE(plan1 != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan0, d_k, d_v, 0, s0) == VKERNELS_OK);
  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan1, d_k, d_v, 0, s1) == VKERNELS_OK);
  ASSERT_TRUE(cudaStreamSynchronize(s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamSynchronize(s1) == cudaSuccess);

  ASSERT_TRUE(device_equal(d_pa, d_pb, h_pa.size()));

  vkernels_p2p_kv_donate_plan_destroy(plan0);
  vkernels_p2p_kv_donate_plan_destroy(plan1);
  cudaStreamDestroy(s0); cudaStreamDestroy(s1);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_pa); cudaFree(d_pb);
}

// The KVAAS donate pattern (issue #36): one run list reused across all model
// layers, each layer reading its OWN (k_src, v_src) source pair. The source
// is no longer bound at create — execute_offset() takes it per call — so one
// plan reads distinct per-layer source buffers and writes into the shared
// peer page buffer at offset `l * layer_bytes`, byte-exact against the
// one-shot oracle run per layer.
TEST(P2pKvDonateCAbi, PlanDistinctSourcesAcrossLayers) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 64
  constexpr size_t kTokenStr = 2 * kSlotBytes;               // 128
  constexpr size_t kLayerBytes = kPageSize * kTokenStr;      // 256
  constexpr size_t kSlots = 8;
  constexpr int kLayers = 40;

  // One peer page (per model the plan binds num_pages pages; here 1) holding
  // every layer back-to-back: [layers, page_size, 2, heads, head_dim].
  auto h_peer = patterned(page_bytes(kPageSize * kLayers, kHeads, kHeadDim, kElem),
                          0x09);
  const int h_slots[4] = {2, 0, 3, 1};

  uint8_t* d_peer = to_device(h_peer);
  const void* ptrs[1] = {d_peer};

  // One distinct (k_src, v_src) source pair per layer.
  std::vector<uint8_t*> dk(kLayers), dv(kLayers);
  for (int l = 0; l < kLayers; ++l) {
    dk[l] = to_device(patterned(kSlots * kSlotBytes,
                                static_cast<uint8_t>(0x10 + l)));
    dv[l] = to_device(patterned(kSlots * kSlotBytes,
                                static_cast<uint8_t>(0x80 + l)));
  }

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs, 1, kPageSize, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);

  // One prepare, forty executes — each with its own source, at its own layer
  // offset — into the shared peer buffer.
  for (int l = 0; l < kLayers; ++l)
    ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
        plan, dk[l], dv[l], static_cast<size_t>(l) * kLayerBytes, 0) ==
        VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // Per-layer one-shot oracle: donate the SAME source into a scratch peer
  // page at the SAME per-page offset, then compare each layer's region.
  uint8_t* d_peer_o = to_device(h_peer);
  const void* ptrs_o[1] = {d_peer_o};
  const size_t one_offs[1] = {0};
  for (int l = 0; l < kLayers; ++l) {
    // Reset the oracle page, then donate source l at offset l*layer_bytes.
    ASSERT_TRUE(cudaMemcpy(d_peer_o, h_peer.data(), h_peer.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    const size_t offs[1] = {static_cast<size_t>(l) * kLayerBytes};
    ASSERT_EQ(vkernels_p2p_kv_donate_layer(
                  dk[l], dv[l], h_slots, ptrs_o, offs, 1, kPageSize, kHeads,
                  kHeadDim, kElem, 0),
              VKERNELS_OK);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
    // Only the layer l region must match (the rest is the 0x09 sentinel on
    // both buffers).
    ASSERT_TRUE(device_equal(
        d_peer + static_cast<size_t>(l) * kLayerBytes,
        d_peer_o + static_cast<size_t>(l) * kLayerBytes, kLayerBytes));
  }

  vkernels_p2p_kv_donate_plan_destroy(plan);
  for (int l = 0; l < kLayers; ++l) { cudaFree(dk[l]); cudaFree(dv[l]); }
  cudaFree(d_peer); cudaFree(d_peer_o);
}

// Device-slot int32 variant: the plan borrows the caller's CUDA slot
// pointer and must match the one-shot (which uses a host copy of the same
// slots).
TEST(P2pKvDonateCAbi, PlanDeviceSlotsMatchesOneShot) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16;

  auto h_k = patterned(kSlots * kSlotBytes, 0x19);
  auto h_v = patterned(kSlots * kSlotBytes, 0x29);
  auto h_p_plan = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x44);
  auto h_p_shot = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x44);
  const int h_slots[4] = {2, 8, 4, 15};
  int* d_slots = ints_to_device(h_slots, 4);

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_p_plan = to_device(h_p_plan);
  uint8_t* d_p_shot = to_device(h_p_shot);

  const void* ptrs_plan[1] = {d_p_plan};
  const void* ptrs_shot[1] = {d_p_shot};
  const size_t offs[1] = {0};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan =
      vkernels_p2p_kv_donate_plan_create_device_slots(
          kSlots, kHeads, kHeadDim, kElem, d_slots, ptrs_plan, 1, kPageSize,
          &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan, d_k, d_v, 0, 0) == VKERNELS_OK);
  ASSERT_TRUE(vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs_shot, offs, 1, kPageSize, kHeads, kHeadDim,
      kElem, 0) == VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_p_plan, d_p_shot, h_p_plan.size()));

  // The device-slot plan borrows d_slots: keep it alive past destroy (we
  // synced above so destroy is safe).
  vkernels_p2p_kv_donate_plan_destroy(plan);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_p_plan); cudaFree(d_p_shot);
  cudaFree(d_slots);
}

// int64 device-slot variant (torch.int64 in KVAAS/SGLang): the plan converts
// int64 indices to an owned int32 buffer at create and must be byte-identical
// to the one-shot oracle (which uses int32 host slots).
TEST(P2pKvDonateCAbi, PlanDeviceSlotsInt64MatchesOneShot) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16;

  auto h_k = patterned(kSlots * kSlotBytes, 0x1A);
  auto h_v = patterned(kSlots * kSlotBytes, 0x2A);
  auto h_p_plan = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x44);
  auto h_p_shot = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x44);
  const int h_slots[4] = {2, 8, 4, 15};
  const int64_t h_slots_i64[4] = {2, 8, 4, 15};
  int64_t* d_slots_i64 = int64s_to_device(h_slots_i64, 4);

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_p_plan = to_device(h_p_plan);
  uint8_t* d_p_shot = to_device(h_p_shot);

  const void* ptrs_plan[1] = {d_p_plan};
  const void* ptrs_shot[1] = {d_p_shot};
  const size_t offs[1] = {0};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan =
      vkernels_p2p_kv_donate_plan_create_device_slots_int64(
          kSlots, kHeads, kHeadDim, kElem, d_slots_i64, ptrs_plan, 1, kPageSize,
          &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);
  // The int64 source buffer may be freed right after create: the plan owns
  // its converted int32 copy, and the create-time D2D conversion kernel is
  // serialized on the default stream ahead of the execute below.
  cudaFree(d_slots_i64);

  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan, d_k, d_v, 0, 0) == VKERNELS_OK);
  ASSERT_TRUE(vkernels_p2p_kv_donate_layer(
      d_k, d_v, h_slots, ptrs_shot, offs, 1, kPageSize, kHeads, kHeadDim,
      kElem, 0) == VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_p_plan, d_p_shot, h_p_plan.size()));

  vkernels_p2p_kv_donate_plan_destroy(plan);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_p_plan); cudaFree(d_p_shot);
}

// execute_via_scratch: gather to scratch then host-driven copy to peer.
// The fused direct-store result and the via-scratch result must agree.
TEST(P2pKvDonateCAbi, PlanExecuteViaScratchEqualsExecute) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 8;
  const size_t kPb = page_bytes(kPageSize, kHeads, kHeadDim, kElem);

  auto h_k = patterned(kSlots * kSlotBytes, 0x11);
  auto h_v = patterned(kSlots * kSlotBytes, 0x22);
  auto h_p_dir = patterned(kPb, 0x44);
  auto h_p_scr = patterned(kPb, 0x44);
  const int h_slots[2] = {1, 5};

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_p_dir = to_device(h_p_dir);
  uint8_t* d_p_scr = to_device(h_p_scr);
  uint8_t* d_scratch = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_scratch, kPb) == cudaSuccess);

  const void* ptrs_dir[1] = {d_p_dir};
  const void* ptrs_scr[1] = {d_p_scr};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan_dir = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs_dir, 1, kPageSize, &st);
  ASSERT_TRUE(plan_dir != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);
  vkernels_p2p_kv_donate_plan_t* plan_scr = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs_scr, 1, kPageSize, &st);
  ASSERT_TRUE(plan_scr != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan_dir, d_k, d_v, 0, 0) == VKERNELS_OK);
  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_via_scratch(
      plan_scr, d_k, d_v, d_scratch, 0, 0) == VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_p_dir, d_p_scr, kPb));

  vkernels_p2p_kv_donate_plan_destroy(plan_dir);
  vkernels_p2p_kv_donate_plan_destroy(plan_scr);
  cudaFree(d_scratch);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_p_dir); cudaFree(d_p_scr);
}

// Plan accessors report the right sizes.
TEST(P2pKvDonateCAbi, PlanAccessors) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kNumPages = 3, kSlots = 8;
  const size_t kExpected = kNumPages * page_bytes(kPageSize, kHeads, kHeadDim, kElem);

  auto h_p = patterned(kExpected, 0x01);
  uint8_t* d_p = to_device(h_p);
  const void* ptrs[3] = {d_p, d_p + page_bytes(kPageSize, kHeads, kHeadDim, kElem),
                         d_p + 2 * page_bytes(kPageSize, kHeads, kHeadDim, kElem)};
  const int h_slots[6] = {0, 1, 2, 3, 4, 5};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs, kNumPages, kPageSize, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_EQ(vkernels_p2p_kv_donate_plan_total_bytes(plan), kExpected);
  ASSERT_EQ(vkernels_p2p_kv_donate_plan_scratch_bytes(plan), kExpected);

  vkernels_p2p_kv_donate_plan_destroy(plan);
  cudaFree(d_p);
}

// ---------------------------------------------------------------------------
// Plan contract checks now return status codes at create time
// ---------------------------------------------------------------------------
TEST(P2pKvDonateCAbi, PlanRejectsNegativeSlot) {
  uint8_t* d_k = to_device(patterned(8, 0));
  uint8_t* d_v = to_device(patterned(8, 0));
  const int h_slots[2] = {1, -1};  // negative
  const void* ptrs[1] = {d_k};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      4, 2, 8, 2, h_slots, ptrs, 1, 2, &st);
  ASSERT_TRUE(plan == nullptr);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_k); cudaFree(d_v);
}

TEST(P2pKvDonateCAbi, PlanRejectsOutOfRangeSlot) {
  uint8_t* d_k = to_device(patterned(8, 0));
  uint8_t* d_v = to_device(patterned(8, 0));
  const int h_slots[2] = {1, 4};  // num_slots=4, slot 4 is out of range
  const void* ptrs[1] = {d_k};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      4, 2, 8, 2, h_slots, ptrs, 1, 2, &st);
  ASSERT_TRUE(plan == nullptr);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_k); cudaFree(d_v);
}

TEST(P2pKvDonateCAbi, PlanRejectsNonBF16) {
  const void* ptrs[1] = {(void*)0x1000};
  int slot = 0;
  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      4, 2, 8, 4, &slot, ptrs, 1, 1, &st);
  ASSERT_TRUE(plan == nullptr);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(P2pKvDonateCAbi, PlanRejectsNullPeerPtr) {
  const int h_slots[2] = {0, 1};
  const void* ptrs[1] = {nullptr};
  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      4, 2, 8, 2, h_slots, ptrs, 1, 1, &st);
  ASSERT_TRUE(plan == nullptr);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

// Uniqueness is NOT required for a gather: repeated slots are accepted by
// the host-input plan (issue #36).
TEST(P2pKvDonateCAbi, PlanAcceptsRepeatedSlots) {
  constexpr size_t kPageSize = 4, kHeads = 1, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16
  constexpr size_t kSlots = 4;

  auto h_k = patterned(kSlots * kSlotBytes, 0x50);
  auto h_v = patterned(kSlots * kSlotBytes, 0x70);
  auto h_p = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x01);
  const int h_slots[4] = {1, 1, 3, 3};  // repeated — allowed for a gather

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_p = to_device(h_p);
  const void* ptrs[1] = {d_p};

  vkernels_status_t st = VKERNELS_OK;
  vkernels_p2p_kv_donate_plan_t* plan = vkernels_p2p_kv_donate_plan_create(
      kSlots, kHeads, kHeadDim, kElem, h_slots, ptrs, 1, kPageSize, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(vkernels_p2p_kv_donate_plan_execute_offset(
      plan, d_k, d_v, 0, 0) == VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  vkernels_p2p_kv_donate_plan_destroy(plan);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_p);
}

// ---------------------------------------------------------------------------
// vkernels_kv_gather: gather indexed K/V into a contiguous scratch buffer.
// Must equal the gather half of the two-stage path (and the full donate
// when the peer pages are laid out contiguously at offset 0).
// ---------------------------------------------------------------------------
TEST(P2pKvDonateCAbi, GatherMatchesDonate) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 8;
  const size_t kPb = page_bytes(kPageSize, kHeads, kHeadDim, kElem);

  auto h_k = patterned(kSlots * kSlotBytes, 0x11);
  auto h_v = patterned(kSlots * kSlotBytes, 0x22);
  auto h_p_g = patterned(kPb, 0x00);
  auto h_p_d = patterned(kPb, 0x00);
  const int h_slots[4] = {0, 3, 6, 2};

  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  uint8_t* d_p_g = to_device(h_p_g);
  uint8_t* d_p_d = to_device(h_p_d);
  int* d_slots = ints_to_device(h_slots, 4);  // gather takes a device pointer

  // Gather-only: writes the interleaved [K,V] layout into d_p_g (one page).
  ASSERT_EQ(vkernels_kv_gather(
                d_p_g, d_k, d_v, d_slots, 1, kPageSize, kHeads, kHeadDim,
                kElem, 0),
            VKERNELS_OK);

  // Full donate: writes the same bytes into d_p_d (page at offset 0).
  const void* ptrs_d[1] = {d_p_d};
  const size_t offs[1] = {0};
  ASSERT_EQ(vkernels_p2p_kv_donate_layer(
                d_k, d_v, h_slots, ptrs_d, offs, 1, kPageSize, kHeads,
                kHeadDim, kElem, 0),
            VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_p_g, d_p_d, kPb));

  cudaFree(d_k); cudaFree(d_v); cudaFree(d_p_g); cudaFree(d_p_d);
  cudaFree(d_slots);
}

// Gather rejects null scratch / sources (contract checks fire).
TEST(P2pKvDonateCAbi, GatherNullScratchReturnsInvalidArgument) {
  int slot = 0;
  vkernels_status_t st = vkernels_kv_gather(
      nullptr, (void*)0x1000, (void*)0x2000, &slot, 1, 1, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(P2pKvDonateCAbi, GatherNonBF16ReturnsInvalidArgument) {
  uint8_t* d_scratch = to_device(patterned(64, 0));
  uint8_t* d_k = to_device(patterned(64, 0));
  uint8_t* d_v = to_device(patterned(64, 0));
  int* d_slots = ints_to_device((const int[]){0}, 1);
  vkernels_status_t st = vkernels_kv_gather(
      d_scratch, d_k, d_v, d_slots, 1, 1, 2, 4, 4, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_scratch); cudaFree(d_k); cudaFree(d_v); cudaFree(d_slots);
}

#endif  // VKERNELS_C_HAS_CUDA
