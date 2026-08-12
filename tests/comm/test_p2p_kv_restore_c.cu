// tests/comm/test_p2p_kv_restore_c.cu
//
// Runtime tests for the `extern "C"` P2P KV restore ABI. These are CUDA-only
// and run on a single GPU using same-device pointers as a stand-in for
// cross-GPU peer memory — the same stand-in the p2p_gather_c tests use, and
// sufficient to exercise the validators, the kernel launch, and the
// status-code return path.
#include "vkernels/comm/p2p_kv_restore_c.h"

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

// Byte size per destination slot.
inline size_t slot_bytes(size_t heads, size_t head_dim, size_t elem) {
  return heads * head_dim * elem;
}

// Byte size of one page's KV data.
inline size_t page_bytes(size_t page_size, size_t heads, size_t head_dim, size_t elem) {
  return page_size * 2 * heads * head_dim * elem;
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

// Fill device memory with a sentinel byte.
void fill_device(uint8_t* d, size_t n, uint8_t fill) {
  std::vector<uint8_t> h(n, fill);
  cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice);
}

}  // namespace

// ---------------------------------------------------------------------------
// Fused equals two-stage (byte-exact)
// ---------------------------------------------------------------------------
TEST(P2pKvRestoreCAbi, FusedEqualsTwoStageSinglePage) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16;

  auto h_page = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x42);
  const int h_slots[4] = {3, 7, 1, 12};

  uint8_t *d_page = nullptr, *dk_f = nullptr, *dv_f = nullptr;
  uint8_t *dk_t = nullptr, *dv_t = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_page, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_f, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_f, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_t, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_t, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, 4 * sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(d_page, h_page.data(), h_page.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, 4 * sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  fill_device(dk_f, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_f, kSlots * kSlotBytes, 0xCC);
  fill_device(dk_t, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_t, kSlots * kSlotBytes, 0xCC);

  const void* ptrs[1] = {d_page};
  const size_t offs[1] = {0};

  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      dk_f, dv_f, d_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  st = vkernels_p2p_kv_restore_layer_twostage(
      dk_t, dv_t, d_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(dk_f, dk_t, kSlots * kSlotBytes));
  ASSERT_TRUE(device_equal(dv_f, dv_t, kSlots * kSlotBytes));

  cudaFree(d_page); cudaFree(dk_f); cudaFree(dv_f);
  cudaFree(dk_t); cudaFree(dv_t); cudaFree(d_slots);
}

TEST(P2pKvRestoreCAbi, FusedEqualsTwoStageMultiPage) {
  constexpr size_t kPageSize = 2, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 128
  constexpr size_t kNumPages = 3, kSlots = 16;

  auto h0 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);
  auto h1 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x80);
  auto h2 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0xF0);
  const int h_slots[6] = {0, 5, 9, 2, 14, 7};

  uint8_t *d0 = nullptr, *d1 = nullptr, *d2 = nullptr;
  uint8_t *dk_f = nullptr, *dv_f = nullptr, *dk_t = nullptr, *dv_t = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, h0.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d1, h1.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d2, h2.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_f, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_f, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_t, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_t, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, 6 * sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(d0, h0.data(), h0.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, 6 * sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  fill_device(dk_f, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_f, kSlots * kSlotBytes, 0xCC);
  fill_device(dk_t, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_t, kSlots * kSlotBytes, 0xCC);

  const void* ptrs[3] = {d0, d1, d2};
  const size_t offs[3] = {0, 0, 0};

  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      dk_f, dv_f, d_slots, ptrs, offs, kNumPages, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  st = vkernels_p2p_kv_restore_layer_twostage(
      dk_t, dv_t, d_slots, ptrs, offs, kNumPages, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(dk_f, dk_t, kSlots * kSlotBytes));
  ASSERT_TRUE(device_equal(dv_f, dv_t, kSlots * kSlotBytes));

  cudaFree(d0); cudaFree(d1); cudaFree(d2);
  cudaFree(dk_f); cudaFree(dv_f); cudaFree(dk_t); cudaFree(dv_t); cudaFree(d_slots);
}

// Test that the vec tail path works on unaligned slot_bytes.
TEST(P2pKvRestoreCAbi, UnalignedSlotBytesFusedEqualsTwoStage) {
  constexpr size_t kPageSize = 1, kHeads = 3, kHeadDim = 5, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 30 bytes
  constexpr size_t kSlots = 4;

  auto h_page = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x37);
  const int h_slots[1] = {2};

  uint8_t *d_page = nullptr, *dk_f = nullptr, *dv_f = nullptr;
  uint8_t *dk_t = nullptr, *dv_t = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_page, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_f, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_f, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_t, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_t, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(d_page, h_page.data(), h_page.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  fill_device(dk_f, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_f, kSlots * kSlotBytes, 0xCC);
  fill_device(dk_t, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_t, kSlots * kSlotBytes, 0xCC);

  const void* ptrs[1] = {d_page};
  const size_t offs[1] = {0};

  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      dk_f, dv_f, d_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  st = vkernels_p2p_kv_restore_layer_twostage(
      dk_t, dv_t, d_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(dk_f, dk_t, kSlots * kSlotBytes));
  ASSERT_TRUE(device_equal(dv_f, dv_t, kSlots * kSlotBytes));

  cudaFree(d_page); cudaFree(dk_f); cudaFree(dv_f);
  cudaFree(dk_t); cudaFree(dv_t); cudaFree(d_slots);
}

// Test with a non-zero page offset.
TEST(P2pKvRestoreCAbi, PageOffsetIsHonoured) {
  constexpr size_t kPageSize = 1, kHeads = 2, kHeadDim = 4, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16
  constexpr size_t kTokenStr = 2 * kSlotBytes;               // 32
  constexpr size_t kSlots = 4;

  auto h_buf = patterned(page_bytes(2, kHeads, kHeadDim, kElem), 0x21);
  const int h_slots[1] = {1};

  uint8_t *d_buf = nullptr, *dk = nullptr, *dv = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_buf, h_buf.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(d_buf, h_buf.data(), h_buf.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  fill_device(dk, kSlots * kSlotBytes, 0xCC);
  fill_device(dv, kSlots * kSlotBytes, 0xCC);

  const void* ptrs[1] = {d_buf};
  const size_t offs[1] = {kTokenStr};  // skip the first token

  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      dk, dv, d_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // Verify: dst slot 1 should match the second token's K and V.
  std::vector<uint8_t> h_k(kSlots * kSlotBytes), h_v(kSlots * kSlotBytes);
  cudaMemcpy(h_k.data(), dk, h_k.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_v.data(), dv, h_v.size(), cudaMemcpyDeviceToHost);

  const uint8_t* expect_k = h_buf.data() + kTokenStr;
  const uint8_t* expect_v = expect_k + kSlotBytes;
  for (size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(h_k[kSlotBytes + i], expect_k[i]);
    ASSERT_EQ(h_v[kSlotBytes + i], expect_v[i]);
  }

  cudaFree(d_buf); cudaFree(dk); cudaFree(dv); cudaFree(d_slots);
}

// ---------------------------------------------------------------------------
// C ABI: error-code returns
// ---------------------------------------------------------------------------
TEST(P2pKvRestoreCAbi, SuccessReturnsOk) {
  constexpr size_t kPageSize = 1, kHeads = 2, kHeadDim = 4, kElem = 2;
  auto h_page = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);
  const int h_slots[1] = {0};
  const size_t kSlotBytes = slot_bytes(kHeads, kHeadDim, kElem);
  const size_t kSlots = 2;

  uint8_t *d_page = nullptr, *dk = nullptr, *dv = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_page, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, sizeof(int)) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_page, h_page.data(), h_page.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  const void* ptrs[1] = {d_page};
  const size_t offs[1] = {0};
  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      dk, dv, d_slots, ptrs, offs, 1, kPageSize, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  cudaFree(d_page); cudaFree(dk); cudaFree(dv); cudaFree(d_slots);
}

TEST(P2pKvRestoreCAbi, NullKDstReturnsInvalidArgument) {
  const size_t offs[1] = {0};
  int slot = 0;
  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      nullptr, (void*)0x1000, &slot, (const void**)0x2000, offs,
      1, 16, 8, 128, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(P2pKvRestoreCAbi, NonUniqueSlotsReturnsInvalidArgument) {
  constexpr size_t kHeads = 2, kHeadDim = 4, kElem = 2;
  auto h_page = patterned(page_bytes(2, kHeads, kHeadDim, kElem), 0x10);
  int h_slots[2] = {1, 1};  // duplicate
  uint8_t *d_page = nullptr, *dk = nullptr, *dv = nullptr;
  int* d_slots = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_page, h_page.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk, 64) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv, 64) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_slots, 2 * sizeof(int)) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_page, h_page.data(), h_page.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_slots, h_slots, 2 * sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  const void* ptrs[1] = {d_page};
  const size_t offs[1] = {0};
  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      dk, dv, d_slots, ptrs, offs, 1, 2, kHeads, kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_page); cudaFree(dk); cudaFree(dv); cudaFree(d_slots);
}

// ---------------------------------------------------------------------------
// Async stream: two concurrent streams on the fused path
// ---------------------------------------------------------------------------
TEST(P2pKvRestoreCAbi, ConcurrentStreams) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 8;

  auto h_p0 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x10);
  auto h_p1 = patterned(page_bytes(kPageSize, kHeads, kHeadDim, kElem), 0x80);
  const int h_slots[4] = {0, 4, 1, 5};

  uint8_t *dp0 = nullptr, *dp1 = nullptr;
  uint8_t *dk0 = nullptr, *dv0 = nullptr, *dk1 = nullptr, *dv1 = nullptr;
  int *ds0 = nullptr, *ds1 = nullptr;
  ASSERT_TRUE(cudaMalloc(&dp0, h_p0.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dp1, h_p1.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk0, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv0, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk1, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv1, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ds0, 4 * sizeof(int)) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ds1, 4 * sizeof(int)) == cudaSuccess);

  ASSERT_TRUE(cudaMemcpy(dp0, h_p0.data(), h_p0.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(dp1, h_p1.data(), h_p1.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(ds0, h_slots, 4 * sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(ds1, h_slots, 4 * sizeof(int),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  fill_device(dk0, kSlots * kSlotBytes, 0xCC);
  fill_device(dv0, kSlots * kSlotBytes, 0xCC);
  fill_device(dk1, kSlots * kSlotBytes, 0xCC);
  fill_device(dv1, kSlots * kSlotBytes, 0xCC);

  cudaStream_t s0, s1;
  ASSERT_TRUE(cudaStreamCreate(&s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamCreate(&s1) == cudaSuccess);

  const void* ptrs0[1] = {dp0};
  const void* ptrs1[1] = {dp1};
  const size_t offs[1] = {0};

  vkernels_status_t st = vkernels_p2p_kv_restore_layer(
      dk0, dv0, ds0, ptrs0, offs, 1, kPageSize, kHeads, kHeadDim, kElem, s0);
  ASSERT_EQ(st, VKERNELS_OK);
  st = vkernels_p2p_kv_restore_layer(
      dk1, dv1, ds1, ptrs1, offs, 1, kPageSize, kHeads, kHeadDim, kElem, s1);
  ASSERT_EQ(st, VKERNELS_OK);

  ASSERT_TRUE(cudaStreamSynchronize(s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamSynchronize(s1) == cudaSuccess);

  // Verify both streams have correct data.
  std::vector<uint8_t> hk0(kSlots * kSlotBytes), hv0(kSlots * kSlotBytes);
  std::vector<uint8_t> hk1(kSlots * kSlotBytes), hv1(kSlots * kSlotBytes);
  cudaMemcpy(hk0.data(), dk0, hk0.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hv0.data(), dv0, hv0.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hk1.data(), dk1, hk1.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hv1.data(), dv1, hv1.size(), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < hk0.size(); ++i) {
    ASSERT_EQ(hk0[i], hk1[i]);
    ASSERT_EQ(hv0[i], hv1[i]);
  }

  cudaStreamDestroy(s0); cudaStreamDestroy(s1);
  cudaFree(dp0); cudaFree(dp1); cudaFree(dk0); cudaFree(dv0);
  cudaFree(dk1); cudaFree(dv1); cudaFree(ds0); cudaFree(ds1);
}

#endif  // VKERNELS_C_HAS_CUDA
