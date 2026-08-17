// tests/comm/test_kv_gather_c.cu
//
// Runtime tests for the `extern "C` fused indexed K/V layer gather ABI
// (issue #2). CUDA-only: they run on a single GPU, exercising the host-input
// entry point (validate + upload + launch) and the device-slot entry point
// (check-free, caller-owned device pointer) for both int32 and int64 slot
// ids, byte-exactly against the PyTorch two-gather reference
// (`dst[:, :, 0] = k_src[slot_ids]; dst[:, :, 1] = v_src[slot_ids]`), and
// the status-code return path.
//
// Destination layout: [num_pages, page_size, 2, num_kv_heads, head_dim]
// row-major; within a token, K is at offset 0 and V at offset slot_bytes.
#include "vkernels/comm/kv_gather_c.h"

#include "minitest.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

namespace {

std::vector<uint8_t> patterned(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + (i % 251));
  return v;
}

inline size_t slot_bytes(size_t heads, size_t head_dim, size_t elem) {
  return heads * head_dim * elem;
}
inline size_t token_stride(size_t heads, size_t head_dim, size_t elem) {
  return 2 * slot_bytes(heads, head_dim, elem);
}
inline size_t page_bytes(size_t page_size, size_t heads, size_t head_dim,
                         size_t elem) {
  return page_size * token_stride(heads, head_dim, elem);
}

bool device_equal(const uint8_t* d_a, const uint8_t* d_b, size_t n) {
  std::vector<uint8_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_a, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, n, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i)
    if (ha[i] != hb[i]) return false;
  return true;
}

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

// The PyTorch two-gather reference, in bytes.
std::vector<uint8_t> two_gather_ref(const std::vector<uint8_t>& k,
                                    const std::vector<uint8_t>& v,
                                    const int* slots, size_t num_pages,
                                    size_t page_size, size_t heads,
                                    size_t head_dim, size_t elem) {
  const size_t sb = slot_bytes(heads, head_dim, elem);
  const size_t ts = token_stride(heads, head_dim, elem);
  const size_t pb = page_bytes(page_size, heads, head_dim, elem);
  std::vector<uint8_t> ref(num_pages * pb);
  for (size_t p = 0; p < num_pages; ++p) {
    for (size_t t = 0; t < page_size; ++t) {
      size_t slot = static_cast<size_t>(slots[p * page_size + t]);
      size_t dst = p * pb + t * ts;
      std::memcpy(ref.data() + dst, k.data() + slot * sb, sb);
      std::memcpy(ref.data() + dst + sb, v.data() + slot * sb, sb);
    }
  }
  return ref;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host-input entry point: validate + upload + launch (int32)
// ---------------------------------------------------------------------------
TEST(KvGatherLayerCAbi, HostInputInt32MatchesReference) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16, kNumPages = 2;

  auto h_k = patterned(kSlots * kSlotBytes, 0x11);
  auto h_v = patterned(kSlots * kSlotBytes, 0x22);
  const int h_slots[8] = {3, 7, 1, 12, 0, 15, 8, 4};  // non-monotonic
  auto h_ref = two_gather_ref(h_k, h_v, h_slots, kNumPages, kPageSize, kHeads,
                              kHeadDim, kElem);

  uint8_t *d_k = to_device(h_k), *d_v = to_device(h_v), *d_dst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_dst, kNumPages * page_bytes(kPageSize, kHeads,
                                                        kHeadDim, kElem)) ==
              cudaSuccess);

  vkernels_status_t st = vkernels_kv_gather_layer(
      d_dst, d_k, d_v, h_slots, 0, kSlots, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref = to_device(h_ref);
  ASSERT_TRUE(device_equal(d_dst, d_ref, h_ref.size()));

  cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst); cudaFree(d_ref);
}

// Host-input, int64 slot ids.
TEST(KvGatherLayerCAbi, HostInputInt64MatchesReference) {
  constexpr size_t kPageSize = 2, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 128
  constexpr size_t kSlots = 8, kNumPages = 2;

  auto h_k = patterned(kSlots * kSlotBytes, 0x33);
  auto h_v = patterned(kSlots * kSlotBytes, 0x55);
  const int64_t h_slots64[4] = {5LL, 2LL, 7LL, 0LL};
  std::vector<int> h_slots32(h_slots64, h_slots64 + 4);
  auto h_ref = two_gather_ref(h_k, h_v, h_slots32.data(), kNumPages, kPageSize,
                              kHeads, kHeadDim, kElem);

  uint8_t *d_k = to_device(h_k), *d_v = to_device(h_v), *d_dst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_dst, kNumPages * page_bytes(kPageSize, kHeads,
                                                        kHeadDim, kElem)) ==
              cudaSuccess);

  vkernels_status_t st = vkernels_kv_gather_layer(
      d_dst, d_k, d_v, h_slots64, 1, kSlots, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref = to_device(h_ref);
  ASSERT_TRUE(device_equal(d_dst, d_ref, h_ref.size()));

  cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst); cudaFree(d_ref);
}

// ---------------------------------------------------------------------------
// Device-slot entry point: check-free, caller-owned device pointer (int32)
// ---------------------------------------------------------------------------
TEST(KvGatherLayerCAbi, DeviceSlotsInt32MatchesReference) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16, kNumPages = 2;

  auto h_k = patterned(kSlots * kSlotBytes, 0x11);
  auto h_v = patterned(kSlots * kSlotBytes, 0x22);
  const int h_slots[8] = {3, 7, 1, 12, 0, 15, 8, 4};
  auto h_ref = two_gather_ref(h_k, h_v, h_slots, kNumPages, kPageSize, kHeads,
                              kHeadDim, kElem);

  uint8_t *d_k = to_device(h_k), *d_v = to_device(h_v), *d_dst = nullptr;
  int* d_slots = ints_to_device(h_slots, 8);
  ASSERT_TRUE(cudaMalloc(&d_dst, kNumPages * page_bytes(kPageSize, kHeads,
                                                        kHeadDim, kElem)) ==
              cudaSuccess);

  vkernels_status_t st = vkernels_kv_gather_layer_device_slots(
      d_dst, d_k, d_v, d_slots, 0, kSlots, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref = to_device(h_ref);
  ASSERT_TRUE(device_equal(d_dst, d_ref, h_ref.size()));

  cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst); cudaFree(d_ref);
  cudaFree(d_slots);
}

// Device-slot, int64.
TEST(KvGatherLayerCAbi, DeviceSlotsInt64MatchesReference) {
  constexpr size_t kPageSize = 2, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 128
  constexpr size_t kSlots = 8, kNumPages = 2;

  auto h_k = patterned(kSlots * kSlotBytes, 0x33);
  auto h_v = patterned(kSlots * kSlotBytes, 0x55);
  const int64_t h_slots64[4] = {5LL, 2LL, 7LL, 0LL};
  std::vector<int> h_slots32(h_slots64, h_slots64 + 4);
  auto h_ref = two_gather_ref(h_k, h_v, h_slots32.data(), kNumPages, kPageSize,
                              kHeads, kHeadDim, kElem);

  uint8_t *d_k = to_device(h_k), *d_v = to_device(h_v), *d_dst = nullptr;
  int64_t* d_slots = int64s_to_device(h_slots64, 4);
  ASSERT_TRUE(cudaMalloc(&d_dst, kNumPages * page_bytes(kPageSize, kHeads,
                                                        kHeadDim, kElem)) ==
              cudaSuccess);

  vkernels_status_t st = vkernels_kv_gather_layer_device_slots(
      d_dst, d_k, d_v, d_slots, 1, kSlots, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref = to_device(h_ref);
  ASSERT_TRUE(device_equal(d_dst, d_ref, h_ref.size()));

  cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst); cudaFree(d_ref);
  cudaFree(d_slots);
}

// ---------------------------------------------------------------------------
// Contract checks return VKERNELS_ERR_INVALID_ARGUMENT
// ---------------------------------------------------------------------------
TEST(KvGatherLayerCAbi, NullPointersReturnInvalidArgument) {
  int slot = 0;
  vkernels_status_t st = vkernels_kv_gather_layer(
      nullptr, (void*)0x1000, (void*)0x2000, &slot, 0, 1, 1, 1, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(KvGatherLayerCAbi, OutOfRangeSlotReturnInvalidArgument) {
  uint8_t* d_k = to_device(patterned(4 * 2 * 4 * 2, 0));
  uint8_t* d_v = to_device(patterned(4 * 2 * 4 * 2, 0));
  uint8_t* d_dst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_dst, 1 * page_bytes(1, 2, 4, 2)) == cudaSuccess);
  const int too_big[1] = {4};  // == num_slots
  vkernels_status_t st = vkernels_kv_gather_layer(
      d_dst, d_k, d_v, too_big, 0, 4, 1, 1, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst);
}

TEST(KvGatherLayerCAbi, NonBf16Fp16ElemReturnInvalidArgument) {
  uint8_t* d_k = to_device(patterned(64, 0));
  uint8_t* d_v = to_device(patterned(64, 0));
  uint8_t* d_dst = to_device(patterned(64, 0));
  int slot = 0;
  vkernels_status_t st = vkernels_kv_gather_layer(
      d_dst, d_k, d_v, &slot, 0, 1, 1, 1, 2, 4, 4, 0);  // FP32
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst);
}

TEST(KvGatherLayerCAbi, ZeroPagesIsNoOp) {
  vkernels_status_t st = vkernels_kv_gather_layer(
      nullptr, nullptr, nullptr, nullptr, 0, 1, 0, 1, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_OK);
}

#endif  // VKERNELS_C_HAS_CUDA
