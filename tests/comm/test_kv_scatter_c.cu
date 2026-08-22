// tests/comm/test_kv_scatter_c.cu
//
// Runtime tests for the `extern "C"` fused indexed K/V layer scatter ABI
// (issue #1). CUDA-only: they run on a single GPU, exercising the host-input
// entry point (validate, incl. UNIQUENESS, + upload + launch) and the
// device-slot entry point (check-free, caller-owned device pointer) for both
// int32 and int64 slot ids, byte-exactly against the PyTorch two-write
// reference (`k_dst[slot_ids] = src[:, :, 0]; v_dst[slot_ids] = src[:, :, 1]`),
// and the status-code return path (including the duplicate-slot rejection
// the gather ABI does not have).
//
// Destination layout: k_dst / v_dst are separate [num_slots, num_kv_heads,
// head_dim] allocations; src is [num_pages, page_size, 2, num_kv_heads,
// head_dim] row-major (index 0 of the "2" dim is K, index 1 is V).
#include "vkernels/comm/kv_scatter_c.h"

#include "minitest.hpp"

#include <cstdint>
#include <cstring>
#include <utility>
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

// The PyTorch two-write reference, in bytes. Returns (k, v) each
// [num_slots * slot_bytes], sentinel-filled so untouched slots match the
// kernel (which leaves them alone).
template <typename SlotT>
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> two_scatter_ref(
    const std::vector<uint8_t>& src, const SlotT* slots, size_t num_slots,
    size_t num_pages, size_t page_size, size_t heads, size_t head_dim,
    size_t elem) {
  const size_t sb = slot_bytes(heads, head_dim, elem);
  const size_t ts = token_stride(heads, head_dim, elem);
  const size_t pb = page_bytes(page_size, heads, head_dim, elem);
  std::vector<uint8_t> k(num_slots * sb, 0xCC), v(num_slots * sb, 0xCC);
  for (size_t p = 0; p < num_pages; ++p) {
    const uint8_t* sp = src.data() + p * pb;
    for (size_t t = 0; t < page_size; ++t) {
      size_t slot = static_cast<size_t>(slots[p * page_size + t]);
      memcpy(k.data() + slot * sb, sp + t * ts, sb);
      memcpy(v.data() + slot * sb, sp + t * ts + sb, sb);
    }
  }
  return {k, v};
}

}  // namespace

// ---------------------------------------------------------------------------
// Host-input entry point: validate (+ uniqueness) + upload + launch (int32)
// ---------------------------------------------------------------------------
TEST(KvScatterLayerCAbi, HostInputInt32MatchesReference) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16, kNumPages = 2;

  auto h_src = patterned(kNumPages * page_bytes(kPageSize, kHeads, kHeadDim,
                                                kElem), 0x11);
  const int h_slots[8] = {3, 7, 1, 12, 0, 15, 8, 4};  // unique, non-monotonic
  auto ref = two_scatter_ref(h_src, h_slots, kSlots, kNumPages, kPageSize,
                             kHeads, kHeadDim, kElem);

  uint8_t *d_src = to_device(h_src), *d_k = nullptr, *d_v = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_k, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_v, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_k, 0xCC, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_v, 0xCC, kSlots * kSlotBytes) == cudaSuccess);

  vkernels_status_t st = vkernels_kv_scatter_layer(
      d_k, d_v, h_slots, 0, kSlots, d_src, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref_k = to_device(ref.first), d_ref_v = to_device(ref.second);
  ASSERT_TRUE(device_equal(d_k, d_ref_k, ref.first.size()));
  ASSERT_TRUE(device_equal(d_v, d_ref_v, ref.second.size()));

  cudaFree(d_src); cudaFree(d_k); cudaFree(d_v);
  cudaFree(d_ref_k); cudaFree(d_ref_v);
}

// Host-input, int64 slot ids.
TEST(KvScatterLayerCAbi, HostInputInt64MatchesReference) {
  constexpr size_t kPageSize = 2, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 128
  constexpr size_t kSlots = 8, kNumPages = 2;

  auto h_src = patterned(kNumPages * page_bytes(kPageSize, kHeads, kHeadDim,
                                                kElem), 0x33);
  const int64_t h_slots64[4] = {5LL, 2LL, 7LL, 0LL};  // unique
  auto ref = two_scatter_ref(h_src, h_slots64, kSlots, kNumPages, kPageSize,
                             kHeads, kHeadDim, kElem);

  uint8_t *d_src = to_device(h_src), *d_k = nullptr, *d_v = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_k, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_v, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_k, 0xCC, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_v, 0xCC, kSlots * kSlotBytes) == cudaSuccess);

  vkernels_status_t st = vkernels_kv_scatter_layer(
      d_k, d_v, h_slots64, 1, kSlots, d_src, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref_k = to_device(ref.first), d_ref_v = to_device(ref.second);
  ASSERT_TRUE(device_equal(d_k, d_ref_k, ref.first.size()));
  ASSERT_TRUE(device_equal(d_v, d_ref_v, ref.second.size()));

  cudaFree(d_src); cudaFree(d_k); cudaFree(d_v);
  cudaFree(d_ref_k); cudaFree(d_ref_v);
}

// ---------------------------------------------------------------------------
// Device-slot entry point: check-free, caller-owned device pointer (int32)
// ---------------------------------------------------------------------------
TEST(KvScatterLayerCAbi, DeviceSlotsInt32MatchesReference) {
  constexpr size_t kPageSize = 4, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr size_t kSlots = 16, kNumPages = 2;

  auto h_src = patterned(kNumPages * page_bytes(kPageSize, kHeads, kHeadDim,
                                                kElem), 0x11);
  const int h_slots[8] = {3, 7, 1, 12, 0, 15, 8, 4};
  auto ref = two_scatter_ref(h_src, h_slots, kSlots, kNumPages, kPageSize,
                             kHeads, kHeadDim, kElem);

  uint8_t *d_src = to_device(h_src), *d_k = nullptr, *d_v = nullptr;
  int* d_slots = ints_to_device(h_slots, 8);
  ASSERT_TRUE(cudaMalloc(&d_k, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_v, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_k, 0xCC, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_v, 0xCC, kSlots * kSlotBytes) == cudaSuccess);

  vkernels_status_t st = vkernels_kv_scatter_layer_device_slots(
      d_k, d_v, d_slots, 0, kSlots, d_src, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref_k = to_device(ref.first), d_ref_v = to_device(ref.second);
  ASSERT_TRUE(device_equal(d_k, d_ref_k, ref.first.size()));
  ASSERT_TRUE(device_equal(d_v, d_ref_v, ref.second.size()));

  cudaFree(d_src); cudaFree(d_k); cudaFree(d_v); cudaFree(d_slots);
  cudaFree(d_ref_k); cudaFree(d_ref_v);
}

// Device-slot, int64.
TEST(KvScatterLayerCAbi, DeviceSlotsInt64MatchesReference) {
  constexpr size_t kPageSize = 2, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 128
  constexpr size_t kSlots = 8, kNumPages = 2;

  auto h_src = patterned(kNumPages * page_bytes(kPageSize, kHeads, kHeadDim,
                                                kElem), 0x33);
  const int64_t h_slots64[4] = {5LL, 2LL, 7LL, 0LL};
  auto ref = two_scatter_ref(h_src, h_slots64, kSlots, kNumPages, kPageSize,
                             kHeads, kHeadDim, kElem);

  uint8_t *d_src = to_device(h_src), *d_k = nullptr, *d_v = nullptr;
  int64_t* d_slots = int64s_to_device(h_slots64, 4);
  ASSERT_TRUE(cudaMalloc(&d_k, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_v, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_k, 0xCC, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(d_v, 0xCC, kSlots * kSlotBytes) == cudaSuccess);

  vkernels_status_t st = vkernels_kv_scatter_layer_device_slots(
      d_k, d_v, d_slots, 1, kSlots, d_src, kNumPages, kPageSize, kHeads,
      kHeadDim, kElem, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto d_ref_k = to_device(ref.first), d_ref_v = to_device(ref.second);
  ASSERT_TRUE(device_equal(d_k, d_ref_k, ref.first.size()));
  ASSERT_TRUE(device_equal(d_v, d_ref_v, ref.second.size()));

  cudaFree(d_src); cudaFree(d_k); cudaFree(d_v); cudaFree(d_slots);
  cudaFree(d_ref_k); cudaFree(d_ref_v);
}

// ---------------------------------------------------------------------------
// Contract checks return VKERNELS_ERR_INVALID_ARGUMENT
// ---------------------------------------------------------------------------
TEST(KvScatterLayerCAbi, NullPointersReturnInvalidArgument) {
  int slot = 0;
  // k_dst null with num_pages=1 and a non-null src pointer.
  vkernels_status_t st = vkernels_kv_scatter_layer(
      nullptr, (void*)0x1000, &slot, 0, 1, (void*)0x2000, 1, 1, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
}

TEST(KvScatterLayerCAbi, OutOfRangeSlotReturnInvalidArgument) {
  uint8_t* d_src = to_device(patterned(4 * 2 * 4 * 2, 0));
  uint8_t* d_k = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_k, 4 * slot_bytes(2, 4, 2)) == cudaSuccess);
  uint8_t* d_v = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_v, 4 * slot_bytes(2, 4, 2)) == cudaSuccess);
  const int too_big[1] = {4};  // == num_slots
  vkernels_status_t st = vkernels_kv_scatter_layer(
      d_k, d_v, too_big, 0, 4, d_src, 1, 1, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_src); cudaFree(d_k); cudaFree(d_v);
}

// Duplicates are a scatter contract violation (the gather ABI, which may
// repeat source slots, does NOT reject these).
TEST(KvScatterLayerCAbi, DuplicateSlotReturnInvalidArgument) {
  uint8_t* d_src = to_device(patterned(4 * 2 * 4 * 2, 0));
  uint8_t* d_k = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_k, 4 * slot_bytes(2, 4, 2)) == cudaSuccess);
  uint8_t* d_v = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_v, 4 * slot_bytes(2, 4, 2)) == cudaSuccess);
  const int dup[2] = {1, 1};
  vkernels_status_t st = vkernels_kv_scatter_layer(
      d_k, d_v, dup, 0, 4, d_src, 1, 2, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_src); cudaFree(d_k); cudaFree(d_v);
}

TEST(KvScatterLayerCAbi, NonBf16Fp16ElemReturnInvalidArgument) {
  uint8_t* d_src = to_device(patterned(64, 0));
  uint8_t* d_k = to_device(patterned(64, 0));
  uint8_t* d_v = to_device(patterned(64, 0));
  int slot = 0;
  vkernels_status_t st = vkernels_kv_scatter_layer(
      d_k, d_v, &slot, 0, 1, d_src, 1, 1, 2, 4, 4, 0);  // FP32
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);

  cudaFree(d_src); cudaFree(d_k); cudaFree(d_v);
}

TEST(KvScatterLayerCAbi, ZeroPagesIsNoOp) {
  vkernels_status_t st = vkernels_kv_scatter_layer(
      nullptr, nullptr, nullptr, 0, 1, nullptr, 0, 1, 2, 4, 2, 0);
  ASSERT_EQ(st, VKERNELS_OK);
}

#endif  // VKERNELS_C_HAS_CUDA
