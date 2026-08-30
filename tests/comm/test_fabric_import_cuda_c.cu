// tests/comm/test_fabric_import_cuda_c.cu
//
// Runtime tests for the `extern "C"` on-device fabric-import bounce ABI
// (issue #49). CUDA-only, run on a single GPU. The host planning C ABI
// (classify / eager-break / cost model) is exercised by the always-compiled
// test_fabric_import_c.cpp; these tests cover the CUDA-only entries:
// vkernels_fabric_bounce_scratch_alloc / free and the stream-ordered
// device<->pinned copies. The real CU_MEM_HANDLE_TYPE_FABRIC import
// (vkernels_fabric_import_device_ptr) needs live driver handles and is
// exercised on H-CLARIDEN / H-JSC, not in a container.
#include "vkernels/comm/fabric_import_c.h"

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

bool device_equal(const uint8_t* d_a, const uint8_t* d_b, size_t n) {
  std::vector<uint8_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_a, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, n, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i)
    if (ha[i] != hb[i]) return false;
  return true;
}

void fill_device(uint8_t* d, size_t n, uint8_t fill) {
  std::vector<uint8_t> h(n, fill);
  cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice);
}

}  // namespace

// ---------------------------------------------------------------------------
// Bounce scratch: alloc/free, and a device<->pinned round trip.
// ---------------------------------------------------------------------------
TEST(FabricImportBounceCAbi, ScratchAllocFreeRoundTrip) {
  constexpr size_t kN = 4096;
  auto h_pat = patterned(kN, 0x5A);

  int status = -1;
  void* pinned = vkernels_fabric_bounce_scratch_alloc(kN, &status);
  ASSERT_TRUE(pinned != nullptr);
  ASSERT_EQ(status, VKERNELS_FI_OK);

  // Fill the pinned buffer with the pattern, mirror it to a device buffer,
  // copy it back into a second pinned buffer, and compare. The plan owns
  // neither buffer after this round trip (caller frees with _scratch_free).
  memcpy(pinned, h_pat.data(), kN);
  uint8_t* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, kN) == cudaSuccess);
  fill_device(d, kN, 0xCC);

  vkernels_fabric_bounce_pinned_to_device(d, pinned, kN, /*stream=*/0);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  void* pinned2 = vkernels_fabric_bounce_scratch_alloc(kN, &status);
  ASSERT_TRUE(pinned2 != nullptr);
  ASSERT_EQ(status, VKERNELS_FI_OK);
  memset(pinned2, 0, kN);

  vkernels_fabric_bounce_device_to_pinned(pinned2, d, kN, /*stream=*/0);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(std::memcmp(pinned2, h_pat.data(), kN) == 0);

  vkernels_fabric_bounce_scratch_free(pinned);
  vkernels_fabric_bounce_scratch_free(pinned2);
  cudaFree(d);
}

// A zero-size alloc is a valid no-op returning nullptr + OK (the host
// reference contract for an empty transfer).
TEST(FabricImportBounceCAbi, ScratchZeroSizeIsNoOp) {
  int status = -1;
  void* pinned = vkernels_fabric_bounce_scratch_alloc(0, &status);
  ASSERT_TRUE(pinned == nullptr);
  ASSERT_EQ(status, VKERNELS_FI_OK);
  vkernels_fabric_bounce_scratch_free(nullptr);  // no-op on null
}

// A null status_out is allowed: alloc still returns the pointer and
// (critically) does not dereference the null status.
TEST(FabricImportBounceCAbi, ScratchAllocNullStatus) {
  void* pinned = vkernels_fabric_bounce_scratch_alloc(64, nullptr);
  ASSERT_TRUE(pinned != nullptr);
  vkernels_fabric_bounce_scratch_free(pinned);
}

// Null pointers / zero size are no-ops for the stream-ordered copies
// (the plan relies on this when num_pages == 0).
TEST(FabricImportBounceCAbi, BounceNullIsNoOp) {
  uint8_t* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, 64) == cudaSuccess);
  fill_device(d, 64, 0xAB);
  // null pinned -> no-op (device unchanged)
  vkernels_fabric_bounce_device_to_pinned(nullptr, d, 64, 0);
  // null device -> no-op
  void* pinned = vkernels_fabric_bounce_scratch_alloc(64, nullptr);
  ASSERT_TRUE(pinned != nullptr);
  vkernels_fabric_bounce_pinned_to_device(nullptr, pinned, 64, 0);
  // zero size -> no-op
  vkernels_fabric_bounce_pinned_to_device(d, pinned, 0, 0);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<uint8_t> h(64);
  cudaMemcpy(h.data(), d, 64, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < 64; ++i) ASSERT_EQ(h[i], 0xAB);
  vkernels_fabric_bounce_scratch_free(pinned);
  cudaFree(d);
}

// The copies are stream-ordered: a round trip on a non-default stream must
// complete before the stream is synchronised.
TEST(FabricImportBounceCAbi, BounceOnStream) {
  constexpr size_t kN = 2048;
  auto h_pat = patterned(kN, 0x77);
  cudaStream_t s;
  ASSERT_TRUE(cudaStreamCreate(&s) == cudaSuccess);
  void* pinned = vkernels_fabric_bounce_scratch_alloc(kN, nullptr);
  ASSERT_TRUE(pinned != nullptr);
  uint8_t* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, kN) == cudaSuccess);
  memcpy(pinned, h_pat.data(), kN);
  vkernels_fabric_bounce_pinned_to_device(d, pinned, kN, s);
  ASSERT_TRUE(cudaStreamSynchronize(s) == cudaSuccess);
  std::vector<uint8_t> h(kN);
  cudaMemcpy(h.data(), d, kN, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < kN; ++i) ASSERT_EQ(h[i], h_pat[i]);
  vkernels_fabric_bounce_scratch_free(pinned);
  cudaFree(d);
  cudaStreamDestroy(s);
}

#endif  // VKERNELS_C_HAS_CUDA
