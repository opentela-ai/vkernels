// tests/comm/test_p2p_gather_c.cu
//
// Runtime tests for the `extern "C"` P2P gather ABI. These are CUDA-only
// (the entry points take a raw cudaStream_t) and run on a single GPU using
// same-device pointers as a stand-in for cross-GPU peer memory — the same
// stand-in the benchmark uses, and sufficient to exercise the staging
// validators, the kernel launch, and the status-code return path.
#include "vkernels/comm/p2p_gather_c.h"

#include "minitest.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

namespace {

constexpr size_t kCap = 4096;

// Fill a host vector with byte `b` repeated, cast to a device source.
std::vector<uint8_t> filled(size_t n, uint8_t b) {
  return std::vector<uint8_t>(n, b);
}

}  // namespace

TEST(P2pGatherC, OneRunCopiesByteExact) {
  std::vector<uint8_t> hsrc = filled(256, 0xAB);
  uint8_t* dsrc = nullptr;
  uint8_t* ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&dsrc, 256)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(dsrc, hsrc.data(), 256, cudaMemcpyHostToDevice)  == cudaSuccess);

  const void* ptrs[1] = {dsrc};
  size_t offs[1] = {0};
  size_t lens[1] = {256};

  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kCap, ptrs, offs, lens, 1, 0);
  ASSERT_EQ(st, VKERNELS_OK);

  std::vector<uint8_t> hdst(kCap, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kCap, cudaMemcpyDeviceToHost)  == cudaSuccess);
  for (size_t i = 0; i < 256; ++i) ASSERT_EQ(hdst[i], 0xAB);
  for (size_t i = 256; i < kCap; ++i) ASSERT_EQ(hdst[i], 0);  // untouched tail

  cudaFree(dsrc);
  cudaFree(ddst);
}

TEST(P2pGatherC, MultipleRunsNonOverlapping) {
  std::vector<uint8_t> h0 = filled(128, 0x10);
  std::vector<uint8_t> h1 = filled(64, 0x20);
  uint8_t* d0 = nullptr, *d1 = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, 128)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d1, 64)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d0, h0.data(), 128, cudaMemcpyHostToDevice)  == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d1, h1.data(), 64, cudaMemcpyHostToDevice)  == cudaSuccess);

  const void* ptrs[2] = {d0, d1};
  size_t offs[2] = {0, 200};
  size_t lens[2] = {128, 64};

  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kCap, ptrs, offs, lens, 2, 0);
  ASSERT_EQ(st, VKERNELS_OK);

  std::vector<uint8_t> hdst(kCap, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kCap, cudaMemcpyDeviceToHost)  == cudaSuccess);
  for (size_t i = 0; i < 128; ++i) ASSERT_EQ(hdst[i], 0x10);
  for (size_t i = 128; i < 200; ++i) ASSERT_EQ(hdst[i], 0);  // gap
  for (size_t i = 200; i < 264; ++i) ASSERT_EQ(hdst[i], 0x20);

  cudaFree(d0);
  cudaFree(d1);
  cudaFree(ddst);
}

TEST(P2pGatherC, EmptyRunListIsNoOp) {
  uint8_t* ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  // Zero runs: valid no-op, must not touch dst or launch.
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kCap, nullptr, nullptr, nullptr, 0, 0);
  ASSERT_EQ(st, VKERNELS_OK);
  cudaFree(ddst);
}

TEST(P2pGatherC, NullSrcForNonEmptyRunThrows) {
  uint8_t* ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  const void* ptrs[1] = {nullptr};
  size_t offs[1] = {0};
  size_t lens[1] = {16};
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kCap, ptrs, offs, lens, 1, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
  cudaFree(ddst);
}

TEST(P2pGatherC, OverlappingDstRunsThrow) {
  uint8_t* d0 = nullptr, *d1 = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, 32)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d1, 32)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  // [0,16) and [8,24) overlap at [8,16).
  const void* ptrs[2] = {d0, d1};
  size_t offs[2] = {0, 8};
  size_t lens[2] = {16, 16};
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kCap, ptrs, offs, lens, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
  cudaFree(d0);
  cudaFree(d1);
  cudaFree(ddst);
}

TEST(P2pGatherC, CapacityExceededThrows) {
  uint8_t* d0 = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, 64)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, 32)  == cudaSuccess);
  const void* ptrs[1] = {d0};
  size_t offs[1] = {0};
  size_t lens[1] = {64};  // 64 > cap 32
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, 32, ptrs, offs, lens, 1, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
  cudaFree(d0);
  cudaFree(ddst);
}

TEST(P2pGatherC2d, OneTileCopiesByteExact) {
  std::vector<uint8_t> hsrc = filled(64, 0x5A);  // 2 rows x 32 bytes
  uint8_t* dsrc = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&dsrc, 64)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(dsrc, hsrc.data(), 64, cudaMemcpyHostToDevice)  == cudaSuccess);

  vkernels_gather_2d_run_t runs[1] = {{dsrc, 32, 0, 32, 32, 2}};
  vkernels_status_t st = vkernels_p2p_gather_runs_2d(ddst, kCap, runs, 1, 0);
  ASSERT_EQ(st, VKERNELS_OK);

  std::vector<uint8_t> hdst(kCap, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kCap, cudaMemcpyDeviceToHost)  == cudaSuccess);
  for (size_t r = 0; r < 2; ++r)
    for (size_t c = 0; c < 32; ++c) ASSERT_EQ(hdst[r * 32 + c], 0x5A);

  cudaFree(dsrc);
  cudaFree(ddst);
}

TEST(P2pGatherC2d, OverlappingTilesThrow) {
  uint8_t* d0 = nullptr, *d1 = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, 32)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d1, 32)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  // tile0 dst box [0,4), tile1 dst box [2,6) — overlap at [2,4).
  vkernels_gather_2d_run_t runs[2] = {{d0, 4, 0, 4, 4, 1},
                                       {d1, 4, 2, 4, 4, 1}};
  vkernels_status_t st = vkernels_p2p_gather_runs_2d(ddst, kCap, runs, 2, 0);
  ASSERT_EQ(st, VKERNELS_ERR_INVALID_ARGUMENT);
  cudaFree(d0);
  cudaFree(d1);
  cudaFree(ddst);
}

#else  // !VKERNELS_C_HAS_CUDA

// When no CUDA toolkit is on the include path the C ABI entry points are not
// declared; the test file still compiles (as a host TU) so CTest doesn't lose
// a registered target, but there is nothing to exercise.
TEST(P2pGatherC, SkippedWithoutCuda) { EXPECT_TRUE(true); }

#endif  // VKERNELS_C_HAS_CUDA
