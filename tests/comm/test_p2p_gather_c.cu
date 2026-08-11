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

#  include "vkernels/comm/p2p_gather.hpp"  // set_gather_dispatch / force modes

namespace {

constexpr size_t kCap = 4096;

// Fill a host vector with byte `b` repeated, cast to a device source.
std::vector<uint8_t> filled(size_t n, uint8_t b) {
  return std::vector<uint8_t>(n, b);
}

// A pseudo-random but deterministic byte pattern (i % 251) so mis-routed
// copies and off-by-one tails are obvious.
std::vector<uint8_t> patterned(size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(i % 251);
  return v;
}

}  // namespace

TEST(P2pGatherCAdaptive, FewRunsUseCopyEngineByteExact) {
  // Default dispatch: 1-2 runs sit below the crossover, so the CUDA path
  // must take the per-run copy engine and still be byte-exact.
  std::vector<uint8_t> h0 = filled(128, 0x10);
  std::vector<uint8_t> h1 = filled(64, 0x20);
  uint8_t* d0 = nullptr, *d1 = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, 128) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d1, 64) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d0, h0.data(), 128, cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d1, h1.data(), 64, cudaMemcpyHostToDevice) == cudaSuccess);

  const void* ptrs[2] = {d0, d1};
  size_t offs[2] = {0, 200};
  size_t lens[2] = {128, 64};
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kCap, ptrs, offs, lens, 2, 0);
  ASSERT_EQ(st, VKERNELS_OK);

  std::vector<uint8_t> hdst(kCap, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kCap, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < 128; ++i) ASSERT_EQ(hdst[i], 0x10);
  for (size_t i = 200; i < 264; ++i) ASSERT_EQ(hdst[i], 0x20);
  cudaFree(d0); cudaFree(d1); cudaFree(ddst);
}

TEST(P2pGatherCAdaptive, ManyRunsUseKernelByteExact) {
  // 32 runs x 48 KiB: above the 24-run floor and the fitted crossover, so
  // the adaptive path must take the single-launch kernel.
  constexpr size_t kRuns = 32, kRun = 48 * 1024;
  std::vector<uint8_t> hsrc = patterned(kRuns * kRun);
  uint8_t* dsrc = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&dsrc, kRuns * kRun) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kRuns * kRun) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(dsrc, hsrc.data(), kRuns * kRun, cudaMemcpyHostToDevice) == cudaSuccess);

  std::vector<const void*> ptrs(kRuns);
  std::vector<size_t> offs(kRuns), lens(kRuns);
  for (size_t i = 0; i < kRuns; ++i) {
    ptrs[i] = dsrc + i * kRun;
    offs[i] = i * kRun;
    lens[i] = kRun;
  }
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kRuns * kRun, ptrs.data(),
                                                  offs.data(), lens.data(), kRuns, 0);
  ASSERT_EQ(st, VKERNELS_OK);

  std::vector<uint8_t> hdst(kRuns * kRun, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kRuns * kRun, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t r = 0; r < kRuns; ++r)
    for (size_t i = 0; i < kRun; ++i)
      ASSERT_EQ(hdst[r * kRun + i], static_cast<uint8_t>((r * kRun + i) % 251));
  cudaFree(dsrc); cudaFree(ddst);
}

TEST(P2pGatherCAdaptive, ForcedKernelVectorizedTailAndUnaligned) {
  // Force the kernel and mix vectorized and scalar runs in ONE launch:
  //   run0: src aligned, dst off 0,      len 48  -> vec, 3 chunks, tail 0
  //   run1: src unaligned (+3), dst off 64, len 50 -> scalar path
  //   run2: src aligned (+16), dst off 128, len 17 -> vec, 1 chunk + 1 tail
  // (Offsets leave disjoint dst spans [0,48) / [64,114) / [128,145): the
  // staging validators reject overlapping outputs.)
  vkernels::comm::set_gather_dispatch(vkernels::comm::GatherDispatchMode::kForceKernel, 1);

  std::vector<uint8_t> hsrc = patterned(256);
  uint8_t* dsrc = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&dsrc, 256) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap) == cudaSuccess);
  ASSERT_TRUE(cudaMemset(ddst, 0, kCap) == cudaSuccess);  // gap checks need a known fill
  ASSERT_TRUE(cudaMemcpy(dsrc, hsrc.data(), 256, cudaMemcpyHostToDevice) == cudaSuccess);

  const void* ptrs[3] = {dsrc, dsrc + 3, dsrc + 16};
  size_t offs[3] = {0, 64, 128};
  size_t lens[3] = {48, 50, 17};
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kCap, ptrs, offs, lens, 3, 0);
  ASSERT_EQ(st, VKERNELS_OK);

  std::vector<uint8_t> hdst(kCap, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kCap, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < 48; ++i) ASSERT_EQ(hdst[i], hsrc[i]);
  for (size_t i = 0; i < 50; ++i) ASSERT_EQ(hdst[64 + i], hsrc[3 + i]);
  for (size_t i = 0; i < 17; ++i) ASSERT_EQ(hdst[128 + i], hsrc[16 + i]);
  ASSERT_EQ(hdst[48], 0);   // gaps stay untouched
  ASSERT_EQ(hdst[114], 0);
  ASSERT_EQ(hdst[145], 0);

  cudaFree(dsrc); cudaFree(ddst);
  vkernels::comm::set_gather_dispatch();  // restore adaptive defaults
}

TEST(P2pGatherCAdaptive, ForcedCopyEngineManyRuns) {
  // Force the copy-engine path even at 32 runs: byte-exact equivalence with
  // the kernel result (the same list through the forced kernel above).
  constexpr size_t kRuns = 32, kRun = 256;
  std::vector<uint8_t> hsrc = patterned(kRuns * kRun);
  uint8_t* dsrc = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&dsrc, kRuns * kRun) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kRuns * kRun) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(dsrc, hsrc.data(), kRuns * kRun, cudaMemcpyHostToDevice) == cudaSuccess);

  vkernels::comm::set_gather_dispatch(vkernels::comm::GatherDispatchMode::kForceCopyEngine, 24);
  std::vector<const void*> ptrs(kRuns);
  std::vector<size_t> offs(kRuns), lens(kRuns);
  for (size_t i = 0; i < kRuns; ++i) {
    ptrs[i] = dsrc + i * kRun;
    offs[i] = i * kRun;
    lens[i] = kRun;
  }
  vkernels_status_t st = vkernels_p2p_gather_runs(ddst, kRuns * kRun, ptrs.data(),
                                                  offs.data(), lens.data(), kRuns, 0);
  ASSERT_EQ(st, VKERNELS_OK);

  std::vector<uint8_t> hdst(kRuns * kRun, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kRuns * kRun, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < kRuns * kRun; ++i) ASSERT_EQ(hdst[i], hsrc[i]);

  cudaFree(dsrc); cudaFree(ddst);
  vkernels::comm::set_gather_dispatch();  // restore adaptive defaults
}

// ---------------------------------------------------------------------------
// Prepared plans (CUDA): validate + upload once, reuse across launches
// ---------------------------------------------------------------------------
TEST(P2pGatherPlan1dC, ReuseAcrossManyLaunchesAndStreams) {
  // The KVAAS pattern: one run list bound to one scratch, reused across 40
  // layer launches with no per-launch metadata allocation or H2D copy.
  constexpr size_t kRuns = 8, kRun = 512, kLayers = 40;
  std::vector<uint8_t> hsrc = patterned(kRuns * kRun);
  uint8_t* dsrc = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&dsrc, kRuns * kRun) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kRuns * kRun) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(dsrc, hsrc.data(), kRuns * kRun, cudaMemcpyHostToDevice) == cudaSuccess);

  std::vector<const void*> ptrs(kRuns);
  std::vector<size_t> offs(kRuns), lens(kRuns);
  for (size_t i = 0; i < kRuns; ++i) {
    ptrs[i] = dsrc + i * kRun;
    offs[i] = i * kRun;
    lens[i] = kRun;
  }

  vkernels_status_t status = VKERNELS_OK;
  vkernels_p2p_plan_1d_t* plan =
      vkernels_p2p_plan_1d_create(ddst, kRuns * kRun, ptrs.data(), offs.data(),
                                  lens.data(), kRuns, &status);
  ASSERT_EQ(status, VKERNELS_OK);
  ASSERT_TRUE(plan != nullptr);

  // 40 sequential layer launches on one stream.
  cudaStream_t s;
  ASSERT_TRUE(cudaStreamCreate(&s) == cudaSuccess);
  for (size_t layer = 0; layer < kLayers; ++layer) {
    ASSERT_EQ(vkernels_p2p_plan_1d_execute(plan, s), VKERNELS_OK);
  }
  ASSERT_TRUE(cudaStreamSynchronize(s) == cudaSuccess);

  std::vector<uint8_t> hdst(kRuns * kRun, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kRuns * kRun, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < kRuns * kRun; ++i) ASSERT_EQ(hdst[i], hsrc[i]);

  // Concurrent execution on two streams is safe: the plan's metadata is
  // read-only after create (8 interleaved executes per stream, 3 runs).
  cudaStream_t s2;
  ASSERT_TRUE(cudaStreamCreate(&s2) == cudaSuccess);
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(vkernels_p2p_plan_1d_execute(plan, s), VKERNELS_OK);
    ASSERT_EQ(vkernels_p2p_plan_1d_execute(plan, s2), VKERNELS_OK);
  }
  ASSERT_TRUE(cudaStreamSynchronize(s) == cudaSuccess);
  ASSERT_TRUE(cudaStreamSynchronize(s2) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kRuns * kRun, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < kRuns * kRun; ++i) ASSERT_EQ(hdst[i], hsrc[i]);

  vkernels_p2p_plan_1d_destroy(plan);
  cudaStreamDestroy(s);
  cudaStreamDestroy(s2);
  cudaFree(dsrc); cudaFree(ddst);
}

TEST(P2pGatherPlan1dC, ValidationAtCreateReturnsNullWithStatus) {
  uint8_t* d0 = nullptr, *d1 = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, 32) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d1, 32) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap) == cudaSuccess);

  // Overlapping output runs: rejected at create, nothing allocated/launched.
  const void* ptrs[2] = {d0, d1};
  size_t offs[2] = {0, 8};
  size_t lens[2] = {16, 16};
  vkernels_status_t status = VKERNELS_OK;
  vkernels_p2p_plan_1d_t* plan =
      vkernels_p2p_plan_1d_create(ddst, kCap, ptrs, offs, lens, 2, &status);
  ASSERT_EQ(status, VKERNELS_ERR_INVALID_ARGUMENT);
  ASSERT_TRUE(plan == nullptr);

  // Capacity violation at create.
  size_t lens2[1] = {64};
  vkernels_status_t status2 = VKERNELS_OK;
  vkernels_p2p_plan_1d_t* plan2 =
      vkernels_p2p_plan_1d_create(ddst, 32, ptrs, offs, lens2, 1, &status2);
  ASSERT_EQ(status2, VKERNELS_ERR_INVALID_ARGUMENT);
  ASSERT_TRUE(plan2 == nullptr);

  // A valid plan still round-trips through execute after the failures.
  size_t lens3[1] = {16};
  vkernels_status_t status3 = VKERNELS_OK;
  vkernels_p2p_plan_1d_t* plan3 =
      vkernels_p2p_plan_1d_create(ddst, kCap, ptrs, offs, lens3, 1, &status3);
  ASSERT_EQ(status3, VKERNELS_OK);
  ASSERT_TRUE(plan3 != nullptr);
  ASSERT_EQ(vkernels_p2p_plan_1d_execute(plan3, 0), VKERNELS_OK);
  vkernels_p2p_plan_1d_destroy(plan3);

  cudaFree(d0); cudaFree(d1); cudaFree(ddst);
}

TEST(P2pGatherPlan2dC, PrepareAndExecuteStridedTiles) {
  // Two strided tiles with distinct sources, prepared once and executed
  // twice (a tiny stand-in for the layer-reuse pattern).
  std::vector<uint8_t> h0 = filled(64, 0x5A);  // 2 rows x 32 bytes
  std::vector<uint8_t> h1 = filled(24, 0x3C);  // 2 rows x 12 bytes
  uint8_t* d0 = nullptr, *d1 = nullptr, *ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&d0, 64) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d1, 24) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d0, h0.data(), 64, cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d1, h1.data(), 24, cudaMemcpyHostToDevice) == cudaSuccess);

  vkernels_gather_2d_run_t runs[2] = {{d0, 32, 0, 32, 32, 2},
                                       {d1, 12, 200, 12, 12, 2}};
  vkernels_status_t status = VKERNELS_OK;
  vkernels_p2p_plan_2d_t* plan =
      vkernels_p2p_plan_2d_create(ddst, kCap, runs, 2, &status);
  ASSERT_EQ(status, VKERNELS_OK);
  ASSERT_TRUE(plan != nullptr);

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(vkernels_p2p_plan_2d_execute(plan, 0), VKERNELS_OK);
  }

  std::vector<uint8_t> hdst(kCap, 0);
  ASSERT_TRUE(cudaMemcpy(hdst.data(), ddst, kCap, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t r = 0; r < 2; ++r)
    for (size_t c = 0; c < 32; ++c) ASSERT_EQ(hdst[r * 32 + c], 0x5A);
  for (size_t r = 0; r < 2; ++r)
    for (size_t c = 0; c < 12; ++c) ASSERT_EQ(hdst[200 + r * 12 + c], 0x3C);

  vkernels_p2p_plan_2d_destroy(plan);
  cudaFree(d0); cudaFree(d1); cudaFree(ddst);
}

TEST(P2pGatherC, OneRunCopiesByteExact) {
  std::vector<uint8_t> hsrc = filled(256, 0xAB);
  uint8_t* dsrc = nullptr;
  uint8_t* ddst = nullptr;
  ASSERT_TRUE(cudaMalloc(&dsrc, 256)  == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&ddst, kCap)  == cudaSuccess);
  // cudaMalloc memory is not guaranteed zeroed; the untouched-tail check
  // below needs a known fill outside the copied range.
  ASSERT_TRUE(cudaMemset(ddst, 0, kCap) == cudaSuccess);
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
  // Known fill outside the runs so the gap check is meaningful (fresh
  // cudaMalloc memory is not guaranteed zeroed).
  ASSERT_TRUE(cudaMemset(ddst, 0, kCap) == cudaSuccess);
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
