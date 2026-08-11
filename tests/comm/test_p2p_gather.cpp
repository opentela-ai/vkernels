// tests/comm/test_p2p_gather.cpp
//
// Host-reference tests for the single-launch P2P run-list gather. These are
// the correctness oracle: they exercise byte-exact results for 0/1/many runs
// (non-uniform lengths, multiple peer allocations), the 2-D strided variant,
// the "return without synchronising" stream contract, and the defining
// property of the primitive — that p2p_gather_runs issues one stream task
// regardless of run count, against the legacy memcpy_peer_batch_async seam
// which issues one per run.
#include "minitest.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

#include "vkernels/comm/p2p_gather.hpp"
#include "vkernels/core/stream.hpp"

using vkernels::Span;
using vkernels::Stream;
using vkernels::comm::Gather2DRun;
using vkernels::comm::GatherDispatchMode;
using vkernels::comm::P2PGatherPlan1D;
using vkernels::comm::P2PGatherPlan2D;
using vkernels::comm::est_copy_engine_us;
using vkernels::comm::est_gather_kernel_us;
using vkernels::comm::gather_dispatch_config;
using vkernels::comm::memcpy_peer_batch_async;
using vkernels::comm::p2p_gather_runs;
using vkernels::comm::p2p_gather_runs_2d;
using vkernels::comm::prefer_gather_kernel;
using vkernels::comm::set_gather_dispatch;

namespace {
// A small "peer allocation": distinct bytes so a mis-routed copy is obvious.
std::vector<std::uint8_t> peer(std::size_t n, std::uint8_t seed) {
  std::vector<std::uint8_t> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(seed + i);
  return v;
}

// Fill a scratch buffer with a sentinel that a real copy would never write.
std::vector<std::uint8_t> scratch(std::size_t cap, std::uint8_t fill = 0xEE) {
  return std::vector<std::uint8_t>(cap, fill);
}

Span<std::uint8_t> span(std::vector<std::uint8_t>& v) { return Span<std::uint8_t>(v); }

Span<const Gather2DRun> gruns(const Gather2DRun* p, std::size_t n) {
  return Span<const Gather2DRun>(p, n);
}

bool same(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b,
          std::size_t off, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    if (a[off + i] != b[i]) return false;
  return true;
}

// Byte-for-byte equality with a printable per-byte diagnostic (minitest can
// only stringify scalars, so vectors are compared element-wise).
void expect_equal_bytes(const std::vector<std::uint8_t>& a,
                        const std::vector<std::uint8_t>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i], b[i]);
}
}  // namespace

// ---------------------------------------------------------------------------
// 1-D: byte-exact correctness
// ---------------------------------------------------------------------------
TEST(P2pGather1d, ZeroRunsIsNoOp) {
  auto dst = scratch(16);
  auto before = dst;
  p2p_gather_runs(span(dst), nullptr, nullptr, nullptr, 0);
  expect_equal_bytes(dst, before);
}

TEST(P2pGather1d, SingleRun) {
  auto src = peer(4, 10);
  auto dst = scratch(16, 0);
  const void* ps = src.data();
  std::size_t off = 0, len = 4;
  p2p_gather_runs(span(dst), &ps, &off, &len, 1);
  EXPECT_TRUE(same(dst, src, 0, 4));
  EXPECT_EQ(dst[4], 0);
}

TEST(P2pGather1d, ManyRunsNonUniformMultiplePeers) {
  // Three disjoint runs of lengths 2, 4, 1 from three separate allocations.
  auto a = peer(2, 100);   // -> dst[0..2)
  auto b = peer(4, 200);   // -> dst[8..12)
  auto c = peer(1, 250);   // -> dst[24..25)
  auto dst = scratch(32, 0);
  const void* srcs[3] = {a.data(), b.data(), c.data()};
  std::size_t offs[3] = {0, 8, 24};
  std::size_t lens[3] = {2, 4, 1};
  p2p_gather_runs(span(dst), srcs, offs, lens, 3);
  EXPECT_TRUE(same(dst, a, 0, 2));
  EXPECT_TRUE(same(dst, b, 8, 4));
  EXPECT_TRUE(same(dst, c, 24, 1));
  // Untouched gaps keep the fill.
  EXPECT_EQ(dst[2], 0);
  EXPECT_EQ(dst[12], 0);
}

TEST(P2pGather1d, EmptyRunNeedsNoSource) {
  auto a = peer(2, 1);
  auto dst = scratch(16, 0);
  const void* srcs[2] = {a.data(), nullptr};  // run 1 is empty: null src is fine
  std::size_t offs[2] = {0, 10};
  std::size_t lens[2] = {2, 0};
  p2p_gather_runs(span(dst), srcs, offs, lens, 2);
  EXPECT_TRUE(same(dst, a, 0, 2));
  EXPECT_EQ(dst[10], 0);
}

TEST(P2pGather1d, AsyncStreamIsCorrectAfterWait) {
  auto a = peer(3, 5);
  auto b = peer(3, 9);
  auto dst = scratch(16, 0);
  const void* srcs[2] = {a.data(), b.data()};
  std::size_t offs[2] = {0, 4};
  std::size_t lens[2] = {3, 3};
  Stream s;
  std::size_t before = s.submitted();
  p2p_gather_runs(span(dst), srcs, offs, lens, 2, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one task, not one per run
  s.wait();
  EXPECT_TRUE(same(dst, a, 0, 3));
  EXPECT_TRUE(same(dst, b, 4, 3));
}

// ---------------------------------------------------------------------------
// 1-D: contract violations
// ---------------------------------------------------------------------------
TEST(P2pGather1d, NullSrcPtrsThrows) {
  auto dst = scratch(8);
  std::size_t o = 0, l = 1;
  EXPECT_THROW(p2p_gather_runs(span(dst), nullptr, &o, &l, 1), std::invalid_argument);
}

TEST(P2pGather1d, NullDstOffsetsThrows) {
  auto dst = scratch(8);
  const void* ps = dst.data();
  std::size_t l = 1;
  EXPECT_THROW(p2p_gather_runs(span(dst), &ps, nullptr, &l, 1), std::invalid_argument);
}

TEST(P2pGather1d, NullLengthsThrows) {
  auto dst = scratch(8);
  const void* ps = dst.data();
  std::size_t o = 0;
  EXPECT_THROW(p2p_gather_runs(span(dst), &ps, &o, nullptr, 1), std::invalid_argument);
}

TEST(P2pGather1d, RunExceedsCapacityThrows) {
  auto dst = scratch(4, 0);
  const void* ps = dst.data();
  std::size_t o = 0, l = 8;
  EXPECT_THROW(p2p_gather_runs(span(dst), &ps, &o, &l, 1), std::invalid_argument);
}

TEST(P2pGather1d, NullSrcForNonEmptyRunThrows) {
  auto dst = scratch(8, 0);
  const void* srcs[1] = {nullptr};
  std::size_t o = 0, l = 4;
  EXPECT_THROW(p2p_gather_runs(span(dst), srcs, &o, &l, 1), std::invalid_argument);
}

TEST(P2pGather1d, SrcOverlapsDstThrows) {
  auto dst = scratch(16, 0);
  const void* srcs[1] = {dst.data() + 4};  // overlaps dst[0..8)
  std::size_t o = 0, l = 8;
  EXPECT_THROW(p2p_gather_runs(span(dst), srcs, &o, &l, 1), std::invalid_argument);
}

TEST(P2pGather1d, OverlappingOutputRunsThrow) {
  auto src = peer(8, 1);
  auto dst = scratch(16, 0);
  const void* srcs[2] = {src.data(), src.data()};
  std::size_t offs[2] = {0, 4};   // [0,8) and [4,12) overlap
  std::size_t lens[2] = {8, 8};
  EXPECT_THROW(p2p_gather_runs(span(dst), srcs, offs, lens, 2), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Legacy seam: one copy per run (the overhead p2p_gather_runs removes)
// ---------------------------------------------------------------------------
TEST(MemcpyPeerBatch, OneSubmitPerRun) {
  auto a = peer(2, 1);
  auto b = peer(2, 9);
  auto dst = scratch(16, 0);
  const void* srcs[2] = {a.data(), b.data()};
  std::size_t offs[2] = {0, 4};
  std::size_t lens[2] = {2, 2};
  Stream s;
  std::size_t before = s.submitted();
  memcpy_peer_batch_async(span(dst), srcs, offs, lens, 2, &s);
  EXPECT_EQ(s.submitted() - before, 2u);  // one task per run
  s.wait();
  EXPECT_TRUE(same(dst, a, 0, 2));
  EXPECT_TRUE(same(dst, b, 4, 2));
}

TEST(MemcpyPeerBatch, SyncDefaultStream) {
  auto a = peer(2, 1);
  auto b = peer(2, 9);
  auto dst = scratch(16, 0);
  const void* srcs[2] = {a.data(), b.data()};
  std::size_t offs[2] = {0, 4};
  std::size_t lens[2] = {2, 2};
  memcpy_peer_batch_async(span(dst), srcs, offs, lens, 2);  // stream == nullptr
  EXPECT_TRUE(same(dst, a, 0, 2));
  EXPECT_TRUE(same(dst, b, 4, 2));
}

TEST(MemcpyPeerBatch, ZeroRunsIsNoOp) {
  auto dst = scratch(8);
  auto before = dst;
  Stream s;
  memcpy_peer_batch_async(span(dst), nullptr, nullptr, nullptr, 0, &s);
  expect_equal_bytes(dst, before);
}

TEST(NoPerRunCalls, GatherIsOneLaunchBatchIsOnePerRun) {
  // The defining property: a single-launch gather submits exactly one task
  // for N runs; the legacy seam submits N. Same metadata in both cases.
  auto a = peer(2, 1);
  auto b = peer(2, 9);
  auto c = peer(2, 17);
  const void* srcs[3] = {a.data(), b.data(), c.data()};
  std::size_t offs[3] = {0, 4, 8};
  std::size_t lens[3] = {2, 2, 2};

  Stream gs, bs;
  auto g_dst = scratch(16, 0);
  std::size_t g_before = gs.submitted();
  p2p_gather_runs(span(g_dst), srcs, offs, lens, 3, &gs);
  EXPECT_EQ(gs.submitted() - g_before, 1u);

  auto b_dst = scratch(16, 0);
  std::size_t b_before = bs.submitted();
  memcpy_peer_batch_async(span(b_dst), srcs, offs, lens, 3, &bs);
  EXPECT_EQ(bs.submitted() - b_before, 3u);

  gs.wait();
  bs.wait();
  EXPECT_TRUE(same(g_dst, a, 0, 2));
  EXPECT_TRUE(same(g_dst, b, 4, 2));
  EXPECT_TRUE(same(g_dst, c, 8, 2));
  expect_equal_bytes(g_dst, b_dst);  // byte-exact equivalence
}

// ---------------------------------------------------------------------------
// 2-D: byte-exact correctness with per-run source stride
// ---------------------------------------------------------------------------
TEST(P2pGather2d, SingleTileContiguous) {
  auto src = peer(16, 32);  // 4x4, stride 4
  auto dst = scratch(64, 0);
  Gather2DRun runs[1] = {{src.data(), 4, 0, 4, 4, 4}};
  p2p_gather_runs_2d(span(dst), gruns(runs, 1));
  EXPECT_TRUE(same(dst, src, 0, 16));
}

TEST(P2pGather2d, StridedSourceRows) {
  // Source is a 3-row tile in a wider row-major region (src stride 8); the
  // run copies only the first 3 bytes of each of 3 rows into a 4-byte-stride
  // destination. This is the layer-major page layout the 2-D variant exists
  // for: the per-run source stride is honoured without host staging.
  auto src = peer(24, 50);  // 3 rows x 8 bytes
  auto dst = scratch(32, 0);
  Gather2DRun runs[1] = {{src.data(), 8, 0, 4, 3, 3}};
  p2p_gather_runs_2d(span(dst), gruns(runs, 1));
  for (std::size_t y = 0; y < 3; ++y)
    for (std::size_t x = 0; x < 3; ++x)
      ASSERT_EQ(dst[y * 4 + x], src[y * 8 + x]);
  // Untouched padding in each destination row stays zero.
  ASSERT_EQ(dst[3], 0);
  ASSERT_EQ(dst[7], 0);
}

TEST(P2pGather2d, ManyTilesDifferentPeers) {
  auto a = peer(16, 1);  // 4x4 tile
  auto b = peer(12, 9);  // 3x3 tile in 3x4 region (src stride 4)
  auto dst = scratch(64, 0);
  Gather2DRun runs[2] = {{a.data(), 4, 0, 4, 4, 4},
                         {b.data(), 4, 24, 4, 3, 3}};
  p2p_gather_runs_2d(span(dst), gruns(runs, 2));
  EXPECT_TRUE(same(dst, a, 0, 16));
  for (std::size_t y = 0; y < 3; ++y)
    for (std::size_t x = 0; x < 3; ++x)
      ASSERT_EQ(dst[24 + y * 4 + x], b[y * 4 + x]);
}

TEST(P2pGather2d, EmptyTileIsSkipped) {
  auto a = peer(4, 1);
  auto dst = scratch(16, 0);
  Gather2DRun runs[2] = {{a.data(), 2, 0, 2, 2, 2},   // 2x2 real tile
                         {nullptr, 0, 8, 0, 0, 3}};   // zero-width: skipped
  p2p_gather_runs_2d(span(dst), gruns(runs, 2));
  EXPECT_TRUE(same(dst, a, 0, 4));
  EXPECT_EQ(dst[8], 0);
}

TEST(P2pGather2d, ZeroTilesIsNoOp) {
  auto dst = scratch(16);
  auto before = dst;
  p2p_gather_runs_2d(span(dst), {});
  expect_equal_bytes(dst, before);
}

TEST(P2pGather2d, AsyncIsCorrectAfterWait) {
  auto src = peer(16, 32);
  auto dst = scratch(64, 0);
  Gather2DRun runs[1] = {{src.data(), 4, 0, 4, 4, 4}};
  Stream s;
  std::size_t before = s.submitted();
  p2p_gather_runs_2d(span(dst), gruns(runs, 1), &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // single launch for the whole tile list
  s.wait();
  EXPECT_TRUE(same(dst, src, 0, 16));
}

// ---------------------------------------------------------------------------
// 2-D: contract violations
// ---------------------------------------------------------------------------
TEST(P2pGather2d, WidthExceedsSrcStrideThrows) {
  auto src = peer(8, 1);
  auto dst = scratch(32, 0);
  Gather2DRun runs[1] = {{src.data(), 2, 0, 2, 4, 1}};  // width 4 > src_stride 2
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(runs, 1)), std::invalid_argument);
}

TEST(P2pGather2d, WidthExceedsDstStrideThrows) {
  auto src = peer(8, 1);
  auto dst = scratch(32, 0);
  Gather2DRun bad[1] = {{src.data(), 2, 0, 1, 2, 1}};   // width 2 > dst_stride 1
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(bad, 1)), std::invalid_argument);
}

TEST(P2pGather2d, NullSrcThrows) {
  auto dst = scratch(32, 0);
  Gather2DRun runs[1] = {{nullptr, 2, 0, 2, 2, 2}};
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(runs, 1)), std::invalid_argument);
}

TEST(P2pGather2d, StartsPastCapacityThrows) {
  auto src = peer(8, 1);
  auto dst = scratch(8, 0);
  Gather2DRun runs[1] = {{src.data(), 2, 10, 2, 2, 1}};  // dst_offset 10 > cap 8
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(runs, 1)), std::invalid_argument);
}

TEST(P2pGather2d, RowStrideExceedsCapacityThrows) {
  auto src = peer(64, 1);
  auto dst = scratch(100, 0);
  Gather2DRun runs[1] = {{src.data(), 64, 0, 64, 8, 3}};  // 2 rows * 64 = 128 > 100
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(runs, 1)), std::invalid_argument);
}

TEST(P2pGather2d, LastRowExceedsCapacityThrows) {
  auto src = peer(64, 1);
  auto dst = scratch(100, 0);
  // row0 [0,30), row1 [80,110) -> last byte 109 > 99, but rows*stride=80<=100
  // so it passes the row-stride check and fails the last-row check.
  Gather2DRun runs[1] = {{src.data(), 64, 0, 80, 30, 2}};
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(runs, 1)), std::invalid_argument);
}

// 2-D now mirrors 1-D: per-tile src/dst non-overlap and mutually-disjoint
// output tiles, so the concurrent CUDA kernel (no inter-run ordering) can
// trust the staged metadata.
TEST(P2pGather2d, SrcOverlapsDstThrows) {
  // Source tile points into the destination buffer (4 bytes in), so the
  // src and dst bounding boxes overlap.
  auto dst = scratch(64, 0);
  Gather2DRun runs[1] = {{dst.data() + 4, 8, 0, 8, 4, 4}};
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(runs, 1)), std::invalid_argument);
}

TEST(P2pGather2d, OverlappingOutputTilesThrow) {
  // Two tiles with overlapping destination bounding boxes: tile 0 owns
  // [0,12), tile 1 owns [8,20) — they overlap at [8,12).
  auto a = peer(32, 1);
  auto b = peer(32, 9);
  auto dst = scratch(64, 0);
  Gather2DRun runs[2] = {{a.data(), 8, 0, 8, 4, 2},   // dst box [0, 12)
                         {b.data(), 8, 8, 8, 4, 2}};  // dst box [8, 20)
  EXPECT_THROW(p2p_gather_runs_2d(span(dst), gruns(runs, 2)), std::invalid_argument);
}

TEST(P2pGather2d, AdjacentTilesAreAllowed) {
  // Tile 0 owns [0,12), tile 1 owns [12,24): adjacent, not overlapping, so
  // the disjointness check must accept them and copy both byte-exact.
  auto a = peer(16, 1);
  auto b = peer(16, 9);
  auto dst = scratch(64, 0);
  Gather2DRun runs[2] = {{a.data(), 8, 0, 8, 4, 2},    // dst box [0, 12)
                         {b.data(), 8, 12, 8, 4, 2}};  // dst box [12, 24)
  p2p_gather_runs_2d(span(dst), gruns(runs, 2));
  for (std::size_t y = 0; y < 2; ++y)
    for (std::size_t x = 0; x < 4; ++x) {
      ASSERT_EQ(dst[y * 8 + x], a[y * 8 + x]);
      ASSERT_EQ(dst[12 + y * 8 + x], b[y * 8 + x]);
    }
}

// ---------------------------------------------------------------------------
// Adaptive dispatch policy (pure, host-testable)
// ---------------------------------------------------------------------------
namespace {
constexpr std::size_t k48MiB = 48u * 1024u * 1024u;
}

TEST(GatherDispatch, DefaultsAreAdaptiveWithFloor4) {
  auto [mode, min_runs] = gather_dispatch_config();
  EXPECT_EQ(static_cast<int>(mode), static_cast<int>(GatherDispatchMode::kAdaptive));
  EXPECT_EQ(min_runs, 4u);
}

TEST(GatherDispatch, CostModelMatchesH100NVLMeasurements) {
  // Copy engine: 201.7 us @ 1 run / 48 MiB + ~7.37 us per extra run, with
  // a ~20 us per-call floor (measured on sgs-gpu07, real NVLink peer).
  EXPECT_NEAR(est_copy_engine_us(1, k48MiB), 201.6, 1.0);
  EXPECT_NEAR(est_copy_engine_us(192, k48MiB), 1609.3, 1.0);
  EXPECT_NEAR(est_copy_engine_us(16, k48MiB), 312.2, 1.0);
  EXPECT_NEAR(est_copy_engine_us(1, 4096), 20.0, 0.5);  // fixed floor
  EXPECT_EQ(est_copy_engine_us(0, 0), 0.0);
  EXPECT_EQ(est_copy_engine_us(1, 0), 0.0);  // one run, zero bytes: no cost

  // Gather kernel: ~210 us flat at 48 MiB, independent of run count; the
  // fixed launch floor covers tiny transfers (8.58 us measured at 4 KiB).
  EXPECT_NEAR(est_gather_kernel_us(1, k48MiB), 210.2, 1.0);
  EXPECT_NEAR(est_gather_kernel_us(192, k48MiB), 210.2, 1.0);
  EXPECT_NEAR(est_gather_kernel_us(1, 4096), 8.6, 0.5);
  EXPECT_EQ(est_gather_kernel_us(1, 0), 0.0);

  // Monotonicity: the loop grows with run count, the kernel does not.
  EXPECT_LT(est_copy_engine_us(1, k48MiB), est_copy_engine_us(2, k48MiB));
  EXPECT_EQ(est_gather_kernel_us(1, k48MiB), est_gather_kernel_us(2, k48MiB));
}

TEST(GatherDispatch, AdaptiveChoosesKernelAboveMeasuredCrossover) {
  set_gather_dispatch(GatherDispatchMode::kAdaptive, 4);
  // 1-3 runs at 48 MiB: copy engine — the floor guards the ~1% margins
  // measured below the ~3-run crossover.
  for (std::size_t n : {1u, 2u, 3u}) {
    EXPECT_FALSE(prefer_gather_kernel(n, k48MiB));
  }
  // From 4 runs the model wins: kernel was 1.07x/1.23x/1.49x/2.17x/3.59x/
  // 8.40x over the copy loop at 4/8/16/32/64/192 runs (measured).
  for (std::size_t n : {4u, 8u, 16u, 32u, 64u, 192u}) {
    EXPECT_TRUE(prefer_gather_kernel(n, k48MiB));
  }
}

TEST(GatherDispatch, FloorHoldsBelowMinRunsForLargePayloads) {
  set_gather_dispatch(GatherDispatchMode::kAdaptive, 4);
  // 48 MiB (>= the 1 MiB floor threshold): below 4 runs the floor forces
  // the copy engine even where the model's margin is small.
  EXPECT_FALSE(prefer_gather_kernel(2, k48MiB));
  EXPECT_FALSE(prefer_gather_kernel(3, k48MiB));
  EXPECT_TRUE(prefer_gather_kernel(4, k48MiB));
  // Below the threshold the copy engine never wins: the model decides from
  // one run (2.3x kernel advantage measured at 4 KiB).
  EXPECT_TRUE(prefer_gather_kernel(1, 4096));
  EXPECT_TRUE(prefer_gather_kernel(2, 4096));
}

TEST(GatherDispatch, ZeroBytesNeverTakesTheKernel) {
  set_gather_dispatch(GatherDispatchMode::kAdaptive, 1);
  EXPECT_FALSE(prefer_gather_kernel(100, 0));
  EXPECT_FALSE(prefer_gather_kernel(1, 0));
}

TEST(GatherDispatch, ForcedModesOverrideTheModel) {
  set_gather_dispatch(GatherDispatchMode::kForceKernel, 24);
  EXPECT_TRUE(prefer_gather_kernel(1, 1024));  // kernel even for one tiny run
  EXPECT_TRUE(prefer_gather_kernel(1, k48MiB));
  set_gather_dispatch(GatherDispatchMode::kForceCopyEngine, 24);
  EXPECT_FALSE(prefer_gather_kernel(192, k48MiB));  // loop even at 192 runs
  EXPECT_FALSE(prefer_gather_kernel(0, k48MiB));

  // Setter round-trips both parameters.
  set_gather_dispatch(GatherDispatchMode::kAdaptive, 7);
  auto [mode, min_runs] = gather_dispatch_config();
  EXPECT_EQ(static_cast<int>(mode), static_cast<int>(GatherDispatchMode::kAdaptive));
  EXPECT_EQ(min_runs, 7u);
  set_gather_dispatch();  // restore the defaults for later tests
  auto [mode2, min_runs2] = gather_dispatch_config();
  EXPECT_EQ(static_cast<int>(mode2), static_cast<int>(GatherDispatchMode::kAdaptive));
  EXPECT_EQ(min_runs2, 4u);
}

// Host mirror of the CUDA kernel's grid-sizing contract (p2p_gather.cu): the
// 1-D kernel is launched with grid.y = num_runs and grid.x = tiles(max_units)
// where a vectorized run occupies ceil(length/16) units and a scalar run
// occupies `length` units. Every run's units must fit inside grid.x*256
// threads, and the vectorized tail thread (index length/16) must exist in
// the grid so the <16-byte tail is always copied. This test locks that
// invariant for the boundary lengths around the 16-byte chunk size.
TEST(GatherDispatch, KernelGridContractCoversEveryRun) {
  const std::size_t lens[] = {1u, 15u, 16u, 17u, 31u, 32u, 47u, 48u, 49u, 63u,
                              64u, 65u, 255u, 256u, 257u, 4095u, 4096u, 4097u};
  const bool vecs[] = {false, true};
  for (std::size_t len : lens) {
    for (bool vec : vecs) {
      // units_1d: vectorized -> ceil(len/16), scalar -> len.
      const std::size_t units = vec ? (len + 15u) / 16u : len;
      // tiles(): ceil(units / 256) blocks of 256 threads, at least one.
      const std::size_t blocks = units > 0 ? (units + 255u) / 256u : 1u;
      const std::size_t threads = blocks * 256u;
      // Every unit index is covered.
      EXPECT_GE(threads, units);
      // A vectorized run with a <16-byte tail needs its tail thread (index
      // len/16) inside the grid; lengths that are multiples of 16 have no
      // tail and need no such thread.
      if (vec && (len & 15u) != 0u) {
        EXPECT_LT(len / 16u, threads);
        EXPECT_LE((len / 16u) * 16u, len);  // tail offset stays in range
      }
      // Scalar path: the last byte index len-1 is covered.
      if (!vec && len > 0) EXPECT_LT(len - 1u, threads);
    }
  }
}

// ---------------------------------------------------------------------------
// Prepared plan API (host reference): validate once, execute many
// ---------------------------------------------------------------------------
TEST(P2PGatherPlan1d, PrepareOnceExecuteReusesNoValidation) {
  auto a = peer(2, 100);   // -> dst[0..2)
  auto b = peer(4, 200);   // -> dst[8..12)
  auto c = peer(1, 250);   // -> dst[24..25)
  auto dst = scratch(32, 0);
  const void* srcs[3] = {a.data(), b.data(), c.data()};
  std::size_t offs[3] = {0, 8, 24};
  std::size_t lens[3] = {2, 4, 1};

  P2PGatherPlan1D plan(dst.data(), dst.size(), srcs, offs, lens, 3);
  EXPECT_EQ(plan.num_runs(), 3u);
  EXPECT_EQ(plan.total_bytes(), 7u);
  EXPECT_EQ(plan.dst(), dst.data());
  EXPECT_EQ(plan.dst_capacity(), 32u);

  // The KVAAS pattern: one run list reused for 40 layer launches, each a
  // single stream task with no per-call metadata work.
  Stream s;
  for (int layer = 0; layer < 40; ++layer) {
    std::size_t before = s.submitted();
    plan.execute(&s);
    EXPECT_EQ(s.submitted() - before, 1u);  // exactly one task per execute
  }
  s.wait();
  EXPECT_TRUE(same(dst, a, 0, 2));
  EXPECT_TRUE(same(dst, b, 8, 4));
  EXPECT_TRUE(same(dst, c, 24, 1));
  EXPECT_EQ(dst[2], 0);  // untouched gaps keep the fill
  EXPECT_EQ(dst[12], 0);
}

TEST(P2PGatherPlan1d, ExecuteWithoutStreamRunsSynchronously) {
  auto a = peer(3, 5);
  auto dst = scratch(16, 0);
  const void* srcs[1] = {a.data()};
  std::size_t offs[1] = {0};
  std::size_t lens[1] = {3};
  P2PGatherPlan1D plan(dst.data(), dst.size(), srcs, offs, lens, 1);
  plan.execute();  // stream == nullptr: runs to completion
  EXPECT_TRUE(same(dst, a, 0, 3));
}

TEST(P2PGatherPlan1d, ConcurrentExecutesOnTwoStreams) {
  auto a = peer(2, 1);
  auto b = peer(2, 9);
  auto dst = scratch(16, 0);
  const void* srcs[2] = {a.data(), b.data()};
  std::size_t offs[2] = {0, 4};
  std::size_t lens[2] = {2, 2};
  P2PGatherPlan1D plan(dst.data(), dst.size(), srcs, offs, lens, 2);

  // Two streams submit the same (read-only) plan concurrently; the host
  // Stream model runs tasks on separate worker threads, so this is a real
  // concurrency exercise of the reuse contract.
  Stream s1, s2;
  for (int i = 0; i < 8; ++i) {
    plan.execute(&s1);
    plan.execute(&s2);
  }
  s1.wait();
  s2.wait();
  EXPECT_TRUE(same(dst, a, 0, 2));
  EXPECT_TRUE(same(dst, b, 4, 2));
}

TEST(P2PGatherPlan1d, ValidationHappensOnceAtPrepare) {
  auto src = peer(8, 1);
  auto dst = scratch(16, 0);

  // Null source for a non-empty run.
  const void* bad_srcs[1] = {nullptr};
  std::size_t o = 0, l = 4;
  EXPECT_THROW(P2PGatherPlan1D(dst.data(), dst.size(), bad_srcs, &o, &l, 1),
               std::invalid_argument);

  // Capacity violation.
  const void* srcs[1] = {src.data()};
  std::size_t big = 32;
  EXPECT_THROW(P2PGatherPlan1D(dst.data(), dst.size(), srcs, &o, &big, 1),
               std::invalid_argument);

  // Overlapping output runs.
  const void* two[2] = {src.data(), src.data()};
  std::size_t offs[2] = {0, 4};
  std::size_t lens[2] = {8, 8};
  EXPECT_THROW(P2PGatherPlan1D(dst.data(), dst.size(), two, offs, lens, 2),
               std::invalid_argument);
}

TEST(P2PGatherPlan1d, EmptyRunListIsANoOpPlan) {
  auto dst = scratch(8, 0);
  P2PGatherPlan1D plan(dst.data(), dst.size(), nullptr, nullptr, nullptr, 0);
  EXPECT_EQ(plan.num_runs(), 0u);
  EXPECT_EQ(plan.total_bytes(), 0u);
  Stream s;
  std::size_t before = s.submitted();
  plan.execute(&s);
  EXPECT_EQ(s.submitted() - before, 0u);  // nothing enqueued
  s.wait();
  EXPECT_EQ(dst[0], 0);
}

TEST(P2PGatherPlan2d, PrepareOnceExecuteManyTiles) {
  auto a = peer(16, 1);  // 4x4 tile
  auto b = peer(12, 9);  // 3x3 tile in 3x4 region (src stride 4)
  auto dst = scratch(64, 0);
  Gather2DRun runs[2] = {{a.data(), 4, 0, 4, 4, 4},
                         {b.data(), 4, 24, 4, 3, 3}};
  P2PGatherPlan2D plan(dst.data(), dst.size(), runs, 2);
  EXPECT_EQ(plan.num_runs(), 2u);
  EXPECT_EQ(plan.total_bytes(), 25u);

  Stream s;
  for (int i = 0; i < 5; ++i) {
    std::size_t before = s.submitted();
    plan.execute(&s);
    EXPECT_EQ(s.submitted() - before, 1u);
  }
  s.wait();
  EXPECT_TRUE(same(dst, a, 0, 16));
  for (std::size_t y = 0; y < 3; ++y)
    for (std::size_t x = 0; x < 3; ++x)
      ASSERT_EQ(dst[24 + y * 4 + x], b[y * 4 + x]);
}

TEST(P2PGatherPlan2d, SyncExecuteAndValidation) {
  auto src = peer(8, 1);
  auto dst = scratch(8, 0);
  Gather2DRun ok[1] = {{src.data(), 4, 0, 4, 4, 1}};
  P2PGatherPlan2D plan(dst.data(), dst.size(), ok, 1);
  plan.execute();  // synchronous
  EXPECT_TRUE(same(dst, src, 0, 4));

  // A bad 2-D list is rejected at prepare, not at execute.
  auto dst2 = scratch(8, 0);
  Gather2DRun bad[1] = {{nullptr, 4, 0, 4, 4, 1}};
  EXPECT_THROW(P2PGatherPlan2D(dst2.data(), dst2.size(), bad, 1),
               std::invalid_argument);
}
