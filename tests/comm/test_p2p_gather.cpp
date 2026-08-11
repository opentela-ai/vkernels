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
using vkernels::comm::memcpy_peer_batch_async;
using vkernels::comm::p2p_gather_runs;
using vkernels::comm::p2p_gather_runs_2d;

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
