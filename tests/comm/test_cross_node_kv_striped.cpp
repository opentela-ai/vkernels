// tests/comm/test_cross_node_kv_striped.cpp
//
// Host-reference tests for the 2-rank multi-HCA striped variant (issue
// #152). The deterministic chunk layout, the per-lane link plumbing, and
// the striped restore / donate plans are pinned here against the SAME
// byte-exact contract as the single-lane host-bounce path:
//
//   * The chunk layout tiles [0, total_bytes) contiguously, one chunk per
//     lane (remainder folded into the first lanes), and collapses to a
//     single lane below lane_count * min_chunk_bytes.
//   * A striped donate -> striped restore round-trip produces byte-identical
//     K/V to the single-lane bounce round-trip (the chunks concatenate to
//     the same bytes, so kv_scatter sees the same payload).
//   * The traffic is actually distributed: lane l carries exactly chunk l.
//   * Geometry mismatches cannot silently corrupt (per-chunk size check).
//   * The striped path eager-breaks a GraphCapture exactly like the
//     single-lane bounce path (#10) and records no graph nodes.
//
// The real-RDMA-fabric striping measurement (per-HCA QP lanes, 2-node HDR)
// is the on-site step on H-JSC; the bench carries the mode
// (bench_cross_node_nccl.cu, nccl-stripe rows). The host oracle here is
// the byte-exact model the lane transport must reproduce.
#include "minitest.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "vkernels/comm/cross_node_kv.hpp"

using vkernels::Stream;
using vkernels::comm::ByteBlockingQueue;
using vkernels::comm::ByteChannel;
using vkernels::comm::compute_cross_node_kv_stripes;
using vkernels::comm::CrossNodeKvDonatePlan;
using vkernels::comm::CrossNodeKvRestorePlan;
using vkernels::comm::CrossNodeKvStripeChunk;
using vkernels::comm::CrossNodeKvStripeConfig;
using vkernels::comm::CrossNodeKvStripedDonatePlan;
using vkernels::comm::CrossNodeKvStripedRestorePlan;
using vkernels::comm::FabricHandle;
using vkernels::comm::GraphCapture;
using vkernels::comm::make_byte_link;
using vkernels::comm::make_striped_byte_links;

namespace {

// Same layout constants as test_cross_node_kv.cpp (one shape, BF16).
constexpr std::size_t kPages = 4, kPageSize = 3, kHeads = 2, kHeadDim = 4;
constexpr std::size_t kElem = 2, kSlots = 16;
constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;      // 16
constexpr std::size_t kTokenStride = 2 * kSlotBytes;               // 32
constexpr std::size_t kPageLayerBytes = kPageSize * kTokenStride;  // 96
constexpr std::size_t kLayerBytes = kPages * kPageLayerBytes;      // 384

const int* unique_slots() {
  static const int s[kPages * kPageSize] = {3, 1, 14, 7, 0, 9, 6, 12, 2, 5, 8, 11};
  return s;
}

// Non-unique map: the donate's gather semantics accept repeats.
const int* repeat_slots() {
  static const int s[kPages * kPageSize] = {0, 1, 0,  4, 5, 4,  8, 9, 8,  12, 13, 12};
  return s;
}

struct LocalKv {
  std::vector<std::uint8_t> k, v;
};

LocalKv make_local(std::uint8_t kseed = 0x80, std::uint8_t vseed = 0xC0) {
  const std::size_t n = kSlots * kHeads * kHeadDim * kElem;
  LocalKv kv;
  kv.k.resize(n);
  kv.v.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    kv.k[i] = static_cast<std::uint8_t>(kseed + i);
    kv.v[i] = static_cast<std::uint8_t>(vseed + i);
  }
  return kv;
}

bool same_bytes(const std::vector<std::uint8_t>& a,
                const std::vector<std::uint8_t>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size()) == 0;
}

// A forwarding ByteChannel that counts per-lane traffic -- the observable
// that proves the chunks are actually distributed across the lanes.
struct CountingChannel : ByteChannel {
  ByteChannel* inner;
  std::size_t sent_chunks = 0, sent_bytes = 0;
  std::size_t recv_chunks = 0, recv_bytes = 0;
  explicit CountingChannel(ByteChannel* c) : inner(c) {}
  void send(std::vector<std::uint8_t> chunk) override {
    sent_chunks += 1;
    sent_bytes += chunk.size();
    inner->send(std::move(chunk));
  }
  std::vector<std::uint8_t> recv() override {
    std::vector<std::uint8_t> chunk = inner->recv();
    recv_chunks += 1;
    recv_bytes += chunk.size();
    return chunk;
  }
  bool closed() const override { return inner->closed(); }
};

// A tiny payload distinct from every K/V fill pattern, for mismatch tests.
std::vector<std::uint8_t> payload(std::size_t bytes, std::uint8_t tag) {
  return std::vector<std::uint8_t>(bytes, tag);
}

}  // namespace

// ---------------------------------------------------------------------------
// Deterministic chunk layout
// ---------------------------------------------------------------------------

TEST(StripedLayout, EqualSplitTilesExactly) {
  // 384 bytes over 4 lanes: 96 per lane, contiguous, lane l = chunk l.
  auto chunks = compute_cross_node_kv_stripes(
      kLayerBytes, CrossNodeKvStripeConfig{4, 8});
  ASSERT_EQ(chunks.size(), 4u);
  std::size_t offset = 0;
  for (std::size_t l = 0; l < 4; ++l) {
    EXPECT_EQ(chunks[l].offset, offset);
    EXPECT_EQ(chunks[l].bytes, kLayerBytes / 4);
    EXPECT_EQ(chunks[l].lane, l);
    offset += chunks[l].bytes;
  }
  EXPECT_EQ(offset, kLayerBytes);
}

TEST(StripedLayout, RemainderFoldsIntoFirstLanes) {
  // 1002 bytes over 4 lanes: lanes 0..1 get 251, lanes 2..3 get 250, and
  // the chunks still tile [0, 1002) contiguously in order.
  auto chunks = compute_cross_node_kv_stripes(
      1002, CrossNodeKvStripeConfig{4, 8});
  ASSERT_EQ(chunks.size(), 4u);
  EXPECT_EQ(chunks[0].bytes, 251u);
  EXPECT_EQ(chunks[1].bytes, 251u);
  EXPECT_EQ(chunks[2].bytes, 250u);
  EXPECT_EQ(chunks[3].bytes, 250u);
  std::size_t offset = 0;
  for (const CrossNodeKvStripeChunk& c : chunks) {
    EXPECT_EQ(c.offset, offset);
    offset += c.bytes;
  }
  EXPECT_EQ(offset, 1002u);
}

TEST(StripedLayout, SmallTransferCollapsesToOneLane) {
  // 100 bytes < 4 lanes * 64 min: a single chunk on lane 0 (issue #152:
  // small transfers stay on a single QP to avoid per-lane setup).
  auto chunks = compute_cross_node_kv_stripes(
      100, CrossNodeKvStripeConfig{4, 64});
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0].offset, 0u);
  EXPECT_EQ(chunks[0].bytes, 100u);
  EXPECT_EQ(chunks[0].lane, 0u);
}

TEST(StripedLayout, ExactlyMinPerLaneSplits) {
  // total == lane_count * min is NOT below the threshold: split.
  auto chunks = compute_cross_node_kv_stripes(
      256, CrossNodeKvStripeConfig{4, 64});
  ASSERT_EQ(chunks.size(), 4u);
  for (const CrossNodeKvStripeChunk& c : chunks) EXPECT_EQ(c.bytes, 64u);
}

TEST(StripedLayout, ZeroBytesIsEmpty) {
  auto chunks = compute_cross_node_kv_stripes(
      0, CrossNodeKvStripeConfig{4, 64});
  EXPECT_TRUE(chunks.empty());
}

TEST(StripedLayout, RejectsInvalidConfig) {
  EXPECT_THROW(compute_cross_node_kv_stripes(
                   100, CrossNodeKvStripeConfig{0, 64}),
               std::invalid_argument);
  EXPECT_THROW(compute_cross_node_kv_stripes(
                   100, CrossNodeKvStripeConfig{4, 0}),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Per-lane link plumbing
// ---------------------------------------------------------------------------

TEST(StripedLinks, BuildNIndependentLanes) {
  auto link = make_striped_byte_links(3);
  ASSERT_EQ(link.first.size(), 3u);
  ASSERT_EQ(link.second.size(), 3u);
  // Lane 1 carries its own bytes; lanes 0 and 2 stay empty.
  link.first[1]->send(payload(4, 0x5A));
  EXPECT_TRUE(same_bytes(link.second[1]->recv(), payload(4, 0x5A)));
  EXPECT_FALSE(link.second[0]->closed());
  EXPECT_FALSE(link.second[2]->closed());
}

TEST(StripedLinks, RejectsZeroLanes) {
  EXPECT_THROW(make_striped_byte_links(0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Striped restore / donate plans
// ---------------------------------------------------------------------------

TEST(StripedRoundTrip, MatchesSingleLaneBounceByteForByte) {
  auto local = make_local();
  const int* slots = unique_slots();

  // Single-lane reference round-trip (the existing host-bounce path).
  auto ref_link = make_byte_link();
  CrossNodeKvDonatePlan donor1(kSlots, kHeads, kHeadDim, kElem, slots,
                               kPages, kPageSize,
                               vkernels::comm::FabricImportTransport::kHostBounce,
                               nullptr);
  donor1.execute(local.k.data(), local.v.data(), /*remote=*/nullptr, 0,
                 /*stream=*/nullptr, /*graph=*/nullptr, ref_link.first.get());
  CrossNodeKvRestorePlan restorer1(kSlots, kHeads, kHeadDim, kElem, slots,
                                   kPages, kPageSize,
                                   vkernels::comm::FabricImportTransport::kHostBounce,
                                   nullptr);
  auto ref = make_local(0, 0); ref.k.assign(ref.k.size(), 0); ref.v.assign(ref.v.size(), 0);
  restorer1.execute(ref.k.data(), ref.v.data(), 0, nullptr, nullptr,
                    ref_link.second.get());

  // Striped round-trip over 4 lanes (min_chunk small enough to split 384).
  const CrossNodeKvStripeConfig stripe{4, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  std::vector<ByteChannel*> a, b;
  for (auto& c : lanes.first) a.push_back(c.get());
  for (auto& c : lanes.second) b.push_back(c.get());

  CrossNodeKvStripedDonatePlan sdonor(kSlots, kHeads, kHeadDim, kElem, slots,
                                      kPages, kPageSize, stripe);
  sdonor.execute(local.k.data(), local.v.data(), /*remote=*/nullptr, 0,
                 nullptr, nullptr, a);
  CrossNodeKvStripedRestorePlan srestorer(kSlots, kHeads, kHeadDim, kElem,
                                          slots, kPages, kPageSize, stripe);
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  srestorer.execute(out.k.data(), out.v.data(), 0, nullptr, nullptr, b);

  // Byte-identical to the single-lane bounce result.
  ASSERT_TRUE(same_bytes(out.k, ref.k));
  ASSERT_TRUE(same_bytes(out.v, ref.v));
  // The mapped slots landed (slots outside the map stay zeroed).
  EXPECT_TRUE(std::any_of(out.k.begin(), out.k.end(),
                          [](std::uint8_t b) { return b != 0; }));
}

TEST(StripedRoundTrip, TrafficIsDistributedPerLane) {
  auto local = make_local();
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{4, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  std::vector<ByteChannel*> a;
  std::vector<std::unique_ptr<CountingChannel>> counted;
  counted.reserve(lanes.first.size());
  for (auto& c : lanes.first) {
    counted.push_back(std::make_unique<CountingChannel>(c.get()));
    a.push_back(counted.back().get());
  }

  CrossNodeKvStripedDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize, stripe);
  donor.execute(local.k.data(), local.v.data(), nullptr, 0, nullptr, nullptr,
                a);
  ASSERT_EQ(counted.size(), 4u);
  std::size_t total = 0;
  for (std::size_t l = 0; l < 4; ++l) {
    EXPECT_EQ(counted[l]->sent_chunks, 1u);   // exactly one chunk per lane
    EXPECT_EQ(counted[l]->sent_bytes, kLayerBytes / 4);
    total += counted[l]->sent_bytes;
  }
  EXPECT_EQ(total, kLayerBytes);
}

TEST(StripedRoundTrip, AcceptsRepeatedSlotsWithGatherSemantics) {
  auto local = make_local();
  const int* slots = repeat_slots();
  const CrossNodeKvStripeConfig stripe{2, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  std::vector<ByteChannel*> a;
  for (auto& c : lanes.first) a.push_back(c.get());

  // Reference: the SAME repeated-slot map through the single-lane bounce
  // donate -- the contiguous payload the striped chunks must reproduce.
  auto ref_link = make_byte_link();
  CrossNodeKvDonatePlan donor1(kSlots, kHeads, kHeadDim, kElem, slots,
                               kPages, kPageSize,
                               vkernels::comm::FabricImportTransport::kHostBounce,
                               nullptr);
  donor1.execute(local.k.data(), local.v.data(), nullptr, 0, nullptr,
                 nullptr, ref_link.first.get());
  std::vector<std::uint8_t> ref_payload = ref_link.second->recv();

  // Striped donate with the repeated map (gather semantics: repeats are
  // allowed on the DONATE side; the chunk list still tiles the payload).
  CrossNodeKvStripedDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize, stripe);
  donor.execute(local.k.data(), local.v.data(), nullptr, 0, nullptr, nullptr,
                a);
  // Reassemble the striped chunks from the far end and compare byte-for-byte.
  std::vector<std::uint8_t> got;
  for (auto& c : lanes.second) {
    std::vector<std::uint8_t> chunk = c->recv();
    got.insert(got.end(), chunk.begin(), chunk.end());
  }
  ASSERT_TRUE(same_bytes(got, ref_payload));
  ASSERT_EQ(got.size(), kLayerBytes);
}

TEST(StripedRestore, StreamSubmitsOneTask) {
  auto local = make_local();
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{2, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  std::vector<ByteChannel*> a, b;
  for (auto& c : lanes.first) a.push_back(c.get());
  for (auto& c : lanes.second) b.push_back(c.get());

  CrossNodeKvStripedDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize, stripe);
  // Two full donate passes: the reference restore consumes the first
  // chunk set, the stream-ordered restore the second (FIFO per lane).
  donor.execute(local.k.data(), local.v.data(), nullptr, 0, nullptr, nullptr,
                a);
  donor.execute(local.k.data(), local.v.data(), nullptr, 0, nullptr, nullptr,
                a);

  CrossNodeKvStripedRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem,
                                         slots, kPages, kPageSize, stripe);
  auto ref = make_local(0, 0); ref.k.assign(ref.k.size(), 0); ref.v.assign(ref.v.size(), 0);
  restorer.execute(ref.k.data(), ref.v.data(), 0, nullptr, nullptr, b);
  // Stream-ordered result: the recv+scatter is ONE deferred task.
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  Stream s;
  const std::size_t before = s.submitted();
  restorer.execute(out.k.data(), out.v.data(), 0, &s, nullptr, b);
  EXPECT_EQ(s.submitted(), before + 1u);  // exactly one task
  s.wait();
  ASSERT_TRUE(same_bytes(out.k, ref.k));
  ASSERT_TRUE(same_bytes(out.v, ref.v));
}

TEST(StripedDonate, StreamSubmitsOneTask) {
  auto local = make_local();
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{2, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  std::vector<ByteChannel*> a, b;
  for (auto& c : lanes.first) a.push_back(c.get());
  for (auto& c : lanes.second) b.push_back(c.get());

  CrossNodeKvStripedDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize, stripe);
  Stream s;
  const std::size_t before = s.submitted();
  donor.execute(local.k.data(), local.v.data(), nullptr, 0, &s, nullptr, a);
  EXPECT_EQ(s.submitted(), before + 1u);  // exactly one deferred task
  s.wait();

  CrossNodeKvStripedRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem,
                                         slots, kPages, kPageSize, stripe);
  auto ref = make_local(0, 0); ref.k.assign(ref.k.size(), 0); ref.v.assign(ref.v.size(), 0);
  restorer.execute(ref.k.data(), ref.v.data(), 0, nullptr, nullptr, b);
  // Mapped slots landed with the donor's bytes (others stay zeroed).
  const std::uint8_t first = local.k[0];
  EXPECT_TRUE(std::any_of(ref.k.begin(), ref.k.end(),
                          [first](std::uint8_t x) { return x == first; }));
}

TEST(StripedRestore, ChunkSizeMismatchThrows) {
  // A far side built with DIFFERENT geometry sends wrong-sized chunks; the
  // per-chunk recv check must reject it instead of corrupting the scatter.
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{2, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  // Send oversized chunks on both lanes.
  lanes.first[0]->send(payload(kLayerBytes, 0x11));
  lanes.first[1]->send(payload(kLayerBytes, 0x22));
  std::vector<ByteChannel*> b;
  for (auto& c : lanes.second) b.push_back(c.get());

  CrossNodeKvStripedRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem,
                                         slots, kPages, kPageSize, stripe);
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  EXPECT_THROW(restorer.execute(out.k.data(), out.v.data(), 0, nullptr,
                                nullptr, b),
               std::invalid_argument);
}

TEST(StripedRestore, RejectsTooFewLanes) {
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{4, 8};
  CrossNodeKvStripedRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem,
                                         slots, kPages, kPageSize, stripe);
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  auto link = make_byte_link();
  std::vector<ByteChannel*> one{link.second.get()};
  EXPECT_THROW(restorer.execute(out.k.data(), out.v.data(), 0, nullptr,
                                nullptr, one),
               std::invalid_argument);
  EXPECT_THROW(restorer.execute(out.k.data(), out.v.data(), 0, nullptr,
                                nullptr, {}),
               std::invalid_argument);
}

TEST(StripedDonate, RejectsTooFewLanes) {
  auto local = make_local();
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{4, 8};
  CrossNodeKvStripedDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize, stripe);
  auto link = make_byte_link();
  std::vector<ByteChannel*> one{link.first.get()};
  EXPECT_THROW(donor.execute(local.k.data(), local.v.data(), nullptr, 0,
                             nullptr, nullptr, one),
               std::invalid_argument);
}

TEST(StripedPlans, ZeroPagesIsNoOp) {
  const int slots[1] = {0};
  const CrossNodeKvStripeConfig stripe{4, 8};
  CrossNodeKvStripedRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem,
                                         slots, 0, kPageSize, stripe);
  CrossNodeKvStripedDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                                     0, kPageSize, stripe);
  EXPECT_EQ(restorer.num_pages(), 0u);
  EXPECT_EQ(restorer.total_bytes(), 0u);
  EXPECT_TRUE(restorer.stripes().empty());
  EXPECT_EQ(donor.total_bytes(), 0u);
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0x42); out.v.assign(out.v.size(), 0x42);
  // No-op executes need no lanes (they return before the lane check).
  restorer.execute(out.k.data(), out.v.data(), 0, nullptr, nullptr, {});
  donor.execute(out.k.data(), out.v.data(), nullptr, 0, nullptr, nullptr, {});
  EXPECT_EQ(out.k.front(), 0x42);  // untouched
}

TEST(StripedPlans, RejectsInvalidGeometry) {
  const int good[kPages * kPageSize] = {3, 1, 14, 7, 0, 9, 6, 12, 2, 5, 8, 11};
  const CrossNodeKvStripeConfig stripe{2, 8};
  EXPECT_THROW(CrossNodeKvStripedRestorePlan(kSlots, kHeads, kHeadDim, 0,
                                             good, kPages, kPageSize, stripe),
               std::invalid_argument);
  // Duplicate slots are a restore-contract violation.
  const int dup[kPages * kPageSize] = {0, 0, 14, 7, 0, 9, 6, 12, 2, 5, 8, 11};
  EXPECT_THROW(CrossNodeKvStripedRestorePlan(kSlots, kHeads, kHeadDim, kElem,
                                             dup, kPages, kPageSize, stripe),
               std::invalid_argument);
  EXPECT_THROW(CrossNodeKvStripedRestorePlan(kSlots, kHeads, kHeadDim, kElem,
                                             nullptr, kPages, kPageSize,
                                             stripe),
               std::invalid_argument);
  // Out-of-range slot for both plans.
  const int oob[kPages * kPageSize] = {3, 1, 14, 7, 0, 9, 6, 99, 2, 5, 8, 11};
  EXPECT_THROW(CrossNodeKvStripedRestorePlan(kSlots, kHeads, kHeadDim, kElem,
                                             oob, kPages, kPageSize, stripe),
               std::invalid_argument);
  EXPECT_THROW(CrossNodeKvStripedDonatePlan(kSlots, kHeads, kHeadDim, kElem,
                                            oob, kPages, kPageSize, stripe),
               std::invalid_argument);
  EXPECT_THROW(CrossNodeKvStripedDonatePlan(kSlots, kHeads, kHeadDim, kElem,
                                            nullptr, kPages, kPageSize,
                                            stripe),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Graph integration: eager-break, identical to the single-lane bounce path
// ---------------------------------------------------------------------------

TEST(StripedGraph, DonateEagerBreaksAndRecordsNothing) {
  auto local = make_local();
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{2, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  std::vector<ByteChannel*> a;
  for (auto& c : lanes.first) a.push_back(c.get());

  CrossNodeKvStripedDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize, stripe);
  GraphCapture g;
  g.begin();
  EXPECT_EQ(g.num_nodes(), 0u);
  donor.execute(local.k.data(), local.v.data(), nullptr, 0, nullptr, &g, a);
  EXPECT_EQ(g.num_nodes(), 0u);      // nothing recorded (host-staged)
  EXPECT_EQ(g.num_segments(), 1u);   // the segment ended by the break
  EXPECT_TRUE(g.in_capture());       // a new segment was begun
  g.end();
  EXPECT_EQ(g.num_segments(), 2u);
  const std::size_t before = g.replays();
  g.replay();
  EXPECT_EQ(g.replays(), before + 1u);  // replay runs no striped work
}

TEST(StripedGraph, RestoreEagerBreaksAndRecordsNothing) {
  const int* slots = unique_slots();
  const CrossNodeKvStripeConfig stripe{2, 8};
  auto lanes = make_striped_byte_links(stripe.lane_count);
  // Pre-send both chunks so the eager-broken recv is non-blocking.
  lanes.first[0]->send(payload(kLayerBytes / 2, 0x33));
  lanes.first[1]->send(payload(kLayerBytes / 2, 0x33));
  std::vector<ByteChannel*> b;
  for (auto& c : lanes.second) b.push_back(c.get());

  CrossNodeKvStripedRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem,
                                         slots, kPages, kPageSize, stripe);
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  GraphCapture g;
  g.begin();
  EXPECT_EQ(g.num_nodes(), 0u);
  restorer.execute(out.k.data(), out.v.data(), 0, nullptr, &g, b);
  EXPECT_EQ(g.num_nodes(), 0u);      // nothing recorded (host-staged)
  EXPECT_EQ(g.num_segments(), 1u);   // the segment ended by the break
  EXPECT_TRUE(g.in_capture());       // a new segment was begun
  g.end();
  EXPECT_EQ(g.num_segments(), 2u);
  const std::size_t before = g.replays();
  g.replay();
  EXPECT_EQ(g.replays(), before + 1u);
}
