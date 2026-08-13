// tests/comm/test_p2p_kv_restore.cpp
//
// Host-reference tests for the fused peer-to-indexed-KV restore. These are
// the correctness oracle: they exercise byte-exact results against the
// two-stage reference across various shapes, validate the contract checks
// (null pointers, unique slots, capacity, element size), and test the
// async stream contract.
#include "minitest.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <vector>

#include "vkernels/comm/p2p_kv_restore.hpp"
#include "vkernels/core/stream.hpp"

using vkernels::Stream;
using vkernels::comm::from_device_slots;
using vkernels::comm::kv_scatter;
using vkernels::comm::P2PKvRestorePlan;
using vkernels::comm::p2p_kv_restore_layer;
using vkernels::comm::p2p_kv_restore_layer_twostage;

namespace {

// A single peer "allocation" — a flat byte array representing one page's
// worth of KV data for one layer. Layout: [page_size, 2, num_kv_heads, head_dim]
// with elem_size bytes per element, filled with a per-page seed so mis-routed
// reads are obvious.
std::vector<std::uint8_t> make_page(std::size_t page_size, std::size_t num_kv_heads,
                                    std::size_t head_dim, std::size_t elem_size,
                                    std::uint8_t seed) {
  const std::size_t total = page_size * 2 * num_kv_heads * head_dim * elem_size;
  std::vector<std::uint8_t> v(total);
  for (std::size_t i = 0; i < total; ++i)
    v[i] = static_cast<std::uint8_t>(seed + i);
  return v;
}

// Fill a destination K/V buffer with a sentinel byte.
std::vector<std::uint8_t> make_dst(std::size_t num_slots, std::size_t num_kv_heads,
                                   std::size_t head_dim, std::size_t elem_size,
                                   std::uint8_t fill = 0xCC) {
  const std::size_t total = num_slots * num_kv_heads * head_dim * elem_size;
  return std::vector<std::uint8_t>(total, fill);
}

}  // namespace

// ---------------------------------------------------------------------------
// Prepared plan: host reference (issue #27)
// ---------------------------------------------------------------------------
//
// The plan validates and stages metadata once, then execute() adds a single
// scalar source-layer offset to every peer page base and copies — the KVAAS
// "one run list, many layers" reuse pattern. These tests pin the host plan
// to the one-shot oracle byte-for-byte, including the offset path, the
// device-slot (borrowed pointer) variant, the stream contract and the
// contract checks moved out of the hot path into the constructor.

TEST(KvRestorePlan, SyncExecuteEqualsOneShot) {
  constexpr std::size_t kPageSize = 4, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;
  constexpr std::size_t kSlots = 16;

  auto p0 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x11);
  auto p1 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x77);
  const int slot_ids[8] = {3, 1, 14, 7, 0, 9, 6, 12};
  const void* ptrs[2] = {p0.data(), p1.data()};

  auto kp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto ko = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vo = make_dst(kSlots, kHeads, kHeadDim, kElem);

  P2PKvRestorePlan plan(kp.data(), vp.data(), kSlots, kHeads, kHeadDim, kElem,
                        slot_ids, ptrs, 2, kPageSize);
  EXPECT_EQ(plan.num_pages(), 2u);
  EXPECT_EQ(plan.total_bytes(), 2u * kPageSize * 2u * kSlotBytes);
  plan.execute(0);

  const std::size_t offs[2] = {0, 0};
  p2p_kv_restore_layer(ko.data(), vo.data(), slot_ids, ptrs, offs, 2,
                       kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < kp.size(); ++i) {
    ASSERT_EQ(kp[i], ko[i]);
    ASSERT_EQ(vp[i], vo[i]);
  }
}

// Two layers stored back-to-back in each page buffer; the plan selects layer
// 1 with a single scalar offset, the one-shot uses per-page src_page_offsets.
TEST(KvRestorePlan, OffsetExecuteMatchesPerPageOffsets) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 32
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;                // 64
  constexpr std::size_t kLayerBytes = kPageSize * kTokenStr;       // 128
  constexpr std::size_t kSlots = 8;

  // Two layers per peer buffer.
  auto buf0 = make_page(kPageSize * 2, kHeads, kHeadDim, kElem, 0x20);
  auto buf1 = make_page(kPageSize * 2, kHeads, kHeadDim, kElem, 0xA0);
  const int slot_ids[4] = {2, 5, 1, 6};
  const void* ptrs[2] = {buf0.data(), buf1.data()};

  auto kp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto ko = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vo = make_dst(kSlots, kHeads, kHeadDim, kElem);

  P2PKvRestorePlan plan(kp.data(), vp.data(), kSlots, kHeads, kHeadDim, kElem,
                        slot_ids, ptrs, 2, kPageSize);
  plan.execute(kLayerBytes);  // select layer 1 with one scalar

  const std::size_t offs[2] = {kLayerBytes, kLayerBytes};
  p2p_kv_restore_layer(ko.data(), vo.data(), slot_ids, ptrs, offs, 2,
                       kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < kp.size(); ++i) {
    ASSERT_EQ(kp[i], ko[i]);
    ASSERT_EQ(vp[i], vo[i]);
  }
}

TEST(KvRestorePlan, AsyncExecuteIsOneTaskAndCorrect) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto p0 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x33);
  auto p1 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0xEE);
  const int slot_ids[4] = {1, 4, 7, 2};
  const void* ptrs[2] = {p0.data(), p1.data()};

  auto kp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto ko = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vo = make_dst(kSlots, kHeads, kHeadDim, kElem);

  P2PKvRestorePlan plan(kp.data(), vp.data(), kSlots, kHeads, kHeadDim, kElem,
                        slot_ids, ptrs, 2, kPageSize);
  Stream s;
  const std::size_t before = s.submitted();
  plan.execute(0, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one stream task
  s.wait();

  const std::size_t offs[2] = {0, 0};
  p2p_kv_restore_layer(ko.data(), vo.data(), slot_ids, ptrs, offs, 2,
                       kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < kp.size(); ++i) {
    ASSERT_EQ(kp[i], ko[i]);
    ASSERT_EQ(vp[i], vo[i]);
  }
}

// Two plans share one destination buffer and execute on two streams — the
// KVAAS reuse pattern across layers, and a sanity check that a plan is safe
// to run on more than one stream.
TEST(KvRestorePlan, TwoStreamsSharePlan) {
  constexpr std::size_t kPageSize = 2, kHeads = 1, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 16
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;                // 32
  constexpr std::size_t kSlots = 4;

  // Two layers per peer buffer: layer 0 (seed 0x05) and layer 1 (seed 0x77).
  // The plan's single page has two tokens written to slots 1 and 2. We run
  // it twice concurrently on two streams. To make the outcome deterministic
  // regardless of which stream wins the shared-destination write, BOTH runs
  // read the same source offset (layer 0) — the test then asserts the
  // correct layer-0 bytes, proving the plan is safe to share across
  // streams (no deadlock, no metadata mutation during execute).
  auto buf = make_page(kPageSize * 2, kHeads, kHeadDim, kElem, 0x05);
  const int slot_ids[2] = {1, 2};
  const void* ptrs[1] = {buf.data()};

  auto k_dst = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_dst = make_dst(kSlots, kHeads, kHeadDim, kElem);

  P2PKvRestorePlan plan(k_dst.data(), v_dst.data(), kSlots, kHeads, kHeadDim,
                        kElem, slot_ids, ptrs, 1, kPageSize);
  Stream s0, s1;
  plan.execute(0, &s0);  // layer 0 on stream 0
  plan.execute(0, &s1);  // layer 0 on stream 1 (concurrent, same source)
  s0.wait();
  s1.wait();

  // Both writes read layer 0, so the final state is deterministic: slots 1
  // and 2 hold layer 0's token 0 and token 1 respectively.
  const std::uint8_t* layer0 = buf.data();
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(k_dst[1 * kSlotBytes + i], layer0[0 * kTokenStr + i]);
    ASSERT_EQ(v_dst[1 * kSlotBytes + i], layer0[0 * kTokenStr + kSlotBytes + i]);
    ASSERT_EQ(k_dst[2 * kSlotBytes + i], layer0[1 * kTokenStr + i]);
    ASSERT_EQ(v_dst[2 * kSlotBytes + i], layer0[1 * kTokenStr + kSlotBytes + i]);
  }
}

TEST(KvRestorePlan, ZeroPagesIsNoOp) {
  auto k_dst = make_dst(4, 2, 8, 2);
  auto v_dst = make_dst(4, 2, 8, 2);
  auto before_k = k_dst;
  auto before_v = v_dst;
  P2PKvRestorePlan plan(k_dst.data(), v_dst.data(), 4, 2, 8, 2,
                        nullptr, nullptr, 0, 64);
  EXPECT_EQ(plan.num_pages(), 0u);
  plan.execute(0);
  Stream s;
  plan.execute(0, &s);
  s.wait();
  for (std::size_t i = 0; i < k_dst.size(); ++i) {
    ASSERT_EQ(k_dst[i], before_k[i]);
    ASSERT_EQ(v_dst[i], before_v[i]);
  }
}

// The device-slot variant borrows the caller's pointer and skips slot
// validation — but on the host it still produces byte-identical results.
TEST(KvRestorePlan, DeviceSlotsBorrowsAndMatches) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto p0 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x44);
  auto p1 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x88);
  int slot_ids[4] = {3, 0, 7, 5};
  const void* ptrs[2] = {p0.data(), p1.data()};

  auto kp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vp = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto ko = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vo = make_dst(kSlots, kHeads, kHeadDim, kElem);

  P2PKvRestorePlan plan(from_device_slots, kp.data(), vp.data(), kSlots,
                        kHeads, kHeadDim, kElem, slot_ids, ptrs, 2, kPageSize);
  plan.execute(0);

  const std::size_t offs[2] = {0, 0};
  p2p_kv_restore_layer(ko.data(), vo.data(), slot_ids, ptrs, offs, 2,
                       kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < kp.size(); ++i) {
    ASSERT_EQ(kp[i], ko[i]);
    ASSERT_EQ(vp[i], vo[i]);
  }
}

// Contract checks move OUT of the hot path and INTO the constructor: the
// plan rejects a duplicate slot / out-of-range slot / non-BF16 / null dst at
// create time, so execute() can stay free of checks.
TEST(KvRestorePlan, RejectsDuplicateSlotAtCreate) {
  auto p = make_page(2, 2, 4, 2, 0x10);
  auto k = make_dst(4, 2, 4, 2);
  auto v = make_dst(4, 2, 4, 2);
  const int slot_ids[2] = {1, 1};
  const void* ptrs[1] = {p.data()};
  EXPECT_THROW(P2PKvRestorePlan(k.data(), v.data(), 4, 2, 4, 2, slot_ids,
                                ptrs, 1, 2),
               std::invalid_argument);
}

TEST(KvRestorePlan, RejectsOutOfRangeSlotAtCreate) {
  auto p = make_page(1, 2, 4, 2, 0x10);
  auto k = make_dst(4, 2, 4, 2);   // 4 slots -> valid range [0,3]
  auto v = make_dst(4, 2, 4, 2);
  const int slot_ids[1] = {4};     // == num_slots, out of range
  const void* ptrs[1] = {p.data()};
  EXPECT_THROW(P2PKvRestorePlan(k.data(), v.data(), 4, 2, 4, 2, slot_ids,
                                ptrs, 1, 1),
               std::invalid_argument);
}

TEST(KvRestorePlan, RejectsNonBF16AtCreate) {
  auto k = make_dst(4, 2, 4, 4);
  auto v = make_dst(4, 2, 4, 4);
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {k.data()};
  EXPECT_THROW(P2PKvRestorePlan(k.data(), v.data(), 4, 2, 4, 4, slot_ids,
                                ptrs, 1, 1),
               std::invalid_argument);
}

TEST(KvRestorePlan, RejectsNullDstAtCreate) {
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {(void*)0x1000};
  EXPECT_THROW(P2PKvRestorePlan(nullptr, (void*)0x2000, 4, 2, 4, 2, slot_ids,
                                ptrs, 1, 1),
               std::invalid_argument);
}

TEST(KvRestorePlan, RejectsNegativeSlotAtCreate) {
  auto p = make_page(1, 2, 4, 2, 0x10);
  auto k = make_dst(4, 2, 4, 2);
  auto v = make_dst(4, 2, 4, 2);
  const int slot_ids[1] = {-1};
  const void* ptrs[1] = {p.data()};
  EXPECT_THROW(P2PKvRestorePlan(k.data(), v.data(), 4, 2, 4, 2, slot_ids,
                                ptrs, 1, 1),
               std::invalid_argument);
}

// The device-slot variant does NOT validate slot contents (it would need a
// D2H sync), so a duplicate slot is accepted at create — the caller owns
// that invariant. Shape checks (null dst, non-BF16) still apply.
TEST(KvRestorePlan, DeviceSlotsSkipsSlotValidation) {
  auto p = make_page(2, 2, 4, 2, 0x10);
  auto k = make_dst(4, 2, 4, 2);
  auto v = make_dst(4, 2, 4, 2);
  const int slot_ids[2] = {1, 1};   // would be rejected by the host-input ctor
  const void* ptrs[1] = {p.data()};
  EXPECT_NO_THROW(P2PKvRestorePlan(from_device_slots, k.data(), v.data(), 4,
                                   2, 4, 2, slot_ids, ptrs, 1, 2));
}

TEST(KvRestorePlan, DeviceSlotsStillValidatesShape) {
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {(void*)0x1000};
  EXPECT_THROW(P2PKvRestorePlan(from_device_slots, nullptr, (void*)0x2000,
                                4, 2, 4, 4, slot_ids, ptrs, 1, 1),
               std::invalid_argument);  // non-BF16 elem_size
}

// ---------------------------------------------------------------------------
// kv_scatter host reference (the second stage of the two-path)
// ---------------------------------------------------------------------------
TEST(KvScatter, HostScatterMatchesTwoStage) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;
  constexpr std::size_t kScratchPerPage = kPageSize * 2 * kSlotBytes;
  constexpr std::size_t kSlots = 8;

  // Build a contiguous scratch = [num_pages, page_size, 2, num_kv_heads,
  // head_dim] (two pages back-to-back).
  std::vector<std::uint8_t> scratch(2 * kScratchPerPage);
  for (std::size_t i = 0; i < scratch.size(); ++i)
    scratch[i] = static_cast<std::uint8_t>(0x30 + (i % 200));

  const int slot_ids[4] = {3, 1, 6, 4};
  auto ks = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vs = make_dst(kSlots, kHeads, kHeadDim, kElem);
  kv_scatter(ks.data(), vs.data(), scratch.data(), slot_ids, 2, kPageSize,
             kHeads, kHeadDim, kElem);

  // Reference: same scatter, done by hand.
  auto kr = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vr = make_dst(kSlots, kHeads, kHeadDim, kElem);
  for (int p = 0; p < 2; ++p) {
    const std::uint8_t* pg = scratch.data() + p * kScratchPerPage;
    for (int t = 0; t < 2; ++t) {
      int slot = slot_ids[p * 2 + t];
      const std::uint8_t* sk = pg + t * 2 * kSlotBytes;
      const std::uint8_t* sv = sk + kSlotBytes;
      std::memcpy(kr.data() + slot * kSlotBytes, sk, kSlotBytes);
      std::memcpy(vr.data() + slot * kSlotBytes, sv, kSlotBytes);
    }
  }
  for (std::size_t i = 0; i < ks.size(); ++i) {
    ASSERT_EQ(ks[i], kr[i]);
    ASSERT_EQ(vs[i], vr[i]);
  }
}

// ---------------------------------------------------------------------------
// Byte-exact correctness
// ---------------------------------------------------------------------------
TEST(KvRestore, SinglePageSingleToken) {
  constexpr std::size_t kPageSize = 1, kHeads = 2, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16

  auto page = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x10);
  auto k_dst = make_dst(2, kHeads, kHeadDim, kElem);
  auto v_dst = make_dst(2, kHeads, kHeadDim, kElem);
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {page.data()};
  const std::size_t offs[1] = {0};

  p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slot_ids, ptrs, offs,
                       1, kPageSize, kHeads, kHeadDim, kElem);

  // K dst[0] = page K for token 0; V dst[0] = page V for token 0.
  const std::uint8_t* src = page.data();
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(k_dst[i], src[i]);
    ASSERT_EQ(v_dst[i], src[kSlotBytes + i]);
  }
  // Untouched slot 1 stays at sentinel.
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(k_dst[kSlotBytes + i], 0xCC);
    ASSERT_EQ(v_dst[kSlotBytes + i], 0xCC);
  }
}

TEST(KvRestore, SinglePageManyTokens) {
  constexpr std::size_t kPageSize = 8, kHeads = 1, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;               // 32

  auto page = make_page(kPageSize, kHeads, kHeadDim, kElem, 0xAB);
  constexpr std::size_t kSlots = 16;
  auto k_dst = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_dst = make_dst(kSlots, kHeads, kHeadDim, kElem);

  // Scatter tokens to non-contiguous slots: 3, 7, 1, 15, 0, 8, 12, 4.
  const int slot_ids[8] = {3, 7, 1, 15, 0, 8, 12, 4};
  const void* ptrs[1] = {page.data()};
  const std::size_t offs[1] = {0};

  p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slot_ids, ptrs, offs,
                       1, kPageSize, kHeads, kHeadDim, kElem);

  const std::uint8_t* src = page.data();
  for (int t = 0; t < 8; ++t) {
    int slot = slot_ids[t];
    const std::uint8_t* src_k = src + t * kTokenStr;
    const std::uint8_t* src_v = src_k + kSlotBytes;
    for (std::size_t i = 0; i < kSlotBytes; ++i) {
      ASSERT_EQ(k_dst[slot * kSlotBytes + i], src_k[i]);
      ASSERT_EQ(v_dst[slot * kSlotBytes + i], src_v[i]);
    }
  }
  // Unused slots stay at sentinel.
  for (int s : {2, 5, 6, 9, 10, 11, 13, 14}) {
    for (std::size_t i = 0; i < kSlotBytes; ++i) {
      ASSERT_EQ(k_dst[s * kSlotBytes + i], 0xCC);
      ASSERT_EQ(v_dst[s * kSlotBytes + i], 0xCC);
    }
  }
}

TEST(KvRestore, MultiplePagesMultipleAllocations) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 32
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;               // 64

  auto p0 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x10);
  auto p1 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x80);
  constexpr std::size_t kSlots = 8;
  auto k_dst = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_dst = make_dst(kSlots, kHeads, kHeadDim, kElem);

  // Page 0 tokens → slots 0, 1; page 1 tokens → slots 3, 6.
  const int slot_ids[4] = {0, 1, 3, 6};
  const void* ptrs[2] = {p0.data(), p1.data()};
  const std::size_t offs[2] = {0, 0};

  p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slot_ids, ptrs, offs,
                       2, kPageSize, kHeads, kHeadDim, kElem);

  const std::uint8_t* s0 = p0.data();
  const std::uint8_t* s1 = p1.data();
  for (int t = 0; t < 2; ++t) {
    // Page 0
    {
      int slot = slot_ids[t];
      const std::uint8_t* sk = s0 + t * kTokenStr;
      const std::uint8_t* sv = sk + kSlotBytes;
      for (std::size_t i = 0; i < kSlotBytes; ++i) {
        ASSERT_EQ(k_dst[slot * kSlotBytes + i], sk[i]);
        ASSERT_EQ(v_dst[slot * kSlotBytes + i], sv[i]);
      }
    }
    // Page 1
    {
      int slot = slot_ids[2 + t];
      const std::uint8_t* sk = s1 + t * kTokenStr;
      const std::uint8_t* sv = sk + kSlotBytes;
      for (std::size_t i = 0; i < kSlotBytes; ++i) {
        ASSERT_EQ(k_dst[slot * kSlotBytes + i], sk[i]);
        ASSERT_EQ(v_dst[slot * kSlotBytes + i], sv[i]);
      }
    }
  }
  // Unused slots stay at sentinel.
  for (int s : {2, 4, 5, 7}) {
    for (std::size_t i = 0; i < kSlotBytes; ++i) {
      ASSERT_EQ(k_dst[s * kSlotBytes + i], 0xCC);
    }
  }
}

TEST(KvRestore, PageOffsetIsHonoured) {
  constexpr std::size_t kPageSize = 1, kHeads = 1, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 8
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;               // 16

  // Make a buffer with two "pages" back-to-back; use an offset to select
  // the second one.
  auto buf = make_page(2, kHeads, kHeadDim, kElem, 0x30);
  auto k_dst = make_dst(2, kHeads, kHeadDim, kElem);
  auto v_dst = make_dst(2, kHeads, kHeadDim, kElem);

  const int slot_ids[1] = {1};
  const void* ptrs[1] = {buf.data()};
  const std::size_t offs[1] = {kTokenStr};  // skip the first token

  p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slot_ids, ptrs, offs,
                       1, kPageSize, kHeads, kHeadDim, kElem);

  const std::uint8_t* src = buf.data() + kTokenStr;
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(k_dst[kSlotBytes + i], src[i]);
    ASSERT_EQ(v_dst[kSlotBytes + i], src[kSlotBytes + i]);
  }
}

TEST(KvRestore, ZeroPagesIsNoOp) {
  auto k_dst = make_dst(4, 2, 8, 2);
  auto v_dst = make_dst(4, 2, 8, 2);
  auto before_k = k_dst;
  auto before_v = v_dst;
  p2p_kv_restore_layer(k_dst.data(), v_dst.data(), nullptr, nullptr, nullptr,
                       0, 64, 8, 128, 2);
  for (std::size_t i = 0; i < k_dst.size(); ++i) {
    ASSERT_EQ(k_dst[i], before_k[i]);
    ASSERT_EQ(v_dst[i], before_v[i]);
  }
}

TEST(KvRestore, AsyncStreamIsCorrectAfterWait) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16

  auto p0 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x40);
  auto p1 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x90);
  auto k_dst = make_dst(8, kHeads, kHeadDim, kElem);
  auto v_dst = make_dst(8, kHeads, kHeadDim, kElem);

  const int slot_ids[4] = {2, 5, 1, 7};
  const void* ptrs[2] = {p0.data(), p1.data()};
  const std::size_t offs[2] = {0, 0};

  Stream s;
  std::size_t before = s.submitted();
  p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slot_ids, ptrs, offs,
                       2, kPageSize, kHeads, kHeadDim, kElem, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one task
  s.wait();

  // Verify a few key positions.
  const std::uint8_t* s0 = p0.data();
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(k_dst[2 * kSlotBytes + i], s0[i]);
    ASSERT_EQ(v_dst[2 * kSlotBytes + i], s0[kSlotBytes + i]);
  }
}

// ---------------------------------------------------------------------------
// Two-stage reference: fused kernel must produce byte-identical results
// ---------------------------------------------------------------------------
TEST(KvRestore, FusedEqualsTwoStageSinglePage) {
  constexpr std::size_t kPageSize = 4, kHeads = 4, kHeadDim = 16, kElem = 2;
  auto page = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x55);
  constexpr std::size_t kSlots = 16;
  int slot_ids[4] = {2, 7, 11, 3};
  const void* ptrs[1] = {page.data()};
  const std::size_t offs[1] = {0};

  auto k_fused = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_fused = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto k_twostage = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_twostage = make_dst(kSlots, kHeads, kHeadDim, kElem);

  p2p_kv_restore_layer(k_fused.data(), v_fused.data(), slot_ids, ptrs, offs,
                       1, kPageSize, kHeads, kHeadDim, kElem);
  p2p_kv_restore_layer_twostage(k_twostage.data(), v_twostage.data(),
                                slot_ids, ptrs, offs,
                                1, kPageSize, kHeads, kHeadDim, kElem);

  ASSERT_EQ(k_fused.size(), k_twostage.size());
  for (std::size_t i = 0; i < k_fused.size(); ++i) {
    ASSERT_EQ(k_fused[i], k_twostage[i]);
    ASSERT_EQ(v_fused[i], v_twostage[i]);
  }
}

TEST(KvRestore, FusedEqualsTwoStageMultiPage) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 32, kElem = 2;
  auto p0 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x12);
  auto p1 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0xAB);
  auto p2 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x77);
  constexpr std::size_t kSlots = 16;

  // Deliberately non-sequential slots across pages.
  int slot_ids[6] = {7, 2, 13, 0, 5, 9};
  const void* ptrs[3] = {p0.data(), p1.data(), p2.data()};
  const std::size_t offs[3] = {0, 0, 0};

  auto kf = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vf = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto kt = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vt = make_dst(kSlots, kHeads, kHeadDim, kElem);

  p2p_kv_restore_layer(kf.data(), vf.data(), slot_ids, ptrs, offs,
                       3, kPageSize, kHeads, kHeadDim, kElem);
  p2p_kv_restore_layer_twostage(kt.data(), vt.data(), slot_ids, ptrs, offs,
                                3, kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < kf.size(); ++i) {
    ASSERT_EQ(kf[i], kt[i]);
    ASSERT_EQ(vf[i], vt[i]);
  }
}

TEST(KvRestore, TwoStageAsyncEqualsSync) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  auto p0 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x21);
  auto p1 = make_page(kPageSize, kHeads, kHeadDim, kElem, 0x9A);
  constexpr std::size_t kSlots = 8;
  const int slot_ids[4] = {1, 6, 3, 7};
  const void* ptrs[2] = {p0.data(), p1.data()};
  const std::size_t offs[2] = {0, 0};

  auto ka = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto va = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto ks = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto vs = make_dst(kSlots, kHeads, kHeadDim, kElem);

  Stream s;
  std::size_t before = s.submitted();
  p2p_kv_restore_layer_twostage(ka.data(), va.data(), slot_ids, ptrs, offs,
                                2, kPageSize, kHeads, kHeadDim, kElem, &s);
  EXPECT_EQ(s.submitted() - before, 2u);  // one task per page
  s.wait();
  p2p_kv_restore_layer_twostage(ks.data(), vs.data(), slot_ids, ptrs, offs,
                                2, kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < ka.size(); ++i) {
    ASSERT_EQ(ka[i], ks[i]);
    ASSERT_EQ(va[i], vs[i]);
  }
}

// Test realistic shapes from the acceptance criteria.
TEST(KvRestore, RealisticShapesBF16) {
  // head_dim 64, 128, 256; page_size 16, 32, 64; num_kv_heads = 4
  struct Shape {
    std::size_t heads, head_dim, page_size;
  };
  for (Shape s : {Shape{4, 64, 16}, Shape{4, 128, 32}, Shape{4, 256, 64},
                  Shape{8, 128, 64}, Shape{1, 256, 16}}) {
    auto page = make_page(s.page_size, s.heads, s.head_dim, 2, 0x42);
    const std::size_t kSlots = s.page_size * 2;
    std::vector<int> slot_ids(s.page_size);
    // Sequential slots for simplicity.
    for (std::size_t i = 0; i < s.page_size; ++i) slot_ids[i] = static_cast<int>(i);

    auto kf = make_dst(kSlots, s.heads, s.head_dim, 2);
    auto vf = make_dst(kSlots, s.heads, s.head_dim, 2);
    auto kt = make_dst(kSlots, s.heads, s.head_dim, 2);
    auto vt = make_dst(kSlots, s.heads, s.head_dim, 2);

    const void* ptrs[1] = {page.data()};
    const std::size_t offs[1] = {0};

    p2p_kv_restore_layer(kf.data(), vf.data(), slot_ids.data(), ptrs, offs,
                         1, s.page_size, s.heads, s.head_dim, 2);
    p2p_kv_restore_layer_twostage(kt.data(), vt.data(), slot_ids.data(), ptrs, offs,
                                  1, s.page_size, s.heads, s.head_dim, 2);

    for (std::size_t i = 0; i < kf.size(); ++i) {
      ASSERT_EQ(kf[i], kt[i]);
      ASSERT_EQ(vf[i], vt[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Contract violations
// ---------------------------------------------------------------------------
TEST(KvRestore, NullKDstThrows) {
  const void* ptrs[1] = {(void*)0x1000};
  const std::size_t offs[1] = {0};
  const int slots[1] = {0};
  EXPECT_THROW(p2p_kv_restore_layer(nullptr, (void*)0x2000, slots, ptrs, offs,
                                    1, 16, 8, 128, 2),
               std::invalid_argument);
}

TEST(KvRestore, NonUniqueSlotsThrows) {
  auto page = make_page(2, 2, 4, 2, 0x10);
  auto k_dst = make_dst(4, 2, 4, 2);
  auto v_dst = make_dst(4, 2, 4, 2);
  const int slot_ids[2] = {1, 1};  // duplicate
  const void* ptrs[1] = {page.data()};
  const std::size_t offs[1] = {0};
  EXPECT_THROW(p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slot_ids, ptrs, offs,
                                    1, 2, 2, 4, 2),
               std::invalid_argument);
}

TEST(KvRestore, NegativeSlotThrows) {
  auto page = make_page(1, 2, 4, 2, 0x10);
  auto k_dst = make_dst(4, 2, 4, 2);
  auto v_dst = make_dst(4, 2, 4, 2);
  const int slot_ids[1] = {-1};
  const void* ptrs[1] = {page.data()};
  const std::size_t offs[1] = {0};
  EXPECT_THROW(p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slot_ids, ptrs, offs,
                                    1, 1, 2, 4, 2),
               std::invalid_argument);
}

TEST(KvRestore, ZeroPageSizeThrows) {
  auto k_dst = make_dst(4, 2, 4, 2);
  auto v_dst = make_dst(4, 2, 4, 2);
  const int slots[1] = {0};
  const void* ptrs[1] = {k_dst.data()};
  const std::size_t offs[1] = {0};
  EXPECT_THROW(p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slots, ptrs, offs,
                                    1, 0, 8, 128, 2),
               std::invalid_argument);
}

TEST(KvRestore, NonBF16ElemSizeThrows) {
  auto k_dst = make_dst(4, 2, 4, 2);
  auto v_dst = make_dst(4, 2, 4, 2);
  const int slots[1] = {0};
  const void* ptrs[1] = {k_dst.data()};
  const std::size_t offs[1] = {0};
  EXPECT_THROW(p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slots, ptrs, offs,
                                    1, 1, 2, 4, 4),  // elem_size=4 (fp32)
               std::invalid_argument);
}

TEST(KvRestore, NullPagePtrForNonZeroPagesThrows) {
  auto k_dst = make_dst(4, 2, 4, 2);
  auto v_dst = make_dst(4, 2, 4, 2);
  const int slots[1] = {0};
  const void* ptrs[1] = {nullptr};
  const std::size_t offs[1] = {0};
  EXPECT_THROW(p2p_kv_restore_layer(k_dst.data(), v_dst.data(), slots, ptrs, offs,
                                    1, 1, 2, 4, 2),
               std::invalid_argument);
}
