// tests/comm/test_p2p_kv_donate.cpp
//
// Host-reference tests for the fused indexed-KV-to-peer donation (issue
// #36). These are the correctness oracle: they exercise byte-exact results
// against the "pack_pages + per-page peer copy" reference (the data flow
// the fused kernel replaces), validate the contract checks (null pointers,
// non-negative slots, capacity, element size), test the gather semantics
// (repeated / non-monotonic source slots are allowed, unlike the restore's
// unique-destination scatter), the copy-engine fallback path, the adaptive
// dispatch helpers, and the async stream contract.
#include "minitest.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vkernels/comm/p2p_kv_donate.hpp"
#include "vkernels/core/stream.hpp"

using vkernels::Stream;
using vkernels::comm::from_device_slots;
using vkernels::comm::from_device_slots_int64;
using vkernels::comm::kv_gather;
using vkernels::comm::P2PKvDonatePlan;
using vkernels::comm::p2p_kv_donate_layer;
using vkernels::comm::p2p_kv_donate_layer_twostage;

namespace {

// Per-slot bytes: num_kv_heads * head_dim * elem_size.
inline std::size_t slot_bytes(std::size_t heads, std::size_t head_dim,
                              std::size_t elem) {
  return heads * head_dim * elem;
}

// Per-token destination stride: [K, V] = 2 * slot_bytes.
inline std::size_t token_stride(std::size_t heads, std::size_t head_dim,
                                std::size_t elem) {
  return 2 * slot_bytes(heads, head_dim, elem);
}

// A local K (or V) source buffer: [num_slots, num_kv_heads, head_dim],
// row-major, filled with a per-buffer seed so mis-routed reads are obvious.
std::vector<std::uint8_t> make_src(std::size_t num_slots, std::size_t heads,
                                   std::size_t head_dim, std::size_t elem,
                                   std::uint8_t seed) {
  const std::size_t total = num_slots * heads * head_dim * elem;
  std::vector<std::uint8_t> v(total);
  for (std::size_t i = 0; i < total; ++i)
    v[i] = static_cast<std::uint8_t>(seed + (i % 251));
  return v;
}

// A peer destination page. For tests we use one layer, so the page is
// [page_size, 2, num_kv_heads, head_dim] in row-major order, pre-filled
// with a sentinel so untouched bytes are obvious.
std::vector<std::uint8_t> make_page(std::size_t page_size, std::size_t heads,
                                    std::size_t head_dim, std::size_t elem,
                                    std::uint8_t fill = 0xDD) {
  const std::size_t total = page_size * 2 * heads * head_dim * elem;
  return std::vector<std::uint8_t>(total, fill);
}

// The "pack_pages + per-page peer copy" reference (the data flow KVAAS uses
// today and the fused kernel replaces). Gathers K/V from indexed local
// slots into a [num_pages, page_size, 2, heads, head_dim] scratch, then
// copies each scratch page to its peer destination
// (peer_dst_ptrs[p] + dst_page_offsets[p]).
void pack_pages_ref(const std::uint8_t* k_src, const std::uint8_t* v_src,
                    const int* slot_ids,
                    const void* const* peer_dst_ptrs,
                    const std::size_t* dst_page_offsets,
                    std::size_t num_pages, std::size_t page_size,
                    std::size_t heads, std::size_t head_dim,
                    std::size_t elem) {
  const std::size_t sb = slot_bytes(heads, head_dim, elem);
  const std::size_t ts = token_stride(heads, head_dim, elem);
  const std::size_t scratch_per_page = page_size * ts;
  std::vector<std::uint8_t> scratch(num_pages * scratch_per_page);
  for (std::size_t p = 0; p < num_pages; ++p) {
    for (std::size_t t = 0; t < page_size; ++t) {
      int slot = slot_ids[p * page_size + t];
      std::size_t src_off = static_cast<std::size_t>(slot) * sb;
      std::size_t dst_off = (p * scratch_per_page) + t * ts;
      std::memcpy(scratch.data() + dst_off, k_src + src_off, sb);
      std::memcpy(scratch.data() + dst_off + sb, v_src + src_off, sb);
    }
  }
  for (std::size_t p = 0; p < num_pages; ++p) {
    auto* dst = static_cast<std::uint8_t*>(const_cast<void*>(peer_dst_ptrs[p])) +
                dst_page_offsets[p];
    std::memcpy(dst, scratch.data() + p * scratch_per_page, scratch_per_page);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Basic correctness: fused donate equals pack_pages + peer copy
// ---------------------------------------------------------------------------
TEST(KvDonate, SinglePageSingleToken) {
  constexpr std::size_t kPageSize = 1, kHeads = 2, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16

  auto k_src = make_src(2, kHeads, kHeadDim, kElem, 0x10);
  auto v_src = make_src(2, kHeads, kHeadDim, kElem, 0x20);
  auto dst = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto ref = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[1] = {1};
  const void* ptrs[1] = {dst.data()};
  const void* rptrs[1] = {ref.data()};
  const std::size_t offs[1] = {0};

  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, ptrs, offs,
                      1, kPageSize, kHeads, kHeadDim, kElem);
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 1, kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], ref[i]);
  // dst[0][K] = k_src[slot 1], dst[0][V] = v_src[slot 1].
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(dst[i], k_src[1 * kSlotBytes + i]);
    ASSERT_EQ(dst[kSlotBytes + i], v_src[1 * kSlotBytes + i]);
  }
}

TEST(KvDonate, SinglePageManyTokens) {
  constexpr std::size_t kPageSize = 8, kHeads = 1, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;               // 32

  auto k_src = make_src(16, kHeads, kHeadDim, kElem, 0xAB);
  auto v_src = make_src(16, kHeads, kHeadDim, kElem, 0xCD);
  auto dst = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto ref = make_page(kPageSize, kHeads, kHeadDim, kElem);

  // Non-contiguous, non-monotonic source slots.
  const int slot_ids[8] = {3, 7, 1, 15, 0, 8, 12, 4};
  const void* ptrs[1] = {dst.data()};
  const void* rptrs[1] = {ref.data()};
  const std::size_t offs[1] = {0};

  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, ptrs, offs,
                      1, kPageSize, kHeads, kHeadDim, kElem);
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 1, kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], ref[i]);
  // Spot-check token 0 -> slot 3.
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(dst[i], k_src[3 * kSlotBytes + i]);
    ASSERT_EQ(dst[kSlotBytes + i], v_src[3 * kSlotBytes + i]);
  }
  // Token 7 -> slot 4.
  const std::uint8_t* t7 = dst.data() + 7 * kTokenStr;
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(t7[i], k_src[4 * kSlotBytes + i]);
    ASSERT_EQ(t7[kSlotBytes + i], v_src[4 * kSlotBytes + i]);
  }
}

TEST(KvDonate, MultiplePagesMultipleAllocations) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlots = 8;

  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x11);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x22);
  auto d0 = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto d1 = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto r0 = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto r1 = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {0, 1, 3, 6};
  const void* ptrs[2] = {d0.data(), d1.data()};
  const void* rptrs[2] = {r0.data(), r1.data()};
  const std::size_t offs[2] = {0, 0};

  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, ptrs, offs,
                      2, kPageSize, kHeads, kHeadDim, kElem);
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 2, kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < d0.size(); ++i) ASSERT_EQ(d0[i], r0[i]);
  for (std::size_t i = 0; i < d1.size(); ++i) ASSERT_EQ(d1[i], r1[i]);
}

TEST(KvDonate, PageOffsetIsHonoured) {
  constexpr std::size_t kPageSize = 1, kHeads = 1, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 8
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;               // 16

  // Two tokens back-to-back; offset selects the second.
  auto k_src = make_src(2, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(2, kHeads, kHeadDim, kElem, 0x50);
  auto buf = make_page(2, kHeads, kHeadDim, kElem);
  auto ref = make_page(2, kHeads, kHeadDim, kElem);
  const int slot_ids[1] = {1};
  const void* ptrs[1] = {buf.data()};
  const void* rptrs[1] = {ref.data()};
  const std::size_t offs[1] = {kTokenStr};  // write into token 1

  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, ptrs, offs,
                      1, kPageSize, kHeads, kHeadDim, kElem);
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 1, kPageSize, kHeads, kHeadDim, kElem);

  for (std::size_t i = 0; i < buf.size(); ++i) ASSERT_EQ(buf[i], ref[i]);
  // Token 1 holds slot 1; token 0 stays at sentinel.
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(buf[kTokenStr + i], k_src[1 * kSlotBytes + i]);
    ASSERT_EQ(buf[kTokenStr + kSlotBytes + i], v_src[1 * kSlotBytes + i]);
    ASSERT_EQ(buf[i], 0xDD);
    ASSERT_EQ(buf[kSlotBytes + i], 0xDD);
  }
}

TEST(KvDonate, ZeroPagesIsNoOp) {
  auto k_src = make_src(4, 2, 8, 2, 0x10);
  auto v_src = make_src(4, 2, 8, 2, 0x20);
  auto dst = make_page(4, 2, 8, 2);
  auto before = dst;
  p2p_kv_donate_layer(k_src.data(), v_src.data(), nullptr, nullptr, nullptr,
                      0, 64, 2, 8, 2);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], before[i]);
}

// ---------------------------------------------------------------------------
// Gather semantics: repeated / non-monotonic source slots are allowed
// (unlike the restore's unique-destination scatter)
// ---------------------------------------------------------------------------
TEST(KvDonate, RepeatedSourceSlotsAreAllowed) {
  constexpr std::size_t kPageSize = 4, kHeads = 1, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 16
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;               // 32
  auto k_src = make_src(4, kHeads, kHeadDim, kElem, 0x77);
  auto v_src = make_src(4, kHeads, kHeadDim, kElem, 0x99);
  auto dst = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto ref = make_page(kPageSize, kHeads, kHeadDim, kElem);
  // Slots 1, 1, 3, 3 -- two tokens read the same source slot (gather).
  const int slot_ids[4] = {1, 1, 3, 3};
  const void* ptrs[1] = {dst.data()};
  const void* rptrs[1] = {ref.data()};
  const std::size_t offs[1] = {0};

  EXPECT_NO_THROW(p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids,
                                       ptrs, offs, 1, kPageSize, kHeads,
                                       kHeadDim, kElem));
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 1, kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], ref[i]);
  // Tokens 0 and 1 are identical (both read slot 1).
  for (std::size_t i = 0; i < kTokenStr; ++i)
    ASSERT_EQ(dst[i], dst[kTokenStr + i]);
  for (std::size_t i = 0; i < kSlotBytes; ++i)
    ASSERT_EQ(dst[i], k_src[kSlotBytes + i]);
}

// Non-monotonic order across pages: slot_ids need not be sorted.
TEST(KvDonate, NonMonotonicSlotsAcrossPages) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  auto k_src = make_src(16, kHeads, kHeadDim, kElem, 0x0A);
  auto v_src = make_src(16, kHeads, kHeadDim, kElem, 0x1A);
  std::vector<std::vector<std::uint8_t>> dst(3), ref(3);
  for (int i = 0; i < 3; ++i) {
    dst[i] = make_page(kPageSize, kHeads, kHeadDim, kElem);
    ref[i] = make_page(kPageSize, kHeads, kHeadDim, kElem);
  }
  // Deliberately non-monotonic: 15, 0, 7, 2, 11, 4.
  const int slot_ids[6] = {15, 0, 7, 2, 11, 4};
  const void* ptrs[3] = {dst[0].data(), dst[1].data(), dst[2].data()};
  const void* rptrs[3] = {ref[0].data(), ref[1].data(), ref[2].data()};
  const std::size_t offs[3] = {0, 0, 0};

  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, ptrs, offs,
                      3, kPageSize, kHeads, kHeadDim, kElem);
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 3, kPageSize, kHeads, kHeadDim, kElem);
  for (int p = 0; p < 3; ++p)
    for (std::size_t i = 0; i < dst[p].size(); ++i) ASSERT_EQ(dst[p][i], ref[p][i]);
}

// ---------------------------------------------------------------------------
// Two-stage reference: fused must produce byte-identical results, and the
// two-stage is the copy-engine fallback.
// ---------------------------------------------------------------------------
TEST(KvDonate, FusedEqualsTwoStageSinglePage) {
  constexpr std::size_t kPageSize = 4, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 16;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x55);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x66);
  auto fused = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto two = make_page(kPageSize, kHeads, kHeadDim, kElem);
  int slot_ids[4] = {2, 7, 11, 3};
  const void* fptrs[1] = {fused.data()};
  const void* tptrs[1] = {two.data()};
  const std::size_t offs[1] = {0};

  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, fptrs, offs,
                      1, kPageSize, kHeads, kHeadDim, kElem);
  p2p_kv_donate_layer_twostage(k_src.data(), v_src.data(), slot_ids, tptrs,
                               offs, 1, kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < fused.size(); ++i) ASSERT_EQ(fused[i], two[i]);
}

TEST(KvDonate, FusedEqualsTwoStageMultiPage) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 32, kElem = 2;
  constexpr std::size_t kSlots = 16;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x12);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x34);
  std::vector<std::vector<std::uint8_t>> f(3), t(3);
  for (int i = 0; i < 3; ++i) {
    f[i] = make_page(kPageSize, kHeads, kHeadDim, kElem);
    t[i] = make_page(kPageSize, kHeads, kHeadDim, kElem);
  }
  int slot_ids[6] = {7, 2, 13, 0, 5, 9};
  const void* fptrs[3] = {f[0].data(), f[1].data(), f[2].data()};
  const void* tptrs[3] = {t[0].data(), t[1].data(), t[2].data()};
  const std::size_t offs[3] = {0, 0, 0};

  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, fptrs, offs,
                      3, kPageSize, kHeads, kHeadDim, kElem);
  p2p_kv_donate_layer_twostage(k_src.data(), v_src.data(), slot_ids, tptrs,
                               offs, 3, kPageSize, kHeads, kHeadDim, kElem);
  for (int p = 0; p < 3; ++p)
    for (std::size_t i = 0; i < f[p].size(); ++i) ASSERT_EQ(f[p][i], t[p][i]);
}

TEST(KvDonate, TwoStageAsyncEqualsSync) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x21);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x9A);
  auto a = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto b = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {1, 6, 3, 7};
  const void* aptrs[1] = {a.data()};
  const void* bptrs[1] = {b.data()};
  const std::size_t offs[1] = {0};

  Stream s;
  std::size_t before = s.submitted();
  p2p_kv_donate_layer_twostage(k_src.data(), v_src.data(), slot_ids, aptrs,
                               offs, 1, kPageSize, kHeads, kHeadDim, kElem, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // one task per page (1 page here)
  s.wait();
  p2p_kv_donate_layer_twostage(k_src.data(), v_src.data(), slot_ids, bptrs,
                               offs, 1, kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
}

TEST(KvDonate, FusedAsyncMatchesSync) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x40);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x80);
  auto a = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto b = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {2, 5, 1, 7};
  const void* aptrs[1] = {a.data()};
  const void* bptrs[1] = {b.data()};
  const std::size_t offs[1] = {0};

  Stream s;
  std::size_t before = s.submitted();
  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, aptrs, offs,
                      1, kPageSize, kHeads, kHeadDim, kElem, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one stream task
  s.wait();
  p2p_kv_donate_layer(k_src.data(), v_src.data(), slot_ids, bptrs, offs,
                      1, kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
}

// ---------------------------------------------------------------------------
// kv_gather (the first stage of the two-path)
// ---------------------------------------------------------------------------
TEST(KvGather, MatchesPackPagesScratch) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;
  constexpr std::size_t kScratchPerPage = kPageSize * 2 * kSlotBytes;
  constexpr std::size_t kSlots = 8;

  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x70);
  std::vector<std::uint8_t> scratch(2 * kScratchPerPage);
  const int slot_ids[4] = {3, 1, 6, 4};
  kv_gather(scratch.data(), k_src.data(), v_src.data(), slot_ids, 2,
            kPageSize, kHeads, kHeadDim, kElem);

  // Reference: same gather by hand.
  std::vector<std::uint8_t> ref(2 * kScratchPerPage);
  for (int p = 0; p < 2; ++p) {
    for (int t = 0; t < 2; ++t) {
      int slot = slot_ids[p * 2 + t];
      std::size_t src_off = static_cast<std::size_t>(slot) * kSlotBytes;
      std::size_t dst_off = static_cast<std::size_t>(p) * kScratchPerPage +
                            static_cast<std::size_t>(t) * 2 * kSlotBytes;
      std::memcpy(ref.data() + dst_off, k_src.data() + src_off, kSlotBytes);
      std::memcpy(ref.data() + dst_off + kSlotBytes, v_src.data() + src_off,
                  kSlotBytes);
    }
  }
  for (std::size_t i = 0; i < scratch.size(); ++i) ASSERT_EQ(scratch[i], ref[i]);
}

TEST(KvGather, AsyncMatchesSync) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kScratchPerPage = kPageSize * 2 * kHeads * kHeadDim * kElem;
  constexpr std::size_t kSlots = 8;

  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x70);
  const int slot_ids[4] = {3, 1, 6, 4};

  std::vector<std::uint8_t> sync(2 * kScratchPerPage);
  std::vector<std::uint8_t> asy(2 * kScratchPerPage);
  kv_gather(sync.data(), k_src.data(), v_src.data(), slot_ids, 2,
            kPageSize, kHeads, kHeadDim, kElem);
  Stream s;
  std::size_t before = s.submitted();
  kv_gather(asy.data(), k_src.data(), v_src.data(), slot_ids, 2,
            kPageSize, kHeads, kHeadDim, kElem, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one task
  s.wait();
  for (std::size_t i = 0; i < sync.size(); ++i) ASSERT_EQ(sync[i], asy[i]);
}

TEST(KvGather, RepeatedSlots) {
  constexpr std::size_t kPageSize = 2, kHeads = 1, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 8
  constexpr std::size_t kScratchPerPage = kPageSize * 2 * kSlotBytes;  // 32
  auto k_src = make_src(4, kHeads, kHeadDim, kElem, 0x10);
  auto v_src = make_src(4, kHeads, kHeadDim, kElem, 0x20);
  std::vector<std::uint8_t> scratch(kScratchPerPage);
  const int slot_ids[2] = {1, 1};  // both tokens read slot 1
  EXPECT_NO_THROW(kv_gather(scratch.data(), k_src.data(), v_src.data(),
                            slot_ids, 1, kPageSize, kHeads, kHeadDim, kElem));
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(scratch[i], k_src[kSlotBytes + i]);                  // token 0 K
    ASSERT_EQ(scratch[kSlotBytes + i], v_src[kSlotBytes + i]);     // token 0 V
    ASSERT_EQ(scratch[2 * kSlotBytes + i], k_src[kSlotBytes + i]); // token 1 K
    ASSERT_EQ(scratch[3 * kSlotBytes + i], v_src[kSlotBytes + i]); // token 1 V
  }
}

// ---------------------------------------------------------------------------
// Adaptive dispatch helpers (pure, host-testable)
// ---------------------------------------------------------------------------
TEST(DonateDispatch, DefaultsAndConfig) {
  auto [mode, min_pages] = vkernels::comm::donate_dispatch_config();
  EXPECT_EQ(static_cast<int>(mode),
            static_cast<int>(vkernels::comm::DonateDispatchMode::kAdaptive));
  EXPECT_EQ(min_pages, 1u);
}

TEST(DonateDispatch, ForceDirect) {
  vkernels::comm::set_donate_dispatch(vkernels::comm::DonateDispatchMode::kForceDirect);
  EXPECT_TRUE(vkernels::comm::prefer_direct_store(4, 1024));
  EXPECT_TRUE(vkernels::comm::prefer_direct_store(1, 16));  // any size
  vkernels::comm::set_donate_dispatch();  // reset to defaults
}

TEST(DonateDispatch, ForceCopyEngine) {
  vkernels::comm::set_donate_dispatch(vkernels::comm::DonateDispatchMode::kForceCopyEngine);
  EXPECT_FALSE(vkernels::comm::prefer_direct_store(4, 1024 * 1024));
  EXPECT_FALSE(vkernels::comm::prefer_direct_store(16, 16));
  vkernels::comm::set_donate_dispatch();  // reset
}

TEST(DonateDispatch, ZeroBytesNeverTakesKernel) {
  EXPECT_FALSE(vkernels::comm::prefer_direct_store(8, 0));
  vkernels::comm::set_donate_dispatch(vkernels::comm::DonateDispatchMode::kForceDirect);
  EXPECT_FALSE(vkernels::comm::prefer_direct_store(8, 0));  // even forced
  vkernels::comm::set_donate_dispatch();  // reset
}

TEST(DonateDispatch, MinPagesFloor) {
  // With a high min-pages floor, small page counts fall back to the copy
  // engine even in adaptive mode.
  vkernels::comm::set_donate_dispatch(vkernels::comm::DonateDispatchMode::kAdaptive,
                                      100);
  EXPECT_FALSE(vkernels::comm::prefer_direct_store(4, 1024 * 1024));
  EXPECT_TRUE(vkernels::comm::prefer_direct_store(200, 1024 * 1024));
  vkernels::comm::set_donate_dispatch();  // reset to defaults
}

TEST(DonateDispatch, EstimatesAreNonNegative) {
  for (std::size_t np : {0u, 1u, 4u, 16u}) {
    for (std::size_t tb : {0u, 16u, 1024u, 1024u * 1024u, 64u * 1024u * 1024u}) {
      EXPECT_GE(vkernels::comm::est_direct_store_us(np, tb), 0.0);
      EXPECT_GE(vkernels::comm::est_copy_engine_donate_us(np, tb), 0.0);
      if (tb > 0)
        EXPECT_GT(vkernels::comm::est_copy_engine_donate_us(np, tb),
                  vkernels::comm::est_direct_store_us(np, tb));
    }
  }
}

// ---------------------------------------------------------------------------
// Prepared plan (host reference, issue #36)
// ---------------------------------------------------------------------------
TEST(KvDonatePlan, SyncExecuteEqualsOneShot) {
  constexpr std::size_t kPageSize = 4, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 16;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x11);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x22);
  auto dst = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto ref = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {3, 1, 14, 7};
  const void* ptrs[1] = {dst.data()};
  const void* rptrs[1] = {ref.data()};

  P2PKvDonatePlan plan(kSlots, kHeads, kHeadDim, kElem, slot_ids, ptrs, 1,
                       kPageSize);
  EXPECT_EQ(plan.num_pages(), 1u);
  EXPECT_EQ(plan.page_size(), kPageSize);
  EXPECT_EQ(plan.num_slots(), kSlots);
  EXPECT_EQ(plan.num_kv_heads(), kHeads);
  EXPECT_EQ(plan.head_dim(), kHeadDim);
  EXPECT_EQ(plan.elem_size(), kElem);
  EXPECT_EQ(plan.total_bytes(), kPageSize * 2 * kHeads * kHeadDim * kElem);
  EXPECT_EQ(plan.scratch_bytes(), plan.total_bytes());
  plan.execute(k_src.data(), v_src.data(), 0);

  const std::size_t offs[1] = {0};
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 1, kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], ref[i]);
}

// The plan's single scalar offset must match per-page dst_page_offsets.
TEST(KvDonatePlan, OffsetExecuteMatchesPerPageOffsets) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 32
  constexpr std::size_t kTokenStr = 2 * kSlotBytes;                // 64
  constexpr std::size_t kLayerBytes = kPageSize * kTokenStr;       // 128
  constexpr std::size_t kSlots = 8;

  // Each peer buffer holds two layers back-to-back.
  auto buf0 = make_page(kPageSize * 2, kHeads, kHeadDim, kElem, 0x00);
  auto buf1 = make_page(kPageSize * 2, kHeads, kHeadDim, kElem, 0x00);
  auto ref0 = make_page(kPageSize * 2, kHeads, kHeadDim, kElem, 0x00);
  auto ref1 = make_page(kPageSize * 2, kHeads, kHeadDim, kElem, 0x00);
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x20);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0xA0);
  const int slot_ids[4] = {2, 5, 1, 6};
  const void* ptrs[2] = {buf0.data(), buf1.data()};
  const void* rptrs[2] = {ref0.data(), ref1.data()};

  P2PKvDonatePlan plan(kSlots, kHeads, kHeadDim, kElem, slot_ids, ptrs, 2,
                       kPageSize);
  // Write into the SECOND layer of each peer buffer.
  plan.execute(k_src.data(), v_src.data(), kLayerBytes);

  // Reference: per-page offset = kLayerBytes into the same buffers.
  const std::size_t offs[2] = {kLayerBytes, kLayerBytes};
  pack_pages_ref(k_src.data(), v_src.data(), slot_ids, rptrs, offs,
                 2, kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < buf0.size(); ++i) ASSERT_EQ(buf0[i], ref0[i]);
  for (std::size_t i = 0; i < buf1.size(); ++i) ASSERT_EQ(buf1[i], ref1[i]);
  // The first layer of each peer buffer is untouched (zero).
  for (std::size_t i = 0; i < kLayerBytes; ++i) {
    ASSERT_EQ(buf0[i], 0x00);
    ASSERT_EQ(buf1[i], 0x00);
  }
}

TEST(KvDonatePlan, AsyncExecuteEqualsSync) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x31);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x42);
  auto a = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto b = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {0, 4, 7, 2};
  const void* aptrs[1] = {a.data()};
  const void* bptrs[1] = {b.data()};

  P2PKvDonatePlan plan(kSlots, kHeads, kHeadDim, kElem, slot_ids, aptrs, 1,
                       kPageSize);
  Stream s;
  std::size_t before = s.submitted();
  plan.execute(k_src.data(), v_src.data(), 0, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one stream task
  s.wait();

  P2PKvDonatePlan sync(kSlots, kHeads, kHeadDim, kElem, slot_ids, bptrs, 1,
                       kPageSize);
  sync.execute(k_src.data(), v_src.data(), 0);
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
}

TEST(KvDonatePlan, ExecuteViaScratchEqualsExecute) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x51);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x62);
  auto exec = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto scr = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {3, 0, 6, 5};
  const void* eptrs[1] = {exec.data()};
  const void* sptrs[1] = {scr.data()};

  P2PKvDonatePlan plan(kSlots, kHeads, kHeadDim, kElem, slot_ids, eptrs, 1,
                       kPageSize);
  P2PKvDonatePlan plan_s(kSlots, kHeads, kHeadDim, kElem, slot_ids, sptrs, 1,
                         kPageSize);
  std::vector<std::uint8_t> scratch(plan.scratch_bytes());
  plan.execute(k_src.data(), v_src.data(), 0);
  plan_s.execute_via_scratch(k_src.data(), v_src.data(), scratch.data(), 0);
  for (std::size_t i = 0; i < exec.size(); ++i) ASSERT_EQ(exec[i], scr[i]);
}

TEST(KvDonatePlan, ExecuteViaScratchAsyncEqualsSync) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x71);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x82);
  auto a = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto b = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {1, 5, 2, 7};
  const void* aptrs[1] = {a.data()};
  const void* bptrs[1] = {b.data()};

  P2PKvDonatePlan plan_a(kSlots, kHeads, kHeadDim, kElem, slot_ids, aptrs, 1,
                         kPageSize);
  P2PKvDonatePlan plan_b(kSlots, kHeads, kHeadDim, kElem, slot_ids, bptrs, 1,
                         kPageSize);
  std::vector<std::uint8_t> scratch(plan_a.scratch_bytes());
  Stream s;
  std::size_t before = s.submitted();
  plan_a.execute_via_scratch(k_src.data(), v_src.data(), scratch.data(), 0, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one stream task
  s.wait();
  plan_b.execute_via_scratch(k_src.data(), v_src.data(), scratch.data(), 0);
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
}

// The headline benefit of issue #36: build the plan ONCE and reuse it across
// 40 layer buffers (Qwen3-14B has 40 KV-cache layers) with no per-layer
// allocation, H2D descriptor upload, or local packed-KV scratch.
TEST(KvDonatePlan, PrepareOnceExecuteFortyTimes) {
  constexpr std::size_t kNumLayers = 40;
  constexpr std::size_t kPageSize = 4, kHeads = 8, kHeadDim = 128, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 2048
  constexpr std::size_t kLayerBytes = kPageSize * 2 * kSlotBytes;
  constexpr std::size_t kSlots = 256;

  // 40 peer buffers, each holding all 40 layers back-to-back.
  std::vector<std::vector<std::uint8_t>> dst(kNumLayers);
  std::vector<const void*> ptrs(kNumLayers);
  for (std::size_t i = 0; i < kNumLayers; ++i) {
    dst[i].assign(kNumLayers * kLayerBytes, 0xEE);
    ptrs[i] = dst[i].data();
  }
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x10);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x20);

  // Page 0 reads slots [0..3], page 1 reads [4..7], ..., page 39 reads
  // [156..159] (within kSlots = 256).
  std::vector<int> slot_ids(kNumLayers * kPageSize);
  for (std::size_t p = 0; p < kNumLayers; ++p)
    for (std::size_t t = 0; t < kPageSize; ++t)
      slot_ids[p * kPageSize + t] = static_cast<int>((p * 4 + t) % kSlots);

  // Prepare ONCE.
  P2PKvDonatePlan plan(kSlots, kHeads, kHeadDim, kElem, slot_ids.data(),
                       ptrs.data(), kNumLayers, kPageSize);
  EXPECT_EQ(plan.num_pages(), kNumLayers);
  EXPECT_EQ(plan.total_bytes(), kNumLayers * kLayerBytes);
  EXPECT_EQ(plan.scratch_bytes(), plan.total_bytes());

  // Execute 40 times, once per layer, into layer L of every peer buffer.
  // The same metadata is reused; only the source buffers and the scalar
  // layer offset change.
  for (std::size_t layer = 0; layer < kNumLayers; ++layer) {
    // Vary the source seed per layer so cross-layer crosstalk is visible.
    auto k_l = make_src(kSlots, kHeads, kHeadDim, kElem,
                        static_cast<std::uint8_t>(0x30 + layer));
    auto v_l = make_src(kSlots, kHeads, kHeadDim, kElem,
                        static_cast<std::uint8_t>(0x80 + layer));
    plan.execute(k_l.data(), v_l.data(), layer * kLayerBytes);
  }

  // Verify each peer buffer's layer L came from source slot for page L.
  for (std::size_t p = 0; p < kNumLayers; ++p) {
    auto k_l = make_src(kSlots, kHeads, kHeadDim, kElem,
                        static_cast<std::uint8_t>(0x30 + p));
    auto v_l = make_src(kSlots, kHeads, kHeadDim, kElem,
                        static_cast<std::uint8_t>(0x80 + p));
    const std::uint8_t* page = dst[p].data() + p * kLayerBytes;
    for (std::size_t t = 0; t < kPageSize; ++t) {
      int slot = slot_ids[p * kPageSize + t];
      const std::uint8_t* expect_k = k_l.data() + slot * kSlotBytes;
      const std::uint8_t* expect_v = v_l.data() + slot * kSlotBytes;
      const std::uint8_t* got_k = page + t * 2 * kSlotBytes;
      const std::uint8_t* got_v = got_k + kSlotBytes;
      for (std::size_t i = 0; i < kSlotBytes; ++i) {
        ASSERT_EQ(got_k[i], expect_k[i]);
        ASSERT_EQ(got_v[i], expect_v[i]);
      }
    }
  }
}

TEST(KvDonatePlan, ZeroPagesIsNoOp) {
  auto k_src = make_src(4, 2, 8, 2, 0x10);
  auto v_src = make_src(4, 2, 8, 2, 0x20);
  auto dst = make_page(1, 2, 8, 2);
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {dst.data()};
  P2PKvDonatePlan plan(4, 2, 8, 2, slot_ids, ptrs, 0, 64);
  EXPECT_EQ(plan.num_pages(), 0u);
  EXPECT_EQ(plan.total_bytes(), 0u);
  EXPECT_EQ(plan.scratch_bytes(), 0u);
  auto before = dst;
  plan.execute(k_src.data(), v_src.data(), 0);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], before[i]);
}

// ---------------------------------------------------------------------------
// Device-slot plans (host reference mirrors the CUDA path's contract)
// ---------------------------------------------------------------------------
TEST(KvDonatePlan, DeviceSlotsSkipsSlotValidation) {
  auto p = make_page(2, 2, 4, 2, 0x10);
  const int slot_ids[2] = {1, 1};   // repeated -- would be fine anyway, but
  // demonstrates the device-slot variant accepts whatever the caller gives.
  const void* ptrs[1] = {p.data()};
  EXPECT_NO_THROW(P2PKvDonatePlan(from_device_slots, 4,
                                   2, 4, 2, slot_ids, ptrs, 1, 2));
}

TEST(KvDonatePlan, DeviceSlotsStillValidatesShape) {
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {(void*)0x1000};
  EXPECT_THROW(P2PKvDonatePlan(from_device_slots,
                                4, 2, 4, 4, slot_ids, ptrs, 1, 1),
               std::invalid_argument);  // non-BF16 elem_size
}

TEST(KvDonatePlan, DeviceSlotsInt64MatchesInt32) {
  constexpr std::size_t kPageSize = 2, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x90);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0xA0);
  auto a = make_page(kPageSize, kHeads, kHeadDim, kElem);
  auto b = make_page(kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[4] = {3, 1, 6, 4};
  const std::int64_t slot_ids_i64[4] = {3, 1, 6, 4};
  const void* aptrs[1] = {a.data()};
  const void* bptrs[1] = {b.data()};

  P2PKvDonatePlan plan_i32(from_device_slots, kSlots, kHeads, kHeadDim,
                           kElem, slot_ids, aptrs, 1, kPageSize);
  P2PKvDonatePlan plan_i64(from_device_slots_int64, kSlots, kHeads, kHeadDim,
                           kElem, slot_ids_i64, bptrs, 1, kPageSize);
  plan_i32.execute(k_src.data(), v_src.data(), 0);
  plan_i64.execute(k_src.data(), v_src.data(), 0);
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
}

TEST(KvDonatePlan, DeviceSlotsInt64StillValidatesShape) {
  const std::int64_t slot_ids[1] = {0};
  const void* ptrs[1] = {(void*)0x1000};
  EXPECT_THROW(P2PKvDonatePlan(from_device_slots_int64,
                                4, 2, 4, 4, slot_ids, ptrs, 1, 1),
               std::invalid_argument);  // non-BF16 elem_size
}

TEST(KvDonatePlan, DeviceSlotsZeroPagesIsNoOp) {
  auto dst = make_page(1, 2, 8, 2);
  int slot_ids[1] = {0};
  const void* ptrs[1] = {dst.data()};
  P2PKvDonatePlan plan(from_device_slots, 4, 2, 8, 2, slot_ids, ptrs, 0, 64);
  EXPECT_EQ(plan.num_pages(), 0u);
  auto before = dst;
  plan.execute(nullptr, nullptr, 0);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], before[i]);
}

// ---------------------------------------------------------------------------
// Contract checks move OUT of the hot path and INTO the constructor: the
// plan rejects a negative / out-of-range slot / non-BF16 / null dst at create
// time, so execute() can stay free of checks (only k_src/v_src null). The
// donate does NOT require uniqueness (gather semantics).
// ---------------------------------------------------------------------------
TEST(KvDonatePlan, RejectsNegativeSlotAtCreate) {
  auto p = make_page(1, 2, 4, 2, 0x10);
  const int slot_ids[1] = {-1};
  const void* ptrs[1] = {p.data()};
  EXPECT_THROW(P2PKvDonatePlan(4, 2, 4, 2, slot_ids, ptrs, 1, 1),
               std::invalid_argument);
}

TEST(KvDonatePlan, RejectsOutOfRangeSlotAtCreate) {
  auto p = make_page(1, 2, 4, 2, 0x10);
  const int slot_ids[1] = {4};     // == num_slots, out of range
  const void* ptrs[1] = {p.data()};
  EXPECT_THROW(P2PKvDonatePlan(4, 2, 4, 2, slot_ids, ptrs, 1, 1),
               std::invalid_argument);
}

TEST(KvDonatePlan, RejectsNonBF16AtCreate) {
  auto p = make_page(1, 2, 4, 4, 0x10);  // elem_size 4 -> non-BF16
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {p.data()};
  EXPECT_THROW(P2PKvDonatePlan(4, 2, 4, 4, slot_ids, ptrs, 1, 1),
               std::invalid_argument);
}

TEST(KvDonatePlan, RejectsNullPeerDstAtCreate) {
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {nullptr};
  EXPECT_THROW(P2PKvDonatePlan(4, 2, 4, 2, slot_ids, ptrs, 1, 1),
               std::invalid_argument);
}

// With the destination moved to execute(), null k_src/v_src is the only check
// left on the hot path and is rejected there (matching the one-shot).
TEST(KvDonatePlan, RejectsNullSrcAtExecute) {
  auto p = make_page(1, 2, 4, 2, 0x10);
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {p.data()};
  P2PKvDonatePlan plan(4, 2, 4, 2, slot_ids, ptrs, 1, 1);
  EXPECT_THROW(plan.execute(nullptr, p.data(), 0), std::invalid_argument);
  EXPECT_THROW(plan.execute(p.data(), nullptr, 0), std::invalid_argument);
}

TEST(KvDonatePlan, RejectsNullScratchAtExecuteViaScratch) {
  auto p = make_page(1, 2, 4, 2, 0x10);
  const int slot_ids[1] = {0};
  const void* ptrs[1] = {p.data()};
  P2PKvDonatePlan plan(4, 2, 4, 2, slot_ids, ptrs, 1, 1);
  EXPECT_THROW(plan.execute_via_scratch(p.data(), p.data(), nullptr, 0),
               std::invalid_argument);
}
