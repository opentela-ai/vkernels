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
