// tests/comm/test_kv_scatter.cpp
//
// Host-reference tests for the fused indexed K/V layer scatter kernel
// (issue #1). These are the correctness oracle: they exercise byte-exact
// results against the PyTorch two-write reference
// (`k_dst[slot_ids] = src[:, :, 0]; v_dst[slot_ids] = src[:, :, 1]`) across
// the full acceptance matrix (BF16/FP16 × head dims 64/128/256 × page sizes
// 1/16/32/64 × 1..128 pages × sparse/non-monotonic UNIQUE slot maps ×
// int32/int64 slot ids), validate the contract checks (null pointers, zero
// dimensions, non-BF16/FP16 elem_size, negative/out-of-range AND DUPLICATE
// destination slots -- the scatter, unlike the gather, writes disjoint
// destinations so a duplicate is a contract violation), and pin the async
// stream contract to exactly one task per launch.
//
// The kernel copies raw bytes (no type-specific arithmetic), so it is
// bit-exact for both BF16 and FP16; the "dtype" axis of the matrix is
// therefore exercised by elem_size == 2 with distinct per-source seeds that
// would surface any mis-routing.
#include "minitest.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "vkernels/comm/kv_scatter.hpp"
#include "vkernels/core/stream.hpp"

using vkernels::Stream;
using vkernels::comm::kv_scatter_layer;

namespace {

inline std::size_t kv_slot_bytes(std::size_t heads, std::size_t head_dim,
                                 std::size_t elem) {
  return heads * head_dim * elem;
}
inline std::size_t kv_token_stride(std::size_t heads, std::size_t head_dim,
                                   std::size_t elem) {
  return 2 * kv_slot_bytes(heads, head_dim, elem);
}

// Packed [num_pages, page_size, 2, heads, head_dim, elem] source filled
// with a per-byte seed so a mis-routed write is immediately obvious.
std::vector<std::uint8_t> make_src(std::size_t num_pages, std::size_t page_size,
                                   std::size_t heads, std::size_t head_dim,
                                   std::size_t elem, std::uint8_t seed) {
  const std::size_t total = num_pages * page_size * 2 * heads * head_dim * elem;
  std::vector<std::uint8_t> v(total);
  for (std::size_t i = 0; i < total; ++i)
    v[i] = static_cast<std::uint8_t>(seed + i);
  return v;
}

// A flat [num_slots, heads, head_dim, elem] destination filled with a
// sentinel so untouched bytes are detected. k_dst and v_dst are separate
// allocations of this shape.
std::vector<std::uint8_t> make_dst(std::size_t num_slots, std::size_t heads,
                                   std::size_t head_dim, std::size_t elem,
                                   std::uint8_t fill = 0xCC) {
  const std::size_t total = num_slots * heads * head_dim * elem;
  return std::vector<std::uint8_t>(total, fill);
}

// The PyTorch two-write reference the kernel replaces, in bytes:
//   k_dst[slot_ids] = src[:, :, 0]
//   v_dst[slot_ids] = src[:, :, 1]
template <typename SlotT>
std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>> two_scatter_ref(
    const std::vector<std::uint8_t>& src, const std::vector<SlotT>& slot_ids,
    std::size_t num_slots, std::size_t num_pages, std::size_t page_size,
    std::size_t heads, std::size_t head_dim, std::size_t elem) {
  const std::size_t sb = kv_slot_bytes(heads, head_dim, elem);
  const std::size_t ts = kv_token_stride(heads, head_dim, elem);
  const std::size_t page_bytes = page_size * ts;
  auto k = make_dst(num_slots, heads, head_dim, elem, 0xCC);
  auto v = make_dst(num_slots, heads, head_dim, elem, 0xCC);
  for (std::size_t p = 0; p < num_pages; ++p) {
    const std::uint8_t* src_page = src.data() + p * page_bytes;
    for (std::size_t t = 0; t < page_size; ++t) {
      const std::size_t slot =
          static_cast<std::size_t>(slot_ids[p * page_size + t]);
      const std::size_t src_off = t * ts;
      const std::size_t dst_off = slot * sb;
      std::memcpy(k.data() + dst_off, src_page + src_off, sb);
      std::memcpy(v.data() + dst_off, src_page + src_off + sb, sb);
    }
  }
  return {k, v};
}

// Deterministic LCG so the same slot map is reproducible across runs.
struct Lcg {
  std::uint64_t s;
  explicit Lcg(std::uint64_t seed) : s(seed) {}
  std::uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(s >> 32);
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Basic correctness
// ---------------------------------------------------------------------------
TEST(KvScatterLayer, SinglePageSingleToken) {
  constexpr std::size_t kHeads = 2, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;
  constexpr std::size_t kSlots = 8;

  auto src = make_src(1, 1, kHeads, kHeadDim, kElem, 0x30);
  const int slot_ids[1] = {3};  // single token -> slot 3
  auto k = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v = make_dst(kSlots, kHeads, kHeadDim, kElem);
  kv_scatter_layer(k.data(), v.data(), slot_ids, false, kSlots, src.data(), 1,
                   1, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(k[3 * kSlotBytes + i], src[i]);                  // slot 3 K
    ASSERT_EQ(v[3 * kSlotBytes + i], src[kSlotBytes + i]);     // slot 3 V
  }
}

TEST(KvScatterLayer, MatchesTwoScatterReference) {
  constexpr std::size_t kSlots = 8, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kPageSize = 2, kNumPages = 2;
  auto src = make_src(kNumPages, kPageSize, kHeads, kHeadDim, kElem, 0x30);
  const int slot_ids[4] = {3, 1, 6, 4};  // unique, non-monotonic across pages
  auto k = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v = make_dst(kSlots, kHeads, kHeadDim, kElem);
  kv_scatter_layer(k.data(), v.data(), slot_ids, false, kSlots, src.data(),
                   kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  std::vector<int> ids(slot_ids, slot_ids + 4);
  auto ref = two_scatter_ref(src, ids, kSlots, kNumPages, kPageSize, kHeads,
                             kHeadDim, kElem);
  for (std::size_t i = 0; i < k.size(); ++i) ASSERT_EQ(k[i], ref.first[i]);
  for (std::size_t i = 0; i < v.size(); ++i) ASSERT_EQ(v[i], ref.second[i]);
}

// ---------------------------------------------------------------------------
// Full acceptance matrix: BF16/FP16 (elem=2) × head dims {64,128,256} ×
// page sizes {1,16,32,64} × page counts {1, 16, 64, 128} ×
// {dense, sparse, non-monotonic} UNIQUE slot maps × {int32, int64}.
//
// num_slots is chosen larger than num_pages * page_size with headroom so
// sparse maps (distinct slots drawn from [0, num_slots)) exercise wide
// destination strides, and the 128-page case clears the issue's ">=128
// pages" bar. Slot maps are UNIQUE by construction (scatter writes disjoint
// destinations; the gather's repeats are not allowed here). Randomness is
// seeded for reproducibility.
// ---------------------------------------------------------------------------
struct MatrixCase {
  std::size_t head_dim;
  std::size_t page_size;
  std::size_t num_pages;
  const char* map_kind;  // "dense" | "sparse" | "nonmono"
};

std::vector<MatrixCase> matrix_cases() {
  return {
      {64, 1, 1, "dense"},
      {64, 16, 16, "dense"},
      {64, 32, 64, "sparse"},
      {64, 64, 128, "nonmono"},
      {128, 1, 16, "sparse"},
      {128, 16, 64, "nonmono"},
      {128, 32, 128, "dense"},
      {128, 64, 1, "sparse"},
      {256, 1, 64, "nonmono"},
      {256, 16, 128, "sparse"},
      {256, 32, 1, "dense"},
      {256, 64, 16, "nonmono"},
  };
}

// Build a UNIQUE slot map of the requested kind from `num_slots` slots.
// `total_tokens <= num_slots` is required by every kind.
std::vector<std::int64_t> make_map(const char* kind, std::size_t total_tokens,
                                   std::size_t num_slots, Lcg& rng) {
  std::vector<std::int64_t> ids(total_tokens);
  if (std::strcmp(kind, "dense") == 0) {
    // A permutation of a contiguous range [0, total_tokens) -- worst case
    // for the kernel's destination-write pattern is fully non-local, so
    // shuffle. All distinct.
    std::vector<std::int64_t> perm(total_tokens);
    std::iota(perm.begin(), perm.end(), 0);
    for (std::size_t i = total_tokens; i > 1; --i) {
      std::size_t j = rng.next() % i;
      std::swap(perm[i - 1], perm[j]);
    }
    for (std::size_t i = 0; i < total_tokens; ++i) ids[i] = perm[i];
  } else if (std::strcmp(kind, "sparse") == 0) {
    // Distinct slots drawn from the whole [0, num_slots) (num_slots >
    // total_tokens): genuinely sparse, non-contiguous destinations.
    std::vector<std::int64_t> perm(num_slots);
    std::iota(perm.begin(), perm.end(), 0);
    for (std::size_t i = num_slots; i > 1; --i) {
      std::size_t j = rng.next() % i;
      std::swap(perm[i - 1], perm[j]);
    }
    for (std::size_t i = 0; i < total_tokens; ++i) ids[i] = perm[i];
  } else {  // nonmono: strictly descending contiguous range.
    for (std::size_t i = 0; i < total_tokens; ++i)
      ids[i] = static_cast<std::int64_t>(total_tokens - 1 - i);
  }
  return ids;
}

void run_matrix_case(const MatrixCase& c) {
  constexpr std::size_t kHeads = 8, kElem = 2;  // heads * head_dim = 512..2048
  const std::size_t total_tokens = c.num_pages * c.page_size;
  // num_slots with headroom beyond the token count so sparse maps are
  // genuinely sparse. No upper cap: scatter REQUIRES num_slots >=
  // total_tokens (max 8192), unlike the gather which could repeat slots.
  const std::size_t num_slots = std::max<std::size_t>(total_tokens + 64, 128);
  Lcg rng(c.head_dim * 131071ULL + c.page_size * 8191ULL +
          c.num_pages * 127ULL + static_cast<unsigned long>(std::strlen(c.map_kind)));
  const auto ids64 = make_map(c.map_kind, total_tokens, num_slots, rng);

  // Sanity: the map really is unique (the kernel also rejects duplicates).
  {
    std::unordered_set<std::int64_t> seen(ids64.begin(), ids64.end());
    ASSERT_EQ(seen.size(), ids64.size());
  }

  // int32 path (slot ids fit in int32 by construction).
  std::vector<int> ids32(ids64.begin(), ids64.end());
  auto src = make_src(c.num_pages, c.page_size, kHeads, c.head_dim, kElem, 0x12);

  auto k32 = make_dst(num_slots, kHeads, c.head_dim, kElem);
  auto v32 = make_dst(num_slots, kHeads, c.head_dim, kElem);
  kv_scatter_layer(k32.data(), v32.data(), ids32.data(), false, num_slots,
                   src.data(), c.num_pages, c.page_size, kHeads, c.head_dim,
                   kElem);
  auto ref = two_scatter_ref(src, ids32, num_slots, c.num_pages, c.page_size,
                             kHeads, c.head_dim, kElem);
  for (std::size_t i = 0; i < k32.size(); ++i) ASSERT_EQ(k32[i], ref.first[i]);
  for (std::size_t i = 0; i < v32.size(); ++i) ASSERT_EQ(v32[i], ref.second[i]);

  // int64 path: same map, same result.
  auto k64 = make_dst(num_slots, kHeads, c.head_dim, kElem);
  auto v64 = make_dst(num_slots, kHeads, c.head_dim, kElem);
  kv_scatter_layer(k64.data(), v64.data(), ids64.data(), true, num_slots,
                   src.data(), c.num_pages, c.page_size, kHeads, c.head_dim,
                   kElem);
  for (std::size_t i = 0; i < k64.size(); ++i) ASSERT_EQ(k64[i], ref.first[i]);
  for (std::size_t i = 0; i < v64.size(); ++i) ASSERT_EQ(v64[i], ref.second[i]);
}

TEST(KvScatterLayer, AcceptanceMatrix) {
  for (const auto& c : matrix_cases()) run_matrix_case(c);
}

// ---------------------------------------------------------------------------
// Slot map semantics: UNIQUE destinations are required; non-monotonic
// order is allowed. (The gather, by contrast, may repeat source slots.)
// ---------------------------------------------------------------------------
TEST(KvScatterLayer, UniqueSlots) {
  constexpr std::size_t kHeads = 1, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 8
  constexpr std::size_t kPageSize = 2;
  auto src = make_src(1, kPageSize, kHeads, kHeadDim, kElem, 0x10);
  auto k = make_dst(4, kHeads, kHeadDim, kElem);
  auto v = make_dst(4, kHeads, kHeadDim, kElem);
  const int slot_ids[2] = {3, 1};  // unique, non-monotonic
  kv_scatter_layer(k.data(), v.data(), slot_ids, false, 4, src.data(), 1,
                   kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(k[3 * kSlotBytes + i], src[i]);                 // token 0 K
    ASSERT_EQ(v[3 * kSlotBytes + i], src[kSlotBytes + i]);    // token 0 V
    ASSERT_EQ(k[1 * kSlotBytes + i], src[2 * kSlotBytes + i]); // token 1 K
    ASSERT_EQ(v[1 * kSlotBytes + i], src[3 * kSlotBytes + i]); // token 1 V
  }
}

// A duplicate destination is a contract violation: two threads would race
// the same bytes. The host reference rejects it (the CUDA kernel trusts the
// already-validated metadata).
TEST(KvScatterLayer, DuplicateSlotsThrow) {
  constexpr std::size_t kHeads = 1, kHeadDim = 4, kElem = 2;
  auto src = make_src(1, 2, kHeads, kHeadDim, kElem, 0x10);
  auto k = make_dst(4, kHeads, kHeadDim, kElem);
  auto v = make_dst(4, kHeads, kHeadDim, kElem);
  const int slot_ids[2] = {1, 1};  // duplicate destination
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), slot_ids, false, 4, src.data(), 1,
                       2, kHeads, kHeadDim, kElem),
      std::invalid_argument);
  // int64 path too.
  const std::int64_t slot_ids64[2] = {1LL, 1LL};
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), slot_ids64, true, 4, src.data(), 1,
                       2, kHeads, kHeadDim, kElem),
      std::invalid_argument);
}

TEST(KvScatterLayer, Int64Slots) {
  constexpr std::size_t kSlots = 8, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kPageSize = 2, kNumPages = 2;
  auto src = make_src(kNumPages, kPageSize, kHeads, kHeadDim, kElem, 0x30);
  const std::int64_t slot_ids[4] = {3LL, 1LL, 6LL, 4LL};  // unique
  auto k = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v = make_dst(kSlots, kHeads, kHeadDim, kElem);
  kv_scatter_layer(k.data(), v.data(), slot_ids, true, kSlots, src.data(),
                   kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  std::vector<std::int64_t> ids(slot_ids, slot_ids + 4);
  auto ref = two_scatter_ref(src, ids, kSlots, kNumPages, kPageSize, kHeads,
                             kHeadDim, kElem);
  for (std::size_t i = 0; i < k.size(); ++i) ASSERT_EQ(k[i], ref.first[i]);
  for (std::size_t i = 0; i < v.size(); ++i) ASSERT_EQ(v[i], ref.second[i]);
}

// ---------------------------------------------------------------------------
// Async stream contract: exactly one task per launch, result matches sync
// ---------------------------------------------------------------------------
TEST(KvScatterLayer, AsyncMatchesSync) {
  constexpr std::size_t kSlots = 8, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kPageSize = 2, kNumPages = 2;
  auto src = make_src(kNumPages, kPageSize, kHeads, kHeadDim, kElem, 0x30);
  const int slot_ids[4] = {3, 1, 6, 4};

  auto k_sync = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_sync = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto k_asy = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_asy = make_dst(kSlots, kHeads, kHeadDim, kElem);
  kv_scatter_layer(k_sync.data(), v_sync.data(), slot_ids, false, kSlots,
                   src.data(), kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  Stream s;
  std::size_t before = s.submitted();
  kv_scatter_layer(k_asy.data(), v_asy.data(), slot_ids, false, kSlots,
                   src.data(), kNumPages, kPageSize, kHeads, kHeadDim, kElem,
                   &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one task
  s.wait();
  for (std::size_t i = 0; i < k_sync.size(); ++i) ASSERT_EQ(k_sync[i], k_asy[i]);
  for (std::size_t i = 0; i < v_sync.size(); ++i) ASSERT_EQ(v_sync[i], v_asy[i]);
}

// Async must capture the slot map: mutating the source array after enqueue
// does not affect the result (matches the documented lifetime rule).
TEST(KvScatterLayer, AsyncCapturesSlotMap) {
  constexpr std::size_t kSlots = 8, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kPageSize = 2, kNumPages = 2;
  auto src = make_src(kNumPages, kPageSize, kHeads, kHeadDim, kElem, 0x30);
  int slot_ids[4] = {3, 1, 6, 4};

  auto k_sync = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_sync = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto k_asy = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v_asy = make_dst(kSlots, kHeads, kHeadDim, kElem);
  kv_scatter_layer(k_sync.data(), v_sync.data(), slot_ids, false, kSlots,
                   src.data(), kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  Stream s;
  kv_scatter_layer(k_asy.data(), v_asy.data(), slot_ids, false, kSlots,
                   src.data(), kNumPages, kPageSize, kHeads, kHeadDim, kElem,
                   &s);
  // Clobber the source slot array AFTER enqueue.
  for (int& x : slot_ids) x = 0;
  s.wait();
  for (std::size_t i = 0; i < k_sync.size(); ++i) ASSERT_EQ(k_sync[i], k_asy[i]);
  for (std::size_t i = 0; i < v_sync.size(); ++i) ASSERT_EQ(v_sync[i], v_asy[i]);
}

// ---------------------------------------------------------------------------
// Edge cases and contract checks
// ---------------------------------------------------------------------------
TEST(KvScatterLayer, ZeroPagesIsNoOp) {
  constexpr std::size_t kHeads = 2, kHeadDim = 4, kElem = 2;
  auto k = make_dst(1, kHeads, kHeadDim, kElem, 0xDD);
  auto v = make_dst(1, kHeads, kHeadDim, kElem, 0xEE);
  auto before_k = k, before_v = v;
  int slot_ids[1] = {0};
  kv_scatter_layer(k.data(), v.data(), slot_ids, false, 1, nullptr, 0, 1,
                   kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < k.size(); ++i) ASSERT_EQ(k[i], before_k[i]);
  for (std::size_t i = 0; i < v.size(); ++i) ASSERT_EQ(v[i], before_v[i]);
}

TEST(KvScatterLayer, NullPointersThrow) {
  constexpr std::size_t kHeads = 2, kHeadDim = 4, kElem = 2;
  std::vector<std::uint8_t> src(2 * 2 * 2 * kHeads * kHeadDim * kElem, 0);
  int slot_ids[1] = {0};
  EXPECT_THROW(
      kv_scatter_layer(nullptr, nullptr, slot_ids, false, 1, src.data(), 1, 1,
                       kHeads, kHeadDim, kElem),
      std::invalid_argument);
  std::vector<std::uint8_t> k(2 * 2 * kHeads * kHeadDim * kElem),
      v(2 * 2 * kHeads * kHeadDim * kElem);
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), nullptr, false, 1, src.data(), 1, 1,
                       kHeads, kHeadDim, kElem),
      std::invalid_argument);
}

TEST(KvScatterLayer, OutOfRangeSlotThrows) {
  constexpr std::size_t kSlots = 4, kHeads = 2, kHeadDim = 4, kElem = 2;
  auto src = make_src(1, 1, kHeads, kHeadDim, kElem, 1);
  auto k = make_dst(kSlots, kHeads, kHeadDim, kElem);
  auto v = make_dst(kSlots, kHeads, kHeadDim, kElem);
  const int too_big[1] = {static_cast<int>(kSlots)};  // == num_slots
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), too_big, false, kSlots, src.data(),
                       1, 1, kHeads, kHeadDim, kElem),
      std::invalid_argument);
  const int negative[1] = {-1};
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), negative, false, kSlots, src.data(),
                       1, 1, kHeads, kHeadDim, kElem),
      std::invalid_argument);
  // int64 out-of-range path.
  const std::int64_t big64[1] = {static_cast<std::int64_t>(kSlots)};
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), big64, true, kSlots, src.data(), 1,
                       1, kHeads, kHeadDim, kElem),
      std::invalid_argument);
}

TEST(KvScatterLayer, NonBf16Fp16ElemSizeThrows) {
  constexpr std::size_t kSlots = 4, kHeads = 2, kHeadDim = 4;
  auto src = make_src(1, 1, kHeads, kHeadDim, 2, 1);
  auto k = make_dst(kSlots, kHeads, kHeadDim, 2);
  auto v = make_dst(kSlots, kHeads, kHeadDim, 2);
  const int slot_ids[1] = {0};
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), slot_ids, false, kSlots, src.data(),
                       1, 1, kHeads, kHeadDim, 1),
      std::invalid_argument);  // FP8/INT8 not supported
  EXPECT_THROW(
      kv_scatter_layer(k.data(), v.data(), slot_ids, false, kSlots, src.data(),
                       1, 1, kHeads, kHeadDim, 4),
      std::invalid_argument);  // FP32 not supported
}
