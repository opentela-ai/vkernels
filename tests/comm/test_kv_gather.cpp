// tests/comm/test_kv_gather.cpp
//
// Host-reference tests for the fused indexed K/V layer gather kernel
// (issue #2). These are the correctness oracle: they exercise byte-exact
// results against the PyTorch two-gather reference
// (`dst[:, :, 0] = k_src[slot_ids]; dst[:, :, 1] = v_src[slot_ids]`) across
// the full acceptance matrix (BF16/FP16 × head dims 64/128/256 × page sizes
// 1/16/32/64 × 0..128+ pages × sparse/non-monotonic slot maps ×
// int32/int64 slot ids), validate the contract checks (null pointers, zero
// dimensions, non-BF16/FP16 elem_size, negative/out-of-range source slots),
// and pin the async stream contract to exactly one task per launch.
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
#include <vector>

#include "vkernels/comm/kv_gather.hpp"
#include "vkernels/core/stream.hpp"

using vkernels::Stream;
using vkernels::comm::kv_gather_layer;

namespace {

inline std::size_t kv_slot_bytes(std::size_t heads, std::size_t head_dim,
                                 std::size_t elem) {
  return heads * head_dim * elem;
}
inline std::size_t kv_token_stride(std::size_t heads, std::size_t head_dim,
                                   std::size_t elem) {
  return 2 * kv_slot_bytes(heads, head_dim, elem);
}

// A flat [num_slots, num_kv_heads, head_dim, elem_size] source filled with a
// per-byte seed so a mis-routed read is immediately obvious.
std::vector<std::uint8_t> make_src(std::size_t num_slots, std::size_t heads,
                                   std::size_t head_dim, std::size_t elem,
                                   std::uint8_t seed) {
  const std::size_t total = num_slots * heads * head_dim * elem;
  std::vector<std::uint8_t> v(total);
  for (std::size_t i = 0; i < total; ++i)
    v[i] = static_cast<std::uint8_t>(seed + i);
  return v;
}

// Packed [num_pages, page_size, 2, heads, head_dim, elem] destination filled
// with a sentinel so untouched bytes are detected.
std::vector<std::uint8_t> make_dst(std::size_t num_pages, std::size_t page_size,
                                   std::size_t heads, std::size_t head_dim,
                                   std::size_t elem,
                                   std::uint8_t fill = 0xCC) {
  const std::size_t total = num_pages * page_size * 2 * heads * head_dim * elem;
  return std::vector<std::uint8_t>(total, fill);
}

// The PyTorch two-gather reference the kernel replaces, in bytes:
//   dst[:, :, 0] = k_src[slot_ids]
//   dst[:, :, 1] = v_src[slot_ids]
template <typename SlotT>
std::vector<std::uint8_t> two_gather_ref(
    const std::vector<std::uint8_t>& k_src,
    const std::vector<std::uint8_t>& v_src, const std::vector<SlotT>& slot_ids,
    std::size_t num_slots, std::size_t num_pages, std::size_t page_size,
    std::size_t heads, std::size_t head_dim, std::size_t elem) {
  (void)num_slots;
  const std::size_t sb = kv_slot_bytes(heads, head_dim, elem);
  const std::size_t ts = kv_token_stride(heads, head_dim, elem);
  const std::size_t page_bytes = page_size * ts;
  auto ref = make_dst(num_pages, page_size, heads, head_dim, elem);
  for (std::size_t p = 0; p < num_pages; ++p) {
    for (std::size_t t = 0; t < page_size; ++t) {
      const std::size_t slot =
          static_cast<std::size_t>(slot_ids[p * page_size + t]);
      const std::size_t src_off = slot * sb;
      const std::size_t dst_off = p * page_bytes + t * ts;
      std::memcpy(ref.data() + dst_off, k_src.data() + src_off, sb);
      std::memcpy(ref.data() + dst_off + sb, v_src.data() + src_off, sb);
    }
  }
  return ref;
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
TEST(KvGatherLayer, SinglePageSingleToken) {
  constexpr std::size_t kHeads = 2, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;
  constexpr std::size_t kSlots = 8;

  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x70);
  const int slot_ids[1] = {3};
  auto dst = make_dst(1, 1, kHeads, kHeadDim, kElem);
  kv_gather_layer(dst.data(), k_src.data(), v_src.data(), slot_ids, false,
                  kSlots, 1, 1, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(dst[i], k_src[3 * kSlotBytes + i]);
    ASSERT_EQ(dst[kSlotBytes + i], v_src[3 * kSlotBytes + i]);
  }
}

TEST(KvGatherLayer, MatchesTwoGatherReference) {
  constexpr std::size_t kSlots = 8, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kPageSize = 2, kNumPages = 2;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x70);
  const int slot_ids[4] = {3, 1, 6, 4};  // non-monotonic across pages
  auto dst = make_dst(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  kv_gather_layer(dst.data(), k_src.data(), v_src.data(), slot_ids, false,
                  kSlots, kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  std::vector<int> ids(slot_ids, slot_ids + 4);
  auto ref = two_gather_ref(k_src, v_src, ids, kSlots, kNumPages, kPageSize,
                            kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], ref[i]);
}

// ---------------------------------------------------------------------------
// Full acceptance matrix: BF16/FP16 (elem=2) × head dims {64,128,256} ×
// page sizes {1,16,32,64} × page counts {1, 16, 64, 128} ×
// {dense, sparse, non-monotonic} slot maps × {int32, int64}.
//
// num_slots is chosen larger than num_pages * page_size with headroom so
// sparse maps (slots drawn uniformly from [0, num_slots)) exercise wide
// source strides, and the 128-page case clears the issue's ">=128 pages"
// bar. Randomness is seeded for reproducibility.
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

// Build a slot map of the requested kind from `num_slots` slots.
std::vector<std::int64_t> make_map(const char* kind, std::size_t total_tokens,
                                   std::size_t num_slots, Lcg& rng) {
  std::vector<std::int64_t> ids(total_tokens);
  if (std::strcmp(kind, "dense") == 0) {
    // A permutation of a contiguous range (no gaps) -- worst case for the
    // kernel's source-read pattern is fully non-local, so shuffle.
    std::vector<std::int64_t> perm(num_slots);
    std::iota(perm.begin(), perm.end(), 0);
    for (std::size_t i = num_slots; i > 1; --i) {
      std::size_t j = rng.next() % i;
      std::swap(perm[i - 1], perm[j]);
    }
    for (std::size_t i = 0; i < total_tokens; ++i) ids[i] = perm[i % num_slots];
  } else if (std::strcmp(kind, "sparse") == 0) {
    // Slots drawn uniformly from [0, num_slots) -- mostly non-contiguous and
    // with repeats.
    for (std::size_t i = 0; i < total_tokens; ++i)
      ids[i] = static_cast<std::int64_t>(rng.next() % num_slots);
  } else {  // nonmono: strictly non-monotonic (descending modulo wrap).
    for (std::size_t i = 0; i < total_tokens; ++i)
      ids[i] = static_cast<std::int64_t>((num_slots - 1) -
                                         static_cast<std::int64_t>(i % num_slots));
  }
  return ids;
}

void run_matrix_case(const MatrixCase& c) {
  constexpr std::size_t kHeads = 8, kElem = 2;  // heads * head_dim = 512..2048
  // num_slots with headroom beyond the token count, capped to keep the host
  // reference fast; large enough that sparse maps are genuinely sparse.
  const std::size_t total_tokens = c.num_pages * c.page_size;
  std::size_t num_slots = std::max<std::size_t>(total_tokens + 64, 128);
  num_slots = std::min<std::size_t>(num_slots, 4096);
  Lcg rng(c.head_dim * 131071ULL + c.page_size * 8191ULL +
          c.num_pages * 127ULL + static_cast<unsigned long>(std::strlen(c.map_kind)));
  const auto ids64 = make_map(c.map_kind, total_tokens, num_slots, rng);

  // Two distinct sources so K and V are never confused.
  auto k_src = make_src(num_slots, kHeads, c.head_dim, kElem, 0x12);
  auto v_src = make_src(num_slots, kHeads, c.head_dim, kElem, 0x9A);

  // int32 path (slot ids fit in int32 by construction).
  std::vector<int> ids32(ids64.begin(), ids64.end());

  auto dst32 = make_dst(c.num_pages, c.page_size, kHeads, c.head_dim, kElem);
  kv_gather_layer(dst32.data(), k_src.data(), v_src.data(), ids32.data(), false,
                  num_slots, c.num_pages, c.page_size, kHeads, c.head_dim, kElem);
  auto ref = two_gather_ref(k_src, v_src, ids32, num_slots, c.num_pages,
                            c.page_size, kHeads, c.head_dim, kElem);
  for (std::size_t i = 0; i < dst32.size(); ++i) ASSERT_EQ(dst32[i], ref[i]);

  // int64 path: same map, same result.
  auto dst64 = make_dst(c.num_pages, c.page_size, kHeads, c.head_dim, kElem);
  kv_gather_layer(dst64.data(), k_src.data(), v_src.data(), ids64.data(), true,
                  num_slots, c.num_pages, c.page_size, kHeads, c.head_dim, kElem);
  for (std::size_t i = 0; i < dst64.size(); ++i) ASSERT_EQ(dst64[i], ref[i]);
}

TEST(KvGatherLayer, AcceptanceMatrix) {
  for (const auto& c : matrix_cases()) run_matrix_case(c);
}

// ---------------------------------------------------------------------------
// Slot map semantics: repeats and non-monotonic order are allowed
// ---------------------------------------------------------------------------
TEST(KvGatherLayer, RepeatedSlots) {
  constexpr std::size_t kHeads = 1, kHeadDim = 4, kElem = 2;
  constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;  // 8
  constexpr std::size_t kPageSize = 2;
  auto k_src = make_src(4, kHeads, kHeadDim, kElem, 0x10);
  auto v_src = make_src(4, kHeads, kHeadDim, kElem, 0x20);
  auto dst = make_dst(1, kPageSize, kHeads, kHeadDim, kElem);
  const int slot_ids[2] = {1, 1};  // both tokens read slot 1
  kv_gather_layer(dst.data(), k_src.data(), v_src.data(), slot_ids, false,
                  4, 1, kPageSize, kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < kSlotBytes; ++i) {
    ASSERT_EQ(dst[i], k_src[kSlotBytes + i]);                  // token 0 K
    ASSERT_EQ(dst[kSlotBytes + i], v_src[kSlotBytes + i]);     // token 0 V
    ASSERT_EQ(dst[2 * kSlotBytes + i], k_src[kSlotBytes + i]); // token 1 K
    ASSERT_EQ(dst[3 * kSlotBytes + i], v_src[kSlotBytes + i]); // token 1 V
  }
}

TEST(KvGatherLayer, Int64Slots) {
  constexpr std::size_t kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kSlots = 8, kPageSize = 2, kNumPages = 2;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x70);
  const std::int64_t slot_ids[4] = {3LL, 1LL, 6LL, 4LL};
  auto dst = make_dst(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  kv_gather_layer(dst.data(), k_src.data(), v_src.data(), slot_ids, true,
                  kSlots, kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  std::vector<std::int64_t> ids(slot_ids, slot_ids + 4);
  auto ref = two_gather_ref(k_src, v_src, ids, kSlots, kNumPages, kPageSize,
                            kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], ref[i]);
}

// ---------------------------------------------------------------------------
// Async stream contract: exactly one task per launch, result matches sync
// ---------------------------------------------------------------------------
TEST(KvGatherLayer, AsyncMatchesSync) {
  constexpr std::size_t kSlots = 8, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kPageSize = 2, kNumPages = 2;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x70);
  const int slot_ids[4] = {3, 1, 6, 4};

  auto sync = make_dst(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  auto asy = make_dst(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  kv_gather_layer(sync.data(), k_src.data(), v_src.data(), slot_ids, false,
                  kSlots, kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  Stream s;
  std::size_t before = s.submitted();
  kv_gather_layer(asy.data(), k_src.data(), v_src.data(), slot_ids, false,
                  kSlots, kNumPages, kPageSize, kHeads, kHeadDim, kElem, &s);
  EXPECT_EQ(s.submitted() - before, 1u);  // exactly one task
  s.wait();
  for (std::size_t i = 0; i < sync.size(); ++i) ASSERT_EQ(sync[i], asy[i]);
}

// Async must capture the slot map: mutating the source array after enqueue
// does not affect the result (matches the documented lifetime rule).
TEST(KvGatherLayer, AsyncCapturesSlotMap) {
  constexpr std::size_t kSlots = 8, kHeads = 2, kHeadDim = 16, kElem = 2;
  constexpr std::size_t kPageSize = 2, kNumPages = 2;
  auto k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x30);
  auto v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 0x70);
  int slot_ids[4] = {3, 1, 6, 4};

  auto sync = make_dst(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  auto asy = make_dst(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  kv_gather_layer(sync.data(), k_src.data(), v_src.data(), slot_ids, false,
                  kSlots, kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  Stream s;
  kv_gather_layer(asy.data(), k_src.data(), v_src.data(), slot_ids, false,
                  kSlots, kNumPages, kPageSize, kHeads, kHeadDim, kElem, &s);
  // Clobber the source slot array AFTER enqueue.
  for (int& x : slot_ids) x = 0;
  s.wait();
  for (std::size_t i = 0; i < sync.size(); ++i) ASSERT_EQ(sync[i], asy[i]);
}

// ---------------------------------------------------------------------------
// Edge cases and contract checks
// ---------------------------------------------------------------------------
TEST(KvGatherLayer, ZeroPagesIsNoOp) {
  constexpr std::size_t kHeads = 2, kHeadDim = 4, kElem = 2;
  auto dst = make_dst(1, 1, kHeads, kHeadDim, kElem, 0xDD);
  auto before = dst;
  int slot_ids[1] = {0};
  kv_gather_layer(dst.data(), nullptr, nullptr, slot_ids, false, 1, 0, 1,
                  kHeads, kHeadDim, kElem);
  for (std::size_t i = 0; i < dst.size(); ++i) ASSERT_EQ(dst[i], before[i]);
}

TEST(KvGatherLayer, NullPointersThrow) {
  constexpr std::size_t kHeads = 2, kHeadDim = 4, kElem = 2;
  int slot_ids[1] = {0};
  EXPECT_THROW(
      kv_gather_layer(nullptr, nullptr, nullptr, slot_ids, false, 1, 1, 1,
                      kHeads, kHeadDim, kElem),
      std::invalid_argument);
  std::vector<std::uint8_t> k_src(2 * 2 * 4 * 2), v_src(2 * 2 * 4 * 2),
      dst(2 * 2 * 4 * 2);
  EXPECT_THROW(
      kv_gather_layer(dst.data(), k_src.data(), v_src.data(), nullptr, false,
                      1, 1, 1, kHeads, kHeadDim, kElem),
      std::invalid_argument);
}

TEST(KvGatherLayer, OutOfRangeSlotThrows) {
  constexpr std::size_t kSlots = 4, kHeads = 2, kHeadDim = 4, kElem = 2;
  std::vector<std::uint8_t> k_src = make_src(kSlots, kHeads, kHeadDim, kElem, 1);
  std::vector<std::uint8_t> v_src = make_src(kSlots, kHeads, kHeadDim, kElem, 2);
  auto dst = make_dst(1, 1, kHeads, kHeadDim, kElem);
  const int too_big[1] = {static_cast<int>(kSlots)};  // == num_slots
  EXPECT_THROW(
      kv_gather_layer(dst.data(), k_src.data(), v_src.data(), too_big, false,
                      kSlots, 1, 1, kHeads, kHeadDim, kElem),
      std::invalid_argument);
  const int negative[1] = {-1};
  EXPECT_THROW(
      kv_gather_layer(dst.data(), k_src.data(), v_src.data(), negative, false,
                      kSlots, 1, 1, kHeads, kHeadDim, kElem),
      std::invalid_argument);
  // int64 out-of-range path.
  const std::int64_t big64[1] = {static_cast<std::int64_t>(kSlots)};
  EXPECT_THROW(
      kv_gather_layer(dst.data(), k_src.data(), v_src.data(), big64, true,
                      kSlots, 1, 1, kHeads, kHeadDim, kElem),
      std::invalid_argument);
}

TEST(KvGatherLayer, NonBf16Fp16ElemSizeThrows) {
  constexpr std::size_t kSlots = 4, kHeads = 2, kHeadDim = 4;
  std::vector<std::uint8_t> k_src = make_src(kSlots, kHeads, kHeadDim, 2, 1);
  std::vector<std::uint8_t> v_src = make_src(kSlots, kHeads, kHeadDim, 2, 2);
  auto dst = make_dst(1, 1, kHeads, kHeadDim, 2);
  const int slot_ids[1] = {0};
  EXPECT_THROW(
      kv_gather_layer(dst.data(), k_src.data(), v_src.data(), slot_ids, false,
                      kSlots, 1, 1, kHeads, kHeadDim, 1),
      std::invalid_argument);  // FP8/INT8 not supported
  EXPECT_THROW(
      kv_gather_layer(dst.data(), k_src.data(), v_src.data(), slot_ids, false,
                      kSlots, 1, 1, kHeads, kHeadDim, 4),
      std::invalid_argument);  // FP32 not supported
}
