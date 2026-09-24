// tests/kernels/moe/test_moe_align_oracle.cpp
//
// Extra CPU-oracle parity coverage for moe_align_block_size (moe_fused.hpp),
// complementary to the MoeAlign.* cases in test_moe_fused.cpp. Two goals:
//
//   1. Contract edges the device path (moe_align_block_size_hip / the
//      vk_hip_moe_align_block_size C ABI) relies on: garbage/negative
//      routing ids are skipped, block_size=64 (prefill regime), the
//      E=256-expert serving shape, and the exact padding sentinels
//      (sorted_ids pads with N = M*top_k, expert_ids pads with -1).
//
//   2. The serving `_align_em_bound` high-water formula (Python,
//      vllm_experts.py) must UPPER-BOUND the oracle EM for every routing —
//      the on-device fast path sizes the GEMM grid at bound/block_size, so
//      a bound below the real EM would silently truncate real tokens.
//      Mirrored here against the C++ oracle over uniform / hot-expert /
//      skewed routings (the test_glm_fp8_grouped_multitile lesson: skewed
//      routing is where layout assumptions break).
#include "minitest.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "vkernels/kernels/moe_fused.hpp"

using vkernels::kernels::moe_align_block_size;

namespace {

// Independent reference (structure mirrors the CPU oracle in
// moe_fused.cpp / the .hip test's with_map ref, no map): group flat
// indices by expert ascending, pad each expert to a block multiple with
// N, expert_ids = e per real block / -1 for padding. Returns EM.
int align_ref(const std::vector<int32_t>& topk_ids, int block_size,
              int num_experts, std::vector<int32_t>& sorted_ids,
              std::vector<int32_t>& expert_ids) {
  const int N = static_cast<int>(topk_ids.size());
  std::vector<std::vector<int32_t>> per(num_experts);
  for (int i = 0; i < N; ++i) {
    int e = topk_ids[i];
    if (e >= 0 && e < num_experts) per[e].push_back(i);
  }
  int EM = 0;
  for (const auto& v : per)
    EM += (static_cast<int>(v.size()) + block_size - 1) / block_size * block_size;
  int idx = 0, blk = 0;
  for (int e = 0; e < num_experts; ++e) {
    const int cnt = static_cast<int>(per[e].size());
    const int padded = (cnt + block_size - 1) / block_size * block_size;
    for (int t : per[e]) sorted_ids[idx++] = t;
    for (int i = cnt; i < padded; ++i) sorted_ids[idx++] = N;
    for (int b = 0; b < padded / block_size; ++b) {
      if (b * block_size < cnt) expert_ids[blk] = e;
      ++blk;
    }
  }
  return EM;
}

// The serving high-water bound (vllm_experts._align_em_bound): worst case
// every expert gets one token (each padded to block_size), remaining
// tokens land in one expert padded to a block multiple. Host-constant.
int align_em_bound(int M, int top_k, int local_n, int block_size) {
  const int N = M * top_k;
  if (N <= 0) return block_size;
  const int head = std::min(N, local_n) * block_size;
  if (N <= local_n) return head;
  const int rem = ((N - local_n) + block_size - 1) / block_size * block_size;
  return head + rem;
}

// Deterministic xorshift so failures are reproducible.
uint32_t next_rand(uint32_t& s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  return s;
}

}  // namespace

// -----------------------------------------------------------------------
// Padding sentinels are load-bearing: downstream moe_aux / fused kernels
// guard `flat < N` (sorted) and `expert_ids[b] < 0` (blocks). Lock the
// exact values in for a padded-heavy case.
// -----------------------------------------------------------------------
TEST(MoeAlignOracle, PaddingSentinelsExact) {
  constexpr int M = 3, top_k = 2, E = 4, BS = 16;
  const int N = M * top_k;  // 6
  // Only expert 1 receives tokens (2 of 6) -> one block, 14 padding rows.
  // Real flats: 2 (token 1 sel 0) and 4 (token 2 sel 0).
  std::vector<int32_t> ids = {-1, -1, 1, -1, 1, -1};
  const int max_EM = ((N + BS - 1) / BS + E) * BS;
  std::vector<int32_t> sids(max_EM, -777), eids(max_EM / BS, -888);

  const int EM = moe_align_block_size(ids.data(), M, top_k, BS, E,
                                      sids.data(), eids.data());
  EXPECT_EQ(EM, BS);
  EXPECT_EQ(eids[0], 1);          // one real block owned by expert 1
  EXPECT_EQ(sids[0], 2);          // flat 2 (token 1 sel 0)
  EXPECT_EQ(sids[1], 4);          // flat 4 (token 2 sel 0)
  for (int i = 2; i < BS; ++i) EXPECT_EQ(sids[i], N);      // pad == M*top_k
  for (int i = BS; i < max_EM; ++i) EXPECT_EQ(sids[i], -777);  // untouched
  for (int i = 1; i < max_EM / BS; ++i) EXPECT_EQ(eids[i], -888);
}

// -----------------------------------------------------------------------
// Garbage routing (negative ids, ids >= num_experts) must be skipped —
// the device kernel encodes exactly this rule (le = -1 -> skip).
// -----------------------------------------------------------------------
TEST(MoeAlignOracle, GarbageAndNegativeIdsSkipped) {
  constexpr int E = 4, BS = 16;
  // 8 flat entries: 0, 99 (out of range), -3 (negative), 1, E, 1, 2, -1.
  std::vector<int32_t> ids = {0, 99, -3, 1, E, 1, 2, -1};
  const int N = static_cast<int>(ids.size());
  const int max_EM = ((N + BS - 1) / BS + E) * BS;
  std::vector<int32_t> sids(max_EM, -1), eids(max_EM / BS, -1);

  const int EM = moe_align_block_size(ids.data(), 1, N, BS, E,
                                      sids.data(), eids.data());
  // Real entries: flats 0 (e0), 3/5 (e1), 6 (e2) -> 3 blocks of 16.
  EXPECT_EQ(EM, 3 * BS);
  EXPECT_EQ(eids[0], 0);
  EXPECT_EQ(eids[1], 1);
  EXPECT_EQ(eids[2], 2);
  EXPECT_EQ(sids[0], 0);
  EXPECT_EQ(sids[16], 3);
  EXPECT_EQ(sids[17], 5);
  EXPECT_EQ(sids[32], 6);
  // Every slot is a real flat index (< N) or the N padding sentinel —
  // garbage ids must never echo into sorted_ids.
  for (int i = 0; i < EM; ++i) EXPECT_TRUE(sids[i] >= 0 && sids[i] <= N);
}

// -----------------------------------------------------------------------
// Prefill regime block_size=64 and the E=256-expert serving shape, each
// cross-checked against the independent reference.
// -----------------------------------------------------------------------
TEST(MoeAlignOracle, BlockSize64AndManyExpertsMatchRef) {
  const int shapes[][4] = {
      // M, top_k, E, block_size
      {17, 2, 8, 64},    // prefill-style block
      {64, 8, 256, 16},  // K3 decode shape family (E=256)
      {5, 16, 256, 64},  // E=256, prefill block, heavy padding
      {33, 1, 7, 32},    // odd expert count / non-16 block
  };
  for (const auto& sh : shapes) {
    const int M = sh[0], top_k = sh[1], E = sh[2], BS = sh[3];
    const int N = M * top_k;
    std::vector<int32_t> ids(N);
    uint32_t rng = 0xBEEFu + static_cast<uint32_t>(N);
    for (int i = 0; i < N; ++i) ids[i] = static_cast<int>(next_rand(rng) % 32) - 2;
    // Re-map out-of-range to in-range so both refs see a legal mix plus a
    // few deliberate garbage entries the oracle must skip.
    for (int i = 0; i < N; ++i)
      if (ids[i] >= E || ids[i] < -2) ids[i] = E - 1;

    const int max_EM = ((N + BS - 1) / BS + E) * BS;
    std::vector<int32_t> sids(max_EM, N), eids(max_EM / BS, -1);
    std::vector<int32_t> ref_sids(max_EM, N), ref_eids(max_EM / BS, -1);

    const int EM = moe_align_block_size(ids.data(), M, top_k, BS, E,
                                        sids.data(), eids.data());
    const int EM_ref = align_ref(ids, BS, E, ref_sids, ref_eids);
    EXPECT_EQ(EM, EM_ref);
    for (int i = 0; i < EM; ++i)
      EXPECT_EQ(sids[i], ref_sids[i]);
    for (int i = 0; i < EM / BS; ++i)
      EXPECT_EQ(eids[i], ref_eids[i]);
  }
}

// -----------------------------------------------------------------------
// The serving EM high-water bound must upper-bound the oracle EM (and stay
// a block multiple) for every routing class — uniform, hot-expert, and
// skewed power-law — on the shapes the Python AlignEmBoundTest covers plus
// the E=256 serving shape. A bound below the real EM would truncate real
// tokens on the on-device fast path (capture-safe max_EM grid).
// -----------------------------------------------------------------------
TEST(MoeAlignOracle, ServingEmBoundUpperBoundsOracle) {
  const int shapes[][4] = {
      {1, 8, 32, 16}, {8, 8, 32, 16}, {50, 8, 32, 64},
      {32, 16, 896, 64}, {1, 16, 256, 16}, {4, 8, 256, 16},
  };
  for (const auto& sh : shapes) {
    const int M = sh[0], top_k = sh[1], E = sh[2], BS = sh[3];
    const int N = M * top_k;
    const int bound = align_em_bound(M, top_k, E, BS);
    EXPECT_GE(bound, N);
    EXPECT_EQ(bound % BS, 0);

    // Uniform random routing.
    {
      uint32_t rng = 0xC0FFEEu + static_cast<uint32_t>(N * 31 + E);
      std::vector<int32_t> ids(N);
      for (int i = 0; i < N; ++i) ids[i] = static_cast<int>(next_rand(rng) % E);
      const int max_EM = ((N + BS - 1) / BS + E) * BS;
      std::vector<int32_t> sids(max_EM), eids(max_EM / BS);
      const int EM = moe_align_block_size(ids.data(), M, top_k, BS, E,
                                          sids.data(), eids.data());
      EXPECT_GE(bound, EM);
    }
    // Hot expert: most tokens on expert 0 (max intra-expert padding).
    {
      uint32_t rng = 0xD00Du + static_cast<uint32_t>(N * 17 + E);
      std::vector<int32_t> ids(N);
      for (int i = 0; i < N; ++i)
        ids[i] = (next_rand(rng) % 10 == 0) ? static_cast<int>(next_rand(rng) % E) : 0;
      const int max_EM = ((N + BS - 1) / BS + E) * BS;
      std::vector<int32_t> sids(max_EM), eids(max_EM / BS);
      const int EM = moe_align_block_size(ids.data(), M, top_k, BS, E,
                                          sids.data(), eids.data());
      EXPECT_GE(bound, EM);
    }
    // Skewed power-law across many experts (max inter-expert padding).
    {
      uint32_t rng = 0x5EEDu + static_cast<uint32_t>(N * 7 + E);
      std::vector<int32_t> ids(N);
      for (int i = 0; i < N; ++i) {
        // p(e) ~ 2^-e, clamped to [0, E).
        int e = 0;
        while (e + 1 < E && (next_rand(rng) & 1)) ++e;
        ids[i] = e;
      }
      const int max_EM = ((N + BS - 1) / BS + E) * BS;
      std::vector<int32_t> sids(max_EM), eids(max_EM / BS);
      const int EM = moe_align_block_size(ids.data(), M, top_k, BS, E,
                                          sids.data(), eids.data());
      EXPECT_GE(bound, EM);
    }
  }
}

// M=0: the C++ oracle returns EM=0 (the Python with_map path and the GPU
// kernel additionally emit one padding block — min-EM = block_size — a
// documented divergence handled at the Python binding layer).
TEST(MoeAlignOracle, M0EmptyRouting) {
  constexpr int E = 4, BS = 16;
  const int max_EM = ((0 + BS - 1) / BS + E) * BS;
  std::vector<int32_t> sids(max_EM, -1), eids(max_EM / BS, -1);
  const int EM = moe_align_block_size(nullptr, 0, 2, BS, E,
                                      sids.data(), eids.data());
  EXPECT_EQ(EM, 0);
}
