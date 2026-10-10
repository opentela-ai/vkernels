// tests/kernels/common/test_common.cpp — epilogue operators and heuristics
// tables (the kernels/common layer shared by the CPU oracles and CUDA
// kernels; see docs/borrowing-tokenspeed-deepgemm.md §3.8–3.11).
#include "vkernels/kernels/common/epilogue.hpp"
#include "vkernels/kernels/common/heuristics.hpp"
#include "vkernels/kernels/gemm_bf16.hpp"

#include <cmath>
#include <limits>

#include "minitest.hpp"

using vkernels::kernels::epilogue::Linear;
using vkernels::kernels::epilogue::LinearRelu;
using vkernels::kernels::heuristics::gemm_bf16_tile_candidates;
using vkernels::kernels::heuristics::is_gemm_bf16_candidate;

static bool nearly(float a, float b) {
  return std::fabs(a - b) <= 1e-6f * std::fmax(1.0f, std::fabs(b));
}

TEST(Epilogue, LinearScalesAndAccumulates) {
  const Linear store{2.0f, 3.0f};
  EXPECT_TRUE(nearly(store.apply(1.5f, 0.25f), 2.0f * 1.5f + 3.0f * 0.25f));
  const Linear pure{1.0f, 0.0f};
  EXPECT_TRUE(nearly(pure.apply(4.0f, 7.0f), 4.0f));  // beta 0: C_prev ignored
}

TEST(Epilogue, LinearBetaZeroNeverReadsCPrev) {
  // The BLAS rule that motivated the gemm oracle fix: 0 * NaN is NaN, so
  // a beta-0 store must not touch the previous output at all.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Linear store{1.0f, 0.0f};
  EXPECT_TRUE(nearly(store.apply(2.0f, nan), 2.0f));
}

TEST(Epilogue, ReluComposesOnLinear) {
  const LinearRelu store{Linear{1.0f, 0.5f}};
  EXPECT_TRUE(nearly(store.apply(4.0f, 1.0f), 4.5f));   // positive passes
  EXPECT_TRUE(nearly(store.apply(-4.0f, 1.0f), 0.0f));  // negative clamped
  EXPECT_TRUE(nearly(store.apply(0.0f, 1.0f), 0.5f));   // beta term survives
}

TEST(Heuristics, CandidatesAreTheSweptTiles) {
  const auto tiles = gemm_bf16_tile_candidates();
  EXPECT_GE(tiles.size(), 2u);
  for (const auto& tile : tiles) {
    EXPECT_GT(tile.bm, 0);
    EXPECT_GT(tile.bn, 0);
    EXPECT_EQ(tile.bk, 64);  // every K3 K is a multiple of 64
    EXPECT_GT(tile.threads, 0);
    // The selector contract: what it returns is what was swept.
    EXPECT_TRUE(is_gemm_bf16_candidate(tile.bm, tile.bn, tile.bk, tile.threads));
  }
  // An unmeasured tile is not a candidate.
  EXPECT_FALSE(is_gemm_bf16_candidate(32, 32, 64, 128));
}

TEST(Heuristics, SelectorReturnsSweptTiles) {
  // gemm_bf16_config_for on this build must return table entries.
  for (std::size_t m : {1u, 16u, 64u, 65u, 8192u}) {
    for (std::size_t n : {16u, 1024u, 1025u, 7168u}) {
      int bm = 0, bn = 0, bk = 0, threads = 0;
      vkernels::kernels::gemm_bf16_config_for(m, n, 512, &bm, &bn, &bk, &threads);
      EXPECT_TRUE(is_gemm_bf16_candidate(bm, bn, bk, threads));
    }
  }
}
