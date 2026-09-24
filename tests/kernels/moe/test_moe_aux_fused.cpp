// tests/kernels/moe/test_moe_aux_fused.cpp
//
// Tests for the fused gather+quantize op `mxfp4_moe_sorted_quant`
// (moe_aux.hpp). The core contract is BIT-EXACT composition parity against
// the three standalone ops it may replace:
//
//   fused(A, sorted_ids)  ==  mxfp4_moe_quant(mxfp4_moe_sort(A, sorted_ids))
//                           (packed AND scales)
//   fused.scales          ==  mxfp4_moe_sort_scales(mxfp4_moe_quant(A).scales,
//                                                     sorted_ids)
//
// The matrix covers: amax == 0 groups, non-finite groups (inf / NaN bf16),
// padding rows (sorted_ids outside [0, M*top_k), which must quantize exactly
// like real all-zero activations), E2M1 tie breakpoints, multiple group
// sizes / group counts (including an odd group count), EM > M*top_k, and
// the K3-like shape. The CPU reference in moe_aux.cpp is the oracle.
#include "minitest.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "vkernels/kernels/moe_aux.hpp"
#include "vkernels/kernels/moe_fused.hpp"  // moe_align_block_size

using namespace vkernels::kernels;

namespace {

uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return static_cast<uint16_t>(b >> 16);
}

// Deterministic pseudo-random bf16 fill (mixed magnitudes and signs, plus
// injected specials every 11th element).
void fill_bf16(std::vector<uint16_t>& A, int seed, bool specials) {
  for (int i = 0; i < static_cast<int>(A.size()); ++i) {
    unsigned x = 2654435761u * static_cast<unsigned>(seed) +
                 40503u * static_cast<unsigned>(i);
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    float v = (static_cast<float>(static_cast<int>(x % 20001)) - 10000.0f) /
              4096.0f;  // ~[-2.44, +2.44]
    if (specials && i % 11 == 10) {
      // Exact-zero / tiny-subnormal / huge groups hit the amax == 0 and
      // clamp branches; inf / NaN hit the non-finite branch.
      switch (i % 44) {
        case 10: v = 0.0f; break;
        case 21: v = -0.0f; break;
        case 32: v = std::numeric_limits<float>::infinity(); break;
        case 43: v = std::nanf(""); break;
      }
    }
    A[i] = f2bf(v);
  }
}

// Reference composition: sort A into sorted order, then quantize the sorted
// rows. Returns packed and scales in sorted row order.
void ref_composition(const std::vector<uint16_t>& A,
                     const std::vector<int32_t>& sorted_ids, int M, int hidden,
                     int group_size, int top_k, int EM,
                     std::vector<uint8_t>* pk_out, std::vector<uint8_t>* sc_out,
                     std::vector<uint16_t>* sorted_out = nullptr) {
  std::vector<uint16_t> A_sorted(static_cast<size_t>(EM) * hidden);
  mxfp4_moe_sort(A.data(), sorted_ids.data(), A_sorted.data(), M, hidden,
                 top_k, EM);
  pk_out->assign(static_cast<size_t>(EM) * (hidden / 2), 0xAA);
  sc_out->assign(static_cast<size_t>(EM) * (hidden / group_size), 0xAA);
  mxfp4_moe_quant(A_sorted.data(), pk_out->data(), sc_out->data(), EM, hidden,
                  group_size);
  if (sorted_out) *sorted_out = std::move(A_sorted);
}

// minitest's EXPECT_EQ stringifies operands, which std::vector does not
// support — compare byte buffers with an explicit first-mismatch report.
void expect_buf_eq(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want,
                   const char* what, const char* ctx) {
  if (got == want) return;
  size_t n = std::min(got.size(), want.size());
  for (size_t i = 0; i < n; ++i)
    if (got[i] != want[i]) {
      std::fprintf(stderr, "  %s mismatch (%s): first diff at %zu: got %u want %u\n",
                   what, ctx, i, unsigned(got[i]), unsigned(want[i]));
      break;
    }
  EXPECT_TRUE(got == want);
}

}  // namespace

// --- Fused == sort -> quant, over a shape matrix (bit-exact) ---------------
TEST(MoeAuxFused, CompositionParityMatrix) {
  struct Case {
    int M, hidden, gs, top_k, E, BS;
  };
  const Case cases[] = {
      {3, 64, 32, 2, 4, 16},    // tiny, multiple groups per row
      {1, 32, 32, 1, 4, 16},    // single token, one group per row
      {2, 96, 32, 3, 4, 16},    // odd group count (3) per row
      {5, 48, 16, 3, 4, 16},    // 3 groups of 16
      {4, 32, 4, 2, 4, 16},     // group_size 4 (8 groups/row)
      {2, 16, 2, 2, 4, 16},     // group_size 2 (nibble-pair boundary)
      {7, 64, 64, 2, 4, 16},    // one group spanning the whole row
      {8, 128, 32, 4, 6, 16},   // EM > M*top_k padding-heavy
      {16, 256, 32, 8, 8, 16},  // wider routing, more padding
      {112, 512, 32, 16, 64, 16},  // K3-like routing geometry
  };
  for (const Case& c : cases) {
    char ctx[96];
    std::snprintf(ctx, sizeof(ctx), "M=%d hidden=%d gs=%d top_k=%d", c.M,
                  c.hidden, c.gs, c.top_k);
    std::vector<int32_t> topk_ids(static_cast<size_t>(c.M) * c.top_k);
    for (int i = 0; i < static_cast<int>(topk_ids.size()); ++i)
      topk_ids[i] = (i * 5 + i / c.top_k) % c.E;  // skewed but full routing
    std::vector<int32_t> sorted_ids(4096), expert_ids(4096 / c.BS);
    int EM = moe_align_block_size(topk_ids.data(), c.M, c.top_k, c.BS, c.E,
                                  sorted_ids.data(), expert_ids.data());
    sorted_ids.resize(EM);

    std::vector<uint16_t> A(static_cast<size_t>(c.M) * c.hidden);
    fill_bf16(A, c.M * 31 + c.hidden, /*specials=*/true);

    std::vector<uint8_t> pk, sc;
    ref_composition(A, sorted_ids, c.M, c.hidden, c.gs, c.top_k, EM, &pk, &sc);

    std::vector<uint8_t> pk_f(static_cast<size_t>(EM) * (c.hidden / 2), 0x55),
        sc_f(static_cast<size_t>(EM) * (c.hidden / c.gs), 0x55);
    mxfp4_moe_sorted_quant(A.data(), sorted_ids.data(), pk_f.data(), sc_f.data(),
                           c.M, c.hidden, c.gs, c.top_k, EM);

    expect_buf_eq(pk_f, pk, "packed", ctx);
    expect_buf_eq(sc_f, sc, "scales", ctx);
  }
}

// --- Fused scales == quant(A) gathered by sort_scales (the doc contract) ---
TEST(MoeAuxFused, ScalesMatchQuantThenSortScales) {
  constexpr int M = 6, hidden = 128, gs = 32, top_k = 2, E = 4, BS = 16;
  std::vector<int32_t> topk_ids(static_cast<size_t>(M) * top_k);
  for (int i = 0; i < static_cast<int>(topk_ids.size()); ++i)
    topk_ids[i] = (i * 3) % E;
  std::vector<int32_t> sorted_ids(1024), expert_ids(1024 / BS);
  int EM = moe_align_block_size(topk_ids.data(), M, top_k, BS, E,
                                sorted_ids.data(), expert_ids.data());
  sorted_ids.resize(EM);

  std::vector<uint16_t> A(static_cast<size_t>(M) * hidden);
  fill_bf16(A, 7, /*specials=*/true);

  const int ng = hidden / gs;
  // quant in token order, then gather scales into sorted order.
  std::vector<uint8_t> pk_tok(static_cast<size_t>(M) * (hidden / 2)),
      sc_tok(static_cast<size_t>(M) * ng);
  mxfp4_moe_quant(A.data(), pk_tok.data(), sc_tok.data(), M, hidden, gs);
  std::vector<uint8_t> sc_sorted(static_cast<size_t>(EM) * ng, 0xAA);
  mxfp4_moe_sort_scales(sc_tok.data(), sorted_ids.data(), sc_sorted.data(), M,
                        ng, top_k, EM);

  std::vector<uint8_t> pk_f(static_cast<size_t>(EM) * (hidden / 2)),
      sc_f(static_cast<size_t>(EM) * ng);
  mxfp4_moe_sorted_quant(A.data(), sorted_ids.data(), pk_f.data(), sc_f.data(),
                         M, hidden, gs, top_k, EM);

  // REAL rows: sort_scales(quant(A).scales) must equal the fused scales.
  // (Padding rows are excluded: the fused op emits the quant-of-a-zero-row
  // encoding (0xFF + zero nibbles, per the sort→quant composition), while
  // the standalone sort_scales writes literal zeros for padding rows — a
  // pre-existing asymmetry between the two standalone routes, documented in
  // docs/kernels/moe_aux.md. The grouped GEMM never reads padding rows.)
  const int N = M * top_k;
  for (int r = 0; r < EM; ++r) {
    if (sorted_ids[r] < 0 || sorted_ids[r] >= N) continue;
    for (int g = 0; g < ng; ++g)
      EXPECT_EQ(sc_f[static_cast<size_t>(r) * ng + g],
                sc_sorted[static_cast<size_t>(r) * ng + g]);
  }
  // And the packed output must equal quantizing the sorted activation.
  std::vector<uint16_t> A_sorted;
  std::vector<uint8_t> pk_ref, sc_ref;
  ref_composition(A, sorted_ids, M, hidden, gs, top_k, EM, &pk_ref, &sc_ref,
                  &A_sorted);
  expect_buf_eq(pk_f, pk_ref, "packed", "sort->quant ref");
  expect_buf_eq(sc_f, sc_ref, "scales", "sort->quant ref");
}

// --- Padding rows quantize exactly like real all-zero activations ---------
TEST(MoeAuxFused, PaddingRowsMatchZeroActivations) {
  constexpr int M = 2, hidden = 64, gs = 32, top_k = 2, EM = 6;
  // Handcrafted ids: rows 0..2 real (flats 0, 3, 1), rows 3..5 padding
  // (flat == M*top_k, > it, and negative — the sort contract treats all
  // three as padding).
  std::vector<int32_t> ids = {0, 3, 1, 4, 9, -2};
  std::vector<uint16_t> A(static_cast<size_t>(M) * hidden);
  fill_bf16(A, 13, /*specials=*/false);

  std::vector<uint8_t> pk_f(static_cast<size_t>(EM) * (hidden / 2)),
      sc_f(static_cast<size_t>(EM) * (hidden / gs));
  mxfp4_moe_sorted_quant(A.data(), ids.data(), pk_f.data(), sc_f.data(), M,
                         hidden, gs, top_k, EM);

  // Reference: an all-zero [1, hidden] activation quantized by the UNFUSED
  // op must produce exactly what each padding row carries.
  std::vector<uint16_t> zero(static_cast<size_t>(hidden), 0);
  std::vector<uint8_t> pk_z(hidden / 2), sc_z(hidden / gs);
  mxfp4_moe_quant(zero.data(), pk_z.data(), sc_z.data(), 1, hidden, gs);
  for (uint8_t s : sc_z) EXPECT_EQ(s, 0xFF);
  for (uint8_t b : pk_z) EXPECT_EQ(b, 0);
  for (int r = 3; r < EM; ++r) {
    for (int j = 0; j < hidden / 2; ++j)
      EXPECT_EQ(pk_f[static_cast<size_t>(r) * (hidden / 2) + j], pk_z[j]);
    for (int g = 0; g < hidden / gs; ++g)
      EXPECT_EQ(sc_f[static_cast<size_t>(r) * (hidden / gs) + g], sc_z[g]);
  }

  // Real rows still match the composition.
  std::vector<uint8_t> pk, sc;
  ref_composition(A, ids, M, hidden, gs, top_k, EM, &pk, &sc);
  expect_buf_eq(pk_f, pk, "packed", "composition");
  expect_buf_eq(sc_f, sc, "scales", "composition");
}

// --- Zero / non-finite groups survive the fusion bit-exactly ---------------
TEST(MoeAuxFused, ZeroAndNonFiniteGroups) {
  constexpr int M = 3, hidden = 64, gs = 16, top_k = 1, EM = M;
  std::vector<int32_t> ids = {0, 1, 2};
  // Row 0: all zeros. Row 1: +inf in every group. Row 2: NaN in group 1
  // only (finite values elsewhere — the oracle's max ignores NaN, so the
  // all-NaN group takes the amax == 0 path while the others quantize).
  std::vector<uint16_t> A(static_cast<size_t>(M) * hidden, 0);
  for (int j = 0; j < hidden; ++j) A[hidden + j] = f2bf(std::numeric_limits<float>::infinity());
  for (int j = 0; j < hidden; ++j) A[2 * hidden + j] = f2bf(0.5f + (j % 3));
  for (int j = gs; j < 2 * gs; ++j) A[2 * hidden + j] = f2bf(std::nanf(""));

  std::vector<uint8_t> pk_f(static_cast<size_t>(EM) * (hidden / 2)),
      sc_f(static_cast<size_t>(EM) * (hidden / gs));
  mxfp4_moe_sorted_quant(A.data(), ids.data(), pk_f.data(), sc_f.data(), M,
                         hidden, gs, top_k, EM);

  std::vector<uint8_t> pk, sc;
  ref_composition(A, ids, M, hidden, gs, top_k, EM, &pk, &sc);
  expect_buf_eq(pk_f, pk, "packed", "composition");
  expect_buf_eq(sc_f, sc, "scales", "composition");

  // Explicit contract checks on the fused output:
  const int ng = hidden / gs;
  for (int g = 0; g < ng; ++g) EXPECT_EQ(sc_f[0 * ng + g], 0xFF);       // zero row
  for (int g = 0; g < ng; ++g) EXPECT_EQ(sc_f[1 * ng + g], 0xFF);       // inf row
  EXPECT_EQ(sc_f[2 * ng + 1], 0xFF);                                    // NaN group
  EXPECT_EQ(pk_f[2 * (hidden / 2) + gs / 2], 0);                        // its nibbles
  for (int g = 0; g < ng; ++g)
    if (g != 1) EXPECT_TRUE(sc_f[2 * ng + g] != 0xFF);  // finite groups quantize
}

// --- E2M1 tie breakpoints compose bit-exactly (ties -> larger magnitude) --
TEST(MoeAuxFused, TieBreakpointsCompose) {
  constexpr int M = 1, hidden = 32, gs = 32, top_k = 1, EM = 1;
  // Values sitting exactly on the round-to-nearest breakpoints when the
  // group scale is 2^0: 0.125, 0.625, 1.25, 1.75, 2.5 and their negatives.
  const float repr[] = {0.125f,  -0.125f, 0.625f,  -0.625f, 1.25f,  -1.25f,
                        1.75f,   -1.75f,  2.5f,    -2.5f,   0.124f,  0.126f,
                        0.624f,  0.626f,  1.24f,   2.49f,   3.0f,    -3.0f,
                        0.0625f, 0.25f,   1.0f,    1.5f,    2.0f,    0.0f,
                        0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,
                        0.0f,    0.0f};
  std::vector<uint16_t> A(hidden);
  for (int j = 0; j < hidden; ++j) A[j] = f2bf(repr[j]);
  std::vector<int32_t> ids = {0};

  std::vector<uint8_t> pk_f(hidden / 2), sc_f(1), pk(hidden / 2), sc(1);
  mxfp4_moe_sorted_quant(A.data(), ids.data(), pk_f.data(), sc_f.data(), M,
                         hidden, gs, top_k, EM);
  std::vector<uint8_t> pk2, sc2;
  ref_composition(A, ids, M, hidden, gs, top_k, EM, &pk2, &sc2);
  expect_buf_eq(pk_f, pk2, "packed", "tie composition");
  expect_buf_eq(sc_f, sc2, "scales", "tie composition");

  // The amax (3.0) must give scale byte 127 (e = ceil(log2(1)) = 0).
  EXPECT_EQ(sc_f[0], 127);
}

// --- Validation mirrors the unfused op's argument contract -----------------
TEST(MoeAuxFused, RejectsBadArgs) {
  constexpr int M = 2, hidden = 32, gs = 32, top_k = 1, EM = 2;
  std::vector<uint16_t> A(static_cast<size_t>(M) * hidden, 0);
  std::vector<int32_t> ids = {0, 1};
  std::vector<uint8_t> pk(static_cast<size_t>(EM) * (hidden / 2)),
      sc(static_cast<size_t>(EM) * (hidden / gs));
  EXPECT_THROW(mxfp4_moe_sorted_quant(nullptr, ids.data(), pk.data(), sc.data(),
                                      M, hidden, gs, top_k, EM),
               std::invalid_argument);
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), nullptr, pk.data(), sc.data(),
                                      M, hidden, gs, top_k, EM),
               std::invalid_argument);
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), ids.data(), nullptr, sc.data(),
                                      M, hidden, gs, top_k, EM),
               std::invalid_argument);
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), ids.data(), pk.data(), nullptr,
                                      M, hidden, gs, top_k, EM),
               std::invalid_argument);
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), ids.data(), pk.data(), sc.data(),
                                      M, hidden, 0, top_k, EM),
               std::invalid_argument);
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), ids.data(), pk.data(), sc.data(),
                                      M, hidden, 7, top_k, EM),
               std::invalid_argument);
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), ids.data(), pk.data(), sc.data(),
                                      M, 33, 11, top_k, EM),
               std::invalid_argument);  // hidden odd
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), ids.data(), pk.data(), sc.data(),
                                      M, hidden, gs, 0, EM),
               std::invalid_argument);  // top_k == 0
  EXPECT_THROW(mxfp4_moe_sorted_quant(A.data(), ids.data(), pk.data(), sc.data(),
                                      M, hidden, gs, top_k, -1),
               std::invalid_argument);  // negative EM
}
