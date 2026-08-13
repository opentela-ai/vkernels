// tests/kernels/moe/test_moe_aux.cpp
//
// Tests for the MXFP4 MoE orchestration ops (moe_aux.hpp): per-token
// quantization, token→expert gather, and routed scatter-reduce. The CPU
// reference in moe_aux.cpp is the oracle; these tests check the round-trip
// (quant is the inverse of the weight decode), gather correctness, and that
// the scatter-reduce matches a naive sequential accumulation.
#include "minitest.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "vkernels/kernels/moe_aux.hpp"
#include "vkernels/kernels/moe_fused.hpp"  // moe_align_block_size

using namespace vkernels::kernels;

namespace {

float bf2f(uint16_t v) {
  uint32_t b = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &b, sizeof(float));
  return f;
}
uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return static_cast<uint16_t>(b >> 16);
}

// E2M1 nibble → float (identical to moe.cpp).
float nib2f(uint8_t n) {
  int s = (n >> 3) & 1, e = (n >> 1) & 3, m = n & 1;
  if (e == 0) return m ? (s ? -0.25f : 0.25f) : 0.0f;
  if (e == 3) return m ? std::nanf("") : (s ? -INFINITY : INFINITY);
  float v = (1.0f + static_cast<float>(m) * 0.5f) * static_cast<float>(1 << (e - 1));
  return s ? -v : v;
}
float ue8m0(uint8_t s) {
  if (s == 0xFF) return 0.0f;
  float f;
  uint32_t b = static_cast<uint32_t>(s) << 23;
  std::memcpy(&f, &b, sizeof(float));
  return f;
}

// Dequant a per-group-packed activation tile to bf16 (the W4A4 A path).
std::vector<uint16_t> dequant_act(const uint8_t* packed, const uint8_t* scales,
                                  int M, int hidden, int group_size) {
  std::vector<uint16_t> out((size_t)M * hidden);
  int ng = hidden / group_size;
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < ng; ++g) {
      float sc = ue8m0(scales[m * ng + g]);
      for (int i = 0; i < group_size; ++i) {
        uint8_t byte = packed[m * (hidden / 2) + (g * group_size + i) / 2];
        uint8_t nib = (i & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        out[m * hidden + g * group_size + i] = f2bf(nib2f(nib) * sc);
      }
    }
  return out;
}

}  // namespace

// --- #16 mxfp4_moe_quant: round-trip + scale invariant ---------------------
TEST(MoeAux, QuantRoundTrip) {
  constexpr int M = 4, hidden = 64, gs = 32;
  const float repr[] = {0.f, 0.25f, -0.25f, 1.f, -1.f, 1.5f, -1.5f, 2.f, -2.f,
                        3.f, -3.f, 0.7f, -0.7f, 2.3f, -2.3f, 0.125f};
  std::vector<uint16_t> A((size_t)M * hidden);
  for (size_t i = 0; i < A.size(); ++i)
    A[i] = f2bf(repr[i % (sizeof(repr) / sizeof(repr[0]))]);

  std::vector<uint8_t> packed((size_t)M * hidden / 2), scales((size_t)M * (hidden / gs));
  mxfp4_moe_quant(A.data(), packed.data(), scales.data(), M, hidden, gs);

  auto dq = dequant_act(packed.data(), scales.data(), M, hidden, gs);
  // Each dequant element is the nearest fp4 to the input, so it must lie
  // within one fp4 step (<= 0.125 at the smallest scale, 0.5 at the top).
  for (size_t i = 0; i < A.size(); ++i) {
    float a = bf2f(A[i]), d = bf2f(dq[i]);
    EXPECT_TRUE(std::isfinite(d));
    float step = (std::fabs(a) <= 0.25f) ? 0.25f : 0.5f;
    EXPECT_NEAR(std::fabs(a), std::fabs(d), step + 1e-6f);
    if (a == 0.0f) EXPECT_EQ(d, 0.0f);
    // Sign preserved (zero maps to zero of the same sign).
    if (a != 0.0f) EXPECT_TRUE((a < 0) == (d < 0));
  }

  // Scale invariant: scale >= amax / 3 (no element saturates), and the
  // largest element of each group lands within [1.5, 3] * scale (tight fit
  // — ceil(log2(amax/3)) guarantees amax/scale in (1.5, 3] when amax >= 3).
  int ng = hidden / gs;
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < ng; ++g) {
      uint8_t sb = scales[m * ng + g];
      if (sb == 0xFF) continue;  // zero group
      float sc = ue8m0(sb);
      float amax = 0.f;
      for (int i = 0; i < gs; ++i) {
        float a = std::fabs(bf2f(A[m * hidden + g * gs + i]));
        if (a > amax) amax = a;
      }
      EXPECT_GE(sc, amax / 3.0f - 1e-6f);
      EXPECT_TRUE(amax / sc <= 3.0f + 1e-6f);
    }
}

TEST(MoeAux, QuantZeroGroup) {
  constexpr int M = 2, hidden = 32, gs = 32;
  std::vector<uint16_t> A((size_t)M * hidden, 0);  // all zero
  std::vector<uint8_t> packed((size_t)M * hidden / 2), scales((size_t)M);
  mxfp4_moe_quant(A.data(), packed.data(), scales.data(), M, hidden, gs);
  for (uint8_t s : scales) EXPECT_EQ(s, 0xFF);
  for (uint8_t b : packed) EXPECT_EQ(b, 0);
}

TEST(MoeAux, QuantRejectsBadGroupSize) {
  constexpr int M = 1, hidden = 30;  // not divisible by 32
  std::vector<uint16_t> A((size_t)M * hidden, 0);
  std::vector<uint8_t> p((size_t)M * hidden / 2), s((size_t)M);
  EXPECT_THROW(mxfp4_moe_quant(A.data(), p.data(), s.data(), M, hidden, 32),
               std::invalid_argument);
}

// --- #17 mxfp4_moe_sort: gather + padding ----------------------------------
TEST(MoeAux, SortGather) {
  constexpr int M = 3, hidden = 4, top_k = 2;
  // sorted_ids: row0 = token1 sel0 (flat 2), row1 = padding (flat 6 >= 6),
  //             row2 = token0 sel1 (flat 1)
  std::vector<int32_t> ids = {2, 6, 1};
  constexpr int EM = 3;
  std::vector<uint16_t> A = {10, 11, 12, 13,   // token 0
                             20, 21, 22, 23,   // token 1
                             30, 31, 32, 33};  // token 2
  std::vector<uint16_t> out((size_t)EM * hidden);
  mxfp4_moe_sort(A.data(), ids.data(), out.data(), M, hidden, top_k, EM);
  // Row 0 = token 1
  EXPECT_EQ(out[0], 20); EXPECT_EQ(out[3], 23);
  // Row 1 = padding (zero)
  for (int j = 0; j < hidden; ++j) EXPECT_EQ(out[hidden + j], 0);
  // Row 2 = token 0
  EXPECT_EQ(out[2 * hidden + 0], 10); EXPECT_EQ(out[2 * hidden + 1], 11);
}

// --- #18 mxfp4_moe_sort_scales: gather + padding ---------------------------
TEST(MoeAux, SortScalesGather) {
  constexpr int M = 2, ng = 3, top_k = 1, EM = 3;
  std::vector<int32_t> ids = {1, 1, 2};  // row2 = padding (>= 2)
  std::vector<uint8_t> scales = {100, 101, 102,   // token 0
                                 110, 111, 112};  // token 1
  std::vector<uint8_t> out((size_t)EM * ng);
  mxfp4_moe_sort_scales(scales.data(), ids.data(), out.data(), M, ng, top_k, EM);
  EXPECT_EQ(out[0], 110); EXPECT_EQ(out[1], 111); EXPECT_EQ(out[2], 112);  // token1
  EXPECT_EQ(out[3], 110); EXPECT_EQ(out[4], 111); EXPECT_EQ(out[5], 112);
  for (int j = 0; j < ng; ++j) EXPECT_EQ(out[2 * ng + j], 0);  // padding
}

// --- #19 mxfp4_moe_scatter_reduce: matches naive sequential accumulation ---
TEST(MoeAux, ScatterReduce) {
  constexpr int M = 3, width = 5, top_k = 2, EM = 4;
  // Two sorted rows map to token 0 (flats 0,1), one to token 1 (flat 2),
  // one is padding (flat 6 >= 6).
  std::vector<int32_t> ids = {0, 1, 2, 6};
  std::vector<float> partial = {1,2,3,4,5,   10,20,30,40,50,
                                100,200,300,400,500,  9,9,9,9,9};
  std::vector<float> w = {0.5f, 0.5f, 1.0f, 0.0f};
  std::vector<float> out((size_t)M * width, 0.0f);
  mxfp4_moe_scatter_reduce(partial.data(), w.data(), ids.data(), out.data(),
                           M, width, top_k, EM);

  // Naive reference (same order → bit-identical).
  std::vector<float> ref((size_t)M * width, 0.0f);
  for (int r = 0; r < EM; ++r) {
    int flat = ids[r];
    if (flat < 0 || flat >= M * top_k) continue;
    int tok = flat / top_k;
    for (int j = 0; j < width; ++j)
      ref[tok * width + j] = std::move(ref[tok * width + j]) + partial[r * width + j] * w[r];
  }
  for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(out[i], ref[i]);

  // Spot checks: token0 = 0.5*[1..5] + 0.5*[10..50] = [5.5, 11, 16.5, 22, 27.5]
  EXPECT_NEAR(out[0], 5.5f, 1e-5f); EXPECT_NEAR(out[4], 27.5f, 1e-5f);
  // token1 = 1.0*[100..500]
  EXPECT_NEAR(out[5], 100.0f, 1e-5f); EXPECT_NEAR(out[9], 500.0f, 1e-5f);
  // token2 untouched (padding row skipped) → 0
  EXPECT_EQ(out[10], 0.0f);
}

// --- #19q mxfp4_moe_scatter_reduce_q: matches float scatter of the dequant -
TEST(MoeAux, ScatterReduceQ) {
  constexpr int M = 2, width = 32, top_k = 1, EM = 2, gs = 32;
  // Build a float partial of fp4-representable values so dequant is exact,
  // then quantize it (per-row, one group) and scatter-reduce both forms.
  std::vector<int32_t> ids = {0, 1};
  std::vector<float> partial_f((size_t)EM * width);
  const float repr[] = {0.f, 0.25f, -0.25f, 1.f, -1.f, 1.5f, 2.f, 3.f};
  for (size_t i = 0; i < partial_f.size(); ++i)
    partial_f[i] = repr[i % (sizeof(repr) / sizeof(repr[0]))];
  std::vector<float> w = {2.0f, 0.5f};

  // Quantize the partial (row-major, group_size = width = one group/row).
  std::vector<uint8_t> pq((size_t)EM * width / 2), ps((size_t)EM);
  // Build bf16 view, run mxfp4_moe_quant on it.
  std::vector<uint16_t> bf16((size_t)EM * width);
  for (size_t i = 0; i < bf16.size(); ++i) bf16[i] = f2bf(partial_f[i]);
  mxfp4_moe_quant(bf16.data(), pq.data(), ps.data(), EM, width, gs);

  std::vector<float> out_q((size_t)M * width, 0.0f);
  mxfp4_moe_scatter_reduce_q(pq.data(), ps.data(), w.data(), ids.data(),
                             out_q.data(), M, width, top_k, EM, gs);

  // Dequantize the quantized partial back to float and scatter-reduce it.
  auto dq = dequant_act(pq.data(), ps.data(), EM, width, gs);
  std::vector<float> dqf((size_t)EM * width);
  for (size_t i = 0; i < dqf.size(); ++i) dqf[i] = bf2f(dq[i]);
  std::vector<float> out_f((size_t)M * width, 0.0f);
  mxfp4_moe_scatter_reduce(dqf.data(), w.data(), ids.data(), out_f.data(),
                           M, width, top_k, EM);

  // Since the partial values are fp4-representable, dequant is exact and the
  // two paths must agree bit-for-bit.
  for (size_t i = 0; i < out_q.size(); ++i) EXPECT_EQ(out_q[i], out_f[i]);
}

// --- #19q mxfp4_moe_scatter_reduce_q: inf/NaN nibble codes propagate ------
// fp4 nibble codes 6=+inf, 7=NaN, 14=-inf, 15=NaN must pass through the
// e==3 dequant branch (fp4_nibble_to_float in moe_aux.cpp) untouched.
TEST(MoeAux, ScatterReduceQNanInf) {
  constexpr int M = 1, width = 4, top_k = 1, EM = 1, gs = 4;
  // packed bytes: low->high nibbles = 6,7,14,15 (inf,NaN,-inf,NaN).
  std::vector<uint8_t> pq = {0x76, 0xFE};
  std::vector<uint8_t> ps = {127};          // scale = 2^0 = 1.0
  std::vector<int32_t> ids = {0};
  std::vector<float> w = {1.0f};
  std::vector<float> out((size_t)M * width, 0.0f);
  mxfp4_moe_scatter_reduce_q(pq.data(), ps.data(), w.data(), ids.data(),
                             out.data(), M, width, top_k, EM, gs);
  EXPECT_EQ(out[0], std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(out[1]));
  EXPECT_EQ(out[2], -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(out[3]));
}

// --- End-to-end: align → sort → quant → dequant ≈ A_sorted ------------------
TEST(MoeAux, PipelineAlignSortQuant) {
  constexpr int M = 8, top_k = 2, E = 4, hidden = 64, gs = 32, BS = 16;
  std::vector<int32_t> topk_ids((size_t)M * top_k);
  for (int i = 0; i < (int)topk_ids.size(); ++i)
    topk_ids[i] = (i * 3) % E;  // pseudo-random routing

  std::vector<int32_t> sorted_ids(1024), expert_ids(1024 / BS);
  int EM = moe_align_block_size(topk_ids.data(), M, top_k, BS, E,
                                sorted_ids.data(), expert_ids.data());
  sorted_ids.resize(EM);

  // Random bf16 activations (int indexing: (i%7)-3 must not underflow).
  std::vector<uint16_t> A((size_t)M * hidden);
  for (int i = 0; i < (int)A.size(); ++i)
    A[i] = f2bf(((i % 7) - 3) * 0.5f);

  // Sort then quantize the sorted activations.
  std::vector<uint16_t> A_s((size_t)EM * hidden);
  mxfp4_moe_sort(A.data(), sorted_ids.data(), A_s.data(), M, hidden, top_k, EM);
  std::vector<uint8_t> pk((size_t)EM * hidden / 2), sc((size_t)EM * (hidden / gs));
  mxfp4_moe_quant(A_s.data(), pk.data(), sc.data(), EM, hidden, gs);

  // Dequant back; for real rows this recovers the original A row (within fp4),
  // and padding rows (sorted_ids >= M*top_k) decode to exactly zero.
  auto dq = dequant_act(pk.data(), sc.data(), EM, hidden, gs);
  const int N = M * top_k;
  for (int r = 0; r < EM; ++r) {
    int flat = sorted_ids[r];
    for (int j = 0; j < hidden; ++j) {
      float got = bf2f(dq[(size_t)r * hidden + j]);
      if (flat >= 0 && flat < N) {
        float want = bf2f(A[(size_t)(flat / top_k) * hidden + j]);
        float step = (std::fabs(want) <= 0.25f) ? 0.25f : 0.5f;
        EXPECT_NEAR(got, want, step + 1e-6f);
      } else {
        EXPECT_EQ(got, 0.0f);  // padding row → zero scale, zero nibbles
      }
    }
  }
}
