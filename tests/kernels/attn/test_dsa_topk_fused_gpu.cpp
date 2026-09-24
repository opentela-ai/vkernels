// tests/kernels/attn/test_dsa_topk_fused_gpu.cpp
//
// GPU test for the VK_DSA_TOPK_FUSED=1 launch-chain fusion of the DSA kpool
// top-k (dsa_topk.hip): the radix transform row is folded into the tail of a
// cooperative grid that also computes the scalar top-k logits, so the
// indexer chain runs 1 launch instead of 2.
//
// Three gates, mirroring the fusion-lane contract (NOTES-fusion-dsa-kda-launches):
//   1. BIT-IDENTITY of the logits: the fused phase-1 GEMV computes the SAME
//      statements in the SAME order as the standalone dsa_topk_logits scalar
//      kernels (only lanes >= B of the 1024-thread cooperative block idle),
//      so the logits buffers must be memcmp-equal -- including when the
//      fused launcher CLAMPS the requested split_kv to the co-residency
//      capacity (the grouped logit is grouping-independent, the same
//      property test_dsa_topk_correct.cu checks at split_kv=2).
//   2. SET-IDENTITY of the transform: the fused phase-2 runs the VERBATIM
//      dsa_topk_transform_row body, so the selected group sets (sorted --
//      atomic append order is nondeterministic, the test_capi convention)
//      must be equal.
//   3. ORACLE PARITY: both chains match dsa_topk_logits_cpu (host dequant
//      through the SHARED fp8e4m3fnuz_to_f32, so the oracle sees bit-exact
//      inputs) to the harness tolerance (1e-3), and
//      dsa_topk_transform_cpu on the produced logits.
//
// The device code only exists under VKERNELS_HAS_HIP (real HIP build on
// gfx942, or the HIP-on-NVIDIA shim in the CUDA build). Without it this TU
// compiles to a self-reported skip so the host-only CI stays green.
#include "minitest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/dsa.hpp"
#include "vkernels/kernels/dsa_topk.hpp"

#if VKERNELS_HAS_HIP

#include <hip/hip_runtime.h>

#include "vkernels/kernels/device_numeric.cuh"  // fp8e4m3fnuz_to_f32 (SHARED)

using vkernels::kernels::fp8e4m3fnuz_to_f32;

// minitest has no streaming asserts; carry the HIP error string in the
// report detail instead.
#define VK_EXPECT_MSG(cond, msg)                                                              \
  do {                                                                                        \
    if (!(cond)) {                                                                            \
      ::vkernels::minitest::report(__FILE__, __LINE__, #cond, msg);                           \
      throw ::vkernels::minitest::AbortTest();                                                \
    }                                                                                         \
  } while (0)

#define CK(expr, msg)                                                                         \
  do {                                                                                        \
    const hipError_t _e = (expr);                                                             \
    if (_e != hipSuccess)                                                                     \
      VK_EXPECT_MSG(false, std::string(msg) + ": " + hipGetErrorString(_e));                  \
  } while (0)

namespace {

float rnd(unsigned seed, int i) {
  unsigned x = seed * 2654435761u + (unsigned)i * 40503u;
  x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
  return static_cast<float>(static_cast<int>(x & 0xFFFFu) - 32768) / 32768.0f;
}

// Raw fp8 e4m3fnuz byte; keep 0x7F/0xFF (the fnuz NaN encodings) out.
uint8_t rnd_fp8(unsigned seed, int i) {
  unsigned x = seed * 2654435761u + (unsigned)i * 40503u;
  x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
  uint8_t b = static_cast<uint8_t>(x & 0xFFu);
  return (b == 0x7F || b == 0xFF) ? static_cast<uint8_t>(0x3F) : b;
}

struct Stats {
  double max_abs = 0.0;
  double max_rel = 0.0;
};

Stats cmp_rel(const std::vector<float>& got, const std::vector<float>& ref) {
  Stats s;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double e = std::fabs(static_cast<double>(got[i]) - ref[i]);
    if (e > s.max_abs) s.max_abs = e;
    const double d = std::fmax(std::fabs(static_cast<double>(ref[i])), 1.0);
    if (e / d > s.max_rel) s.max_rel = e / d;
  }
  return s;
}

// Sort each row (the radix append order is atomic-nondeterministic; sets are
// the contract, the test_capi_dsa_topk.hip convention).
void sort_rows(std::vector<int32_t>& v, int rows, int cols) {
  for (int r = 0; r < rows; ++r)
    std::sort(v.begin() + (size_t)r * cols, v.begin() + (size_t)(r + 1) * cols);
}

struct Cfg {
  int bs, H, D, B, mt, nb, pool, group_topk;
  bool paged;   // page-table remap (else ragged offsets)
  bool tail;    // seq_lens partial-pool tail (out_cols = token_topk + pool-1)
};

}  // namespace

TEST(DsaTopkFused, BitIdentitySetIdentityOracleParity) {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s\n", p.name);

  const Cfg cfgs[] = {
      {1, 32, 128, 64, 8, 8, 4, 128, false, false},  // GLM-5.3 indexer decode (fp32-Q)
      {2, 32, 128, 64, 8, 8, 4, 128, false, false},  // batch > 1
      {1, 32, 128, 64, 16, 16, 4, 128, true, false},  // paged remap, max_seq_len=1024
      {1, 64, 128, 64, 8, 8, 4, 128, false, false},  // 2x indexer (GB10 fp32-Q / gfx942 fp8-Q)
      {1, 4, 16, 64, 4, 4, 4, 128, false, true},     // tiny sanity + seq_lens tail
      {1, 32, 128, 64, 8, 8, 8, 256, false, false},  // K=256 instantiation
  };

  for (const auto& c : cfgs) {
    const int max_seq_len = c.mt * c.B;
    const int token_topk = c.group_topk * c.pool;
    const int out_cols = token_topk + (c.tail ? c.pool - 1 : 0);
    const size_t nq = (size_t)c.bs * c.H * c.D;
    const size_t kvstride = (size_t)c.B * (c.D + 4);
    const size_t nkv = (size_t)c.nb * kvstride;
    const size_t nw = (size_t)c.bs * c.H;
    const size_t nout = (size_t)c.bs * max_seq_len;
    const size_t ndst = (size_t)c.bs * out_cols;

    // ---- inputs (same generators as test_dsa_topk_correct.cu) ----
    std::vector<uint8_t> q_u8(nq), kv_u8(nkv);
    std::vector<float> q_f32(nq), kv_f32((size_t)c.nb * c.B * c.D);
    std::vector<float> k_scale((size_t)c.nb * c.B);
    std::vector<float> weight(nw);
    std::vector<int32_t> sl((size_t)c.bs, max_seq_len);
    std::vector<int32_t> pt((size_t)c.bs * c.mt, 0);
    // The transform remap table has page_table_stride = max_seq_len entries
    // per row (one lookup slot per raw token): a bijection so the remapped
    // dst sets stay 1:1 with the selected raw tokens.
    std::vector<int32_t> pt_remap;
    if (c.paged) {
      pt_remap.resize((size_t)c.bs * max_seq_len);
      for (int b = 0; b < c.bs; ++b)
        for (int t = 0; t < max_seq_len; ++t)
          pt_remap[(size_t)b * max_seq_len + t] = (t * 3 + 1) % max_seq_len;
    }
    for (size_t i = 0; i < nq; ++i) {
      q_u8[i] = rnd_fp8(1, (int)i);
      q_f32[i] = fp8e4m3fnuz_to_f32(q_u8[i]);
    }
    for (size_t i = 0; i < k_scale.size(); ++i)
      k_scale[i] = static_cast<float>(rnd(3, (int)i)) * 0.25f + 1.0f;
    for (int pg = 0; pg < c.nb; ++pg) {
      uint8_t* bp = kv_u8.data() + (size_t)pg * kvstride;
      for (int j = 0; j < c.B; ++j)
        for (int d = 0; d < c.D; ++d)
          bp[(size_t)j * c.D + d] = rnd_fp8(2 + pg, j * c.D + d);
      std::memcpy(bp + (size_t)c.B * c.D, k_scale.data() + (size_t)pg * c.B,
                  (size_t)c.B * sizeof(float));
    }
    for (int pg = 0; pg < c.nb; ++pg)
      for (int j = 0; j < c.B; ++j)
        for (int d = 0; d < c.D; ++d)
          kv_f32[((size_t)pg * c.B + j) * c.D + d] =
              fp8e4m3fnuz_to_f32(kv_u8[(size_t)pg * kvstride + (size_t)j * c.D + d]);
    for (size_t i = 0; i < nw; ++i) weight[i] = rnd(4, (int)i);
    for (int b = 0; b < c.bs; ++b)
      for (int i = 0; i < c.mt; ++i)
        pt[(size_t)b * c.mt + i] =
            static_cast<int32_t>(((unsigned)(b * 7 + i * 13)) % (unsigned)c.nb);

    // ---- transform geometry (the test_capi_dsa_topk.hip ragged style) ----
    // lengths mix the identity path (<= group_topk) and the radix path (>):
    // the K=256 config needs > 256-length rows for the radix rounds.
    const std::vector<int32_t> lengths_all =
        c.group_topk == 256 ? std::vector<int32_t>{300, 260, 257}
                            : std::vector<int32_t>{145, 129, 127};
    const std::vector<int32_t> row_starts_all = {2, 1, 3};
    const std::vector<int32_t> offsets_all = {17, 113, 911};
    std::vector<int32_t> lengths(c.bs), row_starts(c.bs), offsets(c.bs);
    for (int b = 0; b < c.bs; ++b) {
      lengths[b] = lengths_all[b % 3];
      row_starts[b] = row_starts_all[b % 3];
      offsets[b] = offsets_all[b % 3];
      ASSERT_TRUE(row_starts[b] + lengths[b] <= max_seq_len);
    }
    std::vector<int32_t> t_sl;
    if (c.tail) t_sl.assign({7});  // tail_count = 7 % pool = 3

    // ---- device buffers ----
    uint8_t *dq, *dkv;
    float *dw, *lg0, *lg1;
    int32_t *dsl, *dpt, *dlen, *drs, *doff, *dptp, *dtsl, *dst0, *dst1;
    CK(hipMalloc(&dq, nq), "alloc q");
    CK(hipMalloc(&dkv, nkv), "alloc kv");
    CK(hipMalloc(&dw, nw * sizeof(float)), "alloc weight");
    CK(hipMalloc(&lg0, nout * sizeof(float)), "alloc logits0");
    CK(hipMalloc(&lg1, nout * sizeof(float)), "alloc logits1");
    CK(hipMalloc(&dsl, c.bs * sizeof(int32_t)), "alloc sl");
    CK(hipMalloc(&dpt, pt.size() * sizeof(int32_t)), "alloc pt");
    CK(hipMalloc(&dlen, c.bs * sizeof(int32_t)), "alloc lengths");
    CK(hipMalloc(&drs, c.bs * sizeof(int32_t)), "alloc row_starts");
    CK(hipMalloc(&doff, c.bs * sizeof(int32_t)), "alloc offsets");
    CK(hipMalloc(&dptp, c.paged ? pt_remap.size() * sizeof(int32_t) : 1),
       "alloc page_table remap");
    CK(hipMalloc(&dtsl, c.tail ? t_sl.size() * sizeof(int32_t) : 1), "alloc t_seq_lens");
    CK(hipMalloc(&dst0, ndst * sizeof(int32_t)), "alloc dst0");
    CK(hipMalloc(&dst1, ndst * sizeof(int32_t)), "alloc dst1");

    CK(hipMemcpy(dq, q_u8.data(), nq, hipMemcpyHostToDevice), "cpy q");
    CK(hipMemcpy(dkv, kv_u8.data(), nkv, hipMemcpyHostToDevice), "cpy kv");
    CK(hipMemcpy(dw, weight.data(), nw * sizeof(float), hipMemcpyHostToDevice), "cpy w");
    CK(hipMemcpy(dsl, sl.data(), c.bs * sizeof(int32_t), hipMemcpyHostToDevice), "cpy sl");
    CK(hipMemcpy(dpt, pt.data(), pt.size() * sizeof(int32_t), hipMemcpyHostToDevice), "cpy pt");
    CK(hipMemcpy(dlen, lengths.data(), c.bs * sizeof(int32_t), hipMemcpyHostToDevice), "cpy len");
    CK(hipMemcpy(drs, row_starts.data(), c.bs * sizeof(int32_t), hipMemcpyHostToDevice), "cpy rs");
    CK(hipMemcpy(doff, offsets.data(), c.bs * sizeof(int32_t), hipMemcpyHostToDevice), "cpy off");
    if (c.tail)
      CK(hipMemcpy(dtsl, t_sl.data(), t_sl.size() * sizeof(int32_t), hipMemcpyHostToDevice),
         "cpy tsl");
    if (c.paged)
      CK(hipMemcpy(dptp, pt_remap.data(), pt_remap.size() * sizeof(int32_t),
                   hipMemcpyHostToDevice), "cpy remap pt");

    // ---- the entry under test, proven chain first (gate off) ----
    // Canaries: unwritten logits cells stay 0.0f in BOTH runs (the kernels
    // write only t < seq_len[b]; both buffers start memset to the same
    // pattern), so memcmp is exact over the whole buffer.
    CK(hipMemset(lg0, 0, nout * sizeof(float)), "zero lg0");
    CK(hipMemset(lg1, 0, nout * sizeof(float)), "zero lg1");
    setenv("VK_DSA_TOPK_FUSED", "0", 1);
    const int split_req = 2;  // matches the serving formula's shape at decode
    vkernels::kernels::hip::dsa_topk_logits_transform_fused(
        c.bs, c.H, c.D, c.B, c.mt, max_seq_len, split_req, dq, dkv, dw, dsl, dpt, lg0,
        /*q_variant=*/0, dlen, dst0, max_seq_len, c.pool, token_topk, out_cols,
        c.paged ? dptp : nullptr, c.paged ? max_seq_len : 0, nullptr,
        c.paged ? nullptr : doff, drs, c.tail ? dtsl : nullptr);
    CK(hipGetLastError(), "proven chain launch");
    CK(hipDeviceSynchronize(), "proven chain sync");

    setenv("VK_DSA_TOPK_FUSED", "1", 1);
    vkernels::kernels::hip::dsa_topk_logits_transform_fused(
        c.bs, c.H, c.D, c.B, c.mt, max_seq_len, split_req, dq, dkv, dw, dsl, dpt, lg1,
        /*q_variant=*/0, dlen, dst1, max_seq_len, c.pool, token_topk, out_cols,
        c.paged ? dptp : nullptr, c.paged ? max_seq_len : 0, nullptr,
        c.paged ? nullptr : doff, drs, c.tail ? dtsl : nullptr);
    CK(hipGetLastError(), "fused launch");
    CK(hipDeviceSynchronize(), "fused sync");

    std::vector<float> h_lg0(nout), h_lg1(nout);
    std::vector<int32_t> h_dst0(ndst), h_dst1(ndst);
    CK(hipMemcpy(h_lg0.data(), lg0, nout * sizeof(float), hipMemcpyDeviceToHost), "cpy lg0");
    CK(hipMemcpy(h_lg1.data(), lg1, nout * sizeof(float), hipMemcpyDeviceToHost), "cpy lg1");
    CK(hipMemcpy(h_dst0.data(), dst0, ndst * sizeof(int32_t), hipMemcpyDeviceToHost), "cpy dst0");
    CK(hipMemcpy(h_dst1.data(), dst1, ndst * sizeof(int32_t), hipMemcpyDeviceToHost), "cpy dst1");

    // Gate 1: bit-identical logits.
    const int lg_bit = std::memcmp(h_lg0.data(), h_lg1.data(), nout * sizeof(float)) != 0;
    VK_EXPECT_MSG(!lg_bit, "fused logits differ bitwise");

    // Gate 2: set-identical transform (sorted rows).
    sort_rows(h_dst0, c.bs, out_cols);
    sort_rows(h_dst1, c.bs, out_cols);
    const int set_bit = std::memcmp(h_dst0.data(), h_dst1.data(), ndst * sizeof(int32_t)) != 0;
    VK_EXPECT_MSG(!set_bit, "fused transform set differs");

    // Gate 3a: logits oracle parity (host dequant through the SHARED helper).
    std::vector<float> ref(nout, 0.0f);
    vkernels::kernels::dsa_topk_logits_cpu(c.bs, c.H, c.D, c.B, c.mt, c.nb, q_f32.data(),
                                           kv_f32.data(), k_scale.data(), weight.data(),
                                           sl.data(), pt.data(), ref.data());
    const Stats s0 = cmp_rel(h_lg0, ref);
    const Stats s1 = cmp_rel(h_lg1, ref);
    VK_EXPECT_MSG(s0.max_rel < 1e-3, "proven vs oracle (max_rel too high)");
    VK_EXPECT_MSG(s1.max_rel < 1e-3, "fused vs oracle (max_rel too high)");

    // Gate 3b: transform oracle on the (proven) logits.
    std::vector<int32_t> ref_dst(ndst);
    vkernels::kernels::dsa_topk_transform_cpu(
        c.bs, h_lg0.data(), lengths.data(), ref_dst.data(), max_seq_len, c.pool,
        token_topk, out_cols, c.paged ? pt_remap.data() : nullptr,
        c.paged ? max_seq_len : 0,
        nullptr, c.paged ? nullptr : offsets.data(), row_starts.data(),
        c.tail ? t_sl.data() : nullptr);
    sort_rows(ref_dst, c.bs, out_cols);
    const int ref_bit0 = std::memcmp(h_dst0.data(), ref_dst.data(), ndst * sizeof(int32_t)) != 0;
    const int ref_bit1 = std::memcmp(h_dst1.data(), ref_dst.data(), ndst * sizeof(int32_t)) != 0;
    VK_EXPECT_MSG(!ref_bit0, "proven transform vs oracle differs");
    VK_EXPECT_MSG(!ref_bit1, "fused transform vs oracle differs");

    std::printf("  bs=%d H=%d D=%d mt=%d K=%d%s%s: bit(logits)=%d set(dst)=%d; "
                "oracle max_rel %.3e / %.3e\n",
                c.bs, c.H, c.D, c.mt, c.group_topk, c.paged ? " paged" : "",
                c.tail ? " tail" : "", lg_bit, set_bit, s0.max_rel, s1.max_rel);

    // ---- split_kv clamp: request far beyond the co-residency capacity; the
    // fused launcher must clamp (grouping independence) and STILL produce
    // bit-identical logits + the same transform sets.
    CK(hipMemset(lg1, 0, nout * sizeof(float)), "re-zero lg1");
    vkernels::kernels::hip::dsa_topk_logits_transform_fused(
        c.bs, c.H, c.D, c.B, c.mt, max_seq_len, /*split_kv=*/1024, dq, dkv, dw, dsl, dpt,
        lg1, 0, dlen, dst1, max_seq_len, c.pool, token_topk, out_cols,
        c.paged ? dptp : nullptr, c.paged ? max_seq_len : 0, nullptr,
        c.paged ? nullptr : doff, drs, c.tail ? dtsl : nullptr);
    CK(hipGetLastError(), "clamped fused launch");
    CK(hipDeviceSynchronize(), "clamped fused sync");
    CK(hipMemcpy(h_lg1.data(), lg1, nout * sizeof(float), hipMemcpyDeviceToHost), "cpy lg1c");
    CK(hipMemcpy(h_dst1.data(), dst1, ndst * sizeof(int32_t), hipMemcpyDeviceToHost), "cpy dst1c");
    VK_EXPECT_MSG(std::memcmp(h_lg0.data(), h_lg1.data(), nout * sizeof(float)) == 0,
                  "clamped-split logits differ");
    sort_rows(h_dst1, c.bs, out_cols);
    VK_EXPECT_MSG(std::memcmp(h_dst0.data(), h_dst1.data(), ndst * sizeof(int32_t)) == 0,
                  "clamped-split transform set differs");

    // ROCm marks hipFree nodiscard; the shim build does not. Teardown
    // failures are not observable here, so discard explicitly.
    (void)hipFree(dq); (void)hipFree(dkv); (void)hipFree(dw);
    (void)hipFree(lg0); (void)hipFree(lg1); (void)hipFree(dsl);
    (void)hipFree(dpt); (void)hipFree(dlen); (void)hipFree(drs);
    (void)hipFree(doff); (void)hipFree(dptp); (void)hipFree(dtsl);
    (void)hipFree(dst0); (void)hipFree(dst1);
  }
}

// The proven-chain fallback: an unsupported group_topk with the gate ON must
// silently run the proven chain (bit-identical outputs, no error).
TEST(DsaTopkFused, FallbackOnUnsupportedGroupTopk) {
  const int bs = 1, H = 32, D = 128, B = 64, mt = 8, nb = 8;
  const int max_seq_len = mt * B;  // 512
  const int token_topk = 400;      // group_topk = 100: NOT a validated spec
  const int out_cols = token_topk;
  const size_t nq = (size_t)bs * H * D, nkv = (size_t)nb * B * (D + 4);

  std::vector<uint8_t> q_u8(nq), kv_u8(nkv);
  std::vector<float> weight((size_t)bs * H);
  std::vector<int32_t> sl(bs, max_seq_len), pt((size_t)bs * mt, 0), lengths(bs, 145),
      offsets(bs, 17);
  for (size_t i = 0; i < nq; ++i) q_u8[i] = rnd_fp8(1, (int)i);
  for (size_t i = 0; i < nkv; ++i) kv_u8[i] = rnd_fp8(2, (int)i);
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = rnd(4, (int)i);

  uint8_t *dq, *dkv;
  float *dw, *lg0, *lg1;
  int32_t *dsl, *dpt, *dlen, *doff, *dst0, *dst1;
  CK(hipMalloc(&dq, nq), "alloc q");
  CK(hipMalloc(&dkv, nkv), "alloc kv");
  CK(hipMalloc(&dw, weight.size() * sizeof(float)), "alloc w");
  CK(hipMalloc(&lg0, (size_t)bs * max_seq_len * sizeof(float)), "alloc lg0");
  CK(hipMalloc(&lg1, (size_t)bs * max_seq_len * sizeof(float)), "alloc lg1");
  CK(hipMalloc(&dsl, bs * 4), "alloc sl");
  CK(hipMalloc(&dpt, pt.size() * 4), "alloc pt");
  CK(hipMalloc(&dlen, bs * 4), "alloc len");
  CK(hipMalloc(&doff, bs * 4), "alloc off");
  CK(hipMalloc(&dst0, (size_t)bs * out_cols * 4), "alloc dst0");
  CK(hipMalloc(&dst1, (size_t)bs * out_cols * 4), "alloc dst1");
  // group_topk=100 is not a validated spec: the transform silently no-ops in
  // BOTH runs, so zero the dst buffers for a defined memcmp.
  CK(hipMemset(dst0, 0, (size_t)bs * out_cols * 4), "zero dst0");
  CK(hipMemset(dst1, 0, (size_t)bs * out_cols * 4), "zero dst1");
  CK(hipMemcpy(dq, q_u8.data(), nq, hipMemcpyHostToDevice), "cpy q");
  CK(hipMemcpy(dkv, kv_u8.data(), nkv, hipMemcpyHostToDevice), "cpy kv");
  CK(hipMemcpy(dw, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice), "cpy w");
  CK(hipMemcpy(dsl, sl.data(), bs * 4, hipMemcpyHostToDevice), "cpy sl");
  CK(hipMemcpy(dpt, pt.data(), pt.size() * 4, hipMemcpyHostToDevice), "cpy pt");
  CK(hipMemcpy(dlen, lengths.data(), bs * 4, hipMemcpyHostToDevice), "cpy len");
  CK(hipMemcpy(doff, offsets.data(), bs * 4, hipMemcpyHostToDevice), "cpy off");

  std::vector<float> h_lg0((size_t)bs * max_seq_len), h_lg1((size_t)bs * max_seq_len);
  std::vector<int32_t> h_dst0((size_t)bs * out_cols), h_dst1((size_t)bs * out_cols);
  for (int gate = 0; gate <= 1; ++gate) {
    char v[2] = {static_cast<char>('0' + gate), 0};
    setenv("VK_DSA_TOPK_FUSED", v, 1);
    vkernels::kernels::hip::dsa_topk_logits_transform_fused(
        bs, H, D, B, mt, max_seq_len, 2, dq, dkv, dw, dsl, dpt, gate ? lg1 : lg0, 0,
        dlen, gate ? dst1 : dst0, max_seq_len, /*pool_size=*/4, token_topk, out_cols,
        nullptr, 0, nullptr, doff, nullptr, nullptr);
    CK(hipGetLastError(), "fallback launch");
    CK(hipDeviceSynchronize(), "fallback sync");
  }
  CK(hipMemcpy(h_lg0.data(), lg0, h_lg0.size() * sizeof(float), hipMemcpyDeviceToHost), "c0");
  CK(hipMemcpy(h_lg1.data(), lg1, h_lg1.size() * sizeof(float), hipMemcpyDeviceToHost), "c1");
  CK(hipMemcpy(h_dst0.data(), dst0, h_dst0.size() * 4, hipMemcpyDeviceToHost), "c2");
  CK(hipMemcpy(h_dst1.data(), dst1, h_dst1.size() * 4, hipMemcpyDeviceToHost), "c3");
  VK_EXPECT_MSG(std::memcmp(h_lg0.data(), h_lg1.data(), h_lg0.size() * sizeof(float)) == 0,
                "fallback logits differ");
  VK_EXPECT_MSG(std::memcmp(h_dst0.data(), h_dst1.data(), h_dst0.size() * 4) == 0,
                "fallback transform differs");

  (void)hipFree(dq); (void)hipFree(dkv); (void)hipFree(dw);
  (void)hipFree(lg0); (void)hipFree(lg1); (void)hipFree(dsl);
  (void)hipFree(dpt); (void)hipFree(dlen); (void)hipFree(doff);
  (void)hipFree(dst0); (void)hipFree(dst1);
}

// Launch-overhead probe (print-only, no assertion): the fusion collapses the
// two-launch logits->transform chain into one cooperative launch. Compare the
// gate-OFF entry (the proven two-launch chain, same signature) against the
// gate-ON fused launch at the GLM decode geometry, back to back.

// TEMP shape-matrix probe (delete before commit)
TEST(DsaTopkFused, LaunchOverheadProbe) {
  constexpr int H = 32, D = 128, B = 64, mt = 64;  // block=64, 64 pages -> 4096 tokens
  constexpr int max_seq_len = mt * B, group_topk = 512;
  // Valid transform spec per the entry contract: token_topk = group_topk *
  // pool_size and out_cols == token_topk. (An earlier draft passed
  // pool_size=1 / out_cols=token_topk/group_topk, which fails the
  // transform-validity predicate -- the transform silently no-ops on BOTH
  // arms and the memcmp then compares uninitialized hipMalloc garbage.)
  constexpr int pool_size = 4, token_topk = group_topk * pool_size;
  constexpr int out_cols = token_topk;
  constexpr int split_kv = (max_seq_len + 1023) / 1024;
  constexpr int bs = 2;

  auto rnd_u8 = [](unsigned& s) {
    return (unsigned char)((s = s * 1664525u + 1013904223u) >> 24);
  };
  const size_t nq = (size_t)bs * H * D;
  const size_t kvstride = (size_t)B * (D + 4);
  const size_t nkv = (size_t)mt * kvstride;
  std::vector<uint8_t> q_u8(nq), kv_u8(nkv);
  unsigned s = 42;
  for (auto& v : q_u8) v = rnd_u8(s);
  for (auto& v : kv_u8) v = rnd_u8(s);
  std::vector<float> w((size_t)bs * H, 1.0f);
  std::vector<int32_t> sl((size_t)bs, max_seq_len),
      lg0((size_t)bs * max_seq_len, 0.0f), lg1((size_t)bs * max_seq_len, 0.0f);
  std::vector<int32_t> pt((size_t)bs * mt, 0), len((size_t)bs, max_seq_len),
      off((size_t)bs, 0), rs((size_t)bs, 0),
      dst0((size_t)bs * out_cols, 0), dst1((size_t)bs * out_cols, 0);

  void* dq = nullptr; void* dkv = nullptr; void* dw = nullptr; void* dsl = nullptr;
  void* dpt = nullptr; void* dlg0 = nullptr; void* dlg1 = nullptr;
  void* dlen = nullptr; void* doff = nullptr; void* drs = nullptr;
  void* ddst0 = nullptr; void* ddst1 = nullptr;
  CK(hipMalloc(&dq, nq), "alloc"); CK(hipMalloc(&dkv, nkv), "alloc");
  CK(hipMalloc(&dw, w.size() * 4), "alloc"); CK(hipMalloc(&dsl, sl.size() * 4), "alloc");
  CK(hipMalloc(&dpt, pt.size() * 4), "alloc"); CK(hipMalloc(&dlg0, lg0.size() * 4), "alloc");
  CK(hipMalloc(&dlg1, lg1.size() * 4), "alloc"); CK(hipMalloc(&dlen, len.size() * 4), "alloc");
  CK(hipMalloc(&doff, off.size() * 4), "alloc"); CK(hipMalloc(&drs, rs.size() * 4), "alloc");
  CK(hipMalloc(&ddst0, dst0.size() * 4), "alloc"); CK(hipMalloc(&ddst1, dst1.size() * 4), "alloc");
  CK(hipMemcpy(dq, q_u8.data(), nq, hipMemcpyHostToDevice), "cpy");
  CK(hipMemcpy(dkv, kv_u8.data(), nkv, hipMemcpyHostToDevice), "cpy");
  CK(hipMemcpy(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice), "cpy");
  CK(hipMemcpy(dsl, sl.data(), sl.size() * 4, hipMemcpyHostToDevice), "cpy");
  CK(hipMemcpy(dpt, pt.data(), pt.size() * 4, hipMemcpyHostToDevice), "cpy");
  CK(hipMemcpy(dlen, len.data(), len.size() * 4, hipMemcpyHostToDevice), "cpy");
  CK(hipMemcpy(doff, off.data(), off.size() * 4, hipMemcpyHostToDevice), "cpy");
  CK(hipMemcpy(drs, rs.data(), rs.size() * 4, hipMemcpyHostToDevice), "cpy");

  auto run = [&](void* lg, void* dst) {
    vkernels::kernels::hip::dsa_topk_logits_transform_fused(
        bs, H, D, B, mt, max_seq_len, split_kv, dq, dkv, dw, dsl, dpt, lg,
        /*q_variant=*/0, reinterpret_cast<const int32_t*>(dlen),
        reinterpret_cast<int32_t*>(dst), max_seq_len, pool_size, token_topk,
        out_cols, nullptr, 0, nullptr, reinterpret_cast<const int32_t*>(doff),
        reinterpret_cast<const int32_t*>(drs), nullptr);
  };

  // warmup + sanity: gate off/on produce identical logits & dst
  setenv("VK_DSA_TOPK_FUSED", "0", 1);
  run(dlg0, ddst0);
  setenv("VK_DSA_TOPK_FUSED", "1", 1);
  run(dlg1, ddst1);
  // CHAIN-vs-CHAIN determinism check at this geometry (diagnostic):
  {
    std::vector<int32_t> dstc(dst0.size(), 0), lgc(lg0.size(), 0.0f);
    void* ddstc = nullptr; void* dlgc = nullptr;
    CK(hipMalloc(&ddstc, dstc.size() * 4), "alloc dstc");
    CK(hipMalloc(&dlgc, lgc.size() * 4), "alloc lgc");
    CK(hipMemset(ddstc, 0, dstc.size() * 4), "zero dstc");
    setenv("VK_DSA_TOPK_FUSED", "0", 1);
    run(dlgc, ddstc);
    CK(hipDeviceSynchronize(), "sync c");
    CK(hipMemcpy(dstc.data(), ddstc, dstc.size() * 4, hipMemcpyDeviceToHost), "cpy dstc");
    CK(hipMemcpy(lgc.data(), dlgc, lgc.size() * 4, hipMemcpyDeviceToHost), "cpy lgc");
    int nd = 0; double mx = 0.0;
    for (size_t di = 0; di < lg0.size(); ++di) {
      if (std::memcmp(&lg0[di], &lgc[di], 4) != 0) {
        ++nd;
        double d = std::fabs(static_cast<double>(lg0[di]) - static_cast<double>(lgc[di]));
        if (!(d <= mx)) mx = d;  // NaN-safe max
      }
    }
    std::printf("  probe chain-vs-chain: dst equal=%d logits ndiff=%d maxdiff=%g\n",
                std::memcmp(dst0.data(), dstc.data(), dst0.size() * 4) == 0 ? 1 : 0,
                nd, mx);
    (void)hipFree(ddstc); (void)hipFree(dlgc);
  }
  CK(hipDeviceSynchronize(), "warmup");
  CK(hipMemcpy(lg0.data(), dlg0, lg0.size() * 4, hipMemcpyDeviceToHost), "cpy");
  CK(hipMemcpy(lg1.data(), dlg1, lg1.size() * 4, hipMemcpyDeviceToHost), "cpy");
  CK(hipMemcpy(dst0.data(), ddst0, dst0.size() * 4, hipMemcpyDeviceToHost), "cpy");
  CK(hipMemcpy(dst1.data(), ddst1, dst1.size() * 4, hipMemcpyDeviceToHost), "cpy");
  VK_EXPECT_MSG(std::memcmp(lg0.data(), lg1.data(), lg0.size() * 4) == 0,
                "probe: gate-off/on logits differ");
  for (size_t di = 0; di < dst0.size(); ++di) {
    if (dst0[di] != dst1[di]) {
      std::printf("  probe dst diff at [%zu]: gate-off=%d gate-on=%d\n",
                  di, dst0[di], dst1[di]);
      break;
    }
  }
  { auto a = dst0, b = dst1;
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    int nz1 = 0, nz0 = 0;
    for (size_t di = 0; di < dst1.size(); ++di) {
      if (dst1[di]) ++nz1;
      if (dst0[di]) ++nz0;
    }
    std::printf("  probe dst sorted-equal: %d nonzero gate-off=%d gate-on=%d\n",
                a == b ? 1 : 0, nz0, nz1); }
  // The radix transform's WITHIN-ROW ORDER is atomic-scheduling-dependent
  // (indices are appended via atomicAdd); the contract is the same SET per
  // row (the BitExact test sorts rows before comparing for the same
  // reason). Compare sorted.
  for (int b = 0; b < bs; ++b) {
    std::sort(dst0.begin() + (size_t)b * out_cols,
              dst0.begin() + (size_t)(b + 1) * out_cols);
    std::sort(dst1.begin() + (size_t)b * out_cols,
              dst1.begin() + (size_t)(b + 1) * out_cols);
  }
  VK_EXPECT_MSG(std::memcmp(dst0.data(), dst1.data(), dst0.size() * 4) == 0,
                "probe: gate-off/on dst sets differ");

  const int iters = 200;
  auto t0 = std::chrono::steady_clock::now();
  setenv("VK_DSA_TOPK_FUSED", "0", 1);
  for (int i = 0; i < iters; ++i) { run(dlg0, ddst0); CK(hipDeviceSynchronize(), "s"); }
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) { run(dlg1, ddst1); CK(hipDeviceSynchronize(), "s"); }
  auto t2 = std::chrono::steady_clock::now();
  const long long chain_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;
  const long long fused_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / iters;
  std::printf("  GLM decode geometry (bs=%d H=%d D=%d B=%d mt=%d seq=%d): "
              "gate-off chain %lld us vs fused %lld us per iteration\n",
              bs, H, D, B, mt, max_seq_len, chain_us, fused_us);

  (void)hipFree(dq); (void)hipFree(dkv); (void)hipFree(dw);
  (void)hipFree(dsl); (void)hipFree(dpt); (void)hipFree(dlg0);
  (void)hipFree(dlg1); (void)hipFree(dlen); (void)hipFree(doff);
  (void)hipFree(drs); (void)hipFree(ddst0); (void)hipFree(ddst1);
}

#else  // !VKERNELS_HAS_HIP

// Host-only builds (no HIP toolchain, no CUDA shim): the device path is not
// compiled. Report the skip explicitly so the host CI run is honest.
TEST(DsaTopkFused, SkippedWithoutDevicePath) {
  std::printf(
      "  skip: VKERNELS_HAS_HIP not defined (no device dsa_topk.hip in this build)\n");
}

#endif  // VKERNELS_HAS_HIP
