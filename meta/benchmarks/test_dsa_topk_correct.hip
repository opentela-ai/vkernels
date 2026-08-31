// meta/benchmarks/test_dsa_topk_correct.hip
//
// End-to-end correctness test for the DSA paged-MQA gated top-k logits
// (issue #51) on gfx942, against the CPU reference in
// vkernels::kernels::dsa_topk_logits_cpu.
//
//   hip::dsa_topk_logits  vs  dsa_topk_logits_cpu (the gated-logit oracle)
//
// GLM-5.3-Flash indexer (index_n_heads=32, head_dim=128, block=64) at
// batch in {1, 2} and max_table_len in {8, 16} (max_seq_len in {512, 1024}),
// plus smaller sanity shapes. seq_lens are truncated on a couple of cases to
// exercise the t < seq_len write guard, and split_kv=2 on one case to verify
// the grid's perf-only slicing is actually grouping-independent.
//
// Method. The source Q and K are torch.float8_e4m3fnuz (raw uint8). The
// harness generates fp8 bytes directly, then BOTH paths dequant them via the
// SAME helper (fp8e4m3fnuz_to_f32 in moe_device.hip -- __host__ __device__,
// copied verbatim into the anonymous namespace below): the CPU oracle runs on
// the host-dequanted fp32, the device kernel dequants on load. With identical
// dequant, the only residual is FMA vs IEEE mul+add over the D-dot and H-sum
// (a few ULP), so we require max_rel < 1e-3.
//
// Invariant (the caller contract the kernel relies on): page_table[b,i] is
// VALID (in [0, num_blocks)) for every used slot i < ceildiv(seq_len, B).
// Unused slots (i >= ceildiv(seq_len, B)) are never read. The harness only
// generates valid pages in used slots; the CPU oracle ADDITIONALLY masks OOB
// pages (defensive, verified by tests/kernels/attn/test_dsa.cpp::
// DsaTopk::OutOfBoundPageIsUnwritten), but the kernel has no such guard -- it
// relies on sglang's invariant, so the harness must respect it.
//
// NOTE: the GLM-5.3 config (H=32, D=128, B=64) stages 49.5 KB < gfx942's
// 64 KB non-optin LDS cap (its opt-in ceiling EQUALS the non-optin cap -- no
// hipFuncSetAttribute opt-in past 64 KB, verified on a CSCS beverin node;
// see the KB note mi300a-dynamic-lds-no-optin), so the launcher's fp32-Q
// fast path runs it. H=64 (66 KB) and H=128 (99 KB) exceed the fp32-Q cap,
// so the launcher dispatches the fp8-Q variant (Q staged raw, dequantised on
// the fly -- bit-identical output, same cross-check below) and BOTH are
// exercised here alongside the H=32 headline (the index_n_heads GLM-5.3
// actually uses, which works where the tilelang path hangs).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/dsa.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

float rnd(int seed, int i) {  // deterministic pseudo-random in [-1, 1]
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

// fp8 e4m3fnuz byte in [0,254] excluding 0x7F (+NaN) and 0xFF (-NaN) --
// mirrors real KV cache data (no NaNs; 0x00 and 0x80 both decode to +0).
uint8_t rnd_fp8(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  uint8_t b = (uint8_t)(x % 255u);   // 0..254 (excludes 0xFF)
  if (b == 0x7Fu) b = 0u;            // exclude +NaN; 0x00 is +0 (fine)
  return b;
}

// fp8 e4m3fnuz -> fp32. VERBATIM copy of fp8e4m3fnuz_to_f32 in
// moe_device.hip (__host__ __device__); the harness dequants on the host,
// the device kernel dequants on load via its own copy. Both MUST agree --
// the kernel's docstring asserts this, and we cross-check it here.
__host__ __device__ __forceinline__ float fp8e4m3fnuz_to_f32(uint8_t b) {
  const uint32_t s = static_cast<uint32_t>(b >> 7) & 1u;     // sign
  const uint32_t e = static_cast<uint32_t>(b >> 3) & 0xFu;   // exponent (bias 8)
  const uint32_t m = static_cast<uint32_t>(b) & 0x7u;        // mantissa (3)
  if ((b & 0x7Fu) == 0u) return 0.0f;                        // +0 (0x00 AND 0x80)
  float f;
  if (e == 15u && m == 7u) {                                 // 0x7F = NaN -> qNaN
    const uint32_t qnan = 0x7fc00000u;
    __builtin_memcpy(&f, &qnan, sizeof(f));
  } else if (e == 0u) {                                      // subnormal: m*2^-7
    const float v = static_cast<float>(m) * 0x1p-7f;         // m*2^(1-8), exact
    f = s ? -v : v;
  } else {                                                   // normal: 2^(e-8)*(1+m/8)
    const uint32_t bits = (s << 31) | ((e + 119u) << 23) | (m << 20);
    __builtin_memcpy(&f, &bits, sizeof(f));
  }
  return f;
}

struct Stats { double max_abs, max_rel; };

void cmp_out(const std::vector<float>& got, const std::vector<float>& ref,
             Stats* s) {
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs((double)got[i] - ref[i]);
    if (e > s->max_abs) s->max_abs = e;
    double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    if (e / d > s->max_rel) s->max_rel = e / d;
  }
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  // Peak buffers across all cases (Hmax=128 covers the 4x indexer fp8-Q path).
  const int Bmax = 64, Hmax = 128, Dmax = 128;
  const int MTmax = 16, NBmax = 16, BSmax = 2;
  const int MSMmax = MTmax * Bmax;  // 1024
  uint8_t *dq, *dkv, *dout;
  int32_t *dsl, *dpt;
  float *dw;
  CK(hipMalloc(&dq, (size_t)BSmax * Hmax * Dmax), "alloc q");
  CK(hipMalloc(&dkv, (size_t)NBmax * Bmax * (Dmax + 4)), "alloc kv");
  CK(hipMalloc(&dw, (size_t)BSmax * Hmax * 4), "alloc weight");
  CK(hipMalloc(&dsl, (size_t)BSmax * 4), "alloc seq_lens");
  CK(hipMalloc(&dpt, (size_t)BSmax * MTmax * 4), "alloc page_table");
  CK(hipMalloc(&dout, (size_t)BSmax * MSMmax * 4), "alloc out");

  int fails = 0;
  const float OUT_THRESH = 1e-3f;

  struct Cfg { int bs, H, D, B, mt, nb, split_kv, variant; };
  const Cfg cfgs[] = {
      {1, 32, 128, 64, 8, 8, 1, 0},    // GLM-5.3 indexer decode (auto->mfma)
      {1, 32, 128, 64, 16, 16, 1, 0},  // more pages (max_seq_len=1024)
      {2, 32, 128, 64, 8, 8, 1, 0},    // batch > 1
      {1, 32, 128, 64, 8, 8, 2, 0},    // split_kv=2 (perf-only; must match)
      {1, 64, 128, 64, 8, 8, 1, 0},    // GLM-5.3 2x indexer (auto->mfma, 512)
      {1, 64, 128, 64, 16, 16, 1, 0},  // 2x indexer, more pages (1024)
      {2, 64, 128, 64, 8, 8, 1, 0},    // 2x indexer, batch > 1
      {1, 64, 128, 64, 8, 8, 2, 0},    // 2x indexer, split_kv=2 (must match)
      {1, 128, 128, 64, 8, 8, 1, 0},   // 4x indexer (auto->mfma, 41,728 B)
      {1, 4, 16, 8, 32, 32, 1, 0},     // smaller D, sanity (auto->fp32-Q)
      {1, 2, 8, 4, 16, 16, 1, 0},      // small (auto->fp32-Q)
      {1, 1, 2, 2, 4, 4, 1, 0},        // tiny hand-check (auto->fp32-Q)
      // --- FORCE each kernel at the GLM-5.3 shape (H=32 fits ALL three) ---
      // so every path stays exercised even if the auto dispatcher would pick
      // a different one. Cross-checked against the SAME oracle below.
      {1, 32, 128, 64, 8, 8, 1, 1},    // force fp32-Q
      {1, 32, 128, 64, 8, 8, 1, 2},    // force fp8-Q
      {1, 32, 128, 64, 8, 8, 1, 3},    // force mfma
  };

  int ci = 0;
  for (const auto& c : cfgs) {
    const int max_seq_len = c.mt * c.B;
    const size_t nq = (size_t)c.bs * c.H * c.D;
    const size_t nkvbytes = (size_t)c.nb * c.B * (c.D + 4);
    const size_t nw = (size_t)c.bs * c.H;
    const size_t nout = (size_t)c.bs * max_seq_len;

    std::vector<uint8_t> q_u8(nq), kv_u8(nkvbytes);
    std::vector<float> q_f32(nq), kv_f32((size_t)c.nb * c.B * c.D);
    std::vector<float> k_scale((size_t)c.nb * c.B);
    std::vector<float> weight(nw);
    std::vector<int32_t> sl(c.bs);
    std::vector<int32_t> pt((size_t)c.bs * c.mt);

    // Q: raw fp8 bytes + host dequant (both paths see the same value).
    for (size_t i = 0; i < nq; ++i) {
      q_u8[i] = rnd_fp8(1, (int)i);
      q_f32[i] = fp8e4m3fnuz_to_f32(q_u8[i]);
    }
    // k_scale: fp32 in [0.75, 1.25] (packed in the trailing B*4 bytes/block).
    for (size_t i = 0; i < k_scale.size(); ++i)
      k_scale[i] = static_cast<float>(rnd(3, (int)i)) * 0.25f + 1.0f;
    // kvcache_u8 [nb, B*(D+4)]: per block, B*D fp8 keys then B fp32 scales.
    const size_t kvstride = (size_t)c.B * (c.D + 4);
    for (int p = 0; p < c.nb; ++p) {
      uint8_t* bp = kv_u8.data() + (size_t)p * kvstride;
      for (int j = 0; j < c.B; ++j)
        for (int d = 0; d < c.D; ++d)
          bp[(size_t)j * c.D + d] = rnd_fp8(2 + p, j * c.D + d);
      std::memcpy(bp + (size_t)c.B * c.D, k_scale.data() + (size_t)p * c.B,
                  (size_t)c.B * sizeof(float));
    }
    // Host-dequanted keys for the CPU oracle ([nb, B, D] fp32).
    for (int p = 0; p < c.nb; ++p)
      for (int j = 0; j < c.B; ++j)
        for (int d = 0; d < c.D; ++d)
          kv_f32[((size_t)p * c.B + j) * c.D + d] =
              fp8e4m3fnuz_to_f32(kv_u8[(size_t)p * kvstride + (size_t)j * c.D + d]);
    // gate (weight) in [-1, 1].
    for (size_t i = 0; i < nw; ++i) weight[i] = rnd(4, (int)i);
    // seq_lens: mostly full; truncate the first batch of every third config
    // (ci 0,3,6) to exercise the t < seq_len write guard.
    for (int b = 0; b < c.bs; ++b)
      sl[b] = (ci % 3 == 0 && b == 0) ? max_seq_len * 3 / 4 : max_seq_len;
    // page_table: VALID pages in used slots (i < ceildiv(seq_len,B));
    // unused slots (i >= that) are never read -> fill 0.
    for (int b = 0; b < c.bs; ++b) {
      const int np_total = (sl[b] + c.B - 1) / c.B;
      for (int i = 0; i < c.mt; ++i)
        pt[(size_t)b * c.mt + i] =
            (i < np_total) ? (int32_t)(((unsigned)(b * 7 + i * 13)) %
                                        (unsigned)c.nb)
                           : 0;
    }

    // CPU oracle on the host-dequanted fp32 (output zeroed first).
    std::vector<float> ref(nout, 0.0f);
    vkernels::kernels::dsa_topk_logits_cpu(
        c.bs, c.H, c.D, c.B, c.mt, c.nb, q_f32.data(), kv_f32.data(),
        k_scale.data(), weight.data(), sl.data(), pt.data(), ref.data());

    // Device.
    CK(hipMemcpy(dq, q_u8.data(), nq, hipMemcpyHostToDevice), "cpyq");
    CK(hipMemcpy(dkv, kv_u8.data(), nkvbytes, hipMemcpyHostToDevice), "cpykv");
    CK(hipMemcpy(dw, weight.data(), nw * 4, hipMemcpyHostToDevice), "cpyw");
    CK(hipMemcpy(dsl, sl.data(), (size_t)c.bs * 4, hipMemcpyHostToDevice), "cpysl");
    CK(hipMemcpy(dpt, pt.data(), (size_t)c.bs * c.mt * 4, hipMemcpyHostToDevice), "cpypt");
    CK(hipMemset(dout, 0, nout * sizeof(float)), "zero out");
    vkernels::kernels::hip::dsa_topk_logits_with_variant(
        c.bs, c.H, c.D, c.B, c.mt, max_seq_len, c.split_kv, dq, dkv, dw, dsl,
        dpt, dout, c.variant);
    CK(hipDeviceSynchronize(), "sync");

    std::vector<float> got(nout);
    CK(hipMemcpy(got.data(), dout, nout * 4, hipMemcpyDeviceToHost), "rdout");

    Stats st{};
    cmp_out(got, ref, &st);
    bool ok = st.max_rel < OUT_THRESH;
    const char* vn = c.variant == 0 ? "auto" : c.variant == 1 ? "fp32"
                       : c.variant == 2 ? "fp8q" : "mfma";
    std::printf("  bs=%-2d H=%-2d D=%-3d B=%-2d mt=%-2d nb=%-2d split=%d v=%-4s "
                "out[a=%.4f r=%.6f]  %s\n",
                c.bs, c.H, c.D, c.B, c.mt, c.nb, c.split_kv, vn, st.max_abs,
                st.max_rel, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
    ++ci;
  }

  hipFree(dq); hipFree(dkv); hipFree(dw); hipFree(dsl); hipFree(dpt); hipFree(dout);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
