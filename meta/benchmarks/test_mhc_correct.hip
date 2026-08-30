// meta/benchmarks/test_mhc_correct.hip
//
// End-to-end correctness test for the MHC multi-head hybrid-attention
// pre-norm kernels (issue #51, part 2) on gfx942, against the CPU references
// in vkernels::kernels::{mhc_pre_gemm_sqrsum_cpu, mhc_post_cpu}.
//
//   hip::mhc_pre_gemm_sqrsum  vs  mhc_pre_gemm_sqrsum_cpu
//   hip::mhc_post             vs  mhc_post_cpu
//
// Shapes: the GLM-5.3-Flash family the issue calls out —
//   * pre_gemm_sqrsum: hc_mult=4, hidden=4096 -> hc_hidden_size=16384,
//     hc_mult3=24 (the tilelang split-k stage-0 shape that aborts JIT with
//     "98304 exceeds device limit 65536"); also hidden=7168 ->
//     hc_hidden_size=28672 (the larger sglang branch), and tiny sanity.
//   * post: hc=4, hidden=4096 and hidden=7168 (GLM), plus hc=2/8 shapes.
//
// Method (mirrors test_dsa_correct.hip / test_gemm_bf16_correct.hip). The
// CPU oracles are fp32 end-to-end; the device kernels are bf16 storage /
// fp32 accumulation (the established gfx942 convention). To isolate kernel
// correctness from bf16 input quantization, the fp32 source (x for pre, b/d
// for post) is rounded to bf16 FIRST (host RNE), then the CPU oracle runs on
// the recovered fp32 and the device on the bf16 inputs. Only accumulation
// ordering and the single RNE on store (mhc_post) differ:
//   * pre_gemm_sqrsum: out is fp32 on both paths (no store rounding); the
//     GEMM accumulation is left-to-right identical, so out agrees to ~1e-6.
//     The per-token squared-sum uses a strided lane partial + warp shuffle
//     on the device vs a contiguous sum on the CPU, so sqrsum agrees to
//     ~1e-5 (associativity only).
//   * post: out is bf16 on the device vs fp32 on the CPU, so we tolerate
//     ~2e-2 (one bf16 ULP, as test_dsa_correct.hip / test_gemm_bf16 do).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/mhc.hpp"

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

// Host bf16 (uint16 IEEE bit patterns) <-> fp32, RNE — identical to the
// helpers in gemm_bf16.cpp and to f32bits_to_bf16 in moe_device.hip.
uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return (uint16_t)(b >> 16);
}
float bf2f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

struct Stats { double max_abs, max_rel; };

void cmp(const std::vector<float>& got, const std::vector<float>& ref,
         Stats* s) {
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs((double)got[i] - ref[i]);
    if (e > s->max_abs) s->max_abs = e;
    double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    if (e / d > s->max_rel) s->max_rel = e / d;
  }
}

// --- mhc_pre_gemm_sqrsum ---------------------------------------------------
Stats run_pre(int num_tokens, int hc_mult, int hidden,
              const std::vector<float>& x_src,   // fp32 source
              const std::vector<float>& fn,      // fp32 source (no rounding)
              uint16_t* dx, float* dfn, float* dout, float* dsqrsum) {
  const int hc_hidden = hc_mult * hidden;
  const int hc_mult3 = hc_mult * (2 + hc_mult);
  const size_t nx = (size_t)num_tokens * hc_hidden;
  const size_t nfn = (size_t)hc_mult3 * hc_hidden;
  const size_t nout = (size_t)num_tokens * hc_mult3;

  // Round x -> bf16, recover to fp32 (the value BOTH paths see).
  std::vector<uint16_t> xb(nx);
  std::vector<float> xr(nx);
  for (size_t i = 0; i < nx; ++i) { xb[i] = f2bf(x_src[i]); xr[i] = bf2f(xb[i]); }

  // CPU oracle on the bf16-recovered x, fp32 fn.
  std::vector<float> ref(nout, 0.0f), rsq(num_tokens, 0.0f);
  vkernels::kernels::mhc_pre_gemm_sqrsum_cpu(num_tokens, hc_mult, hidden,
                                             xr.data(), fn.data(), ref.data(),
                                             rsq.data());

  // Device.
  CK(hipMemcpy(dx, xb.data(), nx * 2, hipMemcpyHostToDevice), "cpyx");
  CK(hipMemcpy(dfn, fn.data(), nfn * 4, hipMemcpyHostToDevice), "cpyfn");
  vkernels::kernels::hip::mhc_pre_gemm_sqrsum(num_tokens, hc_mult3, hc_hidden,
                                              dx, dfn, dout, dsqrsum);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<float> got(nout), gsq(num_tokens);
  CK(hipMemcpy(got.data(), dout, nout * 4, hipMemcpyDeviceToHost), "rdout");
  CK(hipMemcpy(gsq.data(), dsqrsum, num_tokens * 4,
               hipMemcpyDeviceToHost), "rdsq");

  Stats s{}, sq{};
  cmp(got, ref, &s);
  cmp(gsq, rsq, &sq);
  // Report the worse of out/sqrsum.
  if (sq.max_abs > s.max_abs) s.max_abs = sq.max_abs;
  if (sq.max_rel > s.max_rel) s.max_rel = sq.max_rel;
  return s;
}

// --- mhc_post --------------------------------------------------------------
Stats run_post(int num_tokens, int hc, int hidden,
               const std::vector<float>& a_src,   // fp32 source (no rounding)
               const std::vector<float>& b_src,   // fp32 source
               const std::vector<float>& c_src,   // fp32 source (no rounding)
               const std::vector<float>& d_src,   // fp32 source
               float* da, uint16_t* db, float* dc, uint16_t* dd,
               uint16_t* dout) {
  const size_t na = (size_t)num_tokens * hc * hc;
  const size_t nb = (size_t)num_tokens * hc * hidden;
  const size_t nc = (size_t)num_tokens * hc;
  const size_t nd = (size_t)num_tokens * hidden;
  const size_t nout = (size_t)num_tokens * hc * hidden;

  // Round b, d -> bf16, recover to fp32 (the value BOTH paths see).
  // a, c are fp32 on both paths (no rounding).
  std::vector<uint16_t> bb(nb), dd_(nd);
  std::vector<float> br(nb), dr(nd);
  for (size_t i = 0; i < nb; ++i) { bb[i] = f2bf(b_src[i]); br[i] = bf2f(bb[i]); }
  for (size_t i = 0; i < nd; ++i) { dd_[i] = f2bf(d_src[i]); dr[i] = bf2f(dd_[i]); }

  std::vector<float> ref(nout, 0.0f);
  vkernels::kernels::mhc_post_cpu(num_tokens, hc, hidden, a_src.data(),
                                  br.data(), c_src.data(), dr.data(),
                                  ref.data());

  CK(hipMemcpy(da, a_src.data(), na * 4, hipMemcpyHostToDevice), "cpya");
  CK(hipMemcpy(db, bb.data(), nb * 2, hipMemcpyHostToDevice), "cpyb");
  CK(hipMemcpy(dc, c_src.data(), nc * 4, hipMemcpyHostToDevice), "cpyc");
  CK(hipMemcpy(dd, dd_.data(), nd * 2, hipMemcpyHostToDevice), "cpyd");
  vkernels::kernels::hip::mhc_post(num_tokens, hc, hidden, da, db, dc, dd,
                                   dout);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<uint16_t> hout(nout);
  CK(hipMemcpy(hout.data(), dout, nout * 2, hipMemcpyDeviceToHost), "rdout");
  std::vector<float> got(nout);
  for (size_t i = 0; i < nout; ++i) got[i] = bf2f(hout[i]);

  Stats s{};
  cmp(got, ref, &s);
  return s;
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  // Peak buffers across all cases.
  const int Tnmax = 64, Hcmax = 28672, Hcmult3max = 32;  // pre
  const int Hcmax2 = 8, Hiddenmax = 7168;                // post
  uint16_t *dx, *db, *dd, *doutp;
  float *dfn, *da, *dc, *dout, *dsqrsum;
  CK(hipMalloc(&dx, (size_t)Tnmax * Hcmax * 2), "alloc x");
  CK(hipMalloc(&dfn, (size_t)Hcmult3max * Hcmax * 4), "alloc fn");
  CK(hipMalloc(&dout, (size_t)Tnmax * Hcmult3max * 4), "alloc out(pre)");
  CK(hipMalloc(&dsqrsum, (size_t)Tnmax * 4), "alloc sqrsum");
  CK(hipMalloc(&da, (size_t)Tnmax * Hcmax2 * Hcmax2 * 4), "alloc a");
  CK(hipMalloc(&db, (size_t)Tnmax * Hcmax2 * Hiddenmax * 2), "alloc b");
  CK(hipMalloc(&dc, (size_t)Tnmax * Hcmax2 * 4), "alloc c");
  CK(hipMalloc(&dd, (size_t)Tnmax * Hiddenmax * 2), "alloc d");
  CK(hipMalloc(&doutp, (size_t)Tnmax * Hcmax2 * Hiddenmax * 2), "alloc out(post)");

  int fails = 0;
  const float PRE_THRESH = 1e-4f;   // fp32 both paths; associativity only
  const float POST_THRESH = 2e-2f;  // bf16 store + input rounding

  // --- mhc_pre_gemm_sqrsum ---
  std::printf("\nmhc_pre_gemm_sqrsum (vs CPU reference, fp32 both paths):\n");
  struct PreCfg { int num_tokens, hc_mult, hidden; };
  const PreCfg pcfgs[] = {
      {1, 2, 2},                    // tiny sanity
      {7, 3, 3},
      {1, 4, 4096},                 // GLM-5.3: hc_hidden=16384, hc_mult3=24
      {7, 4, 4096},                 // GLM-5.3, multi-token
      {1, 4, 7168},                 // larger branch: hc_hidden=28672
  };
  for (const auto& c : pcfgs) {
    const int hc_hidden = c.hc_mult * c.hidden;
    const int hc_mult3 = c.hc_mult * (2 + c.hc_mult);
    std::vector<float> x((size_t)c.num_tokens * hc_hidden);
    std::vector<float> fn((size_t)hc_mult3 * hc_hidden);
    for (size_t i = 0; i < x.size(); ++i) x[i] = rnd(1, (int)i);
    for (size_t i = 0; i < fn.size(); ++i) fn[i] = rnd(2, (int)i);
    auto s = run_pre(c.num_tokens, c.hc_mult, c.hidden, x, fn,
                     dx, dfn, dout, dsqrsum);
    bool ok = s.max_rel < PRE_THRESH;
    std::printf("  n=%-2d hc_mult=%d hidden=%-4d (hc_hidden=%-5d hc_mult3=%-2d)"
                "  abs=%.4e rel=%.4e  %s\n",
                c.num_tokens, c.hc_mult, c.hidden, hc_hidden, hc_mult3,
                s.max_abs, s.max_rel, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  // --- mhc_post ---
  std::printf("\nmhc_post (vs CPU reference, bf16 store):\n");
  struct PostCfg { int num_tokens, hc, hidden; };
  const PostCfg qcfgs[] = {
      {1, 2, 2},                    // tiny sanity
      {7, 4, 5},
      {1, 4, 4096},                 // GLM-5.3
      {7, 4, 4096},                 // GLM-5.3, multi-token
      {1, 4, 7168},                 // larger branch
      {4, 8, 1024},                 // wider hc
  };
  for (const auto& c : qcfgs) {
    std::vector<float> a((size_t)c.num_tokens * c.hc * c.hc);
    std::vector<float> b((size_t)c.num_tokens * c.hc * c.hidden);
    std::vector<float> cc((size_t)c.num_tokens * c.hc);
    std::vector<float> d((size_t)c.num_tokens * c.hidden);
    for (size_t i = 0; i < a.size(); ++i) a[i] = rnd(3, (int)i);
    for (size_t i = 0; i < b.size(); ++i) b[i] = rnd(4, (int)i);
    for (size_t i = 0; i < cc.size(); ++i) cc[i] = rnd(5, (int)i);
    for (size_t i = 0; i < d.size(); ++i) d[i] = rnd(6, (int)i);
    auto s = run_post(c.num_tokens, c.hc, c.hidden, a, b, cc, d,
                      da, db, dc, dd, doutp);
    bool ok = s.max_rel < POST_THRESH;
    std::printf("  n=%-2d hc=%d hidden=%-4d  abs=%.4e rel=%.4e  %s\n",
                c.num_tokens, c.hc, c.hidden, s.max_abs, s.max_rel,
                ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  hipFree(dx); hipFree(dfn); hipFree(dout); hipFree(dsqrsum);
  hipFree(da); hipFree(db); hipFree(dc); hipFree(dd); hipFree(doutp);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
