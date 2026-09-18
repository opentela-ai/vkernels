// meta/benchmarks/test_mhc_correct.hip
//
// End-to-end correctness test for the MHC multi-head hybrid-attention
// pre-norm kernels (issue #51, part 2) on gfx942, against the CPU references
// in vkernels::kernels::{mhc_pre_gemm_sqrsum_cpu, mhc_pre_gemm_sqrsum_cpu_f64,
// mhc_post_cpu}.
//
//   hip::mhc_pre_gemm_sqrsum  vs  mhc_pre_gemm_sqrsum_cpu      (fp32 oracle)
//                             vs  mhc_pre_gemm_sqrsum_cpu_f64  (fp64 ref, #138)
//   hip::mhc_pre_gemm_sqrsum_blocked (NSLICE ladder) vs both
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
// CPU oracles are fp32/fp64 end-to-end; the device kernels are bf16 storage /
// fp32 accumulation (the established gfx942 convention). To isolate kernel
// correctness from bf16 input quantization, the fp32 source (x for pre, b/d
// for post) is rounded to bf16 FIRST (host RNE), then the oracles run on the
// recovered fp32 and the device on the bf16 inputs. Only accumulation
// ordering and the single RNE on store (mhc_post) differ:
//   * pre_gemm_sqrsum: out is fp32 on both paths (no store rounding); the
//     GEMM accumulation is left-to-right identical, so out agrees to ~1e-6.
//     The per-token squared-sum uses a strided lane partial + warp shuffle
//     on the device vs a contiguous sum on the CPU, so sqrsum agrees to
//     ~1e-5 (associativity only).
//   * post: out is bf16 on the device vs fp32 on the CPU, so we tolerate
//     ~2e-2 (one bf16 ULP, as test_dsa_correct.hip / test_gemm_bf16 do).
//
// ---------------------------------------------------------------------------
// GATE POLICY (issue #138 — supersedes the report-only blocked ladder of #79)
// ---------------------------------------------------------------------------
// The legacy gate compared the device fp32 output against the fp32 oracle's
// STRICT sequential chain at PRE_THRESH = 1e-4. That gate is order-fragile:
// the fp32 chain (ANY order, including the oracle's own) deviates from the
// exact fp64 sum by up to ~4e-4 rel on the GLM shapes (MI300A-measured,
// 5 seeds — the deviation lives in heavily-cancelled outputs and is
// amplified rounding noise of the fp32 dot product itself, not a kernel
// bug), so PASS/FAIL verdicts are seed-dependent (issue #79: blocked
// ns=2/16/32 FAILED on the measurement seed; this survey: the STRICT split
// kernel itself FAILS 1e-4 on seed 2). Issue #138 replaces the DEFAULT gate
// with an fp64-REFERENCE comparison normalized by the sum's conditioning:
//
//   default gate:  for every out element   |got - ref64| <= K * eps32 * sum|x_h*fn_h|
//                  for every sqrsum        |got - ref64| <= K * eps32 * sum x_h^2
//                  (ref64 = fp64 CPU reference; eps32 = FLT_EPSILON;
//                   K = PRE_GATE_K, default pinned from the measured
//                   multi-seed MI300A envelope, VK_MHC_F64_K env override)
//
//   Normalizing by K*eps32*sum|terms| (the standard a-priori fp32
//   accumulation-error scale) makes the gate order-INVARIANT — every correct
//   accumulation order sits at the same envelope — while keeping detection:
//   a shifted/dropped term perturbs an output by O(|x*fn|), i.e. ~1/(K*eps)
//   normalized units, orders of magnitude above the envelope (measured:
//   corrupted-kernel f64 rel 5.7..79 vs clean <= 4e-4). It is also TIGHTER
//   than the legacy 1e-4 wherever the output is well-conditioned.
//
//   legacy gate:  VK_MHC_STRICT_GATE=1 restores the strict 1e-4
//                 oracle-chain comparison for every row (the #79 policy;
//                 kept available because it is the only gate under which the
//                 NSLICE=1 blocked variant is BITWISE-exact by construction).
//
// All statistics are printed for every row regardless of which gate decides.
// A deliberately corrupted blocked kernel (fn row off by one — a realistic
// index bug) is run as a negative control and MUST fail the active gate.

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

// fp64-reference comparison: the denominator uses the fp64 magnitude (>= 1),
// so the reported rel is directly comparable to the fp32-oracle rows.
void cmp_f64(const std::vector<float>& got, const std::vector<double>& ref,
             Stats* s) {
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs((double)got[i] - ref[i]);
    if (e > s->max_abs) s->max_abs = e;
    double d = std::fmax(std::fabs(ref[i]), 1.0);
    if (e / d > s->max_rel) s->max_rel = e / d;
  }
}

// Conditioning-normalized fp64 error (the issue #138 gate statistic):
//   max_i |got_i - ref64_i| / (eps32 * scale_i)
// with scale = sum|terms| (out) or sum x^2 (sqrsum) — the a-priori fp32
// accumulation-error bound up to the constant K. Zero-scale rows (all-zero
// inputs) require an exactly-zero error.
double cmp_f64_norm(const std::vector<float>& got,
                    const std::vector<double>& ref,
                    const std::vector<double>& scale) {
  const double eps32 = 1.1920929e-07;  // FLT_EPSILON
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs((double)got[i] - ref[i]);
    double d = eps32 * scale[i];
    double q = d > 0.0 ? e / d : (e > 0.0 ? 1e30 : 0.0);
    if (q > worst) worst = q;
  }
  return worst;
}

// --- mhc_pre_gemm_sqrsum ---------------------------------------------------
struct PreStats {
  Stats oracle_out{};  // device out vs fp32 sequential-chain oracle (legacy)
  Stats oracle_sq{};   // device sqrsum vs fp32 contiguous oracle (legacy)
  Stats f64_out{};     // device out vs fp64 reference (rel for the record)
  Stats f64_sq{};      // device sqrsum vs fp64 reference
  double norm_out{};   // GATE statistic: eps-normalized f64 error (out)
  double norm_sq{};    // GATE statistic: eps-normalized f64 error (sqrsum)
                       // measured envelope (MI300A, 5 seeds, job 641091):
                       // clean <= 0.9, corrupted >= 3.2e5, K = 256
};

PreStats run_pre(int num_tokens, int hc_mult, int hidden,
                 const std::vector<float>& x_src,   // fp32 source
                 const std::vector<float>& fn,      // fp32 source (no rounding)
                 uint16_t* dx, float* dfn, float* dout, float* dsqrsum,
                 int variant = 0, int nslice = 256) {
  const int hc_hidden = hc_mult * hidden;
  const int hc_mult3 = hc_mult * (2 + hc_mult);
  const size_t nx = (size_t)num_tokens * hc_hidden;
  const size_t nfn = (size_t)hc_mult3 * hc_hidden;
  const size_t nout = (size_t)num_tokens * hc_mult3;

  // Round x -> bf16, recover to fp32 (the value ALL paths see).
  std::vector<uint16_t> xb(nx);
  std::vector<float> xr(nx);
  for (size_t i = 0; i < nx; ++i) { xb[i] = f2bf(x_src[i]); xr[i] = bf2f(xb[i]); }

  // CPU oracles on the bf16-recovered x: fp32 sequential chain (legacy gate)
  // and fp64 (issue #138 gate).
  std::vector<float> ref(nout, 0.0f), rsq(num_tokens, 0.0f);
  vkernels::kernels::mhc_pre_gemm_sqrsum_cpu(num_tokens, hc_mult, hidden,
                                             xr.data(), fn.data(), ref.data(),
                                             rsq.data());
  std::vector<double> ref64(nout, 0.0), rsq64(num_tokens, 0.0);
  vkernels::kernels::mhc_pre_gemm_sqrsum_cpu_f64(num_tokens, hc_mult, hidden,
                                                 xr.data(), fn.data(),
                                                 ref64.data(), rsq64.data());

  // Conditioning scales for the normalized gate: sum|x*fn| per out element
  // (fp64, the same inputs the oracles see). The sqrsum scale (sum x^2 per
  // token) is computed in the second loop below.
  std::vector<double> s_out(nout, 0.0), s_sq(num_tokens, 0.0);
  for (int n = 0; n < num_tokens; ++n) {
    const float* xn = xr.data() + (size_t)n * hc_hidden;
    for (int o = 0; o < hc_mult3; ++o) {
      const float* fo = fn.data() + (size_t)o * hc_hidden;
      double sa = 0.0;
      for (int h = 0; h < hc_hidden; ++h)
        sa += std::fabs((double)xn[h] * (double)fo[h]);
      s_out[(size_t)n * hc_mult3 + o] = sa;
    }
  }
  // sqrsum scale: sum x^2 per token (computed cleanly here).
  for (int n = 0; n < num_tokens; ++n) {
    const float* xn = xr.data() + (size_t)n * hc_hidden;
    double sq = 0.0;
    for (int h = 0; h < hc_hidden; ++h) sq += (double)xn[h] * (double)xn[h];
    s_sq[n] = sq;
  }

  // Device.
  CK(hipMemcpy(dx, xb.data(), nx * 2, hipMemcpyHostToDevice), "cpyx");
  CK(hipMemcpy(dfn, fn.data(), nfn * 4, hipMemcpyHostToDevice), "cpyfn");
  if (variant == 0)
    vkernels::kernels::hip::mhc_pre_gemm_sqrsum(num_tokens, hc_mult3,
                                                hc_hidden, dx, dfn, dout,
                                                dsqrsum);
  else
    vkernels::kernels::hip::mhc_pre_gemm_sqrsum_blocked(num_tokens, hc_mult3,
                                                        hc_hidden, dx, dfn,
                                                        dout, dsqrsum, nullptr,
                                                        nslice);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<float> got(nout), gsq(num_tokens);
  CK(hipMemcpy(got.data(), dout, nout * 4, hipMemcpyDeviceToHost), "rdout");
  CK(hipMemcpy(gsq.data(), dsqrsum, num_tokens * 4,
               hipMemcpyDeviceToHost), "rdsq");

  PreStats s{};
  cmp(got, ref, &s.oracle_out);
  cmp(gsq, rsq, &s.oracle_sq);
  cmp_f64(got, ref64, &s.f64_out);
  cmp_f64(gsq, rsq64, &s.f64_sq);
  s.norm_out = cmp_f64_norm(got, ref64, s_out);
  s.norm_sq = cmp_f64_norm(gsq, rsq64, s_sq);
  return s;
}

__device__ float tst_bf16_to_f32(uint16_t h) {  // test-local bf16->f32
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

// --- corrupted blocked kernel (negative control, issue #138) ---------------
//
// A copy of the NSLICE=16 blocked kernel body with ONE realistic bug: the fn
// row pointer is off by one element (reads fn[o*hc_hidden + h + 1], clamped
// at the very end of the buffer) — the classic index-shift bug a gate must
// catch. sqrsum is computed correctly so ONLY the out path is corrupted; the
// fp64 gate must still FAIL the row on the f64_out statistic.
template <int NSLICE>
__global__ void mhc_pre_blocked_corrupt_kernel(
    const uint16_t* __restrict__ x, const float* __restrict__ fn,
    float* __restrict__ out, float* __restrict__ sqrsum,
    int num_tokens, int hc_mult3, int hc_hidden_size) {
  const int n = blockIdx.x;
  const int o = blockIdx.y;
  if (n >= num_tokens || o >= hc_mult3) return;
  const int tid = threadIdx.x;
  const uint16_t* x_row = x + (size_t)n * hc_hidden_size;
  const float* fn_row = fn + (size_t)o * hc_hidden_size + 1;  // BUG: +1 shift
  const size_t fn_total = (size_t)hc_mult3 * hc_hidden_size;

  __shared__ float sRed[256];
  __shared__ float sPart[256];

  if (o == 0) {  // correct sqrsum (corruption targets the out path only)
    float sq = 0.0f;
    for (int h = tid; h < hc_hidden_size; h += 256) {
      float xv = tst_bf16_to_f32(x_row[h]);
      sq += xv * xv;
    }
    sRed[tid] = sq;
    __syncthreads();
#pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
      if (tid < off) sRed[tid] += sRed[tid + off];
      __syncthreads();
    }
    if (tid == 0) sqrsum[n] = sRed[0];
  }

  const int slice = (hc_hidden_size + NSLICE - 1) / NSLICE;
  const int h0 = tid * slice;
  float part = 0.0f;
  if (tid < NSLICE && h0 < hc_hidden_size) {
    const int h1 = (h0 + slice < hc_hidden_size) ? h0 + slice
                                                 : hc_hidden_size;
#pragma unroll 4
    for (int h = h0; h < h1; ++h) {
      float fv = ((size_t)o * hc_hidden_size + h + 1 < fn_total)
                     ? fn_row[h] : 0.0f;  // clamp the one-element overrun
      part = fmaf(tst_bf16_to_f32(x_row[h]), fv, part);
    }
  }
  sPart[tid] = part;
  __syncthreads();
  if (tid == 0) {
    float acc = 0.0f;
#pragma unroll 8
    for (int t = 0; t < 256; ++t) acc += sPart[t];
    out[(size_t)n * hc_mult3 + o] = acc;
  }
}

// Negative control harness: the corrupted kernel at NSLICE=16 on the same
// inputs run_pre just used (same device buffers still hold x/fn).
PreStats run_pre_corrupt(int num_tokens, int hc_mult, int hidden,
                         uint16_t* dx, float* dfn, float* dout,
                         float* dsqrsum) {
  const int hc_hidden = hc_mult * hidden;
  const int hc_mult3 = hc_mult * (2 + hc_mult);
  const size_t nx = (size_t)num_tokens * hc_hidden;
  const size_t nout = (size_t)num_tokens * hc_mult3;

  // Rebuild the host-side references from the buffers run_pre left in place:
  // x on device is the bf16-rounded input; recover it (both oracles ran on
  // exactly this recovered fp32 in run_pre).
  std::vector<uint16_t> xb(nx);
  CK(hipMemcpy(xb.data(), dx, nx * 2, hipMemcpyDeviceToHost), "rdx");
  std::vector<float> xr(nx);
  for (size_t i = 0; i < nx; ++i) xr[i] = bf2f(xb[i]);
  std::vector<float> fn((size_t)hc_mult3 * hc_hidden);
  CK(hipMemcpy(fn.data(), dfn, fn.size() * 4, hipMemcpyDeviceToHost),
    "rdfn");

  std::vector<float> ref(nout, 0.0f), rsq(num_tokens, 0.0f);
  vkernels::kernels::mhc_pre_gemm_sqrsum_cpu(num_tokens, hc_mult, hidden,
                                             xr.data(), fn.data(), ref.data(),
                                             rsq.data());
  std::vector<double> ref64(nout, 0.0), rsq64(num_tokens, 0.0);
  vkernels::kernels::mhc_pre_gemm_sqrsum_cpu_f64(num_tokens, hc_mult, hidden,
                                                 xr.data(), fn.data(),
                                                 ref64.data(), rsq64.data());

  // Conditioning scales (same definition as run_pre).
  std::vector<double> s_out(nout, 0.0), s_sq(num_tokens, 0.0);
  for (int n = 0; n < num_tokens; ++n) {
    const float* xn = xr.data() + (size_t)n * hc_hidden;
    for (int o = 0; o < hc_mult3; ++o) {
      const float* fo = fn.data() + (size_t)o * hc_hidden;
      double sa = 0.0;
      for (int h = 0; h < hc_hidden; ++h)
        sa += std::fabs((double)xn[h] * (double)fo[h]);
      s_out[(size_t)n * hc_mult3 + o] = sa;
    }
    double sq = 0.0;
    for (int h = 0; h < hc_hidden; ++h) sq += (double)xn[h] * (double)xn[h];
    s_sq[n] = sq;
  }

  mhc_pre_blocked_corrupt_kernel<16><<<dim3(num_tokens, hc_mult3), 256>>>(
      reinterpret_cast<const uint16_t*>(dx), dfn, dout, dsqrsum,
      num_tokens, hc_mult3, hc_hidden);
  CK(hipDeviceSynchronize(), "sync-corrupt");

  std::vector<float> got(nout), gsq(num_tokens);
  CK(hipMemcpy(got.data(), dout, nout * 4, hipMemcpyDeviceToHost), "rdout");
  CK(hipMemcpy(gsq.data(), dsqrsum, num_tokens * 4,
               hipMemcpyDeviceToHost), "rdsq");

  PreStats s{};
  cmp(got, ref, &s.oracle_out);
  cmp(gsq, rsq, &s.oracle_sq);
  cmp_f64(got, ref64, &s.f64_out);
  cmp_f64(gsq, rsq64, &s.f64_sq);
  s.norm_out = cmp_f64_norm(got, ref64, s_out);
  s.norm_sq = cmp_f64_norm(gsq, rsq64, s_sq);
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

  // Gate policy (issue #138): default = conditioning-normalized fp64-reference
  // gate (K * eps32 * sum|terms|); VK_MHC_STRICT_GATE=1 = legacy 1e-4
  // oracle-chain gate (the #79 policy). K overridable via VK_MHC_F64_K.
  const float PRE_THRESH = 1e-4f;   // legacy oracle-chain gate (kept)
  double gate_k = 256.0;            // pinned from the measured MI300A
                                    // envelope (job 641091: worst clean
                                    // norm 0.9 over 5 seeds x NSLICE 1..256
                                    // x 3 shapes; corrupted control 3.2e5+)
  if (const char* e = std::getenv("VK_MHC_F64_K"))
    gate_k = std::strtod(e, nullptr);
  const char* strict_env = std::getenv("VK_MHC_STRICT_GATE");
  const bool strict_gate = strict_env && strict_env[0] == '1';
  std::printf(
      "\ngate policy: %s (K=%.0f%s)\n",
      strict_gate ? "LEGACY oracle-chain 1e-4 (VK_MHC_STRICT_GATE=1)"
                  : "fp64-reference normalized K*eps*sum|terms| (issue #138)",
      gate_k, getenv("VK_MHC_F64_K") ? " via env" : " default");
  const float POST_THRESH = 2e-2f;  // bf16 store + input rounding

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

  // --- mhc_pre_gemm_sqrsum ---
  std::printf("\nmhc_pre_gemm_sqrsum (vs CPU reference; oracle = fp32 chain,"
              " f64 = issue #138 reference):\n");
  struct PreCfg { int num_tokens, hc_mult, hidden; int nseeds; };
  const PreCfg pcfgs[] = {
      {1, 2, 2, 1},                 // tiny sanity
      {7, 3, 3, 1},
      {1, 4, 4096, 5},              // GLM-5.3: hc_hidden=16384, hc_mult3=24
      {7, 4, 4096, 5},              // GLM-5.3, multi-token
      {1, 4, 7168, 5},              // larger branch: hc_hidden=28672
  };
  for (const auto& c : pcfgs) {
    const int hc_hidden = c.hc_mult * c.hidden;
    const int hc_mult3 = c.hc_mult * (2 + c.hc_mult);
    for (int seed = 1; seed <= c.nseeds; ++seed) {
      // seed 1 reproduces the historical #79 measurement inputs exactly
      // (x = rnd(1, i), fn = rnd(2, i)); seeds 2..n decorrelate both.
      std::vector<float> x((size_t)c.num_tokens * hc_hidden);
      std::vector<float> fn((size_t)hc_mult3 * hc_hidden);
      for (size_t i = 0; i < x.size(); ++i) x[i] = rnd(2 * seed - 1, (int)i);
      for (size_t i = 0; i < fn.size(); ++i) fn[i] = rnd(2 * seed, (int)i);

      // Strict split (the shipped correctness path). Under the default
      // policy it is gated by the SAME normalized fp64 gate (it passes with
      // huge margin — and, unlike the legacy gate, on EVERY seed); the
      // legacy oracle-chain verdict is printed for the #79 record.
      auto s0 = run_pre(c.num_tokens, c.hc_mult, c.hidden, x, fn,
                        dx, dfn, dout, dsqrsum);
      bool ok0_legacy =
          std::fmax(s0.oracle_out.max_rel, s0.oracle_sq.max_rel) < PRE_THRESH;
      bool ok0 = strict_gate
                     ? ok0_legacy
                     : (s0.norm_out <= gate_k && s0.norm_sq <= gate_k);
      std::printf("  [seed %d] strict split n=%-2d hc_mult=%d hidden=%-4d"
                  "  oracle rel=%.2e (out %.2e / sq %.2e)"
                  "  f64 out=%.2e sq=%.2e  norm out=%.1f sq=%.1f  GATE %s"
                  "  [legacy 1e-4: %s]\n",
                  seed, c.num_tokens, c.hc_mult, c.hidden,
                  std::fmax(s0.oracle_out.max_rel, s0.oracle_sq.max_rel),
                  s0.oracle_out.max_rel, s0.oracle_sq.max_rel,
                  s0.f64_out.max_rel, s0.f64_sq.max_rel,
                  s0.norm_out, s0.norm_sq,
                  ok0 ? "PASS" : "FAIL", ok0_legacy ? "PASS" : "FAIL");
      if (!ok0) ++fails;

      // Blocked ladder (issue #138 gate): every NSLICE must pass the
      // order-invariant fp64 gate on EVERY seed; under the legacy gate the
      // same rows are printed for the seed-fragility record.
      std::printf("  [seed %d] blocked ladder n=%-2d hc_mult=%d hidden=%-4d:"
                  "  (gate = %s)\n", seed, c.num_tokens, c.hc_mult, c.hidden,
                  strict_gate ? "legacy 1e-4" : "fp64");
      for (int ns : {256, 128, 64, 32, 16, 8, 4, 2, 1}) {
        auto sb = run_pre(c.num_tokens, c.hc_mult, c.hidden, x, fn,
                          dx, dfn, dout, dsqrsum, 1, ns);
        bool ok_legacy =
            std::fmax(sb.oracle_out.max_rel, sb.oracle_sq.max_rel)
                < PRE_THRESH;
        bool ok = strict_gate
                      ? ok_legacy
                      : (sb.norm_out <= gate_k && sb.norm_sq <= gate_k);
        std::printf("    ns=%-3d oracle rel=%.3e (out %.2e / sq %.2e, %s)"
                    "  f64 out=%.3e sq=%.3e  norm out=%.1f sq=%.1f"
                    "  GATE %s\n",
                    ns, std::fmax(sb.oracle_out.max_rel, sb.oracle_sq.max_rel),
                    sb.oracle_out.max_rel, sb.oracle_sq.max_rel,
                    ok_legacy ? "P1e-4" : "F1e-4",
                    sb.f64_out.max_rel, sb.f64_sq.max_rel,
                    sb.norm_out, sb.norm_sq,
                    ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
      }

      // Negative control (GLM decode shape only): the corrupted kernel MUST
      // fail the active gate. Counted as a failure if it slips through.
      if (c.num_tokens == 1 && c.hc_mult == 4 && c.hidden == 4096) {
        auto sc = run_pre_corrupt(c.num_tokens, c.hc_mult, c.hidden,
                                  dx, dfn, dout, dsqrsum);
        bool ok = strict_gate
                      ? std::fmax(sc.oracle_out.max_rel, sc.oracle_sq.max_rel)
                            < PRE_THRESH
                      : (sc.norm_out <= gate_k && sc.norm_sq <= gate_k);
        std::printf("  [seed %d] NEGATIVE CONTROL (fn row +1) n=1 hc_mult=4"
                    " hidden=4096:  oracle rel=%.3e  f64 out=%.3e"
                    "  norm out=%.1f  GATE %s %s\n", seed,
                    std::fmax(sc.oracle_out.max_rel, sc.oracle_sq.max_rel),
                    sc.f64_out.max_rel, sc.norm_out,
                    ok ? "PASS" : "FAIL",
                    ok ? "(<-- UNEXPECTED: gate missed a corrupted kernel!)"
                       : "(expected: gate catches the corruption)");
        if (ok) ++fails;
      }
    }
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
