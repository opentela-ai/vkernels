// meta/benchmarks/diag_mfma_issue148.hip
//
// Issue #148 verdict tool: glm_fp8_fnuz_mfma_gemv_kernel still diverges
// from the CPU oracle on multi-segment shapes (max_abs ~2e3). Isolate by
// recomputing the expected output FROM THE KERNEL'S OWN dumped payload
// (dbg_sxa / dbg_s2a hooks) and comparing per (n, segment):
//   out_kernel  - the mfma kernel's bf16 output
//   out_pay     - CPU recompute from the dumped payload bytes
//   out_man     - the fnuz gemv reference (passes 24/24)
// If out_pay == out_kernel but != out_man, the math mapping is right and
// the payload/amax is wrong; if out_pay != out_kernel, the mfma mapping
// or its scale application is wrong. Prints per-segment contributions for
// the first mismatched row.

#include "vkernels/kernels/glm_moe.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>

#define CK(e, m)                                                                                   \
  do {                                                                                             \
    hipError_t r = (e);                                                                            \
    if (r != hipSuccess) {                                                                         \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));                                   \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

namespace {

float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f; // [-1, 1)
}

uint16_t f32_to_bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

float bf16_bits_to_f32(uint16_t u) {
  uint32_t v = (uint32_t)u << 16;
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}

// LIFT of glm_fnuz_e4m3_to_f32_cpu (bias 8, no-NaN, subnormals m*2^-10)
float fnuz_decode(uint8_t v) {
  if (v == 0x80u) {
    return std::nanf("");
  }
  const uint32_t sign = (uint32_t)(v >> 7) & 1u;
  const uint32_t exp = (uint32_t)(v >> 3) & 15u;
  const uint32_t man = (uint32_t)v & 7u;
  float value;
  if (exp == 0) {
    value = (float)man * (1.0f / 1024.0f);
  } else {
    value = (float)(8u + man);
    const int e = (int)exp - 11;
    value = e >= 0 ? value * (float)(1ull << e) : value / (float)(1ull << (uint32_t)(-e));
  }
  return sign ? -value : value;
}

} // namespace

namespace {

__global__ void cvt_probe_kernel(uint32_t* o) {
  const float v0 = 64.f, v1 = 128.f, v2 = 192.f, v3 = 224.f;
  // the diag's quantize pattern: two calls, second appends via old+true
  uint32_t u = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, 0u, false);
  u = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, u, true);
  o[0] = u;
  // each pair alone (packs into the LOW half)
  o[1] = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, 0u, false);
  o[2] = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, 0u, false);
  // append with a nonzero old value (append-position probe)
  o[3] = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, 0x00FFFFFFu, true);
  // swapped arg order
  o[4] = __builtin_amdgcn_cvt_pk_fp8_f32(v1, v0, 0u, false);
  // rounding-mode probe: 2.375 ties between 2.25 (m=1, 0x49) and 2.5
  // (m=2, 0x4a): RNE -> 2.5, RTZ -> 2.25. 2.125 ties 2.0/2.25: RNE->2.0.
  // rounding-mode probe below (written after decode probes)
  // signed encode through the same two-call pattern
  uint32_t su = __builtin_amdgcn_cvt_pk_fp8_f32(-64.f, -128.f, 0u, false);
  su = __builtin_amdgcn_cvt_pk_fp8_f32(-192.f, -224.f, su, true);
  o[5] = su;
  // DECODE the encoded halves back: 0x7870 should be (64,128) and
  // 0x7e7c (192,224) if encode/decode agree; raw f32 bits returned
  const auto d0 = __builtin_amdgcn_cvt_pk_f32_fp8((uint16_t)0x7870u, 0);
  const auto d1 = __builtin_amdgcn_cvt_pk_f32_fp8((uint16_t)0x7e7cu, 0);
  const uint32_t a0 = __builtin_bit_cast(uint32_t, d0[0]);
  const uint32_t a1 = __builtin_bit_cast(uint32_t, d0[1]);
  const uint32_t b0 = __builtin_bit_cast(uint32_t, d1[0]);
  const uint32_t b1 = __builtin_bit_cast(uint32_t, d1[1]);
  o[6] = a0 | (a1 << 16);
  o[7] = b0 | (b1 << 16);
  // rounding-mode probe: 2.375 ties between 2.25 (m=1, 0x49) and 2.5
  // (m=2, 0x4a): RNE -> 2.5, RTZ -> 2.25. 2.125 ties 2.0/2.25: RNE->2.0.
  o[8] = __builtin_amdgcn_cvt_pk_fp8_f32(2.375f, 2.125f, 0u, false);
}

} // namespace
int main() {
  int dev;
  CK(hipGetDevice(&dev), "dev");
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, dev), "prop");
  std::printf("GPU: %s\n", p.name);

  // The failing multi-segment shape class: M=1, N=128, K=2048, sk=1
  // (segs = 16, slice = K so the dumped payload covers all k in one
  // block). Same recipe as the test's mfma section.
  const int M = 1, N = 128, K = 2048, SK = 1;
  const int segs = (K / 128) / SK; // 8
  const int slice = segs * 128;    // 1024
  const size_t nx = (size_t)M * K, nw = (size_t)N * K;
  const size_t ns = (size_t)(N / 128) * (K / 128);

  std::vector<float> xf(nx);
  for (size_t i = 0; i < nx; ++i)
    xf[i] = rnd(1, (int)i);
  std::vector<uint16_t> x(nx);
  for (size_t i = 0; i < nx; ++i)
    x[i] = f32_to_bf16_bits(xf[i]);
  std::vector<uint8_t> w(nw);
  for (size_t i = 0; i < nw; ++i) {
    w[i] = (uint8_t)(rnd(2, (int)i) * 127.0f + 128.0f);
    if (w[i] == 0x80u) w[i] = 0x00u;
  }
  std::vector<float> scales(ns);
  for (size_t i = 0; i < ns; ++i)
    scales[i] = 0.0625f * (float)(1 + (i % 128));

  uint16_t *dx, *dout, *doutm;
  uint8_t* dw;
  float *ds, *dbg_s2a, *part_m;
  uint32_t* dbg_sxa;
  CK(hipMalloc(&dx, nx * 2), "ax");
  CK(hipMalloc(&dw, nw), "aw");
  CK(hipMalloc(&ds, ns * 4), "as");
  CK(hipMalloc(&dout, M * N * 2), "ao");
  CK(hipMalloc(&doutm, M * N * 2), "aom");
  CK(hipMalloc(&dbg_s2a, (size_t)M * segs * 4), "ad2");
  CK(hipMalloc(&dbg_sxa, (size_t)M * slice), "adw");
  CK(hipMalloc(&part_m, (size_t)M * N * SK * 4), "apm");
  CK(hipMemcpy(dx, x.data(), nx * 2, hipMemcpyHostToDevice), "cx");
  CK(hipMemcpy(dw, w.data(), nw, hipMemcpyHostToDevice), "cw");
  CK(hipMemcpy(ds, scales.data(), ns * 4, hipMemcpyHostToDevice), "cs");

  // 1) the true reference: fnuz gemv (passes 24/24)
  vkernels::kernels::hip::glm_fp8_fnuz_block_gemv_with_scratch(
      dx, dw, ds, doutm, part_m, M, N, K, SK);
  CK(hipDeviceSynchronize(), "sync1");
  std::vector<uint16_t> out_man(M * N);
  CK(hipMemcpy(out_man.data(), doutm, out_man.size() * 2, hipMemcpyDeviceToHost), "cm");

  // 2) the mfma kernel with dbg hooks
  vkernels::kernels::hip::glm_fp8_fnuz_mfma_block_gemv_debug(
      dx, dw, ds, dout, part_m, M, N, K, SK, dbg_sxa, dbg_s2a);
  CK(hipDeviceSynchronize(), "sync2");
  std::vector<uint16_t> out_k(M * N);
  CK(hipMemcpy(out_k.data(), dout, out_k.size() * 2, hipMemcpyDeviceToHost), "co");
  std::vector<float> s2v((size_t)M * segs);
  CK(hipMemcpy(s2v.data(), dbg_s2a, s2v.size() * 4, hipMemcpyDeviceToHost), "c2");
  std::vector<uint32_t> sxa((size_t)M * slice / 4);
  CK(hipMemcpy(sxa.data(), dbg_sxa, sxa.size() * 4, hipMemcpyDeviceToHost), "cw2");

  // dumped ground truth
  std::printf("s2a dumped:");
  for (int g = 0; g < M * segs; ++g)
    std::printf(" %g", s2v[g]);
  std::printf("\n");
  const auto pay = [&](int m, int k) {
    const size_t wd = (size_t)m * (slice / 4) + k / 4;
    return (uint8_t)(sxa[wd] >> (8 * (k % 4)) & 0xFFu);
  };
  std::printf("payload[0:15]:");
  for (int k = 0; k < 15; ++k)
    std::printf(" %02x", (unsigned)pay(0, k));
  std::printf("\nx[0:15]:");
  for (int k = 0; k < 15; ++k)
    std::printf(" %g", bf16_bits_to_f32(x[k]));
  std::printf("\n");

  // 3) per-segment component comparison (fp32, before reduce)
  // conv: fnuz payload domain = (1 + m/8) * 2^(e-8) = value decoded by
  // fnuz_decode; both that doubled (x2) and plain domains are tried.
  for (const char* conv : {"plain", "x2"}) {
    const double mul = (std::strcmp(conv, "x2") == 0) ? 2.0 : 1.0;
    double worst = 0.0;
    int worst_n = -1;
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int y = 0; y < SK; ++y) {
        for (int s = 0; s < segs; ++s) {
          const int seg = y * segs + s; // global 128-k segment
          const float s2w = scales[(size_t)(n >> 7) * (K >> 7) + seg];
          const float s2a = s2v[s]; // per (token, seg)
          for (int kb = 0; kb < 128; ++kb) {
            const int k = y * slice + s * 128 + kb;
            acc += (double)fnuz_decode(pay(0, k)) * mul *
                   (double)fnuz_decode(w[(size_t)n * K + k]) * (double)s2w * (double)s2v[s];
          }
        }
      }
      const double got = bf16_bits_to_f32(out_k[n]);
      const double d = std::fabs(got - acc);
      if (d > worst) {
        worst = d;
        worst_n = n;
      }
      if (n < 4 && mul == 1.0) std::printf("  n=%d got=%g acc=%g diff=%.6f\n", n, got, acc, d);
    }
    std::printf("conv=%-5s worst |out_pay - out_kernel| = %.6f at n=%d\n", conv, worst, worst_n);
    if (worst < 2.0) {
      std::printf("  -> payload recompute MATCHES the kernel "
                  "(mapping correct, payload is the story)\n");
    } else {
      std::printf("  -> payload recompute MISMATCHES the kernel "
                  "(mfma mapping/scale application is also broken)\n");
      const int n = worst_n;
      std::printf("  n=%d segments: kernel-out=%g\n", n, bf16_bits_to_f32(out_k[n]));
      for (int y = 0; y < SK; ++y)
        for (int s = 0; s < segs; ++s) {
          const int seg = y * segs + s;
          double a = 0.0;
          for (int kb = 0; kb < 128; ++kb) {
            const int k = y * slice + s * 128 + kb;
            a += (double)fnuz_decode(pay(0, k)) * mul * (double)fnuz_decode(w[(size_t)n * K + k]) *
                 (double)scales[(size_t)(n >> 7) * (K >> 7) + seg] * (double)s2v[s];
          }
          std::printf("    seg=%d s2a=%g s2w=%g pay-contrib=%.6f\n",
                      seg,
                      s2v[s],
                      scales[(size_t)(n >> 7) * (K >> 7) + seg],
                      a);
        }
    }
  }

  // 4) overall: kernel vs manual reference
  double ma = 0.0, mr = 0.0;
  for (int i = 0; i < M * N; ++i) {
    const double a = bf16_bits_to_f32(out_k[i]), b = bf16_bits_to_f32(out_man[i]);
    const double e = std::fabs(a - b), d = std::fmax(std::fabs(b), 1.0);
    if (e > ma) ma = e;
    if (e / d > mr) mr = e / d;
  }
  std::printf("kernel vs manual: max_abs=%.4f max_rel=%.6f\n", ma, mr);

  // 5) cvt_pk_fp8_f32 packing probe: expect bytes for (64,128,192,224)
  // = 0x70,0x78,0x7c,0x7e
  uint32_t* dp;
  CK(hipMalloc(&dp, 9 * 4), "aprobe");
  cvt_probe_kernel<<<1, 1>>>(dp);
  CK(hipDeviceSynchronize(), "probe sync");
  std::vector<uint32_t> pv(9);
  CK(hipMemcpy(pv.data(), dp, 36, hipMemcpyDeviceToHost), "cprobe");
  const char* tags[9] = {"two-call pattern",
                         "v0,v1 alone",
                         "v2,v3 alone",
                         "append nonzero old",
                         "v1,v0 alone",
                         "signed enc",
                         "decom u_lo",
                         "decom u_hi",
                         "tie 2.375/2.125"};
  for (int i = 0; i < 9; ++i)
    std::printf("cvt-probe %-18s = %08x (bytes %02x %02x %02x %02x)\n",
                tags[i],
                pv[i],
                pv[i] & 0xFFu,
                (pv[i] >> 8) & 0xFFu,
                (pv[i] >> 16) & 0xFFu,
                (pv[i] >> 24) & 0xFFu);
  return 0;
}
