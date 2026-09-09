// meta/benchmarks/test_glm_fp8_gemv.hip
//
// Correctness test for the GLM block-FP8 decode-fused GEMV (issue #64) on
// gfx942: hip::glm_fp8_block_gemv vs glm_fp8_block_gemv_cpu (the oracle),
// at the GLM decode expert shapes and small edge shapes.
//
// Convention (test_kda_correct.hip): max_rel < 2e-2, PASS/FAIL summary on
// the last line. Weight codes are random bytes with the NaN encodings
// (0x7F/0xFF) replaced so both sides stay finite; the NaN decode itself
// is checked separately for CPU/GPU bit agreement via the shared code
// path (both use exp==15&&man==7 -> canonical NaN).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/glm_moe.hpp"

#define CK(e, m)                                                          \
  do {                                                                    \
    hipError_t r = (e);                                                   \
    if (r != hipSuccess) {                                                \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));          \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

namespace {

float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f;  // [-1, 1)
}

uint16_t f32_to_bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  int fails = 0, passes = 0;
  const float THRESH = 2e-2f;

  struct Cfg { int M, N, K; };
  const Cfg cfgs[] = {
      {1, 128, 128},    {2, 128, 128},    // minimal blocks
      {1, 256, 1152},   // K not a power of two (9 scale segments)
      {1, 4096, 4096},  // gate/up  [E,4096,4096]
      {2, 4096, 4096},
      {1, 4096, 2048},  // down    [E,4096,2048]
      {2, 4096, 2048},
      {1, 2048, 4096},
  };
  std::printf("--- glm_fp8_block_gemv vs CPU oracle ---\n");
  for (const auto& c : cfgs) {
    const size_t nx = (size_t)c.M * c.K;
    const size_t nw = (size_t)c.N * c.K;
    const size_t ns = (size_t)(c.N / 128) * (c.K / 128);
    std::vector<float> xf(nx);
    for (size_t i = 0; i < nx; ++i) xf[i] = rnd(1, (int)i);
    std::vector<uint16_t> x(nx);
    for (size_t i = 0; i < nx; ++i) x[i] = f32_to_bf16_bits(xf[i]);
    std::vector<uint8_t> w(nw);
    for (size_t i = 0; i < nw; ++i) {
      w[i] = (uint8_t)(rnd(2, (int)i) * 127.0f + 128.0f);  // [1,255]
      if ((w[i] & 0x7Fu) == 0x7Fu) w[i] ^= 1u;             // drop NaN codes
    }
    std::vector<float> scales(ns);
    for (size_t i = 0; i < ns; ++i)
      scales[i] = 0.03125f * (float)(1 + (i % 128));       // [1/32, 4]
    std::vector<uint16_t> ref((size_t)c.M * c.N), host((size_t)c.M * c.N, 0);
    vkernels::kernels::glm_fp8_block_gemv_cpu(
        x.data(), w.data(), scales.data(), ref.data(), c.M, c.N, c.K);

    uint16_t* dx; uint8_t* dw; float* ds; uint16_t* dout;
    CK(hipMalloc(&dx, nx * 2), "ax");
    CK(hipMalloc(&dw, nw), "aw");
    CK(hipMalloc(&ds, ns * 4), "as");
    CK(hipMalloc(&dout, (size_t)c.M * c.N * 2), "ao");
    CK(hipMemcpy(dx, x.data(), nx * 2, hipMemcpyHostToDevice), "cx");
    CK(hipMemcpy(dw, w.data(), nw, hipMemcpyHostToDevice), "cw");
    CK(hipMemcpy(ds, scales.data(), ns * 4, hipMemcpyHostToDevice), "cs");
    vkernels::kernels::hip::glm_fp8_block_gemv(dx, dw, ds, dout,
                                               c.M, c.N, c.K);
    CK(hipDeviceSynchronize(), "sync");
    CK(hipMemcpy(host.data(), dout, host.size() * 2,
                 hipMemcpyDeviceToHost), "co");

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < host.size(); ++i) {
      uint32_t a = (uint32_t)host[i] << 16, b = (uint32_t)ref[i] << 16;
      float fa, fb;
      std::memcpy(&fa, &a, 4);
      std::memcpy(&fb, &b, 4);
      double e = std::fabs((double)fa - fb);
      if (e > max_abs) max_abs = e;
      double d = std::fmax(std::fabs((double)fb), 1.0);
      if (e / d > max_rel) max_rel = e / d;
    }
    bool ok = max_rel < THRESH;
    std::printf("  M=%d N=%-4d K=%-4d  max_abs=%.4f max_rel=%.6f  %s\n",
                c.M, c.N, c.K, max_abs, max_rel, ok ? "PASS" : "FAIL");
    ok ? ++passes : ++fails;
    fflush(stdout);
    hipFree(dx); hipFree(dw); hipFree(ds); hipFree(dout);
  }

  std::printf("\n%s (%d/%d PASS)\n", fails ? "FAIL" : "PASS",
              passes, passes + fails);
  return fails ? 1 : 0;
}
