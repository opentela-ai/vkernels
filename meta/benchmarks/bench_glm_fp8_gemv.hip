// meta/benchmarks/bench_glm_fp8_gemv.hip
//
// Micro-benchmark for the GLM block-FP8 decode-fused GEMV (issue #64) on
// gfx942, at the GLM MoE decode expert shapes:
//
//   gate/up [E,4096,4096]   down [E,4096,2048]   (E4M3FN, scales 128x128)
//
// Compares, per (M, N, K):
//   * fused   : glm_fp8_block_gemv (streams FP8 + scales; nothing
//               materialized)  — bytes = N*K + scales + x + out
//   * mat-gemv: the same GEMV structure reading a BF16 weight buffer
//               (the steady-state cost of the dequantize-once path floe
//               uses today) — bytes = 2*N*K + ...
//   * dequant : one-time FP8->BF16 materialization cost (reported; part
//               of the cold path only)
// against the MI300A roof (~5300 GB/s). A decode token pays gate/up + down
// per selected expert; the per-matrix medians here compose to that.
//
// USAGE: bench_glm_fp8_gemv [iters]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

constexpr int kTh = 256;

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
  return __uint_as_float((uint32_t)v << 16);
}
__device__ __forceinline__ uint16_t f32_to_bf16(float v) {
  const uint32_t u = __float_as_uint(v);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float e4m3_f32(uint32_t code) {
  const uint32_t exp = (code >> 3) & 15u;
  if (exp == 0) {
    const float m = (float)(code & 7u) * (1.0f / 512.0f);
    return (code & 0x80u) ? -m : m;
  }
  if (exp == 15u && (code & 7u) == 7u)
    return __uint_as_float(0x7FC00000u | ((code & 0x80u) << 24));
  return __uint_as_float(((code & 0x80u) << 24) | ((exp - 7u + 127u) << 23) |
                         ((code & 7u) << 20));
}

// BF16-weight GEMV, same launch structure as the fused kernel.
__global__ void bf16_gemv_kernel(const uint16_t* __restrict__ x,
                                 const uint16_t* __restrict__ w,
                                 uint16_t* __restrict__ out,
                                 int M, int N, int K) {
  __shared__ float sx[2 * 4096];
  const int tid = threadIdx.x;
  for (int i = tid; i < M * K; i += kTh) sx[i] = bf16_to_f32(x[i]);
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31;
  const int nseg = K >> 7;
  const int row0 = (blockIdx.x * 8 + warp) * 4;
  for (int r = 0; r < 4; ++r) {
    const int row = row0 + r;
    if (row >= N) break;
    const uint16_t* wrow = w + (size_t)row * K;
    for (int m = 0; m < M; ++m) {
      const float* xr = sx + m * K;
      float acc = 0.0f;
      for (int seg = 0; seg < nseg; ++seg) {
        const uint2 code = *(const uint2*)(wrow + seg * 128 + lane * 4);
        const float4 xv = *(const float4*)(xr + seg * 128 + lane * 4);
        acc += bf16_to_f32(code.x & 0xFFFFu) * xv.x +
               bf16_to_f32(code.x >> 16) * xv.y +
               bf16_to_f32(code.y & 0xFFFFu) * xv.z +
               bf16_to_f32(code.y >> 16) * xv.w;
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off);
      if (lane == 0) out[(size_t)m * N + row] = f32_to_bf16(acc);
    }
  }
}

// FP8 -> BF16 materialization with per-128x128-block scales.
__global__ void dequant_kernel(const uint8_t* __restrict__ w,
                               const float* __restrict__ scales,
                               uint16_t* __restrict__ wb, int N, int K) {
  const int nseg = K >> 7;
  const size_t i = (size_t)blockIdx.x * kTh + threadIdx.x;
  if (i >= (size_t)N * K) return;
  const int n = (int)(i / K), k = (int)(i % K);
  const float s = scales[(size_t)(n >> 7) * nseg + (k >> 7)];
  wb[i] = f32_to_bf16(s * e4m3_f32(w[i]));
}

double bench_us(const std::function<void()>& launch, hipEvent_t a, hipEvent_t b,
                int warm, int iters) {
  for (int i = 0; i < warm; ++i) launch();
  hipDeviceSynchronize();
  std::vector<float> t;
  for (int i = 0; i < iters; ++i) {
    hipEventRecord(a);
    launch();
    hipEventRecord(b);
    hipEventSynchronize(b);
    float ms = 0;
    hipEventElapsedTime(&ms, a, b);
    if (ms > 0) t.push_back(ms * 1000.0f);
  }
  std::sort(t.begin(), t.end());
  return t.empty() ? 0.0 : t[t.size() / 2];
}

float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f;
}

}  // namespace

int main(int argc, char** argv) {
  namespace h = vkernels::kernels::hip;
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  const double roof_gbs = std::strstr(p.gcnArchName, "gfx942") ? 5300.0 : 1000.0;
  std::printf("GPU: %s (arch=%s)  roof=%.0f GB/s\n", p.name, p.gcnArchName,
              roof_gbs);
  const int iters = argc > 1 ? std::atoi(argv[1]) : 60;

  struct Cfg { const char* tag; int N, K; };
  const Cfg cfgs[] = {
      {"gate/up [4096,4096]", 4096, 4096},
      {"down    [4096,2048]", 4096, 2048},
  };
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);

  std::printf("%-22s %2s  %9s %8s %9s %8s %9s   %s\n", "shape", "M",
              "fused_us", "f_GB/s", "mat_us", "m_GB/s", "dequant", "steady");
  for (const auto& c : cfgs)
    for (int M : {1, 2}) {
      const size_t nx = (size_t)M * c.K, nw = (size_t)c.N * c.K;
      const size_t ns = (size_t)(c.N / 128) * (c.K / 128);
      std::vector<uint16_t> x(nx);
      for (size_t i = 0; i < nx; ++i) {
        uint32_t u;
        float f = rnd(1, (int)i);
        std::memcpy(&u, &f, 4);
        x[i] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
      }
      std::vector<uint8_t> w(nw);
      for (size_t i = 0; i < nw; ++i) {
        w[i] = (uint8_t)(rnd(2, (int)i) * 127.0f + 128.0f);
        if ((w[i] & 0x7Fu) == 0x7Fu) w[i] ^= 1u;
      }
      std::vector<float> s(ns);
      for (size_t i = 0; i < ns; ++i) s[i] = 0.03125f * (float)(1 + (i % 128));

      uint16_t *dx, *dout, *dwb;
      uint8_t* dw;
      float *ds, *dpart;
      CK(hipMalloc(&dx, nx * 2), "ax");
      CK(hipMalloc(&dw, nw), "aw");
      CK(hipMalloc(&ds, ns * 4), "as");
      CK(hipMalloc(&dout, (size_t)M * c.N * 2), "ao");
      CK(hipMalloc(&dwb, nw * 2), "ab");
      const int sk = h::glm_fp8_gemv_pick_sk(c.N, c.K);
      CK(hipMalloc(&dpart, (size_t)M * c.N * sk * 4), "ap");
      CK(hipMemcpy(dx, x.data(), nx * 2, hipMemcpyHostToDevice), "cx");
      CK(hipMemcpy(dw, w.data(), nw, hipMemcpyHostToDevice), "cw");
      CK(hipMemcpy(ds, s.data(), ns * 4, hipMemcpyHostToDevice), "cs");

      const double fused = bench_us([&] {
        h::glm_fp8_block_gemv_with_scratch(dx, dw, ds, dout, dpart,
                                           M, c.N, c.K, sk);
      }, a, b, 5, iters);
      const double mat = bench_us([&] {
        const int grid = (int)((nw + kTh - 1) / kTh);
        bf16_gemv_kernel<<<(c.N + 31) / 32, kTh>>>(dx, dwb, dout, M, c.N, c.K);
      }, a, b, 5, iters);
      const double dq = bench_us([&] {
        const int grid = (int)((nw + kTh - 1) / kTh);
        dequant_kernel<<<grid, kTh>>>(dw, ds, dwb, c.N, c.K);
      }, a, b, 3, 20);

      const double fused_bytes = (double)nw + (double)ns * 4 + (double)nx * 2 +
                                 (double)M * c.N * 2;
      const double mat_bytes = 2.0 * nw + (double)nx * 2 + (double)M * c.N * 2;
      const double fgb = fused_bytes / (fused / 1e6) / 1e9;
      const double mgb = mat_bytes / (mat / 1e6) / 1e9;
      const double roof_f = fused_bytes / roof_gbs / 1e9 * 1e6;
      const double roof_m = mat_bytes / roof_gbs / 1e9 * 1e6;
      std::printf("%-22s %2d  %9.1f %8.0f %9.1f %8.0f %9.1f   %.2fx "
                  "(sk=%d, roofs %.1f/%.1f us)\n",
                  c.tag, M, fused, fgb, mat, mgb, dq,
                  mat / (fused > 0 ? fused : 1), sk, roof_f, roof_m);
      hipFree(dx); hipFree(dw); hipFree(ds); hipFree(dout); hipFree(dwb);
      hipFree(dpart);
    }
  return 0;
}
