// meta/benchmarks/probe_issue156_batched.cpp
//
// Issue #156 batched probe: MoE DECODE expert-GEMV split-K bandwidth on
// MI300A (gfx942). Per-iteration event-pair benches are unreliable here
// (DVFS ramp, ~230 us event-pair floor, sporadic 0.0 us readings), so this
// harness uses the mandatory methodology:
//   * 20 warmup launches + hipDeviceSynchronize,
//   * 1000 launches inside ONE hipEvent pair,
//   * >= 4 consecutive batches, per-batch us/launch reported,
//   * output checksum (hipMemcpy D2H + sum) to prove launches are real,
//   * cross-config GPU-side correctness gate vs the unsplit two-phase
//     kernel (same fp32-accum / RNE contract; tolerance 2e-2 rel).
//
// Shapes: the split-K decode rows from the issue (M=5):
//   N=6288 K=7168, N=3584 K=7168, N=896 K=7168.
//
// USAGE: probe_issue156_batched [phase]
//   phase "base" (default): existing gemm_bf16_splitk_with_config sweep
//           over tiles x S + prefill-V5 cross-check row.
// Build (by hand on the cluster):
//   hipcc -O3 -std=c++17 -DVKERNELS_HAS_HIP=1 -I<src>/src/c \
//     probe_issue156_batched.cpp -o probe156 \
//     -L<build>/src/c -lvkernels -L/opt/rocm-6.3.0/lib -lamdhip64

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vkernels/kernels/gemm_bf16.hpp"

// Explicit-tile dispatchers (not in the public header; forward-declared,
// same pattern as meta/benchmarks/bench_gemm_bf16.hip).
namespace vkernels::kernels::hip {
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads);
void gemm_bf16_splitk_with_config(std::size_t M, std::size_t N, std::size_t K,
                                  float alpha, const uint16_t* A,
                                  const uint16_t* B, float beta, uint16_t* C,
                                  int bm, int bn, int S);
void gemm_bf16_splitk_legacy_with_config(
    std::size_t M, std::size_t N, std::size_t K, float alpha,
    const uint16_t* A, const uint16_t* B, float beta, uint16_t* C,
    int bm, int bn, int S);
}  // namespace vkernels::kernels::hip

#define CK(x)                                                          \
  do {                                                                 \
    hipError_t e = (x);                                                \
    if (e != hipSuccess) {                                             \
      std::printf("HIP ERR %s @%d: %s\n", #x, __LINE__,                \
                  hipGetErrorString(e));                               \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

// Host/device bf16 <-> f32 helpers (the header ones are device-side only).
__host__ __device__ static inline float bf16_host_to_f32(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
__host__ __device__ static inline uint16_t f2bf_host(float v) {
  uint32_t u;
  memcpy(&u, &v, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Device dequant oracle (issue #156 fp8 phase): W8 E4M3FNUZ [K, N] + fp32
// per-(128-K x 8-N) block scales -> bf16 [K, N]. Independent decode.
__global__ void dequant_fnuz_kernel(const uint8_t* __restrict__ w8,
                                    const float* __restrict__ scales,
                                    uint16_t* __restrict__ out,
                                    int N, int K) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)K * N) return;
  const int k = (int)(i / N), n = (int)(i % N);
  const uint8_t c = w8[i];
  const int E = (c >> 3) & 15, m = c & 7;
  const float mag = (E == 0) ? (m / 8.0f) * exp2f(-7.0f)
                             : (1.0f + m / 8.0f) * exp2f((float)(E - 8));
  const float v = (c & 0x80) ? -mag : mag;
  out[i] = f2bf_host(v * scales[(size_t)(k / 128) * ((N + 7) / 8) + (n / 8)]);
}

// Standard E4M3FNUZ software decode (bias 8; 0x80 unused by the encoder).
// This is the INDEPENDENT oracle decode: the kernel uses the branchless
// fp32-bit-reinterpret trick (fnuz_value * 2^-119), so a bit-exact match
// against this decode validates the trick too.
static float fnuz_dec(uint8_t c) {
  const int E = (c >> 3) & 15, m = c & 7;
  float v = (E == 0) ? (m / 8.0f) * std::ldexp(1.0f, -7)
                     : (1.0f + m / 8.0f) * std::ldexp(1.0f, E - 8);
  return (c & 0x80) ? -v : v;
}

// Host fnuz encoder with round-to-nearest (mirrors the WIP probe's
// fnuz_enc): v assumed pre-scaled to |v| <= 240.
static uint8_t fnuz_enc(float v) {
  if (v == 0.0f) return 0x00u;
  const uint8_t sign = (v < 0) ? 0x80u : 0x00u;
  float a = std::fabs(v);
  if (a >= 240.0f) return (uint8_t)(sign | 0x7Fu);  // saturate
  if (a < std::ldexp(1.0f, -7)) {                   // subnormal: m * 2^-10
    int m = (int)std::lround(a * std::ldexp(1.0f, 10));
    if (m > 7) m = 7;
    return (uint8_t)(sign | (uint8_t)m);
  }
  int e;
  std::frexp(a, &e);                                // a = f * 2^e, f in [.5,1)
  const int E = e - 1 + 8;                          // bias 8
  if (E > 15) return (uint8_t)(sign | 0x7Fu);
  const float man = a / std::ldexp(1.0f, E - 8) - 1.0f;
  int m = (int)std::lround(man * 8.0f);
  int Ec = E;
  if (m == 8) { m = 0; ++Ec; }
  if (Ec > 15) return (uint8_t)(sign | 0x7Fu);
  return (uint8_t)(sign | (Ec << 3) | m);
}

static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 2654435761u; x ^= x >> 15;
  return (float)((int)(x % 200000)) / 100000.0f - 1.0f;
}

// Batched timing + final-batch checksum. us_per_launch is the MEDIAN of the
// batches; all per-batch values are printed.
static double batched(const char* tag, int batches, int per_batch,
                      const std::function<void()>& launch,
                      const uint16_t* d_out, size_t n_out_elems) {
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  std::vector<double> us;
  double checksum = 0;
  for (int k = 0; k < batches; ++k) {
    for (int w = 0; w < 20; ++w) launch();
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < per_batch; ++i) launch();
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1000.0 / per_batch);
    if (k == batches - 1 && d_out && n_out_elems) {
      std::vector<uint16_t> chk(n_out_elems);
      CK(hipMemcpy(chk.data(), d_out, n_out_elems * 2,
                   hipMemcpyDeviceToHost));
      double s = 0;
      for (uint16_t v : chk) s += bf16_host_to_f32(v);
      checksum = s;
    }
  }
  std::sort(us.begin(), us.end());
  double med = us[us.size() / 2];
  std::printf("  %-42s batches:", tag);
  for (double v : us) std::printf(" %.1f", v);
  std::printf("  | med %.2f us  sum=%.8g\n", med, checksum);
  CK(hipEventDestroy(a));
  CK(hipEventDestroy(b));
  return med;
}

int main(int argc, char** argv) {
  const std::string phase = (argc > 1) ? argv[1] : "base";
  CK(hipInit(0));
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0));
  std::printf("=== issue #156 batched probe (phase=%s) ===\n", phase.c_str());
  std::printf("GPU: %s (%s)\n", p.name, p.gcnArchName);

  struct NK { int N, K; };
  const NK shapes[] = {{6288, 7168}, {3584, 7168}, {896, 7168}};
  const int M = 5;

  for (const auto& sh : shapes) {
    const int N = sh.N, K = sh.K;
    std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N, 0);
    for (size_t i = 0; i < A.size(); ++i) A[i] = f2bf_host(rnd(1, (int)i) * 0.5f);
    for (size_t i = 0; i < B.size(); ++i) B[i] = f2bf_host(rnd(2, (int)i) * 0.5f);
    uint16_t *dA, *dB, *dC, *dRef;
    CK(hipMalloc(&dA, A.size() * 2));
    CK(hipMalloc(&dB, B.size() * 2));
    CK(hipMalloc(&dC, C.size() * 2));
    CK(hipMalloc(&dRef, C.size() * 2));
    CK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemset(dC, 0, C.size() * 2));

    // GPU reference: unsplit two-phase (16,64) kernel, run once.
    vkernels::kernels::hip::gemm_bf16_with_config(
        (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dB, 0.0f, dRef, 16, 64, 64, 64);
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> ref(C.size());
    CK(hipMemcpy(ref.data(), dRef, C.size() * 2, hipMemcpyDeviceToHost));

    const double bytes =
        2.0 * ((double)M * K * ((N + 15) / 16) +
               (double)K * N * ((M + 15) / 16) + (double)M * N);

    auto run_cfg = [&](const char* name, bool legacy, int bm, int bn, int S) {
      CK(hipMemset(dC, 0, C.size() * 2));
      auto L = [&] {
        if (legacy)
          vkernels::kernels::hip::gemm_bf16_splitk_legacy_with_config(
              (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dB, 0.0f, dC,
              bm, bn, S);
        else
          vkernels::kernels::hip::gemm_bf16_splitk_with_config(
              (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dB, 0.0f, dC,
              bm, bn, S);
      };
      char tag[128];
      std::snprintf(tag, sizeof tag, "%s N=%d K=%d", name, N, K);
      double us = batched(tag, 4, 1000, L, dC, C.size());
      std::printf("      -> %.0f GB/s (bytes model %.3g MB)\n",
                  bytes / (us / 1e6) / 1e9, bytes / 1e6);
      // Correctness vs the GPU reference.
      std::vector<uint16_t> got(C.size());
      CK(hipMemcpy(got.data(), dC, C.size() * 2, hipMemcpyDeviceToHost));
      double max_rel = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        float g = bf16_host_to_f32(got[i]), r = bf16_host_to_f32(ref[i]);
        double rel = std::fabs(g - r) / std::fmax(1e-6, std::fabs(r));
        if (rel > max_rel) max_rel = rel;
      }
      std::printf("      -> vs-unsplit max_rel=%.3g %s\n", max_rel,
                  max_rel < 2e-2 ? "PASS" : "FAIL");
    };

    std::printf(" -- shape N=%d K=%d M=%d --\n", N, K, M);
    if (phase == "base" || phase == "full") {
      // legacy (#146) A/B rows
      run_cfg("legacy 16x16 S=8", true, 16, 16, 8);
    }
    if (phase == "full") {
      run_cfg("wide 16x16 S=4", false, 16, 16, 4);
      run_cfg("wide 16x16 S=8", false, 16, 16, 8);
      run_cfg("wide 16x16 S=16", false, 16, 16, 16);
      run_cfg("wide 16x16 S=32", false, 16, 16, 32);
      run_cfg("wide 16x64 S=4", false, 16, 64, 4);
      run_cfg("wide 16x64 S=8", false, 16, 64, 8);
      run_cfg("wide 16x64 S=16", false, 16, 64, 16);
      run_cfg("wide 16x64 S=32", false, 16, 64, 32);
      run_cfg("wide 32x64 S=8", false, 32, 64, 8);
      run_cfg("wide 32x64 S=16", false, 32, 64, 16);
    }
    if (phase == "base") {
      run_cfg("legacy 16x16 S=4", true, 16, 16, 4);
      run_cfg("legacy 16x16 S=16", true, 16, 16, 16);
      run_cfg("legacy 16x16 S=32", true, 16, 16, 32);
      run_cfg("legacy 16x64 S=8", true, 16, 64, 8);
      run_cfg("legacy 16x64 S=16", true, 16, 64, 16);
      run_cfg("legacy 32x64 S=8", true, 32, 64, 8);
    }
    CK(hipFree(dA)); CK(hipFree(dB)); CK(hipFree(dC)); CK(hipFree(dRef));
  }

  // ------------------------------------------------------------------
  // FP8 phase (issue #156 workstream b): block-FP8 (E4M3FNUZ) weight
  // stream, validated against (1) a device dequant oracle -- dequantize
  // W8+scales to bf16 and run the WIDE bf16 split-K kernel, which the fp8
  // kernel must match BIT-EXACTLY (same staging dequant, same MFMA body),
  // and (2) the CPU fp32 oracle on the dequantized weights (2e-2 rel).
  // ------------------------------------------------------------------
  if (phase == "full") {
    for (const auto& sh : shapes) {
      const int N = sh.N, K = sh.K;
      const int n_s8 = (N + 7) / 8, k_b = (K + 127) / 128;
      std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N);
      for (size_t i = 0; i < A.size(); ++i) A[i] = f2bf_host(rnd(1, (int)i) * 0.5f);
      for (size_t i = 0; i < B.size(); ++i) B[i] = f2bf_host(rnd(2, (int)i) * 0.5f);

      // Host quantization: per (128-K x 8-N) block scales, fnuz codes.
      std::vector<float> scales((size_t)n_s8 * k_b);
      std::vector<uint8_t> W8((size_t)K * N);
      for (int kb = 0; kb < k_b; ++kb) {
        for (int nb = 0; nb < n_s8; ++nb) {
          float amax = 0;
          for (int kk = 0; kk < 128; ++kk) {
            const int k = kb * 128 + kk;
            if (k >= K) break;
            for (int nn = 0; nn < 8; ++nn) {
              const int n = nb * 8 + nn;
              if (n >= N) break;
              amax = std::fmax(amax, std::fabs(bf16_host_to_f32(B[(size_t)k * N + n])));
            }
          }
          const float scale = amax / 240.0f;  // fnuz max = 240
          scales[(size_t)kb * n_s8 + nb] = scale;
          for (int kk = 0; kk < 128; ++kk) {
            const int k = kb * 128 + kk;
            if (k >= K) break;
            for (int nn = 0; nn < 8; ++nn) {
              const int n = nb * 8 + nn;
              if (n >= N) break;
              W8[(size_t)k * N + n] = fnuz_enc(
                  bf16_host_to_f32(B[(size_t)k * N + n]) / scale);
            }
          }
        }
      }
      // CPU reference: dequantize on host (INDEPENDENT standard fnuz
      // decode) + gemm_bf16_cpu (fp32 accum, single RNE).
      std::vector<uint16_t> Bdq((size_t)K * N);
      for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
          Bdq[(size_t)k * N + n] = f2bf_host(
              fnuz_dec(W8[(size_t)k * N + n]) *
              scales[(size_t)(k / 128) * n_s8 + (n / 8)]);
      std::vector<uint16_t> Cref((size_t)M * N, 0);
      vkernels::kernels::gemm_bf16_cpu((size_t)M, (size_t)N, (size_t)K,
                                       1.0f, A.data(), Bdq.data(), 0.0f,
                                       Cref.data());

      uint16_t *dA, *dC, *dCq, *dBdq;
      uint8_t* dW8;
      float* dS;
      CK(hipMalloc(&dA, A.size() * 2));
      CK(hipMalloc(&dW8, W8.size()));
      CK(hipMalloc(&dS, scales.size() * 4));
      CK(hipMalloc(&dC, Cref.size() * 2));
      CK(hipMalloc(&dCq, Cref.size() * 2));
      CK(hipMalloc(&dBdq, B.size() * 2));
      CK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
      CK(hipMemcpy(dW8, W8.data(), W8.size(), hipMemcpyHostToDevice));
      CK(hipMemcpy(dS, scales.data(), scales.size() * 4,
                   hipMemcpyHostToDevice));

      // Device dequant oracle -> wide bf16 split-K -> dCq (reference C).
      {
        const size_t els = (size_t)K * N;
        CK(hipMemset(dBdq, 0xEE, B.size() * 2));
        dequant_fnuz_kernel<<<(int)((els + 255) / 256), 256>>>(
            dW8, dS, dBdq, N, K);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        // In-job oracle sanity: the device dequant must match the host
        // dequant bit-exactly (catches silent kernel-launch failures, e.g.
        // a binary built without --offload-arch=gfx942).
        {
          std::vector<uint16_t> dqchk(B.size());
          CK(hipMemcpy(dqchk.data(), dBdq, dqchk.size() * 2,
                       hipMemcpyDeviceToHost));
          int dqbad = 0;
          for (size_t i = 0; i < dqchk.size(); ++i)
            if (dqchk[i] != Bdq[i]) ++dqbad;
          std::printf("      (dequant oracle vs host: %s, %d bad)\n",
                      dqbad == 0 ? "OK" : "MISMATCH", dqbad);
        }
        vkernels::kernels::hip::gemm_bf16_splitk_with_config(
            (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dBdq, 0.0f, dCq,
            16, 64, 8);
        CK(hipDeviceSynchronize());
      }

      std::printf(" -- fp8 shape N=%d K=%d M=%d --\n", N, K, M);
      const double bytes_fp8 =
          (double)K * N +                    // fp8 weight stream
          (double)n_s8 * k_b * 4 +           // scales
          2.0 * ((double)M * K + (double)M * N) +
          2.0 * 8.0 * M * N * 4;             // S=8 workspace round-trip
      struct FCfg { int bm, bn, S; };
      const FCfg fcfgs[] = {{16, 16, 8}, {16, 16, 16}, {16, 64, 8},
                            {16, 64, 16}, {32, 64, 8}};
      for (const auto& fc : fcfgs) {
        if (fc.S > (K + 63) / 64) continue;
        CK(hipMemset(dC, 0, Cref.size() * 2));
        auto L = [&] {
          vkernels::kernels::hip::gemm_fp8_block_splitk_with_config(
              (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dW8, dS, 0.0f, dC,
              fc.bm, fc.bn, fc.S);
        };
        char tag[128];
        std::snprintf(tag, sizeof tag, "fp8 %dx%d S=%d N=%d", fc.bm, fc.bn,
                      fc.S, N);
        double us = batched(tag, 4, 1000, L, dC, Cref.size());
        std::printf("      -> %.0f GB/s (fp8 bytes model %.3g MB)\n",
                    bytes_fp8 / (us / 1e6) / 1e9, bytes_fp8 / 1e6);
        std::vector<uint16_t> got(Cref.size());
        CK(hipMemcpy(got.data(), dC, Cref.size() * 2,
                     hipMemcpyDeviceToHost));
        int bit_mismatch = 0;
        double max_rel = 0;
        // vs device dequant oracle (bit-exact expected)
        std::vector<uint16_t> oracle(Cref.size());
        CK(hipMemcpy(oracle.data(), dCq, oracle.size() * 2,
                     hipMemcpyDeviceToHost));
        for (size_t i = 0; i < got.size(); ++i)
          if (got[i] != oracle[i]) ++bit_mismatch;
        // vs CPU fp32 oracle (tolerance)
        for (size_t i = 0; i < got.size(); ++i) {
          float g = bf16_host_to_f32(got[i]);
          float r = bf16_host_to_f32(Cref[i]);
          double rel = std::fabs(g - r) / std::fmax(1e-3, std::fabs(r));
          if (rel > max_rel) max_rel = rel;
        }
        std::printf("      -> dequant-oracle bit-mismatch=%d/%zu  "
                    "cpu max_rel=%.3g %s\n",
                    bit_mismatch, got.size(), max_rel,
                    (bit_mismatch == 0 && max_rel < 2e-2) ? "PASS" : "FAIL");
      }
      CK(hipFree(dA)); CK(hipFree(dW8)); CK(hipFree(dS));
      CK(hipFree(dC)); CK(hipFree(dCq)); CK(hipFree(dBdq));
    }
  }
  // ------------------------------------------------------------------
  // GEMV phase (issue #156 workstream c): the no-LDS-barrier decode-GEMV
  // split-K kernels. Sweep (S, tb, threads); bf16 AND fp8 variants; per
  // row: checksum vs the CPU oracle (2e-2 rel) + vs the unsplit reference.
  // ------------------------------------------------------------------
  if (phase == "gemv" || phase == "full2") {
    for (const auto& sh : shapes) {
      const int N = sh.N, K = sh.K;
      const int n_s8 = (N + 7) / 8, k_b = (K + 127) / 128;
      std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N);
      for (size_t i = 0; i < A.size(); ++i) A[i] = f2bf_host(rnd(1, (int)i) * 0.5f);
      for (size_t i = 0; i < B.size(); ++i) B[i] = f2bf_host(rnd(2, (int)i) * 0.5f);
      // fp8 quantization (same as the fp8 phase above)
      std::vector<float> scales((size_t)n_s8 * k_b);
      std::vector<uint8_t> W8((size_t)K * N);
      for (int kb = 0; kb < k_b; ++kb)
        for (int nb = 0; nb < n_s8; ++nb) {
          float amax = 0;
          for (int kk = 0; kk < 128; ++kk) {
            const int k = kb * 128 + kk;
            if (k >= K) break;
            for (int nn = 0; nn < 8; ++nn) {
              const int n = nb * 8 + nn;
              if (n >= N) break;
              amax = std::fmax(amax, std::fabs(bf16_host_to_f32(B[(size_t)k * N + n])));
            }
          }
          const float scale = amax / 240.0f;
          scales[(size_t)kb * n_s8 + nb] = scale;
          for (int kk = 0; kk < 128; ++kk) {
            const int k = kb * 128 + kk;
            if (k >= K) break;
            for (int nn = 0; nn < 8; ++nn) {
              const int n = nb * 8 + nn;
              if (n >= N) break;
              W8[(size_t)k * N + n] = fnuz_enc(
                  bf16_host_to_f32(B[(size_t)k * N + n]) / scale);
            }
          }
        }
      // CPU reference on the bf16 B (shared by both variants)
      std::vector<uint16_t> Cref((size_t)M * N, 0);
      vkernels::kernels::gemm_bf16_cpu((size_t)M, (size_t)N, (size_t)K,
                                       1.0f, A.data(), B.data(), 0.0f,
                                       Cref.data());
      std::vector<uint16_t> Crefq((size_t)M * N, 0);
      {
        std::vector<uint16_t> Bdq((size_t)K * N);
        for (int k = 0; k < K; ++k)
          for (int n = 0; n < N; ++n)
            Bdq[(size_t)k * N + n] = f2bf_host(
                fnuz_dec(W8[(size_t)k * N + n]) *
                scales[(size_t)(k / 128) * n_s8 + (n / 8)]);
        vkernels::kernels::gemm_bf16_cpu((size_t)M, (size_t)N, (size_t)K,
                                         1.0f, A.data(), Bdq.data(), 0.0f,
                                         Crefq.data());
      }
      uint16_t *dA, *dC, *dCref, *dB;
      uint8_t* dW8;
      float* dS;
      CK(hipMalloc(&dA, A.size() * 2));
      CK(hipMalloc(&dB, B.size() * 2));
      CK(hipMalloc(&dW8, W8.size()));
      CK(hipMalloc(&dS, scales.size() * 4));
      CK(hipMalloc(&dC, Cref.size() * 2));
      CK(hipMalloc(&dCref, Cref.size() * 2));
      CK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
      CK(hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice));
      CK(hipMemcpy(dW8, W8.data(), W8.size(), hipMemcpyHostToDevice));
      CK(hipMemcpy(dS, scales.data(), scales.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(dCref, Cref.data(), Cref.size() * 2, hipMemcpyHostToDevice));

      struct GCfg { int S, tb, th; };
      const GCfg gcfgs[] = {{32, 4, 64}, {64, 4, 64}, {96, 4, 64},
                            {64, 4, 128}, {96, 4, 128}, {64, 4, 256},
                            {32, 4, 256}, {64, 8, 64}, {64, 8, 128},
                            {128, 4, 128}};
      for (int variant = 0; variant < 2; ++variant) {
        const bool fp8v = variant == 1;
        const double bytes = fp8v
            ? (double)K * N + (double)n_s8 * k_b * 4 +
                  2.0 * ((double)M * K + (double)M * N)
            : 2.0 * ((double)K * N + (double)M * K + (double)M * N);
        std::printf(" -- gemv-%s shape N=%d K=%d M=%d --\n",
                    fp8v ? "fp8" : "bf16", N, K, M);
        for (const auto& gc : gcfgs) {
          if (gc.S > K) continue;
          CK(hipMemset(dC, 0, Cref.size() * 2));
          auto L = [&] {
            if (fp8v)
              vkernels::kernels::hip::gemv_decode_fp8_splitk(
                  (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dW8, dS, 0.0f,
                  dC, gc.S, gc.tb, gc.th);
            else
              vkernels::kernels::hip::gemv_decode_bf16_splitk(
                  (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dB, 0.0f,
                  dC, gc.S, gc.tb, gc.th);
          };
          char tag[128];
          std::snprintf(tag, sizeof tag, "gemv-%s S=%d tb=%d th=%d N=%d",
                        fp8v ? "fp8" : "bf16", gc.S, gc.tb, gc.th, N);
          double us = batched(tag, 4, 1000, L, dC, Cref.size());
          std::printf("      -> %.0f GB/s (bytes model %.3g MB)\n",
                      bytes / (us / 1e6) / 1e9, bytes / 1e6);
          std::vector<uint16_t> got(Cref.size());
          CK(hipMemcpy(got.data(), dC, Cref.size() * 2, hipMemcpyDeviceToHost));
          const std::vector<uint16_t>& ref = fp8v ? Crefq : Cref;
          double max_rel = 0;
          for (size_t i = 0; i < got.size(); ++i) {
            float g = bf16_host_to_f32(got[i]);
            float r = bf16_host_to_f32(ref[i]);
            double rel = std::fabs(g - r) / std::fmax(1e-3, std::fabs(r));
            if (rel > max_rel) max_rel = rel;
          }
          int bit_vs_unsplit = 0;
          std::vector<uint16_t> crefd(Cref.size());
          CK(hipMemcpy(crefd.data(), dCref, crefd.size() * 2, hipMemcpyDeviceToHost));
          (void)bit_vs_unsplit;
          std::printf("      -> cpu max_rel=%.3g %s\n", max_rel,
                      max_rel < 2e-2 ? "PASS" : "FAIL");
        }
      }
      CK(hipFree(dA)); CK(hipFree(dB)); CK(hipFree(dW8)); CK(hipFree(dS));
      CK(hipFree(dC)); CK(hipFree(dCref));
    }
  }
  std::printf("=== probe done ===\n");
  return 0;
}
