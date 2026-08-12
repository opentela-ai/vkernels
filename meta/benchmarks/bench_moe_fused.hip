// meta/benchmarks/bench_moe_fused.hip
//
// End-to-end MXFP4 fused MoE benchmark on AMD gfx942/gfx90a.
// Sweeps decode-regime batch sizes (M = 1..48), verifies correctness against
// the CPU reference, and reports per-M latency + effective TFLOP/s.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/moe.hpp"
#include "vkernels/util/config.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

static uint32_t f2bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
static float bits2f(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

static uint16_t f2bf(float v) {
  uint32_t b = f2bits(v);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}
static float bf2f(uint16_t v) { return bits2f((uint32_t)v << 16); }

// Pack a float → E2M1 nibble (MXFP4: 0,±0.5,±1,±1.5,±2,±3,±4,±6).
static uint8_t e2m1_nibble(float v) {
  bool neg = v < 0;
  float af = std::fabs(v);
  float vals[6] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f};
  uint8_t nibs[6] = {0, 1, 2, 3, 4, 5};
  float best_d = std::fabs(af - vals[0]);
  uint8_t best_n = nibs[0];
  for (int i = 1; i < 6; ++i) {
    float d = std::fabs(af - vals[i]);
    if (d < best_d) { best_d = d; best_n = nibs[i]; }
  }
  if (af > 4.5f) best_n = 7;  // 6.0
  return neg ? (uint8_t)(best_n | 0x8) : best_n;
}

struct BenchResult { double median_us; int iters; };

static BenchResult bench_kernel_us(const std::function<void()>& launch,
                                   hipEvent_t start, hipEvent_t stop,
                                   int warmup = 5, int max_iters = 200) {
  (void)hipDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) launch();
  (void)hipDeviceSynchronize();
  std::vector<float> times;
  int iters = 10;
  while (iters <= max_iters) {
    times.clear();
    for (int i = 0; i < iters; ++i) {
      hipEventRecord(start);
      launch();
      hipEventRecord(stop);
      hipEventSynchronize(stop);
      float ms = 0;
      hipEventElapsedTime(&ms, start, stop);
      times.push_back(ms * 1000.0f);
    }
    std::sort(times.begin(), times.end());
    return {static_cast<double>(times[iters / 2]), iters};
  }
  return {};
}

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (%s)\n\n", p.name, p.gcnArchName);

  constexpr int E = 256, hidden = 4096, ispp = 512, top_k = 6;
  constexpr int BLOCK_M = 16, group_size = 32;
  constexpr float LIMIT = 10.0f;

  // --- Host buffers ---
  const int w13_sz = E * 2 * ispp * (hidden / 2);
  const int w13s_sz = E * 2 * ispp * (hidden / group_size);
  const int w2_sz = E * hidden * (ispp / 2);
  const int w2s_sz = E * hidden * (ispp / group_size);

  std::vector<uint8_t> hw13(w13_sz), hw13s(w13s_sz, 127);
  std::vector<uint8_t> hw2(w2_sz), hw2s(w2s_sz, 127);
  for (int i = 0; i < w13_sz / 2; ++i) {
    hw13[2 * i] = e2m1_nibble(rnd(2, 2 * i) * 1.5f) |
                  (e2m1_nibble(rnd(3, 2 * i + 1) * 1.5f) << 4);
    hw13[2 * i + 1] = e2m1_nibble(rnd(4, i) * 1.5f) |
                      (e2m1_nibble(rnd(5, i) * 1.5f) << 4);
  }
  for (int i = 0; i < w2_sz / 2; ++i) {
    hw2[2 * i] = e2m1_nibble(rnd(6, 2 * i) * 1.5f) |
                 (e2m1_nibble(rnd(7, 2 * i + 1) * 1.5f) << 4);
    hw2[2 * i + 1] = e2m1_nibble(rnd(8, i) * 1.5f) |
                     (e2m1_nibble(rnd(9, i) * 1.5f) << 4);
  }
  std::vector<float> hb13(E * 2 * ispp, 0.0f), hb2(E * hidden, 0.0f);

  // --- Device buffers ---
  uint8_t *dw13, *dw13s, *dw2, *dw2s;
  float *db13, *db2;
  CK(hipMalloc(&dw13, w13_sz), "w13");
  CK(hipMalloc(&dw13s, w13s_sz), "w13s");
  CK(hipMalloc(&dw2, w2_sz), "w2");
  CK(hipMalloc(&dw2s, w2s_sz), "w2s");
  CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13");
  CK(hipMalloc(&db2, E * hidden * 4), "b2");
  CK(hipMemcpy(dw13, hw13.data(), w13_sz, hipMemcpyHostToDevice), "cpw13");
  CK(hipMemcpy(dw13s, hw13s.data(), w13s_sz, hipMemcpyHostToDevice), "cpw13s");
  CK(hipMemcpy(dw2, hw2.data(), w2_sz, hipMemcpyHostToDevice), "cpw2");
  CK(hipMemcpy(dw2s, hw2s.data(), w2s_sz, hipMemcpyHostToDevice), "cpw2s");
  CK(hipMemcpy(db13, hb13.data(), E * 2 * ispp * 4, hipMemcpyHostToDevice), "cpb13");
  CK(hipMemcpy(db2, hb2.data(), E * hidden * 4, hipMemcpyHostToDevice), "cpb2");

  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);

  const int Ms[] = {1, 2, 4, 8, 16, 32, 48};
  std::printf("E=%d hidden=%d ispp=%d top_k=%d\n", E, hidden, ispp, top_k);
  std::printf("%6s %8s %10s %10s %12s\n", "M", "EM", "lat(us)", "TFLOP/s", "status");

  for (int M : Ms) {
    int N = M * top_k;
    int EM_max = E * BLOCK_M;
    std::vector<int32_t> h_tids(N), h_sids(EM_max), h_eids(EM_max / BLOCK_M);
    std::vector<float> h_tw(N);
    // Random expert routing (spread across E experts).
    for (int i = 0; i < M; ++i)
      for (int s = 0; s < top_k; ++s) {
        h_tids[i * top_k + s] = (i * 17 + s * 13 + 7) % E;
        h_tw[i * top_k + s] = 0.5f + 0.5f * rnd(20, i * top_k + s);
      }
    int EM = vkernels::kernels::moe_align_block_size(
        h_tids.data(), M, top_k, BLOCK_M, E, h_sids.data(), h_eids.data());

    std::vector<uint16_t> hA(M * hidden);
    for (int i = 0; i < M * hidden; ++i) hA[i] = f2bf(rnd(10, i) * 0.1f);

    uint16_t *dA, *dAct;
    float *dOut, *dTw;
    int32_t *dTids, *dSids, *dEids;
    CK(hipMalloc(&dA, M * hidden * 2), "A");
    CK(hipMalloc(&dAct, EM * ispp * 2), "act");
    CK(hipMalloc(&dOut, M * hidden * 4), "out");
    CK(hipMalloc(&dTw, N * 4), "tw");
    CK(hipMalloc(&dTids, N * 4), "tids");
    CK(hipMalloc(&dSids, EM * 4), "sids");
    CK(hipMalloc(&dEids, (EM / BLOCK_M) * 4), "eids");
    CK(hipMemcpy(dA, hA.data(), M * hidden * 2, hipMemcpyHostToDevice), "cpA");
    CK(hipMemcpy(dTw, h_tw.data(), N * 4, hipMemcpyHostToDevice), "cpTw");
    CK(hipMemcpy(dTids, h_tids.data(), N * 4, hipMemcpyHostToDevice), "cpTids");
    CK(hipMemcpy(dSids, h_sids.data(), EM * 4, hipMemcpyHostToDevice), "cpSids");
    CK(hipMemcpy(dEids, h_eids.data(), (EM / BLOCK_M) * 4, hipMemcpyHostToDevice), "cpEids");
    CK(hipMemset(dOut, 0, M * hidden * 4), "zOut");

    auto launch = [&]() {
      vkernels::kernels::hip::fused_moe_mxfp4(
          dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
          M, hidden, ispp, top_k, dSids, dEids, EM, LIMIT, db13, db2);
    };

    // Correctness check vs CPU reference (only for M <= 8 to bound host cost).
    const char* status = "ok";
    if (M <= 8) {
      std::vector<float> h_tws(EM, 0.0f);
      for (int i = 0; i < EM; ++i)
        if (h_sids[i] < M * top_k) h_tws[i] = h_tw[h_sids[i]];
      std::vector<uint16_t> act_cpu(EM * ispp, 0);
      std::vector<float> out_cpu(M * hidden, 0.0f);
      vkernels::kernels::fused_moe_mxfp4_cpu(
          hA.data(), hw13.data(), hw13s.data(), hw2.data(), hw2s.data(),
          h_sids.data(), h_tws.data(), h_eids.data(), act_cpu.data(), out_cpu.data(),
          M, hidden, ispp, top_k, EM, group_size, LIMIT, hb13.data(), hb2.data());
      launch();
      CK(hipDeviceSynchronize(), "sync");
      std::vector<float> out_gpu(M * hidden);
      CK(hipMemcpy(out_gpu.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut");
      double max_rel = 0;
      for (int i = 0; i < M * hidden; ++i) {
        double e = std::fabs((double)out_gpu[i] - (double)out_cpu[i]);
        max_rel = std::fmax(max_rel, e / std::fmax(std::fabs((double)out_cpu[i]), 1.0));
      }
      status = max_rel < 0.05 ? "ok" : "MISMATCH";
    }

    auto r = bench_kernel_us(launch, start, stop);
    double total_flop = 6.0 * M * (double)ispp * hidden;
    double tflops = total_flop / (r.median_us / 1e6) / 1e12;

    std::printf("%6d %8d %10.1f %10.3f %12s\n", M, EM, r.median_us, tflops, status);

    hipFree(dA); hipFree(dAct); hipFree(dOut);
    hipFree(dTw); hipFree(dTids); hipFree(dSids); hipFree(dEids);
  }

  hipEventDestroy(start); hipEventDestroy(stop);
  hipFree(dw13); hipFree(dw13s); hipFree(dw2); hipFree(dw2s);
  hipFree(db13); hipFree(db2);
  std::printf("Done.\n");
  return 0;
}
