// meta/benchmarks/bench_moe_fused.hip
//
// End-to-end MXFP4 fused MoE benchmark on AMD gfx942/gfx90a.
// Sweeps decode-regime batch sizes (M = 1..48), verifies correctness against
// the CPU reference, and reports per-M latency + effective TFLOP/s.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
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

// Repack a per-expert [E][2][N][K] uint8 slab to K-major [E][2][K][N]
// (issue #41, item 4): swaps the inner two dims so the N (column)
// dimension becomes contiguous for coalesced 16-byte gate/up reads.
static void repack_kmaj(const uint8_t* src, uint8_t* dst,
                        int E, int n_fast, int k_slow) {
  for (int e = 0; e < E; ++e)
    for (int g = 0; g < 2; ++g) {
      const uint8_t* s = src + ((size_t)e * 2 + g) * n_fast * k_slow;
      uint8_t* d = dst + ((size_t)e * 2 + g) * n_fast * k_slow;
      for (int n = 0; n < n_fast; ++n)
        for (int k = 0; k < k_slow; ++k)
          d[(size_t)k * n_fast + n] = s[(size_t)n * k_slow + k];
    }
}

struct BenchResult { double median_us; int iters; double batch_us; };

static BenchResult bench_kernel_us(const std::function<void()>& launch,
                                   hipEvent_t start, hipEvent_t stop,
                                   int warmup = 20, int max_iters = 200) {
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
    iters *= 2;
  }
  // Batched steady-state latency (issue #41): launch the kernel back-to-back
  // with NO per-launch sync, then a single hipDeviceSynchronize.  This
  // amortises all event/host overhead and reveals the true kernel cost,
  // which for tiny M is otherwise hidden by ~5 us of hipEvent overhead.
  const int bn = 100;
  auto bt0 = std::chrono::steady_clock::now();
  for (int i = 0; i < bn; ++i) launch();
  (void)hipDeviceSynchronize();
  double bms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - bt0).count();
  return {static_cast<double>(times[times.size() / 2]),
          static_cast<int>(times.size()), bms * 1000.0 / bn};
}

int main(int argc, char** argv) {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (%s)\n\n", p.name, p.gcnArchName);

  const bool situ = (argc > 1) && std::strcmp(argv[1], "situ") == 0;
  const int activation = situ ? vkernels::kernels::kSiTU : vkernels::kernels::kSwiGLU;
  const float beta = 4.0f, linear_beta = 25.0f;

  // The MI300A enters a low-power state between SLURM jobs.  The first
  // ~50 kernel launches are 50-80x slower while the GPU clock ramps up,
  // so the bench uses warmup=50 and ramps iters 10..200 (issue #41).

  // CLI shape overrides (issue #41 K3 shapes: E=256 hidden=7168 ispp=33792
  // top_k=16).  Defaults match the original harness (E=256 h4096 i512 k6).
  //   --E N --hidden N --ispp N --topk N
  //   --ms 1,2,4,8,16,32,48   (decode M sweep)
  //   --dummy                fill weights with hipMemset (skip a ~90 GB
  //                          realistic host init + hipMemcpy; any data is
  //                          fine for a latency benchmark, and the CPU
  //                          oracle is impractical at ispp=33792).
  //   --no-cpu               skip the CPU-reference correctness check.
  int E = 256, hidden = 4096, ispp = 512, top_k = 6;
  std::vector<int> Ms = {1, 2, 4, 8, 16, 32, 48};
  bool dummy = false, no_cpu = false, kmajor = false;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](int& j) -> long {
      if (j + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[j]); std::exit(2); }
      return std::strtol(argv[++j], nullptr, 10);
    };
    std::string a = argv[i];
    if      (a == "situ")    { /* activation flag, handled above */ }
    else if (a == "--E")     E = (int)next(i);
    else if (a == "--hidden") hidden = (int)next(i);
    else if (a == "--ispp")  ispp = (int)next(i);
    else if (a == "--topk")  top_k = (int)next(i);
    else if (a == "--ms") {
      Ms.clear();
      for (char* p = argv[++i]; ; ++p) {
        char* endp; long v = std::strtol(p, &endp, 10);
        if (endp == p) break; Ms.push_back((int)v); p = endp;
        if (*p != ',') break;
      }
    }
    else if (a == "--dummy") dummy = true;
    else if (a == "--no-cpu") no_cpu = true;
    else if (a == "--kmajor") kmajor = true;
    else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); std::exit(2); }
  }
  // The CPU oracle at large ispp is impractical (seconds→minutes per call).
  // Keep it only for the original small harness shapes unless forced.
  if (ispp > 4096) no_cpu = true;

  const int BLOCK_M = 16, group_size = 32;
  const float LIMIT = 10.0f;

  const int w13_sz = E * 2 * ispp * (hidden / 2);
  const int w13s_sz = E * 2 * ispp * (hidden / group_size);
  const int w2_sz = E * hidden * (ispp / 2);
  const int w2s_sz = E * hidden * (ispp / group_size);

  std::vector<uint8_t> hw13, hw13s, hw2, hw2s;
  std::vector<float> hb13, hb2;
  if (!dummy) {
    hw13.resize(w13_sz); hw13s.resize(w13s_sz, 127);
    hw2.resize(w2_sz);  hw2s.resize(w2s_sz, 127);
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
    hb13.resize(E * 2 * ispp, 0.0f); hb2.resize(E * hidden, 0.0f);
  }

  // --- Device buffers ---
  uint8_t *dw13, *dw13s, *dw2, *dw2s;
  float *db13, *db2;
  CK(hipMalloc(&dw13, w13_sz), "w13");
  CK(hipMalloc(&dw13s, w13s_sz), "w13s");
  CK(hipMalloc(&dw2, w2_sz), "w2");
  CK(hipMalloc(&dw2s, w2s_sz), "w2s");
  CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13");
  CK(hipMalloc(&db2, E * hidden * 4), "b2");
  // K-major gate/up weights (issue #41, item 4): repack w13 and w13s
  // from N-major [E,2,ispp,K] to K-major [E,2,K,ispp] so the 16 columns
  // owned by each decode block read one coalesced cache line per K-step.
  // w2/w2s stay N-major (the down GEMM is not the low-M bottleneck).
  std::vector<uint8_t> hw13_km, hw13s_km;
  if (kmajor && !dummy) {
    hw13_km.resize(w13_sz);
    hw13s_km.resize(w13s_sz);
    repack_kmaj(hw13.data(), hw13_km.data(), E, ispp, hidden / 2);
    repack_kmaj(hw13s.data(), hw13s_km.data(), E, ispp, hidden / group_size);
  }
  const uint8_t* w13_h = (kmajor && !dummy) ? hw13_km.data() : hw13.data();
  const uint8_t* w13s_h = (kmajor && !dummy) ? hw13s_km.data() : hw13s.data();
  if (dummy) {
    CK(hipMemset(dw13, 0x11, w13_sz), "zw13");
    CK(hipMemset(dw13s, 127, w13s_sz), "zw13s");
    CK(hipMemset(dw2, 0x22, w2_sz), "zw2");
    CK(hipMemset(dw2s, 127, w2s_sz), "zw2s");
    CK(hipMemset(db13, 0, E * 2 * ispp * 4), "zb13");
    CK(hipMemset(db2, 0, E * hidden * 4), "zb2");
  } else {
    CK(hipMemcpy(dw13, w13_h, w13_sz, hipMemcpyHostToDevice), "cpw13");
    CK(hipMemcpy(dw13s, w13s_h, w13s_sz, hipMemcpyHostToDevice), "cpw13s");
    CK(hipMemcpy(dw2, hw2.data(), w2_sz, hipMemcpyHostToDevice), "cpw2");
    CK(hipMemcpy(dw2s, hw2s.data(), w2s_sz, hipMemcpyHostToDevice), "cpw2s");
    CK(hipMemcpy(db13, hb13.data(), E * 2 * ispp * 4, hipMemcpyHostToDevice), "cpb13");
    CK(hipMemcpy(db2, hb2.data(), E * hidden * 4, hipMemcpyHostToDevice), "cpb2");
  }

  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);

  // ----------------------------------------------------------------------
  // Heavy GPU clock warmup (issue #41).  MI300A can enter a low-power
  // state between SLURM jobs; run the heaviest decode kernel (largest M in
  // the sweep) a bounded number of times with a single hipDeviceSynchronize
  // so ~250 ms of sustained compute (64 x ~4 us at M=48) ramps the SM
  // clock before any timing below.  Per-M warmups alone suffice in practice
  // (measured M=1 is identical with/without this block), but this is cheap
  // insurance.  A wall-clock loop is avoided because it floods the launch
  // queue and leaves a multi-second hipDeviceSynchronize tail.
  // ----------------------------------------------------------------------
  {
    int wm = Ms.back();
    int wn = wm * top_k;
    std::vector<int32_t> wt(wn), ws(E * BLOCK_M), we(E);
    for (int i = 0; i < wn; ++i) wt[i] = (i * 17 + 7) % E;
    int wem = vkernels::kernels::moe_align_block_size(
        wt.data(), wm, top_k, BLOCK_M, E, ws.data(), we.data());
    uint16_t *wA, *wAct; float *wO, *wTw; int32_t *wT, *wS, *wE2;
    CK(hipMalloc(&wA, wm * hidden * 2), "wa");
    CK(hipMalloc(&wAct, wem * ispp * 2), "wact");
    CK(hipMalloc(&wO, wm * hidden * 4), "wo");
    CK(hipMalloc(&wTw, wn * 4), "wtw");
    CK(hipMalloc(&wT, wn * 4), "wt");
    CK(hipMalloc(&wS, wem * 4), "ws");
    CK(hipMalloc(&wE2, (wem / BLOCK_M) * 4), "we2");
    CK(hipMemcpy(wS, ws.data(), wem * 4, hipMemcpyHostToDevice), "cpws");
    CK(hipMemcpy(wE2, we.data(), (wem / BLOCK_M) * 4, hipMemcpyHostToDevice), "cpwe");
    CK(hipMemcpy(wT, wt.data(), wn * 4, hipMemcpyHostToDevice), "cpwt");
    CK(hipMemset(wA, 0, wm * hidden * 2), "za");
    CK(hipMemset(wAct, 127, wem * ispp * 2), "zact");
    CK(hipMemset(wO, 0, wm * hidden * 4), "zo");
    CK(hipMemset(wTw, 0x3c, wn * 4), "ztw");
    auto wlaunch = [&]() {
      vkernels::kernels::hip::fused_moe_mxfp4(
          wA, dw13, dw13s, dw2, dw2s, wT, wTw, wAct, wO,
          wm, hidden, ispp, top_k, wS, wE2, wem, LIMIT, activation, beta,
          linear_beta, db13, db2, 16, kmajor);
    };
    for (int i = 0; i < 64; ++i) wlaunch();
    CK(hipDeviceSynchronize(), "wsync");
    CK(hipFree(wA), "fa"); CK(hipFree(wAct), "fact"); CK(hipFree(wO), "fo");
    CK(hipFree(wTw), "ftw"); CK(hipFree(wT), "ft"); CK(hipFree(wS), "fs");
    CK(hipFree(wE2), "fe2");
  }

  std::printf("E=%d hidden=%d ispp=%d top_k=%d activation=%s\n", E, hidden, ispp,
              top_k, situ ? "situ" : "swiglu");
  std::printf("%6s %8s %10s %10s %10s %9s %9s\n", "M", "EM",
              "evt(us)", "bch(us)", "TFLOP/s", "cpu-rel", "status");

  // Persistent scratch for act_scratch is allocated per-M inside the
  // sweep below; the host launcher now performs no sorted_w allocation
  // (the combine reads topk_w[flat] directly, issue #41 items 1+2).

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
          M, hidden, ispp, top_k, dSids, dEids, EM, LIMIT, activation, beta,
          linear_beta, db13, db2, 16, kmajor);
    };

    // Correctness check vs CPU reference (only for M <= 8, and only when
    // real weights are available).  cpu_rel < 0 means "not checked".
    double cpu_rel = -1.0;
    if (M <= 8 && !no_cpu && !dummy) {
      std::vector<float> h_tws(EM, 0.0f);
      for (int i = 0; i < EM; ++i)
        if (h_sids[i] < M * top_k) h_tws[i] = h_tw[h_sids[i]];
      std::vector<uint16_t> act_cpu(EM * ispp, 0);
      std::vector<float> out_cpu(M * hidden, 0.0f);
      vkernels::kernels::fused_moe_mxfp4_cpu(
          hA.data(), hw13.data(), hw13s.data(), hw2.data(), hw2s.data(),
          h_sids.data(), h_tws.data(), h_eids.data(), act_cpu.data(), out_cpu.data(),
          M, hidden, ispp, top_k, EM, group_size, LIMIT, activation, beta,
          linear_beta, hb13.data(), hb2.data());
      launch();
      CK(hipDeviceSynchronize(), "sync");
      std::vector<float> out_gpu(M * hidden);
      CK(hipMemcpy(out_gpu.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut");
      double max_rel = 0;
      for (int i = 0; i < M * hidden; ++i) {
        double e = std::fabs((double)out_gpu[i] - (double)out_cpu[i]);
        max_rel = std::fmax(max_rel, e / std::fmax(std::fabs((double)out_cpu[i]), 1.0));
      }
      cpu_rel = max_rel;
    }

    auto r = bench_kernel_us(launch, start, stop);
    double total_flop = 6.0 * M * (double)ispp * hidden;
    double tflops = total_flop / (r.median_us / 1e6) / 1e12;

    if (cpu_rel < 0.0) {
      std::printf("%6d %8d %10.1f %10.1f %9.3f %9s %9s\n", M, EM,
                  r.median_us, r.batch_us, tflops, "-", "skip");
    } else {
      std::printf("%6d %8d %10.1f %10.1f %9.3f %9.3g %9s\n", M, EM,
                  r.median_us, r.batch_us, tflops, cpu_rel,
                  cpu_rel < 0.05 ? "ok" : "MISMATCH");
    }

    hipFree(dA); hipFree(dAct); hipFree(dOut);
    hipFree(dTw); hipFree(dTids); hipFree(dSids); hipFree(dEids);
  }

  hipEventDestroy(start); hipEventDestroy(stop);
  hipFree(dw13); hipFree(dw13s); hipFree(dw2); hipFree(dw2s);
  hipFree(db13); hipFree(db2);
  std::printf("Done.\n");
  return 0;
}
