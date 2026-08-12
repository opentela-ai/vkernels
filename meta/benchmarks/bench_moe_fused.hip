// meta/benchmarks/bench_moe_fused.hip
//
// End-to-end MXFP4 fused MoE benchmark on AMD gfx942/gfx90a.
// Measures the two-stage kernel (gate_up + SwiGLU → down + combine)
// vs xkernels torch reference loop on the same problem sizes.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/moe.hpp"
#include "vkernels/util/config.hpp"

static void check_hip(hipError_t err, const char* ctx) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", ctx, hipGetErrorString(err));
    std::exit(1);
  }
}

// --- benchmarking helpers (same as bench_moe.hip) ---
struct BenchResult { double min_us, median_us, mean_us, std_us; int iters; };

static BenchResult bench_kernel_us(const std::function<void()>& launch,
                                   hipEvent_t start, hipEvent_t stop,
                                   int warmup = 5, int max_iters = 500) {
  (void)hipDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) launch();
  std::vector<float> times;
  int iters = 20;
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
    double sum = 0, sum2 = 0;
    for (double t : times) { sum += t; sum2 += t * t; }
    double mean = sum / iters;
    double var = sum2 / iters - mean * mean;
    double stddev = std::sqrt(std::max(0.0, var));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {static_cast<double>(times[0]), static_cast<double>(times[iters / 2]),
              mean, stddev, iters};
    }
    iters *= 2;
  }
  return {};
}

// --- Synthetic MXFP4 weights ---
// Pack a float value → E2M1 nibble (approximate).
static uint8_t e2m1_nibble(float v) {
  bool neg = v < 0;
  float af = std::fabs(v);
  if (af == 0.0f || std::isnan(af)) return neg ? 0x8 : 0x0;
  if (std::isinf(af)) return neg ? 0xE : 0x6;
  float vals[5] = {0.25f, 1.0f, 1.5f, 2.0f, 3.0f};
  uint8_t nibs[5] = {1, 2, 3, 4, 5};
  float best_d = std::fabs(af - vals[0]);
  uint8_t best_n = nibs[0];
  for (int i = 1; i < 5; ++i) {
    float d = std::fabs(af - vals[i]);
    if (d < best_d) { best_d = d; best_n = nibs[i]; }
  }
  return neg ? (best_n | 0x8) : best_n;
}

// Pack two float values → one packed byte (low nibble = v0, high = v1).
static uint8_t pack_e2m1(float v0, float v1) {
  return e2m1_nibble(v0) | (e2m1_nibble(v1) << 4);
}

// ---------------------------------------------------------------------------
int main() {
  hipDeviceProp_t p;
  check_hip(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (%s)\n\n", p.name, p.gcnArchName);

  // Benchmark config — small decode-regime problem to match xkernels V4
  // but scaled down since we have E=256 experts and we want to test
  // single-expert throughput.
  constexpr int E = 256, hidden = 4096, ispp = 512, top_k = 6;
  constexpr int M = 8;  // small batch
  constexpr int group_size = 32;
  constexpr int BLOCK_M = 16;

  int N = M * top_k;  // 48
  // Pad to BLOCK_M multiple
  int EM = ((N + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;  // 48 → 48 (already multiple)

  // --- Allocate and fill host buffers ---
  // A [M, hidden] bf16
  std::vector<uint16_t> h_A(M * hidden);
  for (auto& v : h_A) { uint32_t b; float f=1.f; std::memcpy(&b,&f,4); v=(uint16_t)((b+0x7FFFu+((b>>16)&1))>>16); }

  // w13 [E, 2*ispp, hidden/2] packed uint8
  int w13_sz = E * 2 * ispp * (hidden / 2);
  std::vector<uint8_t> h_w13(w13_sz);
  std::vector<uint8_t> h_w13_scale(E * 2 * ispp * (hidden / group_size), 127);
  uint8_t one_byte = pack_e2m1(1.0f, 1.0f);
  for (auto& v : h_w13) v = one_byte;

  // w2 [E, hidden, ispp/2] packed uint8
  int w2_sz = E * hidden * (ispp / 2);
  std::vector<uint8_t> h_w2(w2_sz);
  std::vector<uint8_t> h_w2_scale(E * hidden * (ispp / group_size), 127);
  for (auto& v : h_w2) v = one_byte;

  // b13 [E, 2*ispp], b2 [E, hidden]
  std::vector<float> h_b13(E * 2 * ispp, 0.0f);
  std::vector<float> h_b2(E * hidden, 0.0f);

  // topk_ids [M, top_k], topk_w [M, top_k]
  std::vector<int32_t> h_topk_ids(M * top_k);
  std::vector<float> h_topk_w(M * top_k, 1.0f);
  // Route all tokens to expert 0 for throughput measurement
  for (auto& v : h_topk_ids) v = 0;

  // moe_align_block_size
  std::vector<int32_t> h_sorted_ids(EM), h_expert_ids(EM / BLOCK_M);
  int EM_padded = vkernels::kernels::moe_align_block_size(
      h_topk_ids.data(), M, top_k, BLOCK_M, E,
      h_sorted_ids.data(), h_expert_ids.data());
  std::printf("EM=%d padded=%d\n", EM, EM_padded);

  // act scratch [EM, ispp] bf16
  std::vector<uint16_t> h_act(EM * ispp, 0);
  std::vector<float> h_out(M * hidden, 0.0f);

  // --- Device allocations ---
  uint16_t *d_A, *d_act;
  uint8_t *d_w13, *d_w13_scale, *d_w2, *d_w2_scale;
  float *d_out, *d_b13, *d_b2, *d_topk_w;
  int32_t *d_topk_ids, *d_sorted_ids, *d_expert_ids;

  check_hip(hipMalloc(&d_A, M * hidden * 2), "A");
  check_hip(hipMalloc(&d_w13, w13_sz), "w13");
  check_hip(hipMalloc(&d_w13_scale, E * 2 * ispp * (hidden / group_size)), "w13s");
  check_hip(hipMalloc(&d_w2, w2_sz), "w2");
  check_hip(hipMalloc(&d_w2_scale, E * hidden * (ispp / group_size)), "w2s");
  check_hip(hipMalloc(&d_b13, E * 2 * ispp * 4), "b13");
  check_hip(hipMalloc(&d_b2, E * hidden * 4), "b2");
  check_hip(hipMalloc(&d_topk_ids, M * top_k * 4), "topk_ids");
  check_hip(hipMalloc(&d_topk_w, M * top_k * 4), "topk_w");
  check_hip(hipMalloc(&d_sorted_ids, EM * 4), "sid");
  check_hip(hipMalloc(&d_expert_ids, (EM / BLOCK_M) * 4), "eid");
  check_hip(hipMalloc(&d_act, EM * ispp * 2), "act");
  check_hip(hipMalloc(&d_out, M * hidden * 4), "out");

  // Copy to device
  check_hip(hipMemcpy(d_A, h_A.data(), M * hidden * 2, hipMemcpyHostToDevice), "cpyA");
  check_hip(hipMemcpy(d_w13, h_w13.data(), w13_sz, hipMemcpyHostToDevice), "cpyw13");
  check_hip(hipMemcpy(d_w13_scale, h_w13_scale.data(), E * 2 * ispp * (hidden / group_size), hipMemcpyHostToDevice), "cpyw13s");
  check_hip(hipMemcpy(d_w2, h_w2.data(), w2_sz, hipMemcpyHostToDevice), "cpyw2");
  check_hip(hipMemcpy(d_w2_scale, h_w2_scale.data(), E * hidden * (ispp / group_size), hipMemcpyHostToDevice), "cpyw2s");
  check_hip(hipMemcpy(d_topk_ids, h_topk_ids.data(), M * top_k * 4, hipMemcpyHostToDevice), "cpytkid");
  check_hip(hipMemcpy(d_topk_w, h_topk_w.data(), M * top_k * 4, hipMemcpyHostToDevice), "cpytkw");
  check_hip(hipMemcpy(d_sorted_ids, h_sorted_ids.data(), EM * 4, hipMemcpyHostToDevice), "cpysid");
  check_hip(hipMemcpy(d_expert_ids, h_expert_ids.data(), (EM / BLOCK_M) * 4, hipMemcpyHostToDevice), "cpyeid");

  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);

  // --- Benchmark ---
  auto launch = [&]() {
    vkernels::kernels::hip::fused_moe_mxfp4(
        d_A, d_w13, d_w13_scale, d_w2, d_w2_scale,
        d_topk_ids, d_topk_w, d_act, d_out,
        M, hidden, ispp, top_k,
        d_sorted_ids, d_expert_ids, EM,
        10.0f, d_b13, d_b2);
  };

  // Warmup
  for (int i = 0; i < 5; ++i) launch();
  check_hip(hipDeviceSynchronize(), "sync");

  // Compute FLOP for full MoE:
  // Stage 0: gate_up GEMM = 2 * M * 2*ispp * hidden = 4*M*ispp*hidden
  // Stage 1: down GEMM     = 2 * M * hidden * ispp = 2*M*hidden*ispp
  // (but only for routed tokens; here all routes to expert 0)
  // Total = 6 * M * ispp * hidden
  double total_flop = 6.0 * M * static_cast<double>(ispp) * hidden;

  auto r = bench_kernel_us(launch, start, stop, 5, 100);
  double tflops = total_flop / (r.median_us / 1e6) / 1e12;

  std::printf("\n=== Fused MXFP4 MoE (M=%d hidden=%d ispp=%d) ===\n", M, hidden, ispp);
  std::printf("  Time: %.1f us  median  TFLOP/s: %.3f  (raw, incl dequant)\n",
              r.median_us, tflops);
  std::printf("  Effective FLOP (6× M×ispp×hidden): %.1e\n", total_flop);

  // Cleanup
  hipEventDestroy(start); hipEventDestroy(stop);
  for (void* p : {(void*)d_A, (void*)d_w13, (void*)d_w13_scale, (void*)d_w2, (void*)d_w2_scale,
                  (void*)d_b13, (void*)d_b2, (void*)d_topk_ids, (void*)d_topk_w,
                  (void*)d_sorted_ids, (void*)d_expert_ids, (void*)d_act, (void*)d_out})
    hipFree(p);

  std::printf("Done.\n");
  return 0;
}
