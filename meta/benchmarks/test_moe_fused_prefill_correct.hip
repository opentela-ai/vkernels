// meta/benchmarks/test_moe_fused_prefill_correct.hip
//
// Correctness test for the PREFILL tile config (BM=64, BN=64, 256 threads)
// of the fused MXFP4 MoE HIP kernel, checked against the CPU reference.
//
// The CPU oracle always uses the decode tiling (BLOCK_M=16); the GPU prefill
// path aligns with block_size=64.  `out [M, hidden]` is block-size
// independent, so we compare the final output only (act_scratch has different
// per-expert padding and is not row-comparable across the two alignments).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/moe_fused.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

uint32_t f2bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
float bits2f(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

uint8_t e2m1_nibble(float v) {
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
  return neg ? (uint8_t)(best_n | 0x8) : best_n;
}

uint16_t f2bf(float v) {
  uint32_t b = f2bits(v);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}

// Run one prefill correctness case. Returns true on PASS.
bool run_case(const char* name, int M, int hidden, int ispp, int top_k, int E,
             int activation) {
  constexpr int CPU_BLOCK_M = 16;
  constexpr int GPU_BLOCK_M = 64;
  constexpr int group_size = 32;
  constexpr float LIMIT = 4.0f;

  const int N = M * top_k;
  const int EM_gpu_max = N + E * GPU_BLOCK_M;  // safe upper bound
  const int EM_cpu_max = N + E * CPU_BLOCK_M;

  // --- Host data ---
  std::vector<uint16_t> hA(M * hidden);
  for (int i = 0; i < M * hidden; ++i) hA[i] = f2bf(rnd(1, i) * 0.1f);

  const int w13_sz = E * 2 * ispp * (hidden / 2);
  const int w13s_sz = E * 2 * ispp * (hidden / group_size);
  const int w2_sz = E * hidden * (ispp / 2);
  const int w2s_sz = E * hidden * (ispp / group_size);

  std::vector<uint8_t> hw13(w13_sz), hw13s(w13s_sz);
  std::vector<uint8_t> hw2(w2_sz), hw2s(w2s_sz);
  for (int i = 0; i < w13_sz / 2; ++i) {
    hw13[2 * i] = e2m1_nibble(rnd(2, 2 * i) * 0.5f) |
                  (e2m1_nibble(rnd(3, 2 * i + 1) * 0.5f) << 4);
    hw13[2 * i + 1] = e2m1_nibble(rnd(4, i) * 0.5f) |
                      (e2m1_nibble(rnd(5, i) * 0.5f) << 4);
  }
  for (int i = 0; i < w2_sz / 2; ++i) {
    hw2[2 * i] = e2m1_nibble(rnd(6, 2 * i) * 0.5f) |
                 (e2m1_nibble(rnd(7, 2 * i + 1) * 0.5f) << 4);
    hw2[2 * i + 1] = e2m1_nibble(rnd(8, i) * 0.5f) |
                     (e2m1_nibble(rnd(9, i) * 0.5f) << 4);
  }
  for (int i = 0; i < w13s_sz; ++i) hw13s[i] = (uint8_t)(125 + (i % 5));
  for (int i = 0; i < w2s_sz; ++i) hw2s[i] = (uint8_t)(125 + (i % 5));

  std::vector<float> hb13(E * 2 * ispp), hb2(E * hidden);
  for (int i = 0; i < E * 2 * ispp; ++i) hb13[i] = rnd(12, i) * 0.3f;
  for (int i = 0; i < E * hidden; ++i) hb2[i] = rnd(13, i) * 0.3f;

  // Routing: token i -> experts (i % E) and ((i + 1) % E).
  std::vector<int32_t> h_topk_ids(N);
  std::vector<float> h_topk_w(N);
  for (int i = 0; i < M; ++i) {
    h_topk_ids[i * top_k + 0] = i % E;
    h_topk_ids[i * top_k + 1] = (i + 1) % E;
    h_topk_w[i * top_k + 0] = 0.7f;
    h_topk_w[i * top_k + 1] = 0.3f;
  }

  // --- GPU alignment (block_size=64) ---
  std::vector<int32_t> h_sids_gpu(EM_gpu_max), h_eids_gpu(EM_gpu_max / GPU_BLOCK_M);
  int EM_gpu = vkernels::kernels::moe_align_block_size(
      h_topk_ids.data(), M, top_k, GPU_BLOCK_M, E,
      h_sids_gpu.data(), h_eids_gpu.data());

  // --- CPU oracle (block_size=16) ---
  std::vector<int32_t> h_sids_cpu(EM_cpu_max), h_eids_cpu(EM_cpu_max / CPU_BLOCK_M);
  int EM_cpu = vkernels::kernels::moe_align_block_size(
      h_topk_ids.data(), M, top_k, CPU_BLOCK_M, E,
      h_sids_cpu.data(), h_eids_cpu.data());
  std::vector<float> h_tws_cpu(EM_cpu, 0.0f);
  for (int i = 0; i < EM_cpu; ++i)
    if (h_sids_cpu[i] < N) h_tws_cpu[i] = h_topk_w[h_sids_cpu[i]];

  std::vector<uint16_t> act_cpu(EM_cpu * ispp, 0);
  std::vector<float> out_cpu(M * hidden, 0.0f);
  vkernels::kernels::fused_moe_mxfp4_cpu(
      hA.data(), hw13.data(), hw13s.data(), hw2.data(), hw2s.data(),
      h_sids_cpu.data(), h_tws_cpu.data(), h_eids_cpu.data(),
      act_cpu.data(), out_cpu.data(),
      M, hidden, ispp, top_k, EM_cpu, group_size, LIMIT, activation, 4.0f, 25.0f,
      hb13.data(), hb2.data());

  // --- Device ---
  uint16_t *dA, *dAct;
  uint8_t *dw13, *dw13s, *dw2, *dw2s;
  float *dOut, *dTw, *db13, *db2, *dSw;
  int32_t *dTids, *dSids, *dEids;
  CK(hipMalloc(&dA, M * hidden * 2), "A");
  CK(hipMalloc(&dw13, w13_sz), "w13");
  CK(hipMalloc(&dw13s, w13s_sz), "w13s");
  CK(hipMalloc(&dw2, w2_sz), "w2");
  CK(hipMalloc(&dw2s, w2s_sz), "w2s");
  CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13");
  CK(hipMalloc(&db2, E * hidden * 4), "b2");
  CK(hipMalloc(&dAct, EM_gpu * ispp * 2), "act");
  CK(hipMalloc(&dSw, EM_gpu * 4), "sw");
  CK(hipMalloc(&dOut, M * hidden * 4), "out");
  CK(hipMalloc(&dTw, N * 4), "tw");
  CK(hipMalloc(&dTids, N * 4), "tids");
  CK(hipMalloc(&dSids, EM_gpu * 4), "sids");
  CK(hipMalloc(&dEids, (EM_gpu / GPU_BLOCK_M) * 4), "eids");

  CK(hipMemcpy(dA, hA.data(), M * hidden * 2, hipMemcpyHostToDevice), "cpyA");
  CK(hipMemcpy(dw13, hw13.data(), w13_sz, hipMemcpyHostToDevice), "cpw13");
  CK(hipMemcpy(dw13s, hw13s.data(), w13s_sz, hipMemcpyHostToDevice), "cpw13s");
  CK(hipMemcpy(dw2, hw2.data(), w2_sz, hipMemcpyHostToDevice), "cpw2");
  CK(hipMemcpy(dw2s, hw2s.data(), w2s_sz, hipMemcpyHostToDevice), "cpw2s");
  CK(hipMemcpy(db13, hb13.data(), E * 2 * ispp * 4, hipMemcpyHostToDevice), "cpb13");
  CK(hipMemcpy(db2, hb2.data(), E * hidden * 4, hipMemcpyHostToDevice), "cpb2");
  CK(hipMemcpy(dTw, h_topk_w.data(), N * 4, hipMemcpyHostToDevice), "cptw");
  CK(hipMemcpy(dTids, h_topk_ids.data(), N * 4, hipMemcpyHostToDevice), "cptids");
  CK(hipMemcpy(dSids, h_sids_gpu.data(), EM_gpu * 4, hipMemcpyHostToDevice), "cpsids");
  CK(hipMemcpy(dEids, h_eids_gpu.data(), (EM_gpu / GPU_BLOCK_M) * 4, hipMemcpyHostToDevice), "cpeids");
  CK(hipMemset(dOut, 0, M * hidden * 4), "zOut");

  vkernels::kernels::hip::fused_moe_mxfp4(
      dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dSw, dOut,
      M, hidden, ispp, top_k, dSids, dEids, EM_gpu, LIMIT, activation, 4.0f,
      25.0f, db13, db2,
      GPU_BLOCK_M);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<float> out_gpu(M * hidden);
  CK(hipMemcpy(out_gpu.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut");

  double max_rel = 0;
  for (int i = 0; i < M * hidden; ++i) {
    double e = std::fabs((double)out_gpu[i] - (double)out_cpu[i]);
    double d = std::fmax(std::fabs((double)out_cpu[i]), 1.0);
    if (e / d > max_rel) max_rel = e / d;
  }

  std::printf("%-14s M=%d hidden=%d ispp=%d top_k=%d E=%d EM_gpu=%d EM_cpu=%d\n",
              name, M, hidden, ispp, top_k, E, EM_gpu, EM_cpu);
  std::printf("  OUT max_rel=%.6f %s\n", max_rel, max_rel < 0.02 ? "PASS" : "FAIL");

  hipFree(dA); hipFree(dw13); hipFree(dw13s); hipFree(dw2); hipFree(dw2s);
  hipFree(db13); hipFree(db2); hipFree(dAct); hipFree(dSw); hipFree(dOut);
  hipFree(dTw); hipFree(dTids); hipFree(dSids); hipFree(dEids);
  return max_rel < 0.02;
}

}  // namespace

int main(int argc, char** argv) {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s\n", p.name);

  const bool situ = (argc > 1) && std::strcmp(argv[1], "situ") == 0;
  const int activation = situ ? vkernels::kernels::kSiTU : vkernels::kernels::kSwiGLU;
  std::printf("activation=%s\n", situ ? "situ" : "swiglu");

  // Clean prefill: 64 tokens/expert -> one full 64-row block, no padding.
  bool ok1 = run_case("clean", 128, 512, 256, 2, 4, activation);
  // Padded prefill: 40 tokens/expert -> 24 padding rows per expert.
  bool ok2 = run_case("padded", 80, 512, 256, 2, 4, activation);

  std::printf("%s\n", (ok1 && ok2) ? "PASS" : "FAIL");
  return (ok1 && ok2) ? 0 : 1;
}
