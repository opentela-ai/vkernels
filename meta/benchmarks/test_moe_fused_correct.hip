// meta/benchmarks/test_moe_fused_correct.hip
//
// End-to-end correctness test for the fused MXFP4 MoE HIP kernel against the
// CPU reference.  Uses random weights (packed E2M1 + ue8m0), random bias,
// SwiGLU clamp limit, and multi-expert routing with padding.

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
  // deterministic pseudo-random in [-1, 1]
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

// --- fp4 E2M1 pack ---
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

float bf2f(uint16_t v) { return bits2f((uint32_t)v << 16); }

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s\n", p.name);

  // Config: moderate size, exercises multiple M/N/K blocks, padding, and
  // multi-expert routing.
  constexpr int M = 8, hidden = 256, ispp = 128, top_k = 2;
  constexpr int E = 4, BLOCK_M = 16, group_size = 32;
  constexpr float LIMIT = 4.0f;

  int N = M * top_k;
  // Upper bound for per-expert-padded EM: every expert gets its own block.
  int EM_max = E * BLOCK_M;
  (void)N;

  // --- Host data ---
  std::vector<uint16_t> hA(M * hidden);
  for (int i = 0; i < M * hidden; ++i) hA[i] = f2bf(rnd(1, i) * 0.1f);

  int w13_sz = E * 2 * ispp * (hidden / 2);
  int w13s_sz = E * 2 * ispp * (hidden / group_size);
  int w2_sz = E * hidden * (ispp / 2);
  int w2s_sz = E * hidden * (ispp / group_size);

  std::vector<uint8_t> hw13(w13_sz), hw13s(w13s_sz);
  std::vector<uint8_t> hw2(w2_sz), hw2s(w2s_sz);
  for (int i = 0; i < w13_sz / 2; ++i) {
    float v0 = rnd(2, 2 * i) * 0.5f;
    float v1 = rnd(3, 2 * i + 1) * 0.5f;
    hw13[2 * i] = e2m1_nibble(v0) | (e2m1_nibble(v1) << 4);
    hw13[2 * i + 1] = e2m1_nibble(rnd(4, i) * 0.5f) |
                      (e2m1_nibble(rnd(5, i) * 0.5f) << 4);
  }
  for (int i = 0; i < w2_sz / 2; ++i) {
    hw2[2 * i] = e2m1_nibble(rnd(6, 2 * i) * 0.5f) |
                 (e2m1_nibble(rnd(7, 2 * i + 1) * 0.5f) << 4);
    hw2[2 * i + 1] = e2m1_nibble(rnd(8, i) * 0.5f) |
                     (e2m1_nibble(rnd(9, i) * 0.5f) << 4);
  }
  // ue8m0 scales: near 127 (2^0 = 1.0) — realistic range 0.25..4.0
  for (int i = 0; i < w13s_sz; ++i) hw13s[i] = (uint8_t)(125 + (i % 5));
  for (int i = 0; i < w2s_sz; ++i) hw2s[i] = (uint8_t)(125 + (i % 5));

  std::vector<float> hb13(E * 2 * ispp), hb2(E * hidden);
  for (int i = 0; i < E * 2 * ispp; ++i) hb13[i] = rnd(12, i) * 0.3f;
  for (int i = 0; i < E * hidden; ++i) hb2[i] = rnd(13, i) * 0.3f;

  // Routing: token i → experts (i % E) and ((i+1) % E)
  std::vector<int32_t> h_topk_ids(M * top_k);
  std::vector<float> h_topk_w(M * top_k);
  for (int i = 0; i < M; ++i) {
    h_topk_ids[i * top_k + 0] = i % E;
    h_topk_ids[i * top_k + 1] = (i + 1) % E;
    h_topk_w[i * top_k + 0] = 0.7f;
    h_topk_w[i * top_k + 1] = 0.3f;
  }

  std::vector<int32_t> h_sids(EM_max), h_eids(EM_max / BLOCK_M);
  int EM_p = vkernels::kernels::moe_align_block_size(
      h_topk_ids.data(), M, top_k, BLOCK_M, E, h_sids.data(), h_eids.data());
  std::printf("M=%d hidden=%d ispp=%d top_k=%d E=%d EM_padded=%d\n",
              M, hidden, ispp, top_k, E, EM_p);

  std::vector<float> h_tws(EM_p, 0.0f);
  for (int i = 0; i < EM_p; ++i) {
    if (h_sids[i] < M * top_k) h_tws[i] = h_topk_w[h_sids[i]];
  }

  // --- CPU reference ---
  std::vector<uint16_t> act_cpu(EM_p * ispp, 0);
  std::vector<float> out_cpu(M * hidden, 0.0f);
  vkernels::kernels::fused_moe_mxfp4_cpu(
      hA.data(), hw13.data(), hw13s.data(), hw2.data(), hw2s.data(),
      h_sids.data(), h_tws.data(), h_eids.data(), act_cpu.data(), out_cpu.data(),
      M, hidden, ispp, top_k, EM_p, group_size, LIMIT, hb13.data(), hb2.data());

  // --- Device ---
  uint16_t *dA, *dAct;
  uint8_t *dw13, *dw13s, *dw2, *dw2s;
  float *dOut, *dTw, *db13, *db2;
  int32_t *dTids, *dSids, *dEids;
  CK(hipMalloc(&dA, M * hidden * 2), "A");
  CK(hipMalloc(&dw13, w13_sz), "w13");
  CK(hipMalloc(&dw13s, w13s_sz), "w13s");
  CK(hipMalloc(&dw2, w2_sz), "w2");
  CK(hipMalloc(&dw2s, w2s_sz), "w2s");
  CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13");
  CK(hipMalloc(&db2, E * hidden * 4), "b2");
  CK(hipMalloc(&dAct, EM_p * ispp * 2), "act");
  CK(hipMalloc(&dOut, M * hidden * 4), "out");
  CK(hipMalloc(&dTw, M * top_k * 4), "tw");
  CK(hipMalloc(&dTids, M * top_k * 4), "tids");
  CK(hipMalloc(&dSids, EM_p * 4), "sids");
  CK(hipMalloc(&dEids, (EM_p / BLOCK_M) * 4), "eids");

  CK(hipMemcpy(dA, hA.data(), M * hidden * 2, hipMemcpyHostToDevice), "cpyA");
  CK(hipMemcpy(dw13, hw13.data(), w13_sz, hipMemcpyHostToDevice), "cpw13");
  CK(hipMemcpy(dw13s, hw13s.data(), w13s_sz, hipMemcpyHostToDevice), "cpw13s");
  CK(hipMemcpy(dw2, hw2.data(), w2_sz, hipMemcpyHostToDevice), "cpw2");
  CK(hipMemcpy(dw2s, hw2s.data(), w2s_sz, hipMemcpyHostToDevice), "cpw2s");
  CK(hipMemcpy(db13, hb13.data(), E * 2 * ispp * 4, hipMemcpyHostToDevice), "cpb13");
  CK(hipMemcpy(db2, hb2.data(), E * hidden * 4, hipMemcpyHostToDevice), "cpb2");
  CK(hipMemcpy(dTw, h_topk_w.data(), M * top_k * 4, hipMemcpyHostToDevice), "cptw");
  CK(hipMemcpy(dTids, h_topk_ids.data(), M * top_k * 4, hipMemcpyHostToDevice), "cptids");
  CK(hipMemcpy(dSids, h_sids.data(), EM_p * 4, hipMemcpyHostToDevice), "cpsids");
  CK(hipMemcpy(dEids, h_eids.data(), (EM_p / BLOCK_M) * 4, hipMemcpyHostToDevice), "cpeids");
  CK(hipMemset(dOut, 0, M * hidden * 4), "zOut");
  CK(hipMemset(dAct, 0, EM_p * ispp * 2), "zAct");

  vkernels::kernels::hip::fused_moe_mxfp4(
      dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
      M, hidden, ispp, top_k, dSids, dEids, EM_p, LIMIT, db13, db2);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<uint16_t> act_gpu(EM_p * ispp);
  std::vector<float> out_gpu(M * hidden);
  CK(hipMemcpy(act_gpu.data(), dAct, EM_p * ispp * 2, hipMemcpyDeviceToHost), "rdAct");
  CK(hipMemcpy(out_gpu.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut");

  // --- Compare ---
  double max_abs_out = 0, max_rel_out = 0;
  for (int i = 0; i < M * hidden; ++i) {
    double err = std::fabs((double)out_gpu[i] - (double)out_cpu[i]);
    if (err > max_abs_out) max_abs_out = err;
    double d = std::fmax(std::fabs((double)out_cpu[i]), 1.0);
    if (err / d > max_rel_out) max_rel_out = err / d;
  }
  double max_abs_act = 0, max_rel_act = 0;
  for (int i = 0; i < EM_p * ispp; ++i) {
    double err = std::fabs((double)act_gpu[i] - (double)act_cpu[i]);
    if (err > max_abs_act) max_abs_act = err;
    double d = std::fmax(std::fabs((double)act_cpu[i]), 1.0);
    if (err / d > max_rel_act) max_rel_act = err / d;
  }

  std::printf("OUT: max_abs=%.4f max_rel=%.6f\n", max_abs_out, max_rel_out);
  std::printf("ACT: max_abs=%.4f max_rel=%.6f\n", max_abs_act, max_rel_act);

  // bf16 intermediates + fp32 accumulators: allow small relative tolerance
  bool pass = max_rel_out < 0.02 && max_rel_act < 0.02;
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  hipFree(dA); hipFree(dw13); hipFree(dw13s); hipFree(dw2); hipFree(dw2s);
  hipFree(db13); hipFree(db2); hipFree(dAct); hipFree(dOut);
  hipFree(dTw); hipFree(dTids); hipFree(dSids); hipFree(dEids);
  return pass ? 0 : 1;
}
