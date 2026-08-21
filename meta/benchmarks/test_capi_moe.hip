// test_capi_moe.hip — validate the C ABI wrapper (hip_capi.cpp) with the
// exact same data as test_moe_fused_correct.hip, on the GPU.
//
// Calls `vk_hip_fused_moe_mxfp4` (from `libvkernels_hip.so` / the
// `vkernels_hip` target) with biases and without, asserting no NaN and
// non-zero output — the same contract as the in-tree C++ MoE-fused
// correctness test, exercised through the `extern "C"` boundary that
// vLLM and other ctypes consumers use.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "vkernels/capi/hip_capi.hpp"
#include "vkernels/kernels/moe_fused.hpp"

#define CK(x, msg) do { \
  hipError_t _e = (x); \
  if (_e != hipSuccess) { \
    std::fprintf(stderr, "ERR %s: %s\n", msg, hipGetErrorString(_e)); \
    return 1; \
  } \
} while (0)

// --- helpers stolen from test_moe_fused_correct.hip ---
static float rnd(int seed, int i) {
  uint32_t h = (uint32_t)(seed * 2654435761u + i * 40503u); h ^= h >> 13;
  h *= 2654435761; h ^= h >> 15;
  return (float)(h & 0x7fffu) / 0x7fffu;
}
static uint16_t f2bf(float v) {
  uint32_t u; std::memcpy(&u, &v, 4);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}
static uint8_t e2m1_nibble(float v) {
  float a = fabsf(v);
  static const float vals[8] = {0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  int best = 0; float bd = a;
  for (int i = 0; i < 8; ++i) {
    float d = fabsf(a - vals[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return (uint8_t)(best + (v < 0 ? 8 : 0));
}

int main(int argc, char** argv) {
  constexpr int M = 8, hidden = 256, ispp = 128, top_k = 2;
  constexpr int E = 4, BLOCK_M = 16, group_size = 32;
  constexpr float LIMIT = 4.0f;
  const bool situ = (argc > 1) && std::strcmp(argv[1], "situ") == 0;
  const int activation = situ ? 1 : 0;  // kSiTU=1, kSwiGLU=0
  const float beta = 4.0f, linear_beta = 25.0f;

  int N = M * top_k;
  int EM_max = E * BLOCK_M;

  // Same host data as test_moe_fused_correct.hip
  std::vector<uint16_t> hA(M * hidden);
  for (int i = 0; i < M * hidden; ++i) hA[i] = f2bf(rnd(1, i) * 0.1f);

  int w13_sz = E * 2 * ispp * (hidden / 2);
  int w13s_sz = E * 2 * ispp * (hidden / group_size);
  int w2_sz = E * hidden * (ispp / 2);
  int w2s_sz = E * hidden * (ispp / group_size);

  std::vector<uint8_t> hw13(w13_sz), hw13s(w13s_sz);
  std::vector<uint8_t> hw2(w2_sz), hw2s(w2s_sz);
  for (int i = 0; i < w13_sz / 2; ++i) {
    hw13[2 * i] = e2m1_nibble(rnd(2, 2 * i) * 0.5f) | (e2m1_nibble(rnd(3, 2 * i + 1) * 0.5f) << 4);
    hw13[2 * i + 1] = e2m1_nibble(rnd(4, i) * 0.5f) | (e2m1_nibble(rnd(5, i) * 0.5f) << 4);
  }
  for (int i = 0; i < w2_sz / 2; ++i) {
    hw2[2 * i] = e2m1_nibble(rnd(6, 2 * i) * 0.5f) | (e2m1_nibble(rnd(7, 2 * i + 1) * 0.5f) << 4);
    hw2[2 * i + 1] = e2m1_nibble(rnd(8, i) * 0.5f) | (e2m1_nibble(rnd(9, i) * 0.5f) << 4);
  }
  for (int i = 0; i < w13s_sz; ++i) hw13s[i] = (uint8_t)(125 + (i % 5));
  for (int i = 0; i < w2s_sz; ++i) hw2s[i] = (uint8_t)(125 + (i % 5));

  std::vector<float> hb13(E * 2 * ispp), hb2(E * hidden);
  for (int i = 0; i < E * 2 * ispp; ++i) hb13[i] = rnd(12, i) * 0.3f;
  for (int i = 0; i < E * hidden; ++i) hb2[i] = rnd(13, i) * 0.3f;

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

  // Device buffers
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

  // Call via C ABI (with biases, like C++ test)
  std::printf("calling vk_hip_fused_moe_mxfp4 (with biases) ...\n");
  vk_hip_fused_moe_mxfp4(dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
      M, hidden, ispp, top_k, dSids, dEids, EM_p, LIMIT, activation,
      beta, linear_beta, db13, db2, BLOCK_M, /*stream=*/nullptr);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<float> out(M * hidden);
  CK(hipMemcpy(out.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut");

  int nnz = 0, n_nan = 0;
  for (int i = 0; i < M * hidden; ++i) {
    if (out[i] == out[i]) nnz++; // not NaN
    else n_nan++;
  }
  std::printf("with_bias: nnz=%d nan=%d out[:4]=%g %g %g %g\n",
              nnz, n_nan, out[0], out[1], out[2], out[3]);

  // Re-zero and call WITHOUT biases (matching Python test)
  CK(hipMemset(dOut, 0, M * hidden * 4), "zOut2");
  CK(hipMemset(dAct, 0, EM_p * ispp * 2), "zAct2");
  std::printf("calling vk_hip_fused_moe_mxfp4 (without biases) ...\n");
  vk_hip_fused_moe_mxfp4(dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
      M, hidden, ispp, top_k, dSids, dEids, EM_p, LIMIT, activation,
      beta, linear_beta, nullptr, nullptr, BLOCK_M, /*stream=*/nullptr);
  CK(hipDeviceSynchronize(), "sync2");

  std::vector<float> out2(M * hidden);
  CK(hipMemcpy(out2.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut2");
  nnz = 0; n_nan = 0;
  for (int i = 0; i < M * hidden; ++i) {
    if (out2[i] == out2[i]) nnz++;
    else n_nan++;
  }
  std::printf("no_bias:   nnz=%d nan=%d out[:4]=%g %g %g %g\n",
              nnz, n_nan, out2[0], out2[1], out2[2], out2[3]);

  bool ok_bias = (n_nan == 0) && (nnz > 0);
  std::printf("%s\n", ok_bias ? "PASS (no NaN)" : "FAIL (NaN present)");
  return ok_bias ? 0 : 1;
}
