// meta/benchmarks/test_moe_fused_prefill_correct.hip
//
// Correctness test for the PREFILL tile config (BM=64, 256 threads) of the
// fused MXFP4 MoE HIP kernel, checked against the CPU reference.
//
// This runs the *shipped* prefill path, i.e. whichever kernel
// VK_MOE_PF_VARIANT selects (default 5: 16-wide gateup N tile, 32-wide down N
// tile — see the prefill-variants section of docs/kernels/moe_fused.md), so it
// is the regression guard for that default.  Set VK_MOE_PF_VARIANT to a
// non-default value to exercise a specific variant.
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

static float bf16_to_f32(uint16_t h) {
  uint32_t b = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

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

// Persistent device buffers, allocated once at maximum size and reused by
// every case.  On MI300A + ROCm 6.3.0 a per-case hipMalloc/hipFree cycle
// triggers a driver-level "Memory access fault ... Reason: Unknown" at
// phantom addresses (see meta/benchmarks/repro_issue150_fault.hip): the
// fault fires asynchronously, even in waves of *earlier* cases whose
// hipDeviceSynchronize() already returned success, and only when several
// alloc/free rounds precede it.  Single cases, repeated cases and
// churn-free multi-case sequences all pass deterministically, so the test
// allocates once up front and frees at the end.
struct DevBufs {
  uint16_t *dA, *dAct;
  uint8_t *dw13, *dw13s, *dw2, *dw2s;
  float *dOut, *dTw, *db13, *db2;
  int32_t *dTids, *dSids, *dEids;
};

// Worst-case geometry over all cases in main(): M=128, block=128.
static void alloc_dev_bufs(DevBufs& B, int hidden, int ispp, int top_k, int E) {
  const int M_max = 128, N_max = M_max * top_k;
  const int EM_max = N_max + E * 128;  // safe upper bound for both blocks
  CK(hipMalloc(&B.dA, M_max * hidden * 2), "A");
  CK(hipMalloc(&B.dw13, E * 2 * ispp * (hidden / 2)), "w13");
  CK(hipMalloc(&B.dw13s, E * 2 * ispp * (hidden / 32)), "w13s");
  CK(hipMalloc(&B.dw2, E * hidden * (ispp / 2)), "w2");
  CK(hipMalloc(&B.dw2s, E * hidden * (ispp / 32)), "w2s");
  CK(hipMalloc(&B.db13, E * 2 * ispp * 4), "b13");
  CK(hipMalloc(&B.db2, E * hidden * 4), "b2");
  CK(hipMalloc(&B.dAct, EM_max * ispp * 2), "act");
  CK(hipMalloc(&B.dOut, M_max * hidden * 4), "out");
  CK(hipMalloc(&B.dTw, N_max * 4), "tw");
  CK(hipMalloc(&B.dTids, N_max * 4), "tids");
  CK(hipMalloc(&B.dSids, EM_max * 4), "sids");
  CK(hipMalloc(&B.dEids, (EM_max / 64) * 4), "eids");  // smallest block
}

static void free_dev_bufs(DevBufs& B) {
  hipFree(B.dA); hipFree(B.dw13); hipFree(B.dw13s); hipFree(B.dw2);
  hipFree(B.dw2s); hipFree(B.db13); hipFree(B.db2); hipFree(B.dAct);
  hipFree(B.dOut); hipFree(B.dTw); hipFree(B.dTids); hipFree(B.dSids);
  hipFree(B.dEids);
}

// Run one prefill correctness case. Returns true on PASS.
// `gpu_block_m` is the alignment the GPU path consumes: 64 for the shipped
// prefill tiles, 128 for the issue-#150 M-tile config (VK_MOE_PF_VARIANT=6;
// the launcher falls back to the 64-row v5 path when the alignment is 64).
bool run_case(const char* name, int M, int hidden, int ispp, int top_k, int E,
             int activation, int gpu_block_m, DevBufs& B) {
  constexpr int CPU_BLOCK_M = 16;
  const int GPU_BLOCK_M = gpu_block_m;
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
  // Aliases into the persistent pre-allocated buffers (see DevBufs note).
  uint16_t *dA = B.dA, *dAct = B.dAct;
  uint8_t *dw13 = B.dw13, *dw13s = B.dw13s, *dw2 = B.dw2, *dw2s = B.dw2s;
  float *dOut = B.dOut, *dTw = B.dTw, *db13 = B.db13, *db2 = B.db2;
  int32_t *dTids = B.dTids, *dSids = B.dSids, *dEids = B.dEids;

  CK(hipMemset(B.dAct, 0xAB, (size_t)EM_gpu * ispp * 2), "zAct");
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
      dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
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

  if (max_rel >= 0.02) {
    // Worst-element diagnostics.
    int wi = 0; double we = 0;
    for (int i = 0; i < M * hidden; ++i) {
      double e = std::fabs((double)out_gpu[i] - (double)out_cpu[i]);
      double d = std::fmax(std::fabs((double)out_cpu[i]), 1.0);
      if (e / d > we) { we = e / d; wi = i; }
    }
    std::printf("  worst out[%d] (token %d, ch %d): gpu=%.6f cpu=%.6f\n",
                wi, wi / hidden, wi % hidden, out_gpu[wi], out_cpu[wi]);
    // Fraction of badly-wrong elements + per-token worst.
    int bad = 0;
    std::vector<int> bad_tok(M, 0);
    for (int i = 0; i < M * hidden; ++i) {
      double e = std::fabs((double)out_gpu[i] - (double)out_cpu[i]);
      double d = std::fmax(std::fabs((double)out_cpu[i]), 1.0);
      if (e / d > 0.02) { ++bad; ++bad_tok[i / hidden]; }
    }
    std::printf("  bad elements: %d / %d; per-token bad counts:",
                bad, M * hidden);
    for (int t = 0; t < M; ++t) std::printf(" %d", bad_tok[t]);
    std::printf("\n");
    // Optional gateup-stage isolation: compare device act vs oracle act.
    if (std::getenv("VK150_DIAG")) {
      // Re-run the CPU oracle with the GPU's own alignment so act rows are
      // directly comparable (act rows encode the sorted order of their
      // alignment; cross-alignment comparison is meaningless).
      std::vector<float> tws_gpu(EM_gpu, 0.0f);
      for (int i = 0; i < EM_gpu; ++i)
        if (h_sids_gpu[i] < N) tws_gpu[i] = h_topk_w[h_sids_gpu[i]];
      // The CPU oracle tiles M by 16 internally: expand the 128-row eids.
      std::vector<int32_t> eids16(EM_gpu / 16, -1);
      for (size_t j = 0; j < eids16.size(); ++j)
        eids16[j] = h_eids_gpu[j / (128 / 16)];
      std::vector<uint16_t> act_ref(EM_gpu * ispp, 0);
      std::vector<float> out_ref(M * hidden, 0.0f);
      vkernels::kernels::fused_moe_mxfp4_cpu(
          hA.data(), hw13.data(), hw13s.data(), hw2.data(), hw2s.data(),
          h_sids_gpu.data(), tws_gpu.data(), eids16.data(),
          act_ref.data(), out_ref.data(),
          M, hidden, ispp, top_k, EM_gpu, group_size, LIMIT, activation,
          4.0f, 25.0f, hb13.data(), hb2.data());
      double out_ref_rel = 0;
      for (int i = 0; i < M * hidden; ++i) {
        double e = std::fabs((double)out_gpu[i] - (double)out_ref[i]);
        double d = std::fmax(std::fabs((double)out_ref[i]), 1.0);
        if (e / d > out_ref_rel) out_ref_rel = e / d;
      }
      std::printf("  OUT vs aligned-oracle max_rel=%.6f\n", out_ref_rel);
      std::vector<uint16_t> act_gpu(EM_gpu * ispp);
      CK(hipMemcpy(act_gpu.data(), B.dAct, (size_t)EM_gpu * ispp * 2,
                   hipMemcpyDeviceToHost), "rdAct");
      int act_bad = 0; double act_rel = 0; int awi = 0;
      for (int r = 0; r < EM_gpu; ++r) {
        int32_t flat = h_sids_gpu[r];
        if (flat >= N) continue;  // padding row
        for (int c = 0; c < ispp; ++c) {
          float g = bf16_to_f32(act_gpu[r * ispp + c]);
          float o = bf16_to_f32(act_ref[r * ispp + c]);
          double e = std::fabs((double)g - (double)o);
          double d = std::fmax(std::fabs((double)o), 1.0);
          if (e / d > act_rel) { act_rel = e / d; awi = r * ispp + c; }
          if (e / d > 0.05) ++act_bad;
        }
      }
      std::printf("  ACT max_rel=%.6f bad=%d worst=(row %d, ch %d) gpu=%.5f cpu=%.5f\n",
                  act_rel, act_bad, awi / ispp, awi % ispp,
                  bf16_to_f32(act_gpu[awi]), act_ref[awi]);
    }
  }

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

  DevBufs B;
  alloc_dev_bufs(B, 512, 256, 2, 4);

  // Clean prefill: 64 tokens/expert -> one full 64-row block, no padding.
  bool ok1 = run_case("clean64", 128, 512, 256, 2, 4, activation, 64, B);
  // Padded prefill: 40 tokens/expert -> 24 padding rows per expert.
  bool ok2 = run_case("padded64", 80, 512, 256, 2, 4, activation, 64, B);

  // Issue #150: the 128-row M-tile config (VK_MOE_PF_VARIANT=6) plus its
  // 64-row fall-back.  The padded case exercises 128-row alignment with
  // heavy per-expert padding (40 real of 128 rows).
  bool ok3 = true, ok4 = true;
  const char* venv = std::getenv("VK_MOE_PF_VARIANT");
  const int pf_variant = venv ? std::atoi(venv) : 5;
  if (pf_variant == 6) {
    ok3 = run_case("clean128", 128, 512, 256, 2, 4, activation, 128, B);
    ok4 = run_case("padded128", 80, 512, 256, 2, 4, activation, 128, B);
  }

  free_dev_bufs(B);
  std::printf("%s\n", (ok1 && ok2 && ok3 && ok4) ? "PASS" : "FAIL");
  return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
}
