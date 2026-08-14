// meta/benchmarks/test_moe_fused_dist_correct.hip
//
// End-to-end correctness test for the distributed (TP) fused MXFP4 MoE on
// CDNA2 (gfx90a / gfx942): runs the four stage kernels
// (moe_gateup_preact → caller all-reduce → moe_act_epilogue →
//  moe_down_preact → caller all-reduce → moe_combine) over `tp` simulated
// ranks and compares every rank's result against the CPU oracle
// (fused_moe_mxfp4_cpu) — the acceptance criterion of issue #18.
//
// The TP all-reduces are the caller's collective; here they are performed on
// the host (device→host, sum, host→device) since vkernels' ring all-reduce
// is host-side.  A vLLM-style deployment substitutes NCCL/RCCL on device.
//
// Build/run (needs a ROCm toolkit):
//   cmake --preset hip -DVKERNELS_BUILD_BENCHMARKS=ON \
//         -DCMAKE_HIP_ARCHITECTURES=gfx90a
//   cmake --build --preset hip --target test_moe_fused_dist_correct
//   ./build/hip/meta/benchmarks/test_moe_fused_dist_correct [situ]
//
// Exit code 0 on PASS.
//
// Note on the CK macro below: the result variable is deliberately named
// `ck_rc_` (not `r`) — inside `for (int r = 0; ...)` loops, `hipError_t r`
// would shadow the loop index *within the initializer* (`int r = f(r);`
// refers to the new uninitialised `r`), silently corrupting every argument
// that uses the loop index.  (This bit us on real hardware.)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/dist/dist_moe.hpp"
#include "vkernels/kernels/moe_fused.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t ck_rc_ = (e);                                           \
    if (ck_rc_ != hipSuccess) {                                        \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(ck_rc_));  \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

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

float bf2f(uint16_t v) { return bits2f((uint32_t)v << 16); }

double max_rel(const std::vector<float>& got, const std::vector<float>& ref) {
  double m = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs((double)got[i] - (double)ref[i]);
    double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    m = std::fmax(m, e / d);
  }
  return m;
}

// Read the device shard contents directly from a kernel (avoids any
// host-side copy in the integrity check).
__global__ void probe_kernel(const uint8_t* a, const uint8_t* b, uint8_t* out) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < 16; ++i) { out[i] = a[i]; out[16 + i] = b[i]; }
  }
}

int main(int argc, char** argv) {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s\n", p.name);

  // Shape: hidden/ispp keep per-rank shards at multiples of 64 for tp = 2
  // (hidden/2 = 128, ispp/2 = 64).  Exercises multiple M/N/K blocks.
  constexpr int M = 16, hidden = 256, ispp = 128, top_k = 2;
  constexpr int E = 8, BLOCK_M = 16, group_size = 32;
  constexpr int TP = 2;
  constexpr float LIMIT = 4.0f;
  const bool situ = (argc > 1) && std::strcmp(argv[1], "situ") == 0;
  const int activation = situ ? vkernels::kernels::kSiTU : vkernels::kernels::kSwiGLU;
  const float beta = 4.0f, linear_beta = 25.0f;

  // --- Host data (identical to test_moe_fused_correct.hip) ---
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

  std::vector<int32_t> h_sids(M * top_k * 2 + E * BLOCK_M + 1);
  std::vector<int32_t> h_eids((M * top_k * 2 + E * BLOCK_M + 1) / BLOCK_M + 1);
  int EM = vkernels::kernels::moe_align_block_size(
      h_topk_ids.data(), M, top_k, BLOCK_M, E, h_sids.data(), h_eids.data());
  h_sids.resize(EM);
  h_eids.resize(EM / BLOCK_M);
  std::vector<float> h_tws(EM, 0.0f);
  for (int i = 0; i < EM; ++i) {
    if (h_sids[i] < M * top_k) h_tws[i] = h_topk_w[h_sids[i]];
  }
  std::printf("M=%d hidden=%d ispp=%d top_k=%d E=%d TP=%d EM=%d (%s)\n",
              M, hidden, ispp, top_k, E, TP, EM, situ ? "situ" : "swiglu");
  std::fflush(stdout);

  // --- CPU oracle (single rank) ---
  std::vector<uint16_t> act_cpu(EM * ispp, 0);
  std::vector<float> out_cpu(M * hidden, 0.0f);
  vkernels::kernels::fused_moe_mxfp4_cpu(
      hA.data(), hw13.data(), hw13s.data(), hw2.data(), hw2s.data(),
      h_sids.data(), h_tws.data(), h_eids.data(), act_cpu.data(), out_cpu.data(),
      M, hidden, ispp, top_k, EM, group_size, LIMIT, activation, beta,
      linear_beta, hb13.data(), hb2.data());

  // --- Per-rank TP shards (host) ---
  const vkernels::dist::MoeTPPlan plan = vkernels::dist::moe_tp_plan(hidden, ispp, TP);
  std::vector<std::vector<uint8_t>> sh13(TP), sh13s(TP), sh2(TP), sh2s(TP);
  for (int r = 0; r < TP; ++r) {
    sh13[r].resize((size_t)plan.w13_shard_bytes * E);
    sh13s[r].resize((size_t)plan.w13s_shard_bytes * E);
    sh2[r].resize((size_t)plan.w2_shard_bytes * E);
    sh2s[r].resize((size_t)plan.w2s_shard_bytes * E);
    vkernels::dist::moe_tp_shard_w13(hw13.data(), hw13s.data(), E, plan, r,
                                     sh13[r].data(), sh13s[r].data());
    vkernels::dist::moe_tp_shard_w2(hw2.data(), hw2s.data(), E, plan, r,
                                    sh2[r].data(), sh2s[r].data());
  }

  // --- Device buffers ---
  uint16_t *dA, *dAct;
  float *dGateUp, *dPartial, *dOut, *dTw, *db13, *db2;
  int32_t *dSids, *dEids;
  std::vector<uint8_t*> dsh13(TP), dsh13s(TP), dsh2(TP), dsh2s(TP);
  CK(hipMalloc(&dA, M * hidden * 2), "A");
  CK(hipMalloc(&dAct, EM * ispp * 2), "act");
  CK(hipMalloc(&dGateUp, (size_t)EM * 2 * ispp * 4), "gate_up");
  CK(hipMalloc(&dPartial, (size_t)EM * hidden * 4), "partial");
  CK(hipMalloc(&dOut, M * hidden * 4), "out");
  CK(hipMalloc(&dTw, EM * 4), "tw");
  CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13");
  CK(hipMalloc(&db2, E * hidden * 4), "b2");
  CK(hipMalloc(&dSids, EM * 4), "sids");
  CK(hipMalloc(&dEids, (EM / BLOCK_M) * 4), "eids");
  for (int r = 0; r < TP; ++r) {
    CK(hipMalloc(&dsh13[r], sh13[r].size()), "sh13");
    CK(hipMalloc(&dsh13s[r], sh13s[r].size()), "sh13s");
    CK(hipMalloc(&dsh2[r], sh2[r].size()), "sh2");
    CK(hipMalloc(&dsh2s[r], sh2s[r].size()), "sh2s");
  }
  for (int r = 0; r < TP; ++r) {
    CK(hipMemcpy(dsh13[r], sh13[r].data(), sh13[r].size(), hipMemcpyHostToDevice), "cp13");
    CK(hipMemcpy(dsh13s[r], sh13s[r].data(), sh13s[r].size(), hipMemcpyHostToDevice), "cp13s");
    CK(hipMemcpy(dsh2[r], sh2[r].data(), sh2[r].size(), hipMemcpyHostToDevice), "cp2");
    CK(hipMemcpy(dsh2s[r], sh2s[r].data(), sh2s[r].size(), hipMemcpyHostToDevice), "cp2s");
  }
  CK(hipMemcpy(dA, hA.data(), M * hidden * 2, hipMemcpyHostToDevice), "cpyA");
  CK(hipMemcpy(dTw, h_tws.data(), EM * 4, hipMemcpyHostToDevice), "cptw");
  CK(hipMemcpy(db13, hb13.data(), E * 2 * ispp * 4, hipMemcpyHostToDevice), "cpb13");
  CK(hipMemcpy(db2, hb2.data(), E * hidden * 4, hipMemcpyHostToDevice), "cpb2");
  CK(hipMemcpy(dSids, h_sids.data(), EM * 4, hipMemcpyHostToDevice), "cpsids");
  CK(hipMemcpy(dEids, h_eids.data(), (EM / BLOCK_M) * 4, hipMemcpyHostToDevice), "cpeids");
  CK(hipMemset(dOut, 0, M * hidden * 4), "zOut");
  CK(hipDeviceSynchronize(), "sync_copy");

  // --- Probe device shard integrity (kernel-read, bypasses the copy path) ---
  {
    uint8_t* dbg;
    CK(hipMalloc(&dbg, 64), "dbg");
    probe_kernel<<<1, 64>>>(dsh13[0], dsh13[1], dbg);
    CK(hipDeviceSynchronize(), "probe");
    uint8_t dbg_h[64];
    CK(hipMemcpy(dbg_h, dbg, 64, hipMemcpyDeviceToHost), "rddbg");
    int ok0 = std::memcmp(dbg_h, sh13[0].data(), 16) == 0;
    int ok1 = std::memcmp(dbg_h + 16, sh13[1].data(), 16) == 0;
    hipFree(dbg);
    if (!ok0 || !ok1) {
      std::fprintf(stderr, "FAIL: device shard content mismatch\n");
      return 1;
    }
  }

  // --- Run the four stages over TP simulated ranks, all-reducing on host ---
  std::vector<float> gate_up_host((size_t)EM * 2 * ispp, 0.0f);
  std::vector<float> partial_host((size_t)EM * hidden, 0.0f);

  for (int r = 0; r < TP; ++r) {
    // Stage 0a: per-rank gate/up GEMM over the rank's hidden K-slice.
    vkernels::kernels::hip::moe_gateup_preact(
        dA, dsh13[r], dsh13s[r], dSids, dEids, dGateUp,
        M, hidden, ispp, top_k, EM, r * plan.hidden_shard, plan.hidden_shard);
    CK(hipGetLastError(), "err0a");
    CK(hipDeviceSynchronize(), "sync0a");
    std::vector<float> rank_gu((size_t)EM * 2 * ispp);
    CK(hipMemcpy(rank_gu.data(), dGateUp, rank_gu.size() * 4, hipMemcpyDeviceToHost), "rd_gu");
    for (std::size_t i = 0; i < rank_gu.size(); ++i) gate_up_host[i] += rank_gu[i];
  }
  CK(hipMemcpy(dGateUp, gate_up_host.data(), gate_up_host.size() * 4, hipMemcpyHostToDevice), "wr_gu");
  CK(hipDeviceSynchronize(), "sync_wr_gu");

  // Stage 0b: bias + activation.
  vkernels::kernels::hip::moe_act_epilogue(
      dGateUp, db13, dAct, dSids, dEids, M, top_k, EM, ispp, LIMIT,
      activation, beta, linear_beta);
  CK(hipGetLastError(), "err0b");
  CK(hipDeviceSynchronize(), "sync0b");

  for (int r = 0; r < TP; ++r) {
    // Stage 1a: per-rank down GEMM over the rank's ispp K-slice.
    vkernels::kernels::hip::moe_down_preact(
        dAct, dsh2[r], dsh2s[r], dSids, dEids, dPartial,
        M, ispp, hidden, top_k, EM, r * plan.ispp_shard, plan.ispp_shard);
    CK(hipGetLastError(), "err1a");
    CK(hipDeviceSynchronize(), "sync1a");
    std::vector<float> rank_p((size_t)EM * hidden);
    CK(hipMemcpy(rank_p.data(), dPartial, rank_p.size() * 4, hipMemcpyDeviceToHost), "rd_p");
    for (std::size_t i = 0; i < rank_p.size(); ++i) partial_host[i] += rank_p[i];
  }
  CK(hipMemcpy(dPartial, partial_host.data(), partial_host.size() * 4, hipMemcpyHostToDevice), "wr_p");
  CK(hipDeviceSynchronize(), "sync_wr_p");

  // Stage 1b: routed combine.
  vkernels::kernels::hip::moe_combine(
      dPartial, db2, dTw, dSids, dEids, dOut, M, hidden, top_k, EM);
  CK(hipGetLastError(), "err1b");
  CK(hipDeviceSynchronize(), "sync1b");

  // --- Compare against the CPU oracle ---
  std::vector<float> out_gpu(M * hidden);
  CK(hipMemcpy(out_gpu.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut");
  std::vector<uint16_t> act_gpu(EM * ispp);
  CK(hipMemcpy(act_gpu.data(), dAct, EM * ispp * 2, hipMemcpyDeviceToHost), "rdAct");

  double rel_out = max_rel(out_gpu, out_cpu);
  double max_abs_act = 0, rel_act = 0;
  int real = 0;
  for (int r = 0; r < EM; ++r) {
    if (h_sids[r] >= M * top_k) continue;
    for (int c = 0; c < ispp; ++c, ++real) {
      double e = std::fabs((double)bf2f(act_gpu[(size_t)r * ispp + c]) -
                           (double)bf2f(act_cpu[(size_t)r * ispp + c]));
      max_abs_act = std::fmax(max_abs_act, e);
      rel_act = std::fmax(rel_act, e / std::fmax(std::fabs((double)bf2f(act_cpu[(size_t)r * ispp + c])), 1.0));
    }
  }
  std::printf("OUT: max_rel=%.6f\n", rel_out);
  std::printf("ACT: max_rel=%.6f (real rows %d)\n", rel_act, real);

  bool pass = rel_out < 0.02 && rel_act < 0.02;
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  hipFree(dA); hipFree(dAct); hipFree(dGateUp); hipFree(dPartial);
  hipFree(dOut); hipFree(dTw); hipFree(db13); hipFree(db2);
  hipFree(dSids); hipFree(dEids);
  for (int r = 0; r < TP; ++r) {
    hipFree(dsh13[r]); hipFree(dsh13s[r]); hipFree(dsh2[r]); hipFree(dsh2s[r]);
  }
  return pass ? 0 : 1;
}
