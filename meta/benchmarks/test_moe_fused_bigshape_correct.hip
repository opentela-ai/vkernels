// meta/benchmarks/test_moe_fused_bigshape_correct.hip
//
// Regression test for issue #41's "GPU memory-access fault at ispp >= 4096"
// blocker: the fused-MoE HIP kernels computed per-expert weight byte offsets
// (expert * w13_bytes, expert * w2_bytes) in INT32, which wraps negative once
//   expert * hidden * ispp/2  >= 2^31   (down:   e >= 147 at h=7168, ispp=4096)
//   expert * 2 * ispp * hidden/2 >= 2^31 (gateup: e >= 74  at h=7168, ispp=4096)
// and faults (or silently reads the wrong expert) at large shapes.  The
// kernels now compute those offsets in 64-bit; this test validates GPU vs
// CPU oracle exactly at and beyond the old wrap thresholds:
//
//   1. decode, E=256 h7168 ispp=4096, routing bracketing the wrap thresholds
//      (e = 73..255) — both stages previously wrapped;
//   2. decode, all 16 top-k slots on expert 255 (worst-case offset);
//   3. prefill (block_size=64) at the same shape — 64x64-tile kernels had
//      the same defect;
//   4. decode, E=32 h7168 ispp=33792 — the documented "~12 GB resident"
//      per-GPU TP=8 shard shape that faulted on beverin (e >= 18 wraps).
//
// Weights are real (random E2M1 + ue8m0), so this validates numerics, not
// just absence-of-fault.  GPU-only; run on a MI300A/gfx942 node.

#include <hip/hip_runtime.h>

#include <algorithm>
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

int g_failures = 0;

float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

uint32_t f2bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }

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

// Fill packed E2M1 weights + ue8m0 scales with deterministic random data.
// Layouts match the kernel contract: w13 [E, 2*ispp, hidden/2] packed,
// w13_scale [E, 2*ispp, hidden/32], w2 [E, hidden, ispp/2],
// w2_scale [E, hidden, ispp/32].
void fill_weights(int E, int hidden, int ispp,
                  std::vector<uint8_t>& w13, std::vector<uint8_t>& w13s,
                  std::vector<uint8_t>& w2, std::vector<uint8_t>& w2s) {
  const int group = 32;
  w13.assign((size_t)E * 2 * ispp * (hidden / 2), 0);
  w13s.assign((size_t)E * 2 * ispp * (hidden / group), 127);
  w2.assign((size_t)E * hidden * (ispp / 2), 0);
  w2s.assign((size_t)E * hidden * (ispp / group), 127);
  for (size_t i = 0; i < w13.size() / 2; ++i) {
    w13[2 * i] = e2m1_nibble(rnd(2, (int)i) * 1.5f) |
                 (e2m1_nibble(rnd(3, (int)i) * 1.5f) << 4);
  }
  for (size_t i = 0; i < w2.size() / 2; ++i) {
    w2[2 * i] = e2m1_nibble(rnd(6, (int)i) * 1.5f) |
                (e2m1_nibble(rnd(7, (int)i) * 1.5f) << 4);
  }
  // Scales 127 = ue8m0 byte for 2^0 = 1.0 (bit-exact neutral), with some
  // variation so the group-scale path is exercised.
  for (size_t i = 0; i < w13s.size(); ++i)
    if (rnd(11, (int)i) > 0.8f) w13s[i] = (uint8_t)(126 + (int)(rnd(12, (int)i) * 3.0f));
  for (size_t i = 0; i < w2s.size(); ++i)
    if (rnd(13, (int)i) > 0.8f) w2s[i] = (uint8_t)(126 + (int)(rnd(14, (int)i) * 3.0f));
}

// One GPU-vs-oracle case. `expert_ids_route` must hold exactly top_k entries.
void run_case(const char* name, int E, int hidden, int ispp, int top_k, int M,
              int block_size, const std::vector<int>& route, int activation) {
  const int group = 32;
  const float LIMIT = 10.0f, beta = 4.0f, linear_beta = 25.0f;

  std::printf("--- %s: E=%d hidden=%d ispp=%d top_k=%d M=%d block=%d %s\n",
              name, E, hidden, ispp, top_k, M, block_size,
              activation == vkernels::kernels::kSiTU ? "situ" : "swiglu");

  std::vector<uint8_t> w13, w13s, w2, w2s;
  fill_weights(E, hidden, ispp, w13, w13s, w2, w2s);

  // Routing: explicit experts (covers the old INT32 wrap thresholds).
  std::vector<int32_t> tids((size_t)M * top_k);
  std::vector<float> tw((size_t)M * top_k);
  for (int i = 0; i < M; ++i)
    for (int s = 0; s < top_k; ++s) {
      tids[(size_t)i * top_k + s] = route[(size_t)i * top_k + s];
      tw[(size_t)i * top_k + s] = 0.5f + 0.5f * rnd(20, i * top_k + s);
    }

  const int EM_max = E * block_size;
  std::vector<int32_t> sids(EM_max), eids(EM_max / block_size);
  int EM = vkernels::kernels::moe_align_block_size(
      tids.data(), M, top_k, block_size, E, sids.data(), eids.data());
  std::fprintf(stderr, "[phase] align ok EM=%d max_expert=%d\n", EM,
               *std::max_element(route.begin(), route.end()));

  // Oracle alignment: the CPU oracle tiles per-16 (BLOCK_M=16) and indexes
  // expert_ids per 16-row block, so it must be fed the decode alignment even
  // when the GPU path runs the prefill (per-64) tiling — same contract as
  // bench_moe_fused_prefill ("block_size=16 oracle") and
  // test_moe_fused_prefill_correct.  The final out [M, hidden] is
  // block-size invariant.
  constexpr int kOracleBS = 16;
  std::vector<int32_t> sids_o(E * kOracleBS), eids_o(E * kOracleBS / kOracleBS);
  int EM_o = vkernels::kernels::moe_align_block_size(
      tids.data(), M, top_k, kOracleBS, E, sids_o.data(), eids_o.data());

  // Oracle inputs: per-sorted-row weights (per-16 alignment).
  std::vector<float> tws_o(EM_o, 0.0f);
  for (int i = 0; i < EM_o; ++i)
    if (sids_o[i] < M * top_k) tws_o[i] = tw[sids_o[i]];

  std::vector<uint16_t> hA((size_t)M * hidden);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = f2bf(rnd(10, (int)i) * 0.1f);

  // CPU oracle (per-16 alignment).
  std::fprintf(stderr, "[phase] weights filled (w13=%.1f GB)\n",
               (double)w13.size() / (1 << 30));
  std::vector<uint16_t> act_cpu((size_t)EM_o * ispp, 0);
  std::vector<float> out_cpu((size_t)M * hidden, 0.0f);
  vkernels::kernels::fused_moe_mxfp4_cpu(
      hA.data(), w13.data(), w13s.data(), w2.data(), w2s.data(),
      sids_o.data(), tws_o.data(), eids_o.data(), act_cpu.data(), out_cpu.data(),
      M, hidden, ispp, top_k, EM_o, group, LIMIT, activation, beta,
      linear_beta, nullptr, nullptr);
  std::fprintf(stderr, "[phase] CPU oracle done\n");

  // Device buffers (size_t products throughout — the same class of bug).
  uint16_t *dA, *dAct; float *dOut, *dTw;
  int32_t *dTids, *dSids, *dEids;
  uint8_t *dw13, *dw13s, *dw2, *dw2s;
  const size_t w13_sz = (size_t)E * 2 * ispp * (hidden / 2);
  const size_t w13s_sz = (size_t)E * 2 * ispp * (hidden / group);
  const size_t w2_sz = (size_t)E * hidden * (ispp / 2);
  const size_t w2s_sz = (size_t)E * hidden * (ispp / group);
  CK(hipMalloc((void**)&dw13, w13_sz), "w13");
  CK(hipMalloc((void**)&dw13s, w13s_sz), "w13s");
  CK(hipMalloc((void**)&dw2, w2_sz), "w2");
  CK(hipMalloc((void**)&dw2s, w2s_sz), "w2s");
  CK(hipMalloc((void**)&dA, (size_t)M * hidden * 2), "A");
  CK(hipMalloc((void**)&dAct, (size_t)EM * ispp * 2), "act");
  CK(hipMalloc((void**)&dOut, (size_t)M * hidden * 4), "out");
  CK(hipMalloc((void**)&dTw, (size_t)M * top_k * 4), "tw");
  CK(hipMalloc((void**)&dTids, (size_t)M * top_k * 4), "tids");
  CK(hipMalloc((void**)&dSids, (size_t)EM * 4), "sids");
  CK(hipMalloc((void**)&dEids, (size_t)(EM / block_size) * 4), "eids");
  CK(hipMemcpy(dw13, w13.data(), w13_sz, hipMemcpyHostToDevice), "cpw13");
  CK(hipMemcpy(dw13s, w13s.data(), w13s_sz, hipMemcpyHostToDevice), "cpw13s");
  CK(hipMemcpy(dw2, w2.data(), w2_sz, hipMemcpyHostToDevice), "cpw2");
  CK(hipMemcpy(dw2s, w2s.data(), w2s_sz, hipMemcpyHostToDevice), "cpw2s");
  CK(hipMemcpy(dA, hA.data(), (size_t)M * hidden * 2, hipMemcpyHostToDevice), "cpA");
  CK(hipMemcpy(dTw, tw.data(), (size_t)M * top_k * 4, hipMemcpyHostToDevice), "cpTw");
  CK(hipMemcpy(dTids, tids.data(), (size_t)M * top_k * 4, hipMemcpyHostToDevice), "cpTids");
  CK(hipMemcpy(dSids, sids.data(), (size_t)EM * 4, hipMemcpyHostToDevice), "cpSids");
  CK(hipMemcpy(dEids, eids.data(), (size_t)(EM / block_size) * 4, hipMemcpyHostToDevice), "cpEids");
  CK(hipMemset(dOut, 0, (size_t)M * hidden * 4), "zOut");
  std::fprintf(stderr, "[phase] device buffers ok, launching fused_moe_mxfp4\n");

  vkernels::kernels::hip::fused_moe_mxfp4(
      dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
      M, hidden, ispp, top_k, dSids, dEids, EM, LIMIT, activation, beta,
      linear_beta, nullptr, nullptr, block_size, false);
  std::fprintf(stderr, "[phase] launcher returned, syncing\n");
  CK(hipDeviceSynchronize(), "sync");
  std::fprintf(stderr, "[phase] device sync ok\n");

  std::vector<float> out_gpu((size_t)M * hidden);
  CK(hipMemcpy(out_gpu.data(), dOut, (size_t)M * hidden * 4,
               hipMemcpyDeviceToHost), "rdOut");

  double max_rel = 0;
  int worst_t = -1, worst_c = -1;
  float worst_g = 0, worst_c2 = 0;
  for (int t = 0; t < M; ++t) {
    for (int c = 0; c < hidden; ++c) {
      int i = t * hidden + c;
      double e = std::fabs((double)out_gpu[i] - (double)out_cpu[i]);
      double r = e / std::fmax(std::fabs((double)out_cpu[i]), 1.0);
      if (r > max_rel) {
        max_rel = r;
        worst_t = t; worst_c = c;
        worst_g = out_gpu[i]; worst_c2 = out_cpu[i];
      }
    }
  }

  bool ok = max_rel < 0.05;
  if (!ok) ++g_failures;
  std::printf("    EM=%d max_rel=%.4g -> %s", EM, max_rel, ok ? "PASS" : "FAIL");
  if (!ok)
    std::printf("  [worst token=%d col=%d gpu=%.4f cpu=%.4f]", worst_t,
                worst_c, worst_g, worst_c2);
  std::printf("\n");

  CK(hipFree(dw13), "f"); CK(hipFree(dw13s), "f"); CK(hipFree(dw2), "f");
  CK(hipFree(dw2s), "f"); CK(hipFree(dA), "f"); CK(hipFree(dAct), "f");
  CK(hipFree(dOut), "f"); CK(hipFree(dTw), "f"); CK(hipFree(dTids), "f");
  CK(hipFree(dSids), "f"); CK(hipFree(dEids), "f");
}

}  // namespace

int main(int argc, char** argv) {
  // Unbuffered: a GPU fault kills the process and we must not lose the log
  // position (slurm pipes stdout through a block-buffered file).
  setvbuf(stdout, nullptr, _IONBF, 0);
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (%s)\n\n", p.name, p.gcnArchName);

  using vkernels::kernels::kSiTU;
  int only = (argc > 1) ? std::atoi(argv[1]) : 0;  // 0 = all

  // Wrap thresholds at h=7168, ispp=4096 (decode kernels):
  //   down:    e * 7168*2048      >= 2^31 at e >= 147
  //   gateup:  e * 2*4096*3584    >= 2^31 at e >= 74
  // 16 distinct experts bracketing both thresholds, ending at 255.
  const std::vector<int> route_thresholds = {
      7, 73, 74, 100, 145, 146, 147, 180, 200, 220, 240, 250, 253, 254, 255, 74};

  // 1. decode regime at the documented fault threshold.
  if (only == 0 || only == 1)
    run_case("decode ispp=4096 wrap-threshold experts", 256, 7168, 4096, 16, 1,
             16, route_thresholds, kSiTU);

  // 2. worst-case single expert (max offset, all 16 slots on expert 255).
  const std::vector<int> route_e255(16, 255);
  if (only == 0 || only == 2)
    run_case("decode ispp=4096 all-slots expert 255", 256, 7168, 4096, 16, 1,
             16, route_e255, kSiTU);

  // 3. prefill regime (64x64 tiles) at the same shape.
  if (only == 0 || only == 3)
    run_case("prefill ispp=4096 wrap-threshold experts", 256, 7168, 4096, 16, 1,
             64, route_thresholds, kSiTU);

  // 4. the documented serving-shard shape that faulted on beverin
  // (E=32, ~12 GB resident; down wraps at e >= 18, gateup at e >= 9).
  const std::vector<int> route_shard = {
      2, 5, 9, 12, 17, 18, 19, 20, 22, 24, 26, 28, 29, 30, 31, 8};
  if (only == 0 || only == 4)
    run_case("decode ispp=33792 E=32 serving shard", 32, 7168, 33792, 16, 1,
             16, route_shard, kSiTU);

  // 5. prefill at a small shape — isolates a general prefill-path bug from
  //    a big-shape overflow (case 3 = big + padding).
  const std::vector<int> route_small = {0, 1, 2, 3};
  if (only == 0 || only == 5)
    run_case("prefill small E=8 h512 ispp=512", 8, 512, 512, 4, 1,
             64, route_small, kSiTU);

  // 6. prefill big, NO padding: M=64 tokens all routed to the same 16
  //    distinct experts -> EM = 64*16 = 1024 = N, every block full.
  std::vector<int> route_full;
  route_full.reserve(64 * 16);
  for (int i = 0; i < 64; ++i)
    route_full.insert(route_full.end(), route_thresholds.begin(),
                      route_thresholds.end());
  if (only == 0 || only == 6)
    run_case("prefill ispp=4096 M=64 no-padding", 256, 7168, 4096, 16, 64,
             64, route_full, kSiTU);

  // 7. prefill big, M=2 (like case 3 but N=32: mostly-padding blocks).
  std::vector<int> route_m2;
  route_m2.reserve(2 * 16);
  route_m2.insert(route_m2.end(), route_thresholds.begin(),
                  route_thresholds.end());
  route_m2.insert(route_m2.end(), route_thresholds.begin(),
                  route_thresholds.end());
  if (only == 0 || only == 7)
    run_case("prefill ispp=4096 M=2 padded", 256, 7168, 4096, 16, 2,
             64, route_m2, kSiTU);

  // 8. prefill SMALL no-padding (M=64, full 64-row blocks): a small
  //    reproduction of the case-6 failure, if it is block-row-related
  //    rather than shape-related.
  std::vector<int> route_small_full;
  route_small_full.reserve(64 * 4);
  for (int i = 0; i < 64; ++i)
    route_small_full.insert(route_small_full.end(), route_small.begin(),
                            route_small.end());
  if (only == 0 || only == 8)
    run_case("prefill small M=64 no-padding", 8, 512, 512, 4, 64,
             64, route_small_full, kSiTU);

  if (g_failures) {
    std::printf("\n%d case(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("\nAll cases PASS\n");
  return 0;
}
