// meta/benchmarks/bench_moe_fused_oracle.cpp
//
// Host-only benchmark for the MXFP4 fused-MoE CPU reference
// (vkernels::kernels::fused_moe_mxfp4_cpu), which is the golden oracle the
// gfx942/gfx90a HIP kernels are validated against (issue #41, acceptance
// criterion #1: "validated against the CPU oracle at K3 shapes").
//
// This bench is the always-runnable (no GPU, no ROCm) analog of
// bench_moe_fused.hip, exactly as bench_pipeline_boundary.cpp is the
// always-runnable surface for issue #10 and bench_rccl.cpp for issue #19.
// The HIP bench marks the device kernel and notes that "the CPU oracle is
// impractical at ispp=33792" (its --no-cpu flag skips the correctness
// check for that reason).  The latency the oracle spends on padding rows
// (zeroed-A tiles that contribute exactly zero) is the part this bench
// isolates: at K3 decode (M=1, top_k=16) each 16-row block has ~1 real row
// and ~15 padding rows, so ~94% of the oracle's GEMM is wasted unless the
// implementation skips padding rows.
//
//   ./bench_moe_fused_oracle [--hidden N] [--ispp N] [--topk N] [--E N]
//                            [--ms 1,2,4,8] [--iters 3] [--seed 7]
//                            [--no-act]            (SwiGLU instead of SiTU)
//
// Reports, per M: EM (padded sorted-row count), the real / padding row
// split, and the median oracle latency.  The default shape (E=256,
// top_k=16, M=1) reproduces the K3 decode routing structure at a size the
// host can hold; pass --hidden 7168 --ispp 33792 for the full K3 shape
// (the weights are ~6 TB, so only run that on a machine with the RAM —
// the routing/block-structure/top_k=16 paths the bench measures at the
// smaller default are identical).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "vkernels/kernels/moe.hpp"
#include "vkernels/kernels/moe_fused.hpp"

using vkernels::kernels::fused_moe_mxfp4_cpu;
using vkernels::kernels::kSiTU;
using vkernels::kernels::kSwiGLU;
using vkernels::kernels::moe_align_block_size;

namespace {

// ---- tiny E2M1 packer (mirrors tests/kernels/moe/test_moe_fused.cpp) ----
uint8_t e2m1_nibble(float f) {
  bool neg = f < 0.0f;
  float af = std::fabs(f);
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
  return neg ? static_cast<uint8_t>(best_n | 0x8) : best_n;
}

uint8_t pack_e2m1_pair(float v0, float v1) {
  return static_cast<uint8_t>(e2m1_nibble(v0) | (e2m1_nibble(v1) << 4));
}

uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return static_cast<uint16_t>(b >> 16);
}

// Deterministic float in [0, 1) from a 64-bit-ish seed (good enough for a
// bench; the oracle is compared to itself across configs, not to a known
// value here).
float rnd(uint64_t s) {
  s ^= s >> 33;
  s *= 0xff51afd7ed558ccdULL;
  s ^= s >> 33;
  s *= 0xc4ceb9fe1a85ec53ULL;
  s ^= s >> 33;
  return static_cast<float>(s >> 40) * (1.0f / 16777216.0f);  // 24-bit
}

// Median of a sorted-then-indexed vector.
double median_us(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
  int E = 256, hidden = 512, ispp = 1024, top_k = 16;
  int iters = 3, seed = 7;
  int activation = kSiTU;
  std::vector<int> Ms = {1, 2, 4, 8};
  for (int i = 1; i < argc; ++i) {
    auto next = [&](int& j) -> long {
      if (j + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[j]); return 1; }
      return std::strtol(argv[++j], nullptr, 10);
    };
    std::string a = argv[i];
    if      (a == "--hidden") hidden = (int)next(i);
    else if (a == "--ispp")   ispp = (int)next(i);
    else if (a == "--topk")   top_k = (int)next(i);
    else if (a == "--E")      E = (int)next(i);
    else if (a == "--iters")  iters = (int)next(i);
    else if (a == "--seed")   seed = (int)next(i);
    else if (a == "--no-act") activation = kSwiGLU;
    else if (a == "--ms") {
      Ms.clear();
      for (char* p = argv[++i]; ; ++p) {
        char* endp; long v = std::strtol(p, &endp, 10);
        if (endp == p) break;
        Ms.push_back((int)v); p = endp;
        if (*p != ',') break;
      }
    } else if (a == "--help" || a == "-h") {
      std::printf("usage: bench_moe_fused_oracle [--hidden N] [--ispp N] "
                  "[--topk N] [--E N] [--ms 1,2,4,8] [--iters N] [--seed N] "
                  "[--no-act]\n");
      return 0;
    }
  }

  if (hidden % 64 != 0 || ispp % 64 != 0) {
    std::fprintf(stderr, "hidden and ispp must be multiples of 64\n");
    return 2;
  }
  if (top_k <= 0 || E <= 0) { std::fprintf(stderr, "top_k/E must be positive\n"); return 2; }

  // Pack the weights once (they are constant across the M sweep).  Random
  // E2M1 with unit (127) ue8m0 scales, random biases — the same shape as
  // bench_moe_fused.hip, just smaller so the host can hold them.
  const std::size_t w13_bytes  = static_cast<std::size_t>(E) * 2 * ispp * (hidden / 2);
  const std::size_t w13s_bytes = static_cast<std::size_t>(E) * 2 * ispp * (hidden / 32);
  const std::size_t w2_bytes   = static_cast<std::size_t>(E) * hidden * (ispp / 2);
  const std::size_t w2s_bytes  = static_cast<std::size_t>(E) * hidden * (ispp / 32);
  std::vector<uint8_t> w13(w13_bytes), w13s(w13s_bytes, 127);
  std::vector<uint8_t> w2(w2_bytes), w2s(w2s_bytes, 127);
  for (std::size_t i = 0; i < w13_bytes; ++i)
    w13[i] = pack_e2m1_pair(rnd(seed * 101 + i) * 3.0f, rnd(seed * 202 + i) * 3.0f);
  for (std::size_t i = 0; i < w2_bytes; ++i)
    w2[i] = pack_e2m1_pair(rnd(seed * 303 + i) * 3.0f, rnd(seed * 404 + i) * 3.0f);
  std::vector<float> b13(static_cast<std::size_t>(E) * 2 * ispp);
  std::vector<float> b2(static_cast<std::size_t>(E) * hidden);
  for (std::size_t i = 0; i < b13.size(); ++i) b13[i] = rnd(seed * 505 + i) * 0.3f;
  for (std::size_t i = 0; i < b2.size(); ++i)   b2[i]  = rnd(seed * 606 + i) * 0.3f;

  const int block_size = 16;       // K3 decode: 16x64 tiles, 64-thread blocks
  const float beta = 4.0f, linear_beta = 25.0f, limit = 0.0f;

  std::printf("fused-MoE CPU oracle  E=%d hidden=%d ispp=%d top_k=%d "
              "act=%s  iters=%d\n",
              E, hidden, ispp, top_k, activation == kSiTU ? "SiTU" : "SwiGLU",
              iters);
  std::printf("%6s %8s %10s %10s %9s %10s\n",
              "M", "EM", "real", "pad", "pad%", "oracle_us");

  for (int M : Ms) {
    int N = M * top_k;
    // Build the routing table.  For the K3 decode structure at M=1 each of
    // the top_k selections maps to a distinct expert (so EM = top_k * 16
    // padded rows, ~94% padding).  For larger M we spread selections to
    // keep a mix of full and partial expert blocks.
    std::vector<int32_t> topk_ids(static_cast<std::size_t>(N));
    std::vector<float> topk_w(static_cast<std::size_t>(N));
    for (int i = 0; i < M; ++i) {
      for (int s = 0; s < top_k; ++s) {
        std::size_t idx = static_cast<std::size_t>(i * top_k + s);
        topk_ids[idx] = static_cast<int32_t>((static_cast<uint64_t>(i) * 131
                                              + s * 17 + seed) % E);
        topk_w[idx] = 0.1f + rnd(seed * 707 + static_cast<uint64_t>(idx)) * 0.4f;
      }
    }

    int EM_max = ((N + E * block_size) / block_size) * block_size + block_size;
    std::vector<int32_t> sorted_ids(static_cast<std::size_t>(EM_max));
    std::vector<int32_t> expert_ids(static_cast<std::size_t>(EM_max / block_size));
    int EM = moe_align_block_size(topk_ids.data(), M, top_k, block_size, E,
                                  sorted_ids.data(), expert_ids.data());
    sorted_ids.resize(static_cast<std::size_t>(EM));
    expert_ids.resize(static_cast<std::size_t>(EM / block_size));

    // The CPU oracle takes pre-sorted routing weights (gathered into the
    // sorted order); the HIP kernel reads topk_w[flat] directly (item 2).
    std::vector<float> topk_w_sorted(static_cast<std::size_t>(EM), 0.0f);
    int real_rows = 0;
    for (int i = 0; i < EM; ++i) {
      int f = sorted_ids[static_cast<std::size_t>(i)];
      if (f >= 0 && f < N) {
        topk_w_sorted[static_cast<std::size_t>(i)] = topk_w[static_cast<std::size_t>(f)];
        ++real_rows;
      }
    }

    // Inputs A [M, hidden] bf16, act scratch [EM, ispp] bf16, out [M, hidden] fp32.
    std::vector<uint16_t> A(static_cast<std::size_t>(M) * hidden);
    for (std::size_t i = 0; i < A.size(); ++i)
      A[i] = f2bf(rnd(seed * 808 + i) * 0.2f - 0.1f);
    std::vector<uint16_t> act(static_cast<std::size_t>(EM) * ispp, 0);
    std::vector<float> out(static_cast<std::size_t>(M) * hidden, 0.0f);

    std::vector<double> times;
    for (int it = 0; it < iters; ++it) {
      std::fill(out.begin(), out.end(), 0.0f);
      std::fill(act.begin(), act.end(), uint16_t{0});
      auto t0 = std::chrono::steady_clock::now();
      fused_moe_mxfp4_cpu(A.data(), w13.data(), w13s.data(), w2.data(),
                          w2s.data(), sorted_ids.data(), topk_w_sorted.data(),
                          expert_ids.data(), act.data(), out.data(),
                          M, hidden, ispp, top_k, EM, 32 /*group*/, limit,
                          activation, beta, linear_beta, b13.data(), b2.data());
      auto t1 = std::chrono::steady_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    int pad_rows = EM - real_rows;
    double pad_pct = EM ? 100.0 * pad_rows / EM : 0.0;
    std::printf("%6d %8d %10d %10d %8.1f%% %10.1f\n", M, EM, real_rows,
                pad_rows, pad_pct, median_us(times));
  }
  return 0;
}
