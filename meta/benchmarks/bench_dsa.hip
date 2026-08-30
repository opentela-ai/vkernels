// meta/benchmarks/bench_dsa.hip
//
// Micro-benchmark for the DeepseekSparseAttn sparse-MLA forward (issue #51)
// on AMD gfx942 (MI300A). For each GLM-5.3-Flash (tail_dim == 0) and
// DeepSeek-V3 (tail_dim > 0) shape it reports, against the MI300A roof
// (≈1307 TFLOP/s bf16, ≈5300 GB/s HBM3, ridge ≈247 FLOP/byte):
//
//   * median / min latency (us),
//   * achieved TFLOP/s  = 2*S_q*H*topk*(W + d_v) / t,
//   * effective GB/s    = actual HBM bytes / t,
//   * arithmetic intensity and the binding resource (mem / comp).
//
// The baseline streams each query's `topk` selected keys once (one key at a
// time, no double-buffered key tile); kv is shared across the H head-blocks
// and so is re-read once per head. The gap to the roof is what the tiled-key
// prefetch / persistent-kernel follow-ons close. A small autotuner sweeps BQ
// via dsa_sparse_fwd_with_tile.
//
// Built only with VKERNELS_HAS_HIP + VKERNELS_BUILD_BENCHMARKS.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/dsa.hpp"

static void check_hip(hipError_t err, const char* ctx) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", ctx, hipGetErrorString(err));
    std::exit(1);
  }
}

struct GpuInfo { std::string name; int arch; double tflops, bw; };
static GpuInfo get_gpu_info() {
  hipDeviceProp_t p;
  check_hip(hipGetDeviceProperties(&p, 0), "props");
  GpuInfo i;
  i.name = p.name;
  if (std::strstr(p.gcnArchName, "gfx942"))      { i.arch = 942; i.tflops = 1307; i.bw = 5300; }
  else if (std::strstr(p.gcnArchName, "gfx90a")) { i.arch = 90;  i.tflops = 383;  i.bw = 3200; }
  else                                            { i.arch = 0;   i.tflops = 100;  i.bw = 1000; }
  return i;
}

struct BenchResult { double min_us, median_us, mean_us; };

static BenchResult bench_us(const std::function<void()>& launch,
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
    double sum = 0;
    for (double t : times) sum += t;
    double mean = sum / iters;
    double sum2 = 0;
    for (double t : times) sum2 += t * t;
    double var = sum2 / iters - mean * mean;
    double stddev = std::sqrt(std::max(0.0, var));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {times[0], times[iters / 2], mean};
    }
    iters *= 2;
  }
  return {};
}

static uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return (uint16_t)(b >> 16);
}
static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f;  // [-1, 1)
}

// ---------------------------------------------------------------------------
// GLM-5.3-Flash (dim=256, tail_dim=0 -> W=256, d_v=256) and DeepSeek-V3
// (dim=576, tail_dim=64 -> W=640, d_v=512). H sweeps the head count; S_q
// sweeps decode -> prefill. topk is the indexer fan-out per query.
static void bench_shapes(const GpuInfo& info, hipEvent_t start,
                         hipEvent_t stop) {
  struct Cfg { int H, S_q, S_kv, dim, tail_dim, topk; };
  const Cfg cfgs[] = {
      // GLM-5.3-Flash (tail_dim == 0).
      {64,    1, 2048, 256,  0, 2048},  // decode, full topk
      {64,    1,  256, 256,  0,  256},  // decode, topk=256
      {64,    1,  128, 256,  0,  128},  // decode, topk=128
      { 8,   64,  256, 256,  0,  128},  // short
      { 1, 8192, 1024, 256,  0,  128},  // long prefill
      // DeepSeek-V3 (tail_dim > 0).
      {16,    1,  256, 576, 64,  256},  // decode, d_v=512
      {16,    1,  512, 576, 64,  256},  // decode, more keys
      { 1,   64,  256, 576, 64,  256},  // short
      { 1, 8192, 1024, 576, 64,  256},  // long prefill
  };

  std::printf("\n=== hip::dsa_sparse_fwd (GLM-5.3 + DeepSeek-V3) ===\n");
  std::printf("  %4s %6s %6s %5s %5s %5s %9s %9s %9s %9s %5s  %s\n",
              "H", "S_q", "S_kv", "dim", "tail", "topk", "us(min)", "us(med)",
              "TFLOP/s", "GB/s", "AI", "bound");

  for (const auto& c : cfgs) {
    const int H = c.H, S_q = c.S_q, S_kv = c.S_kv, dim = c.dim;
    const int tail_dim = c.tail_dim, topk = c.topk;
    const int W = dim + tail_dim, d_v = dim - tail_dim;
    const float sm_scale = (1.0f / std::sqrt((float)W)) * 1.4426950408889634f;

    const size_t nq = (size_t)S_q * H * W;
    const size_t nkv = (size_t)S_kv * W;
    const size_t nidx = (size_t)S_q * topk;
    const size_t nout = (size_t)S_q * H * d_v;
    std::vector<float> q(nq), kv(nkv);
    std::vector<int32_t> idx(nidx);
    for (size_t i = 0; i < nq; ++i)  q[i]  = rnd(1, (int)i);
    for (size_t i = 0; i < nkv; ++i) kv[i] = rnd(2, (int)i);
    for (int i = 0; i < S_q; ++i)
      for (int k = 0; k < topk; ++k)
        idx[(size_t)i * topk + k] =
            (k % 10 == 9) ? -1 : (int32_t)(((unsigned)(i * 31 + k * 7)) %
                                           (unsigned)S_kv);
    std::vector<uint16_t> qb(nq), kb(nkv), out(nout, 0);
    for (size_t i = 0; i < nq; ++i)  qb[i] = f2bf(q[i]);
    for (size_t i = 0; i < nkv; ++i) kb[i] = f2bf(kv[i]);

    uint16_t *dq, *dkv, *dout;
    int32_t* didx;
    float* dlse;
    check_hip(hipMalloc(&dq,   nq   * 2), "q");
    check_hip(hipMalloc(&dkv,  nkv  * 2), "kv");
    check_hip(hipMalloc(&didx, nidx * 4), "idx");
    check_hip(hipMalloc(&dout, nout * 2), "out");
    check_hip(hipMalloc(&dlse, (size_t)S_q * H * 4), "lse");
    check_hip(hipMemcpy(dq,   qb.data(),  nq   * 2, hipMemcpyHostToDevice),
              "cpyq");
    check_hip(hipMemcpy(dkv,  kb.data(),  nkv  * 2, hipMemcpyHostToDevice),
              "cpykv");
    check_hip(hipMemcpy(didx, idx.data(), nidx * 4, hipMemcpyHostToDevice),
              "cpyidx");

    int bq = 0, th = 0, bi = 0, ii = 0;
    vkernels::kernels::dsa_config_for(S_q, H, dim, topk, &bq, &th, &bi, &ii);

    auto L = [&] {
      vkernels::kernels::hip::dsa_sparse_fwd(
          S_q, S_kv, H, dim, tail_dim, topk, 1, bi, ii, sm_scale,
          /*return_lse=*/true, dq, dkv, didx, dout, dlse);
    };
    auto r = bench_us(L, start, stop);

    // FLOPs: per (i,h): 2*W (q.k) + 2*d_v (w.v), over topk keys.
    double flops = 2.0 * S_q * H * topk * (W + d_v);
    double tflops = flops / (r.median_us / 1e6) / 1e12;
    // Actual HBM: q read once, out written once, kv re-read once per head
    // (shared kv_group==1), indices read once.
    double bytes = 2.0 * ((double)S_q * H * W           // q read once
                        + (double)S_q * H * d_v          // out written once
                        + (double)S_q * H * topk * W)    // kv re-read per head
                 + 4.0 * (double)S_q * topk;             // indices (int32)
    double gbs = bytes / (r.median_us / 1e6) / 1e9;
    double ai = flops / bytes;
    const char* bound = ai < (info.tflops * 1e3 / info.bw) ? "mem" : "comp";
    std::printf("  %4d %6d %6d %5d %5d %5d %9.1f %9.1f %9.3f %9.1f %5.1f  %s"
                "  [BQ=%d]\n",
                H, S_q, S_kv, dim, tail_dim, topk, r.min_us, r.median_us,
                tflops, gbs, ai, bound, bq);

    check_hip(hipFree(dq),   "fq");
    check_hip(hipFree(dkv),  "fkv");
    check_hip(hipFree(didx), "fidx");
    check_hip(hipFree(dout), "fout");
    check_hip(hipFree(dlse), "flse");
  }
  std::printf("  Roof: %.0f TFLOP/s bf16, %.0f GB/s HBM, ridge ~%.0f FLOP/B\n",
              info.tflops, info.bw, info.tflops * 1e3 / info.bw);
}

// ---------------------------------------------------------------------------
// Per-shape autotuner: sweep BQ via dsa_sparse_fwd_with_tile at the key
// serving shapes, printing the median-us row and the best BQ so
// dsa_config_for can be regenerated from measured data. (block_I /
// inner_iter / bn_kv are reserved by the baseline; only BQ is swept.)
static void autotune(const GpuInfo& info, hipEvent_t start, hipEvent_t stop) {
  (void)info;
  struct Cfg { int H, S_q, S_kv, dim, tail_dim, topk; };
  const Cfg cfgs[] = {
      {64,   1, 2048, 256,  0, 2048},  // GLM-5.3 decode, full topk
      {16,   1,  256, 576, 64,  256},  // DeepSeek-V3 decode
      { 8,  64,  256, 256,  0,  128},  // short
      { 1, 8192, 1024, 256,  0,  128}, // long prefill
  };
  const int bqs[]  = {1, 2, 4, 8};
  constexpr int nBQ = sizeof(bqs) / sizeof(bqs[0]);

  for (const auto& c : cfgs) {
    const int H = c.H, S_q = c.S_q, S_kv = c.S_kv, dim = c.dim;
    const int tail_dim = c.tail_dim, topk = c.topk;
    const int W = dim + tail_dim, d_v = dim - tail_dim;
    const float sm_scale = (1.0f / std::sqrt((float)W)) * 1.4426950408889634f;
    const size_t nq = (size_t)S_q * H * W, nkv = (size_t)S_kv * W;
    const size_t nidx = (size_t)S_q * topk, nout = (size_t)S_q * H * d_v;
    std::vector<uint16_t> q(nq), kv(nkv), out(nout, 0);
    std::vector<int32_t> idx(nidx);
    for (size_t i = 0; i < nq; ++i)  q[i]  = f2bf(rnd(1, (int)i));
    for (size_t i = 0; i < nkv; ++i) kv[i] = f2bf(rnd(2, (int)i));
    for (int i = 0; i < S_q; ++i)
      for (int k = 0; k < topk; ++k)
        idx[(size_t)i * topk + k] =
            (k % 10 == 9) ? -1 : (int32_t)(((unsigned)(i * 31 + k * 7)) %
                                           (unsigned)S_kv);
    uint16_t *dq, *dkv, *dout;
    int32_t* didx;
    float* dlse;
    check_hip(hipMalloc(&dq,   nq   * 2), "q");
    check_hip(hipMalloc(&dkv,  nkv  * 2), "kv");
    check_hip(hipMalloc(&didx, nidx * 4), "idx");
    check_hip(hipMalloc(&dout, nout * 2), "out");
    check_hip(hipMalloc(&dlse, (size_t)S_q * H * 4), "lse");
    check_hip(hipMemcpy(dq,   q.data(),   nq   * 2, hipMemcpyHostToDevice),
              "cpyq");
    check_hip(hipMemcpy(dkv,  kv.data(),  nkv  * 2, hipMemcpyHostToDevice),
              "cpykv");
    check_hip(hipMemcpy(didx, idx.data(), nidx * 4, hipMemcpyHostToDevice),
              "cpyidx");

    std::printf("\n=== DSA autotuner (H=%d S_q=%d dim=%d tail=%d topk=%d):"
                " us(med) per BQ ===\n",
                H, S_q, dim, tail_dim, topk);
    int best_bq = bqs[0]; double best_us = 1e18;
    for (int bq : bqs) {
      auto L = [&] {
        vkernels::kernels::hip::dsa_sparse_fwd_with_tile(
            S_q, S_kv, H, dim, tail_dim, topk, 1, sm_scale, /*lse=*/true,
            dq, dkv, didx, dout, dlse, bq, /*block_I=*/64,
            /*inner_iter=*/1, /*bn_kv=*/0);
      };
      auto r = bench_us(L, start, stop, 3, 60);
      std::printf("  BQ=%-2d  %.1f us\n", bq, r.median_us);
      if (r.median_us < best_us) { best_us = r.median_us; best_bq = bq; }
    }
    std::printf("  best: BQ=%d  (%.1f us)\n", best_bq, best_us);
    check_hip(hipFree(dq),   "fq");
    check_hip(hipFree(dkv),  "fkv");
    check_hip(hipFree(didx), "fidx");
    check_hip(hipFree(dout), "fout");
    check_hip(hipFree(dlse), "flse");
  }
}

// ---------------------------------------------------------------------------
int main() {
  auto info = get_gpu_info();
  std::printf("GPU: %s  arch=%d  bf16=%.0f TFLOP/s  HBM=%.0f GB/s\n",
              info.name.c_str(), info.arch, info.tflops, info.bw);
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  bench_shapes(info, start, stop);
  autotune(info, start, stop);
  hipEventDestroy(start); hipEventDestroy(stop);
  return 0;
}
