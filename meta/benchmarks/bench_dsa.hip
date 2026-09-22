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
#include <string>
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
      // ROCm 6.3 event quirk (observed on MI300A with rapid-fire event
      // pairs around DVFS clock transitions): a rare event pair reports
      // elapsed exactly 0.0 even though the kernels ran (verified by
      // output checksum). Discard such samples instead of poisoning
      // min/median; a window where EVERY sample is 0 widens once and
      // ultimately yields the {0,0,0} result (a visible broken row -- the
      // login-node no-gfx-agent signature), rather than a hang.
      if (ms <= 0.0f) continue;
      times.push_back(ms * 1000.0f);
    }
    const int n = (int)times.size();
    if (n == 0) {
      iters *= 2;
      continue;
    }
    double sum = 0;
    for (double t : times) sum += t;
    double mean = sum / n;
    double sum2 = 0;
    for (double t : times) sum2 += t * t;
    double var = sum2 / n - mean * mean;
    double stddev = std::sqrt(std::max(0.0, var));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {times[0], times[n / 2], mean};
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
// Tile sweep + tuning-store persist. The launch tile is (bq, threads,
// block_I, inner_iter) from dsa_config_for; with_tile ignores block_I /
// inner_iter today (reserved for the tiled-key prefetch) so the live
// dimension is bq -- flat at decode (S_q=1 always launches one row per
// block), live at prefill where ceil(S_q/bq)*H sets the grid. The sweep
// therefore covers the decode serving shapes (to pin the store where the
// formula is expected to agree) AND the prefill shapes (where a real
// override can exist). A 6-point degenerate sweep on one decode shape
// documents that the reserved params are timing-invariant. --persist
// writes the per-shape argmin under "dsa_sparse_fwd_tile" (key
// {S_q,H,dim,tail_dim,topk}, params bq/block_I/inner_iter) so the
// dispatch override in dsa.hip picks it up on the next run.
static void tile_sweep(const GpuInfo& info, hipEvent_t start, hipEvent_t stop,
                       bool do_persist) {
  (void)info;
  struct Cfg { int H, S_q, S_kv, dim, tail_dim, topk; const char* tag; };
  const Cfg cfgs[] = {
      {64,    1, 2048, 256,  0, 2048, "GLM decode full-topk"},
      {64,    1,  256, 256,  0,  256, "GLM decode topk=256"},
      {64,    1,  128, 256,  0,  128, "GLM decode topk=128"},
      {16,    1,  512, 576, 64,  256, "DSv3 decode"},
      {16,    1,  256, 576, 64,  256, "DSv3 decode S_kv=256"},
      { 8,   64,  256, 256,  0,  128, "GLM short prefill"},
      { 1, 8192, 1024, 256,  0,  128, "GLM long prefill"},
      { 1, 8192, 1024, 576, 64,  256, "DSv3 long prefill"},
  };
  const int bqs[] = {1, 2, 4, 8};
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

    int fbq = 0, fth = 0, fbi = 0, fii = 0;
    vkernels::kernels::dsa_config_for(S_q, H, dim, topk, &fbq, &fth, &fbi,
                                      &fii);
    std::printf("\n=== DSA tile sweep (H=%d S_q=%d dim=%d tail=%d topk=%d;"
                " %s): formula BQ=%d block_I=%d inner_iter=%d ===\n",
                H, S_q, dim, tail_dim, topk, c.tag, fbq, fbi, fii);
    std::printf("    %3s %6s %9s %9s  %s\n", "bq", "", "us(min)", "us(med)",
                "note");
    int best_bq = bqs[0]; double best_us = 1e18;
    for (int bq : bqs) {
      auto L = [&] {
        vkernels::kernels::hip::dsa_sparse_fwd_with_tile(
            S_q, S_kv, H, dim, tail_dim, topk, 1, sm_scale, /*lse=*/true,
            dq, dkv, didx, dout, dlse, bq, /*block_I=*/fbi,
            /*inner_iter=*/fii, /*bn_kv=*/0);
      };
      auto r = bench_us(L, start, stop, 3, 60);
      std::printf("    %3d %6s %9.1f %9.1f  %s\n", bq, "", r.min_us,
                  r.median_us, bq == fbq ? "<- formula" : "");
      if (r.median_us < best_us) { best_us = r.median_us; best_bq = bq; }
    }
    if (S_q == 1) {
      // Reserved-param flatness probe: block_I x inner_iter at fixed bq.
      // with_tile (void)s both today, so the row MUST be flat; if it ever
      // is not, the baseline grew a live block_I and this probe catches it.
      std::printf("    (reserved-param probe at bq=%d:)", best_bq);
      for (int bi : {32, 64, 128}) {
        for (int ii : {1, 2}) {
          if (topk % (bi * ii) != 0) continue;  // with_tile precondition
          auto L = [&] {
            vkernels::kernels::hip::dsa_sparse_fwd_with_tile(
                S_q, S_kv, H, dim, tail_dim, topk, 1, sm_scale, true,
                dq, dkv, didx, dout, dlse, best_bq, bi, ii, 0);
          };
          auto r = bench_us(L, start, stop, 3, 30);
          std::printf("\n      block_I=%-3d inner_iter=%d  %8.1f us", bi, ii,
                      r.median_us);
        }
      }
      std::printf("\n");
    }
    std::printf("    best: BQ=%d (%.1f us)%s\n", best_bq, best_us,
                best_bq == fbq ? "  formula agrees" : "  OVERRIDE");
    if (do_persist) {
      vkernels::core::tuning::persist(
          "dsa_sparse_fwd_tile", {S_q, H, dim, tail_dim, topk},
          {{"bq", best_bq}, {"block_I", fbi}, {"inner_iter", fii}},
          "bench_dsa");
      std::printf("    -> stored bq=%d block_I=%d inner_iter=%d"
                  "  [dsa_sparse_fwd_tile]\n", best_bq, fbi, fii);
    }
    check_hip(hipFree(dq),   "fq");
    check_hip(hipFree(dkv),  "fkv");
    check_hip(hipFree(didx), "fidx");
    check_hip(hipFree(dout), "fout");
    check_hip(hipFree(dlse), "flse");
  }
}

// ---------------------------------------------------------------------------
// Split-key sweep (decode occupancy fix, issue #51 follow-up): the plain
// forward's grid is (ceil(S_q/BQ), H) -- 64 blocks on 228 CUs at the
// GLM-5.3-Flash decode shape -- so decode is occupancy-bound, not
// bandwidth-bound (docs/performance/dsa/gfx942.md). dsa_sparse_fwd_split
// divides the topk index range across `split` partial blocks + a combine
// pass (grouping-independent, exact log2-domain merge). This sweep measures
// the decode shapes across split and marks the dsa_sparse_fwd_split_for
// recommendation so the launcher's default can be regenerated from data.
static void split_sweep(const GpuInfo& info, hipEvent_t start,
                        hipEvent_t stop) {
  struct Cfg { int H, S_q, S_kv, dim, tail_dim, topk; };
  const Cfg cfgs[] = {
      {64, 1, 2048, 256, 0, 2048},  // GLM-5.3 decode, full topk
      {64, 1,  256, 256, 0,  256},  // GLM-5.3 decode, serving topk
      {64, 1,  128, 256, 0,  128},  // GLM-5.3 decode, small topk
      {16, 1,  512, 576, 64,  256}, // DeepSeek-V3 decode
  };
  const int splits[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  constexpr int nS = sizeof(splits) / sizeof(splits[0]);

  std::printf("\n=== DSA split-key sweep (decode shapes; split x partial +"
              " combine) ===\n");
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
    float *dlse, *dpart_out, *dpart_lse;
    check_hip(hipMalloc(&dq,   nq   * 2), "q");
    check_hip(hipMalloc(&dkv,  nkv  * 2), "kv");
    check_hip(hipMalloc(&didx, nidx * 4), "idx");
    check_hip(hipMalloc(&dout, nout * 2), "out");
    check_hip(hipMalloc(&dlse, (size_t)S_q * H * 4), "lse");
    // Scratch for the largest legal split in the sweep (<= topk).
    int smax = 1;
    for (int s : splits) if (s <= topk && s > smax) smax = s;
    check_hip(hipMalloc(&dpart_out,
                        (size_t)S_q * H * smax * d_v * 4), "part_out");
    check_hip(hipMalloc(&dpart_lse, (size_t)S_q * H * smax * 4), "part_lse");
    check_hip(hipMemcpy(dq,   q.data(),   nq   * 2, hipMemcpyHostToDevice),
              "cpyq");
    check_hip(hipMemcpy(dkv,  kv.data(),  nkv  * 2, hipMemcpyHostToDevice),
              "cpykv");
    check_hip(hipMemcpy(didx, idx.data(), nidx * 4, hipMemcpyHostToDevice),
              "cpyidx");

    const int rec = vkernels::kernels::dsa_sparse_fwd_split_for(
        S_q, H, topk, /*block_I=*/64, /*num_cu=*/228);
    std::printf("\n  H=%d S_q=%d topk=%d (dim=%d tail=%d):"
                " dsa_sparse_fwd_split_for = %d\n",
                H, S_q, topk, dim, tail_dim, rec);
    std::printf("    %5s %9s %9s %9s %8s  %s\n", "split", "us(min)",
                "us(med)", "GB/s", "speedup", "note");
    double base_us = 0.0;
    for (int s : splits) {
      if (s > topk) continue;
      auto L = [&] {
        vkernels::kernels::hip::dsa_sparse_fwd_split(
            S_q, S_kv, H, dim, tail_dim, topk, 1, /*block_I=*/64,
            /*inner_iter=*/1, sm_scale, /*return_lse=*/true, s,
            dq, dkv, didx, dout, dlse, dpart_out, dpart_lse);
      };
      auto r = bench_us(L, start, stop, 3, 60);
      if (s == 1) base_us = r.median_us;
      // Same byte model as the main table: kv re-read once per head.
      double bytes = 2.0 * ((double)S_q * H * W + (double)S_q * H * d_v +
                            (double)S_q * H * topk * W)
                   + 4.0 * (double)S_q * topk;
      double gbs = bytes / (r.median_us / 1e6) / 1e9;
      std::printf("    %5d %9.1f %9.1f %9.1f %7.1fx  %s\n", s, r.min_us,
                  r.median_us, gbs, base_us / r.median_us,
                  s == rec ? "<- split_for" : "");
    }
    check_hip(hipFree(dq),   "fq");
    check_hip(hipFree(dkv),  "fkv");
    check_hip(hipFree(didx), "fidx");
    check_hip(hipFree(dout), "fout");
    check_hip(hipFree(dlse), "flse");
    check_hip(hipFree(dpart_out), "fpart_out");
    check_hip(hipFree(dpart_lse), "fpart_lse");
  }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  // --persist [dir]: write the per-shape tile winners into the native
  // tuning store (same contract as bench_dsa_split137.hip). A bare
  // --persist uses the store's default location; --persist=<dir> pins
  // (and exports) VKERNELS_TUNING_CACHE.
  bool do_persist = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--persist") {
      do_persist = true;
    } else if (arg.rfind("--persist=", 0) == 0) {
      do_persist = true;
      ::setenv("VKERNELS_TUNING_CACHE", arg.substr(10).c_str(), 1);
    }
  }
  if (do_persist && !vkernels::core::tuning::enabled()) {
    std::printf("--persist: tuning store is off (VKERNELS_TUNING_CACHE=off)\n");
    do_persist = false;
  }
  // A winner measured under a diagnostic dispatch variant is the wrong
  // artifact -- only the default env is what serving runs.
  const char* env_override = nullptr;
  for (const char* k : {"VK_DSA_DB", "VK_DSA_VEC", "VK_DSA_FUSE",
                        "VK_DSA_VECG"})
    if (std::getenv(k)) env_override = k;
  if (do_persist && env_override) {
    std::printf("--persist: refused, %s selects a diagnostic dispatch "
                "variant (unset it to tune the serving path)\n",
                env_override);
    do_persist = false;
  }

  auto info = get_gpu_info();
  std::printf("GPU: %s  arch=%d  bf16=%.0f TFLOP/s  HBM=%.0f GB/s\n",
              info.name.c_str(), info.arch, info.tflops, info.bw);
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  bench_shapes(info, start, stop);
  autotune(info, start, stop);
  split_sweep(info, start, stop);
  tile_sweep(info, start, stop, do_persist);
  if (do_persist)
    std::printf("\n  Tile winners stored under %s; delete the file (or the "
                "store) to re-tune.\n",
                vkernels::core::tuning::store_dir().c_str());
  hipEventDestroy(start); hipEventDestroy(stop);
  return 0;
}
