// meta/benchmarks/bench_mla.hip
//
// Micro-benchmark for the MLA forward (issue #21, Kimi-K3 latent attention)
// on AMD gfx942 (MI300A). For each K3-shaped (H, S_q) prefill / decode case
// it reports, against the MI300A roof (≈1307 TFLOP/s bf16, ≈5300 GB/s HBM3,
// ridge ≈247 FLOP/byte):
//
//   * median / min latency (us),
//   * achieved TFLOP/s  = 2*S_q*S_kv*(D_q + kv_lora_rank)*H / t,
//   * effective GB/s    = actual HBM bytes / t,
//   * arithmetic intensity and the binding resource (mem / comp).
//
// The absorbed-form kernel streams keys/values once per query tile (no
// cross-query-tile KV reuse in this baseline), so the ACTUAL HBM traffic is
// ~(S_q/BQ)× larger than the ideal once-only traffic — the gap to the roof
// is what the tiled-key-loop / persistent-kernel follow-ons close. A small
// autotuner sweeps (BQ, BN_kv) via mla_fwd_with_tile so mla_config_for can
// be regenerated from measured data.
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

#include "vkernels/kernels/mla.hpp"

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

static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f;  // [-1, 1)
}

// ---------------------------------------------------------------------------
// K3 latent-attention shapes. kv_lora_rank=512, qk_rope_head_dim=64 →
// D_q=576. H sweeps the head count; S_q sweeps decode→prefill. S_kv=S_q
// (self-attention) for the throughput column; a long-context column
// (S_kv=8192) stresses the per-query-tile KV re-read.
static void bench_shapes(const GpuInfo& info, hipEvent_t start,
                         hipEvent_t stop) {
  const int kv = 512, pe = 64;
  const int Dq = kv + pe;            // 576
  const float scale = 1.0f / std::sqrt((float)Dq);

  struct Cfg { int H, S_q, S_kv; };
  const Cfg cfgs[] = {
      {1,    1, 8192},   // decode, long KV
      {1,   64, 8192},   // small batch, long KV
      {8,   64,   64},   // short context
      {16,  512,  512},  // prefill
      {128,  512, 512},  // wide head count
      {1,  8192, 8192},  // long prefill
  };
  const int B = 1;

  std::printf("\n=== hip::mla_fwd (K3: kv=%d pe=%d D_q=%d) ===\n", kv, pe, Dq);
  std::printf("  %4s %6s %6s %9s %9s %9s %9s %5s  %s\n",
              "H", "S_q", "S_kv", "us(min)", "us(med)", "TFLOP/s", "GB/s",
              "AI", "bound");

  for (const auto& c : cfgs) {
    const int H = c.H, S_q = c.S_q, S_kv = c.S_kv;
    const size_t nq = (size_t)B * H * S_q * Dq;
    const size_t nkc = (size_t)B * S_kv * kv;
    const size_t nkpe = (size_t)B * S_kv * pe;
    const size_t nv = (size_t)B * S_kv * kv;
    const size_t nout = (size_t)B * H * S_q * kv;
    std::vector<float> q(nq), kc(nkc), kpe(nkpe), v(nv), out(nout, 0.0f);
    for (size_t i = 0; i < nq; ++i)   q[i]    = rnd(1, (int)i);
    for (size_t i = 0; i < nkc; ++i)  kc[i]   = rnd(2, (int)i);
    for (size_t i = 0; i < nkpe; ++i) kpe[i]  = rnd(3, (int)i);
    for (size_t i = 0; i < nv; ++i)   v[i]    = rnd(4, (int)i);

    float *dq, *dkc, *dkpe, *dv, *dout;
    check_hip(hipMalloc(&dq,   nq   * 4), "q");
    check_hip(hipMalloc(&dkc,  nkc  * 4), "kc");
    check_hip(hipMalloc(&dkpe, nkpe * 4), "kpe");
    check_hip(hipMalloc(&dv,   nv   * 4), "v");
    check_hip(hipMalloc(&dout, nout * 4), "out");
    check_hip(hipMemcpy(dq,   q.data(),   nq   * 4, hipMemcpyHostToDevice),
              "cpyq");
    check_hip(hipMemcpy(dkc,  kc.data(),  nkc  * 4, hipMemcpyHostToDevice),
              "cpykc");
    check_hip(hipMemcpy(dkpe, kpe.data(), nkpe * 4, hipMemcpyHostToDevice),
              "cpykpe");
    check_hip(hipMemcpy(dv,   v.data(),   nv   * 4, hipMemcpyHostToDevice),
              "cpyv");

    // Tile the kernel will use (for the actual-HBM-traffic model).
    int bq = 0, bn = 0, th = 0;
    vkernels::kernels::mla_config_for(S_q, kv, pe, &bq, &bn, &th);

    auto L = [&] {
      vkernels::kernels::hip::mla_fwd(
          B, H, S_q, S_kv, 0, 0, kv, pe, scale, dq, dkc, dkpe, dv, dout);
    };
    auto r = bench_us(L, start, stop);

    // FLOPs: scores 2*S_q*S_kv*D_q + output 2*S_q*S_kv*kv_lora_rank, per head.
    double flops = 2.0 * S_q * S_kv * (Dq + kv) * H * B;
    double tflops = flops / (r.median_us / 1e6) / 1e12;

    // Actual HBM traffic (streaming baseline): q read once, out written once,
    // and k_c/k_pe/v_c re-read once per query tile (no cross-tile reuse).
    const int n_tiles = (S_q + bq - 1) / bq;
    double bytes = 4.0 * ((double)B * H * S_q * Dq            // q read once
                        + (double)B * H * S_q * kv            // out written once
                        + n_tiles * (double)B * H * S_kv * (kv + pe)   // k_c+k_pe re-read
                        + n_tiles * (double)B * H * S_kv * kv);        // v_c re-read
    double gbs = bytes / (r.median_us / 1e6) / 1e9;
    double ai = flops / bytes;
    const char* bound = ai < (info.tflops * 1e3 / info.bw) ? "mem" : "comp";
    std::printf("  %4d %6d %6d %9.1f %9.1f %9.3f %9.1f %5.1f  %s"
                "  [BQ=%d BN=%d th=%d]\n",
                H, S_q, S_kv, r.min_us, r.median_us, tflops, gbs, ai, bound,
                bq, bn, th);

    check_hip(hipFree(dq),   "fq");
    check_hip(hipFree(dkc),  "fkc");
    check_hip(hipFree(dkpe), "fkpe");
    check_hip(hipFree(dv),   "fv");
    check_hip(hipFree(dout), "fout");
  }
  std::printf("  Roof: %.0f TFLOP/s bf16, %.0f GB/s HBM, ridge ~%.0f FLOP/B\n",
              info.tflops, info.bw, info.tflops * 1e3 / info.bw);
}

// ---------------------------------------------------------------------------
// Per-shape autotuner: sweep (BQ, BN_kv) tiles via mla_fwd_with_tile at the
// key serving shapes, printing the median-us matrix and the best tile so
// mla_config_for can be regenerated from measured data.
static void autotune(const GpuInfo& info, hipEvent_t start, hipEvent_t stop) {
  (void)info;
  const int kv = 512, pe = 64, Dq = kv + pe, B = 1;
  const float scale = 1.0f / std::sqrt((float)Dq);

  struct Cfg { int H, S_q, S_kv; };
  const Cfg cfgs[] = {{1, 64, 8192}, {8, 64, 64}, {16, 512, 512},
                      {128, 512, 512}, {1, 8192, 8192}};
  const int bqs[]  = {1, 2, 4, 8, 16, 32};
  const int bns[]  = {8, 16, 32, 64, 128};
  constexpr int nBQ = sizeof(bqs) / sizeof(bqs[0]);
  constexpr int nBN = sizeof(bns) / sizeof(bns[0]);

  for (const auto& c : cfgs) {
    const int H = c.H, S_q = c.S_q, S_kv = c.S_kv;
    const size_t nq = (size_t)B * H * S_q * Dq;
    const size_t nkc = (size_t)B * S_kv * kv, nkpe = (size_t)B * S_kv * pe;
    const size_t nv = (size_t)B * S_kv * kv, nout = (size_t)B * H * S_q * kv;
    std::vector<float> q(nq), kc(nkc), kpe(nkpe), v(nv);
    for (size_t i = 0; i < nq; ++i)  q[i]   = rnd(1, (int)i);
    for (size_t i = 0; i < nkc; ++i) kc[i]  = rnd(2, (int)i);
    for (size_t i = 0; i < nkpe; ++i) kpe[i] = rnd(3, (int)i);
    for (size_t i = 0; i < nv; ++i)  v[i]   = rnd(4, (int)i);
    float *dq, *dkc, *dkpe, *dv, *dout;
    check_hip(hipMalloc(&dq,   nq   * 4), "q");
    check_hip(hipMalloc(&dkc,  nkc  * 4), "kc");
    check_hip(hipMalloc(&dkpe, nkpe * 4), "kpe");
    check_hip(hipMalloc(&dv,   nv   * 4), "v");
    check_hip(hipMalloc(&dout, nout * 4), "out");
    check_hip(hipMemcpy(dq,   q.data(),   nq   * 4, hipMemcpyHostToDevice),
              "cpyq");
    check_hip(hipMemcpy(dkc,  kc.data(),  nkc  * 4, hipMemcpyHostToDevice),
              "cpykc");
    check_hip(hipMemcpy(dkpe, kpe.data(), nkpe * 4, hipMemcpyHostToDevice),
              "cpykpe");
    check_hip(hipMemcpy(dv,   v.data(),   nv   * 4, hipMemcpyHostToDevice),
              "cpyv");

    std::printf("\n=== MLA autotuner (H=%d S_q=%d S_kv=%d): us(med) per tile ===\n",
                H, S_q, S_kv);
    std::printf("  %5s", "BQ\\BN");
    for (int bn : bns) std::printf(" %8d", bn);
    std::printf("  %s\n", "best(BQ,BN)");
    int best_bq = bqs[0], best_bn = bns[0]; double best_us = 1e18;
    for (int iq = 0; iq < nBQ; ++iq) {
      const int bq = bqs[iq];
      std::printf("  %5d", bq);
      for (int ib = 0; ib < nBN; ++ib) {
        const int bn = bns[ib];
        auto L = [&] {
          vkernels::kernels::hip::mla_fwd_with_tile(
              B, H, S_q, S_kv, 0, 0, kv, pe, scale, dq, dkc, dkpe, dv,
              dout, bq, bn);
        };
        auto r = bench_us(L, start, stop, 3, 60);
        std::printf(" %8.1f", r.median_us);
        if (r.median_us < best_us) {
          best_us = r.median_us; best_bq = bq; best_bn = bn;
        }
      }
      std::printf("\n");
    }
    std::printf("  best: BQ=%d BN=%d  (%.1f us)\n", best_bq, best_bn, best_us);
    check_hip(hipFree(dq),   "fq");
    check_hip(hipFree(dkc),  "fkc");
    check_hip(hipFree(dkpe), "fkpe");
    check_hip(hipFree(dv),   "fv");
    check_hip(hipFree(dout), "fout");
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
