// meta/benchmarks/bench_mhc.hip
//
// Micro-benchmark for the MHC multi-head hybrid-attention pre-norm kernels
// (issue #51, part 2) on AMD gfx942 (MI300A). For each GLM-5.3 shape it
// reports, against the MI300A roof (~1307 TFLOP/s bf16, ~5300 GB/s HBM3,
// ridge ~247 FLOP/byte):
//
//   mhc_pre_gemm_sqrsum : median latency, achieved TFLOP/s, effective GB/s,
//                         arithmetic intensity and the binding resource.
//                         FLOPs = 2*num_tokens*hc_hidden*hc_mult3 (GEMM)
//                                 + 2*num_tokens*hc_hidden (sqrsum).
//                         Bytes = x read once (bf16) + fn read once (fp32)
//                                 + out+sqrsum written once (fp32). The
//                                 baseline streams fn per output column
//                                 (not coalesced) -- the gap to the roof is
//                                 what the tiled, swizzled-shared follow-on
//                                 closes.
//   mhc_post            : median latency, TFLOP/s, GB/s. FLOPs =
//                         2*num_tokens*hc*hidden*(hc+1). Bytes = a,b,c,d read
//                         once + out written once (bf16).
//
// Built only with VKERNELS_HAS_HIP + VKERNELS_BUILD_BENCHMARKS.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/mhc.hpp"

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
// mhc_pre_gemm_sqrsum: GLM-5.3 (hc_mult=4, hidden=4096 -> hc_hidden=16384,
// hc_mult3=24) and the larger branch (hidden=7168 -> hc_hidden=28672).
static void bench_pre(const GpuInfo& info, hipEvent_t start, hipEvent_t stop) {
  struct Cfg { int num_tokens, hc_mult, hidden; };
  const Cfg cfgs[] = {
      {1, 2, 2},                 // tiny
      {1, 4, 4096},              // GLM-5.3 decode
      {7, 4, 4096},              // GLM-5.3, multi-token
      {1, 4, 7168},              // larger branch
  };
  std::printf("\n=== hip::mhc_pre_gemm_sqrsum ===\n");
  std::printf("  %4s %8s %6s %8s %9s %9s %9s %5s  %s\n",
              "n", "hc_hidden", "hc_mult3", "us(min)", "us(med)",
              "TFLOP/s", "GB/s", "AI", "bound");
  for (const auto& c : cfgs) {
    const int hc_hidden = c.hc_mult * c.hidden;
    const int hc_mult3 = c.hc_mult * (2 + c.hc_mult);
    const size_t nx = (size_t)c.num_tokens * hc_hidden;
    const size_t nfn = (size_t)hc_mult3 * hc_hidden;
    const size_t nout = (size_t)c.num_tokens * hc_mult3;
    std::vector<uint16_t> x(nx);
    std::vector<float> fn(nfn);
    for (size_t i = 0; i < nx; ++i) x[i] = f2bf(rnd(1, (int)i));
    for (size_t i = 0; i < nfn; ++i) fn[i] = rnd(2, (int)i);
    uint16_t *dx; float *dfn, *dout, *dsq;
    check_hip(hipMalloc(&dx, nx * 2), "x");
    check_hip(hipMalloc(&dfn, nfn * 4), "fn");
    check_hip(hipMalloc(&dout, nout * 4), "out");
    check_hip(hipMalloc(&dsq, (size_t)c.num_tokens * 4), "sq");
    check_hip(hipMemcpy(dx, x.data(), nx * 2, hipMemcpyHostToDevice), "cpyx");
    check_hip(hipMemcpy(dfn, fn.data(), nfn * 4, hipMemcpyHostToDevice),
              "cpyfn");
    auto L = [&] {
      vkernels::kernels::hip::mhc_pre_gemm_sqrsum(c.num_tokens, hc_mult3,
                                                  hc_hidden, dx, dfn, dout,
                                                  dsq);
    };
    auto r = bench_us(L, start, stop);
    double flops = 2.0 * c.num_tokens * hc_hidden * hc_mult3
                 + 2.0 * c.num_tokens * hc_hidden;
    double bytes = 2.0 * nx                  // x read (bf16)
                 + 4.0 * nfn                 // fn read (fp32)
                 + 4.0 * nout                // out written (fp32)
                 + 4.0 * c.num_tokens;       // sqrsum written (fp32)
    bool subres = r.median_us < 1.0;   // below hipEventElapsedTime floor
    char tbuf[16], gbuf[16];
    if (subres) { std::snprintf(tbuf, sizeof tbuf, "%9s", "<res");
                  std::snprintf(gbuf, sizeof gbuf, "%9s", "<res"); }
    else { std::snprintf(tbuf, sizeof tbuf, "%9.3f", flops / (r.median_us / 1e6) / 1e12);
           std::snprintf(gbuf, sizeof gbuf, "%9.1f", bytes / (r.median_us / 1e6) / 1e9); }
    double ai = flops / bytes;
    const char* bound = ai < (info.tflops * 1e3 / info.bw) ? "mem" : "comp";
    std::printf("  %4d %8d %6d %8.1f %9.1f %s %s %5.1f  %s\n",
                c.num_tokens, hc_hidden, hc_mult3, r.min_us, r.median_us,
                tbuf, gbuf, ai, bound);
    check_hip(hipFree(dx), "fx");
    check_hip(hipFree(dfn), "ffn");
    check_hip(hipFree(dout), "fout");
    check_hip(hipFree(dsq), "fsq");
  }
}

// ---------------------------------------------------------------------------
// mhc_post: GLM-5.3 (hc=4, hidden=4096/7168) and wider hc.
static void bench_post(const GpuInfo& info, hipEvent_t start, hipEvent_t stop) {
  struct Cfg { int num_tokens, hc, hidden; };
  const Cfg cfgs[] = {
      {1, 2, 2},                 // tiny
      {1, 4, 4096},              // GLM-5.3 decode
      {7, 4, 4096},              // GLM-5.3, multi-token
      {1, 4, 7168},              // larger branch
      {4, 8, 1024},              // wider hc
  };
  std::printf("\n=== hip::mhc_post ===\n");
  std::printf("  %4s %4s %6s %8s %9s %9s %9s %5s  %s\n",
              "n", "hc", "hidden", "us(min)", "us(med)", "TFLOP/s", "GB/s",
              "AI", "bound");
  for (const auto& c : cfgs) {
    const size_t na = (size_t)c.num_tokens * c.hc * c.hc;
    const size_t nb = (size_t)c.num_tokens * c.hc * c.hidden;
    const size_t nc = (size_t)c.num_tokens * c.hc;
    const size_t nd = (size_t)c.num_tokens * c.hidden;
    const size_t nout = nb;
    std::vector<float> a(na), c_(nc);
    std::vector<uint16_t> b(nb), d(nd);
    for (size_t i = 0; i < na; ++i) a[i] = rnd(3, (int)i);
    for (size_t i = 0; i < nb; ++i) b[i] = f2bf(rnd(4, (int)i));
    for (size_t i = 0; i < nc; ++i) c_[i] = rnd(5, (int)i);
    for (size_t i = 0; i < nd; ++i) d[i] = f2bf(rnd(6, (int)i));
    float *da, *dc; uint16_t *db, *dd, *dout;
    check_hip(hipMalloc(&da, na * 4), "a");
    check_hip(hipMalloc(&db, nb * 2), "b");
    check_hip(hipMalloc(&dc, nc * 4), "c");
    check_hip(hipMalloc(&dd, nd * 2), "d");
    check_hip(hipMalloc(&dout, nout * 2), "out");
    check_hip(hipMemcpy(da, a.data(), na * 4, hipMemcpyHostToDevice), "cpya");
    check_hip(hipMemcpy(db, b.data(), nb * 2, hipMemcpyHostToDevice), "cpyb");
    check_hip(hipMemcpy(dc, c_.data(), nc * 4, hipMemcpyHostToDevice), "cpyc");
    check_hip(hipMemcpy(dd, d.data(), nd * 2, hipMemcpyHostToDevice), "cpyd");
    auto L = [&] {
      vkernels::kernels::hip::mhc_post(c.num_tokens, c.hc, c.hidden, da, db,
                                       dc, dd, dout);
    };
    auto r = bench_us(L, start, stop);
    double flops = 2.0 * c.num_tokens * c.hc * c.hidden * (c.hc + 1);
    double bytes = 4.0 * (na + nc)              // a, c read (fp32)
                 + 2.0 * (nb + nd)              // b, d read (bf16)
                 + 2.0 * nout;                  // out written (bf16)
    bool subres = r.median_us < 1.0;   // below hipEventElapsedTime floor
    char tbuf[16], gbuf[16];
    if (subres) { std::snprintf(tbuf, sizeof tbuf, "%9s", "<res");
                  std::snprintf(gbuf, sizeof gbuf, "%9s", "<res"); }
    else { std::snprintf(tbuf, sizeof tbuf, "%9.3f", flops / (r.median_us / 1e6) / 1e12);
           std::snprintf(gbuf, sizeof gbuf, "%9.1f", bytes / (r.median_us / 1e6) / 1e9); }
    double ai = flops / bytes;
    const char* bound = ai < (info.tflops * 1e3 / info.bw) ? "mem" : "comp";
    std::printf("  %4d %4d %6d %8.1f %9.1f %s %s %5.1f  %s\n",
                c.num_tokens, c.hc, c.hidden, r.min_us, r.median_us, tbuf,
                gbuf, ai, bound);
    check_hip(hipFree(da), "fa");
    check_hip(hipFree(db), "fb");
    check_hip(hipFree(dc), "fc");
    check_hip(hipFree(dd), "fd");
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
  bench_pre(info, start, stop);
  bench_post(info, start, stop);
  std::printf("\nRoof: %.0f TFLOP/s bf16, %.0f GB/s HBM, ridge ~%.0f FLOP/B\n",
              info.tflops, info.bw, info.tflops * 1e3 / info.bw);
  hipEventDestroy(start); hipEventDestroy(stop);
  return 0;
}
