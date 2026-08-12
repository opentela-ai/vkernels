// meta/benchmarks/bench_moe.hip
//
// Micro-benchmarks for AMD gfx942/gfx90a MXFP4 MoE kernel primitives.
// Measures fp4→bf16 dequant, LDS fill, K16 MFMA, and K32 emulation (2xK16).
// Compares against theoretical roofs for the current GPU.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/moe.hpp"
#include "vkernels/util/config.hpp"

// ---------------------------------------------------------------------------
static void check_hip(hipError_t err, const char* ctx) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", ctx, hipGetErrorString(err));
    std::exit(1);
  }
}

struct BenchResult { double min_us, median_us, mean_us, std_us; int iters; };

// Measure using HIP events for microsecond-precision GPU timing.
static BenchResult bench_kernel_us(const std::function<void()>& launch,
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
      times.push_back(ms * 1000.0f);  // convert to us
    }
    double sum = 0, sum2 = 0;
    for (double t : times) { sum += t; sum2 += t * t; }
    double mean = sum / iters;
    double var = sum2 / iters - mean * mean;
    double stddev = std::sqrt(std::max(0.0, var));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {static_cast<double>(times[0]), static_cast<double>(times[iters / 2]),
              mean, stddev, iters};
    }
    iters *= 2;
  }
  return {};
}

// Millisecond-precision bench (for longer-running kernels).
static BenchResult bench_kernel_ms(const std::function<void()>& launch,
                                   int warmup = 5, int max_iters = 500) {
  (void)hipDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) launch();
  std::vector<double> times;
  int iters = 20;
  while (iters <= max_iters) {
    times.clear();
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      launch();
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      times.push_back(us);
    }
    double sum = 0, sum2 = 0;
    for (double t : times) { sum += t; sum2 += t * t; }
    double mean = sum / iters;
    double var = sum2 / iters - mean * mean;
    double stddev = std::sqrt(std::max(0.0, var));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {times[0], times[iters / 2], mean, stddev, iters};
    }
    iters *= 2;
  }
  return {};
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// fp4→bf16 helpers
// ---------------------------------------------------------------------------
__host__ __device__ static uint32_t fp4nib_f32(int n) {
  int s = (n>>3)&1, e = (n>>1)&3, m = n&1;
  if (e==0) return m ? ((s?0x80000000u:0u)|0x3E800000u) : (s?0x80000000u:0u);
  if (e==3) return m ? 0x7FC00000u : (s?0xFF800000u:0x7F800000u);
  uint32_t eb = (127u+(uint32_t)e-1u)<<23, mb = m?0x400000u:0u;
  return (s?0x80000000u:0u)|eb|mb;
}
__host__ __device__ static uint16_t f32_bf16(uint32_t b) {
  uint32_t lsb = (b>>16)&1; b += 0x7FFFu+lsb; return (uint16_t)(b>>16);
}

__global__ void fp4_dequant_k(const uint8_t* __restrict__ in,
                              uint16_t* __restrict__ out,
                              size_t n, float scale) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint8_t b = in[i];
  uint32_t lo = fp4nib_f32(b&0xF), hi = fp4nib_f32(b>>4);
  float vl, vh;
  __builtin_memcpy(&vl, &lo, 4); __builtin_memcpy(&vh, &hi, 4);
  vl *= scale; vh *= scale;
  uint32_t bl, bh;
  __builtin_memcpy(&bl, &vl, 4); __builtin_memcpy(&bh, &vh, 4);
  out[i*2] = f32_bf16(bl); out[i*2+1] = f32_bf16(bh);
}

static void bench_fp4_dequant(const GpuInfo&) {
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  std::printf("\n=== fp4->bf16 dequant ===\n  %10s %10s %10s %8s %8s\n",
              "elems", "us", "GElem/s", "us(min)", "us(med)");
  for (int ln : {20,22,24,26,28}) {
    size_t np = 1ul<<ln, no = np*2;
    uint8_t *dp, *hp = new uint8_t[np]; uint16_t *dout;
    std::memset(hp, 0x22, np);
    check_hip(hipMalloc(&dp, np), "alloc"); check_hip(hipMalloc(&dout, no*sizeof(uint16_t)), "alloc");
    check_hip(hipMemcpy(dp, hp, np, hipMemcpyHostToDevice), "cpy");
    auto L = [&]{ fp4_dequant_k<<<(np+255)/256,256>>>(dp,dout,np,1.f); };
    auto r = bench_kernel_us(L, start, stop);
    std::printf("  %10zu %8.1f %10.3f %8.1f %8.1f\n", no, r.median_us, (no/1e9)/(r.median_us/1e6), r.min_us, r.median_us);
    check_hip(hipFree(dp), "free"); check_hip(hipFree(dout), "free"); delete[] hp;
  }
  hipEventDestroy(start); hipEventDestroy(stop);
}

// ---------------------------------------------------------------------------
// LDS fill — uint4 load; HIP emits global_load_dwordx4
// ---------------------------------------------------------------------------
__global__ void lds_fill_k(const uint16_t* __restrict__ src, int iters, uint32_t* __restrict__ sink) {
  __shared__ volatile uint16_t buf[512];
  int tid = threadIdx.x;
  const uint32_t* s = (const uint32_t*)src + tid*4;
  volatile uint32_t* d = (volatile uint32_t*)buf + tid*4;
  for (int i = 0; i < iters; ++i) {
    uint4 v = *(const uint4*)s;
    d[0]=v.x; d[1]=v.y; d[2]=v.z; d[3]=v.w;
  }
  if (sink) sink[tid] = (uint32_t)d[0];
}

static void bench_lds_fill(const GpuInfo& info) {
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  std::printf("\n=== LDS fill (global_load_dwordx4 + LDS store) ===\n  %10s %10s %10s %8s %8s\n",
              "MB", "iters", "GB/s", "us(min)", "us(med)");
  constexpr int BD=64, ept=8, NB=1024;
  constexpr size_t AB = NB * BD * ept * sizeof(uint16_t);
  uint16_t* dg; uint32_t* sink;
  check_hip(hipMalloc(&dg, AB), "alloc");
  check_hip(hipMalloc(&sink, NB*BD*sizeof(uint32_t)), "alloc sink");
  check_hip(hipMemset(dg, 0, AB), "set");
  for (int li : {6,8,10,12}) {
    int it = 1<<li;
    double mb = (double)(AB)*it/(1024*1024);
    auto L = [&]{ lds_fill_k<<<NB,BD>>>(dg,it,sink); };
    auto r = bench_kernel_us(L, start, stop, 5, 2000);
    double gbs = mb/1024/(r.median_us/1e6);
    std::printf("  %8.1f %10d %10.1f %8.1f %8.1f\n", mb, it, gbs, r.min_us, r.median_us);
  }
  hipEventDestroy(start); hipEventDestroy(stop);
  check_hip(hipFree(dg), "free"); check_hip(hipFree(sink), "free");
  std::printf("  Roof HBM: %.0f GB/s\n", info.bw);
}

// ---------------------------------------------------------------------------
// K16 MFMA — explicit register allocation for consecutive VGPRs on gfx90a
// ---------------------------------------------------------------------------
__global__ void mfma_k16_k(int iters, float* out) {
  register float c0 asm("v0")=0,c1 asm("v1")=0,c2 asm("v2")=0,c3 asm("v3")=0;
  register unsigned a0 asm("v4")=0x3F803F80,a1 asm("v5")=0x3F803F80;
  register unsigned b0 asm("v6")=0x3F803F80,b1 asm("v7")=0x3F803F80;
  for (int i=0;i<iters;++i)
    __asm__ __volatile__(
      "v_mfma_f32_16x16x16bf16_1k v[0:3], v[4:5], v[6:7], v[0:3]"
      : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3)
      : "v"(a0),"v"(a1),"v"(b0),"v"(b1));
  out[0]=c0;
}

static void bench_mfma_k16(const GpuInfo& info) {
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  std::printf("\n=== K16 MFMA (16x16x16bf16) ===\n  %12s %10s %10s %8s %8s\n",
              "calls", "us", "TFLOP/s", "us(min)", "us(med)");
  constexpr double flops = 2.*16*16*16;
  float* dout; check_hip(hipMalloc(&dout, 4), "alloc");
  for (int li : {10,14,18,22,24}) {
    int it = 1<<li;
    auto L = [&]{ mfma_k16_k<<<1,64>>>(it,dout); };
    auto r = bench_kernel_us(L, start, stop, 5, 1000);
    std::printf("  %12d %8.1f %10.3f %8.1f %8.1f\n", it, r.median_us, flops*it/(r.median_us/1e6)/1e12, r.min_us, r.median_us);
  }
  hipEventDestroy(start); hipEventDestroy(stop);
  check_hip(hipFree(dout), "free");
  std::printf("  Roof: %.0f TFLOP/s\n", info.tflops);
}

// ---------------------------------------------------------------------------
// K32 emulation = 2xK16
// ---------------------------------------------------------------------------
__global__ void mfma_k32_k(int iters, float* out) {
  register float c0 asm("v0")=0,c1 asm("v1")=0,c2 asm("v2")=0,c3 asm("v3")=0;
  register unsigned a0 asm("v4")=0x3F803F80,a1 asm("v5")=0x3F803F80;
  register unsigned b0 asm("v6")=0x3F803F80,b1 asm("v7")=0x3F803F80;
  for (int i=0;i<iters;++i) {
    __asm__("v_mfma_f32_16x16x16bf16_1k v[0:3], v[4:5], v[6:7], v[0:3]"
            : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3) : "v"(a0),"v"(a1),"v"(b0),"v"(b1));
    __asm__("v_mfma_f32_16x16x16bf16_1k v[0:3], v[4:5], v[6:7], v[0:3]"
            : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3) : "v"(a0),"v"(a1),"v"(b0),"v"(b1));
  }
  out[0]=c0;
}

static void bench_mfma_k32(const GpuInfo& info) {
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  std::printf("\n=== K32 emul (2xK16) ===\n  %12s %10s %10s %8s %8s\n",
              "pairs", "us", "TFLOP/s", "us(min)", "us(med)");
  constexpr double flops = 2.*16*16*32;
  float* dout; check_hip(hipMalloc(&dout, 4), "alloc");
  for (int li : {10,14,18,22,24}) {
    int it = 1<<li;
    auto L = [&]{ mfma_k32_k<<<1,64>>>(it,dout); };
    auto r = bench_kernel_us(L, start, stop, 5, 1000);
    std::printf("  %12d %8.1f %10.3f %8.1f %8.1f\n", it, r.median_us, flops*it/(r.median_us/1e6)/1e12, r.min_us, r.median_us);
  }
  hipEventDestroy(start); hipEventDestroy(stop);
  check_hip(hipFree(dout), "free");
  std::printf("  Roof: %.0f TFLOP/s\n", info.tflops);
}

// ---------------------------------------------------------------------------
int main() {
  auto info = get_gpu_info();
  std::printf("GPU: %s  arch=%d  fp16=%.0f TFLOP/s  HBM=%.0f GB/s  async=%s\n\n",
              info.name.c_str(), info.arch, info.tflops, info.bw,
              vkernels::kernels::use_async_copy_default()?"ON":"OFF");
  bench_mfma_k16(info);
  bench_mfma_k32(info);
  bench_fp4_dequant(info);
  bench_lds_fill(info);
  return 0;
}
