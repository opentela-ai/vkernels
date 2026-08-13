// meta/benchmarks/bench_moe_aux.hip
//
// Latency + effective-bandwidth benchmark for the MXFP4 MoE orchestration
// ops (issue #22), run at the K3 decode shape (M=112, hidden=7168,
// ispp=3072, top_k=16, E=64) and a few smaller batch sizes.
//
// These are lightweight data-movement primitives (per-group fp4 quant,
// token->expert gather, routed scatter-reduce) that sit around the heavy
// fused GEMM. They are memory-bound in the roofline sense, but at K3 scale
// several are small enough that kernel-launch overhead dominates -- so we
// report median latency (us) as the primary metric and effective HBM GB/s
// = (bytes_in + bytes_out) / time as a secondary metric against the MI300A
// HBM3 roof (~5.3 TB/s published, verified here by a streaming copy).
//
// Build (MI300A compute node, bare-metal ROCm 6.3):
//   cmake --preset hip -DVKERNELS_BUILD_BENCHMARKS=ON
//   cmake --build build/hip --target moe_aux_bench
//   srun --partition=mi300 --ntasks=1 --cpus-per-task=16 \
//        ./build/hip/meta/benchmarks/moe_aux_bench

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/moe_aux.hpp"    // CPU reference (for setup only)
#include "vkernels/kernels/moe_fused.hpp"  // moe_align_block_size

// The HIP entry points live in moe_aux.hip under vkernels::kernels::hip;
// they are intentionally not re-declared in the header (vkl discovery), so
// the bench forward-declares them here.
namespace vkernels::kernels::hip {
void mxfp4_moe_quant(const uint16_t* A, uint8_t* packed, uint8_t* scales,
                     int M, int hidden, int group_size);
void mxfp4_moe_sort(const uint16_t* A, const int32_t* sorted_ids,
                    uint16_t* A_sorted, int M, int hidden, int top_k, int EM);
void mxfp4_moe_sort_scales(const uint8_t* scales, const int32_t* sorted_ids,
                           uint8_t* scales_sorted, int M, int n_groups,
                           int top_k, int EM);
void mxfp4_moe_scatter_reduce(const float* partial, const float* topk_w,
                              const int32_t* sorted_ids, float* out,
                              int M, int width, int top_k, int EM);
void mxfp4_moe_scatter_reduce_q(const uint8_t* partial_q,
                                const uint8_t* partial_s,
                                const float* topk_w,
                                const int32_t* sorted_ids, float* out,
                                int M, int width, int top_k, int EM,
                                int group_size);
}  // namespace vkernels::kernels::hip

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

uint32_t f2bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
uint16_t f2bf(float v) {
  uint32_t b = f2bits(v);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}

// Median latency (us) over a fixed iteration count, with warmup. Mirrors
// bench_moe_fused.hip.
struct Bench { double median_us; int iters; };
Bench bench_us(const std::function<void()>& launch, hipEvent_t s, hipEvent_t t,
               int warmup = 10, int iters = 100) {
  for (int i = 0; i < warmup; ++i) launch();
  (void)hipDeviceSynchronize();
  std::vector<float> ts(iters);
  for (int i = 0; i < iters; ++i) {
    hipEventRecord(s);
    launch();
    hipEventRecord(t);
    hipEventSynchronize(t);
    float ms = 0;
    hipEventElapsedTime(&ms, s, t);
    ts[i] = ms * 1000.0f;
  }
  std::sort(ts.begin(), ts.end());
  return {(double)ts[iters / 2], iters};
}

// A simple read-only streaming copy to establish the achievable HBM
// bandwidth on this node (the practical roof for the memory-bound ops).
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out,
                            size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = in[i];
}

double hbm_gbs(size_t bytes, double us) {
  return (double)bytes / (us * 1e-6) / 1e9;
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (%s)\n", p.name, p.gcnArchName);

  hipEvent_t s, t;
  hipEventCreate(&s);
  hipEventCreate(&t);

  // --- HBM bandwidth reference (streaming copy) ---
  // Use enough data to exceed L2 (~256 MB) so the copy is HBM-bound.
  {
    constexpr size_t N = 64 * 1024 * 1024;  // 64M floats = 256 MB
    float *din, *dout;
    CK(hipMalloc(&din, N * 4), "cin");
    CK(hipMalloc(&dout, N * 4), "cout");
    CK(hipMemset(din, 1, N * 4), "zin");
    auto copy = [&]() { copy_kernel<<<256, 256>>>(din, dout, N); };
    auto r = bench_us(copy, s, t, 5, 50);
    double bw = hbm_gbs(2.0 * N * 4, r.median_us);  // read + write
    std::printf("HBM streaming copy : %7.1f us  %6.0f GB/s  (%zu MB)\n\n",
                r.median_us, bw, N * 4 / (1024 * 1024));
    hipFree(din);
    hipFree(dout);
  }

  constexpr int hidden = 7168, top_k = 16, E = 64, BLOCK_M = 16, gs = 32;
  const int n_groups = hidden / gs;

  std::printf("K3 shape: hidden=%d top_k=%d E=%d gs=%d n_groups=%d\n",
              hidden, top_k, E, gs, n_groups);
  std::printf("%6s %8s  %-22s %9s %10s %9s\n", "M", "EM", "op", "lat(us)",
              "GB/s", "%HBM");

  const int Ms[] = {8, 32, 64, 112};  // 112 = K3
  for (int M : Ms) {
    const int N = M * top_k;
    const int EM_max = M * top_k + E * BLOCK_M;
    std::vector<int32_t> h_tids(N), h_sids(EM_max), h_eids(EM_max / BLOCK_M + 1);
    std::vector<float> h_tw(N);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < top_k; ++k) {
        h_tids[i * top_k + k] = (i * 17 + k * 13 + 7) % E;
        h_tw[i * top_k + k] = 0.5f + 0.5f * rnd(20, i * top_k + k);
      }
    int EM = vkernels::kernels::moe_align_block_size(
        h_tids.data(), M, top_k, BLOCK_M, E, h_sids.data(), h_eids.data());
    std::vector<float> h_tws(EM, 0.0f);
    for (int r = 0; r < EM; ++r)
      if (h_sids[r] < N) h_tws[r] = h_tw[h_sids[r]];

    // Host inputs (only what's needed to populate device buffers).
    std::vector<uint16_t> hA((size_t)M * hidden);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = f2bf(rnd(1, (int)i) * 0.1f);
    std::vector<uint16_t> hA_sorted((size_t)EM * hidden);  // dst (zero-init ok)
    std::vector<uint8_t> hscales((size_t)M * n_groups);
    std::vector<uint8_t> hscales_sorted((size_t)EM * n_groups);
    std::vector<float> hpartial((size_t)EM * hidden);
    for (size_t i = 0; i < hpartial.size(); ++i)
      hpartial[i] = rnd(20, (int)i) * 0.5f;
    std::vector<uint16_t> hpbf((size_t)EM * hidden);
    for (size_t i = 0; i < hpbf.size(); ++i) hpbf[i] = f2bf(hpartial[i]);
    std::vector<uint8_t> hpq((size_t)EM * hidden / 2), hps((size_t)EM * n_groups);
    vkernels::kernels::mxfp4_moe_quant(hpbf.data(), hpq.data(), hps.data(),
                                       EM, hidden, gs);

    // Device buffers.
    uint16_t *dA, *dA_sorted, *dPbf;
    uint8_t *dPacked, *dScales, *dScalesSorted, *dPq, *dPs;
    int32_t *dSids;
    float *dTw, *dOut, *dPartial;
    CK(hipMalloc(&dA, (size_t)M * hidden * 2), "mA");
    CK(hipMalloc(&dA_sorted, (size_t)EM * hidden * 2), "mAs");
    CK(hipMalloc(&dPbf, (size_t)EM * hidden * 2), "mPbf");
    CK(hipMalloc(&dPacked, (size_t)M * hidden / 2), "mPacked");
    CK(hipMalloc(&dScales, (size_t)M * n_groups), "mScales");
    CK(hipMalloc(&dScalesSorted, (size_t)EM * n_groups), "mScalesSorted");
    CK(hipMalloc(&dPq, (size_t)EM * hidden / 2), "mPq");
    CK(hipMalloc(&dPs, (size_t)EM * n_groups), "mPs");
    CK(hipMalloc(&dSids, (size_t)EM * 4), "mSids");
    CK(hipMalloc(&dTw, (size_t)EM * 4), "mTw");
    CK(hipMalloc(&dOut, (size_t)M * hidden * 4), "mOut");
    CK(hipMalloc(&dPartial, (size_t)EM * hidden * 4), "mPartial");
    CK(hipMemcpy(dA, hA.data(), (size_t)M * hidden * 2, hipMemcpyHostToDevice), "cA");
    CK(hipMemcpy(dPbf, hpbf.data(), (size_t)EM * hidden * 2, hipMemcpyHostToDevice), "cPbf");
    CK(hipMemcpy(dPq, hpq.data(), (size_t)EM * hidden / 2, hipMemcpyHostToDevice), "cPq");
    CK(hipMemcpy(dPs, hps.data(), (size_t)EM * n_groups, hipMemcpyHostToDevice), "cPs");
    CK(hipMemcpy(dSids, h_sids.data(), (size_t)EM * 4, hipMemcpyHostToDevice), "cSids");
    CK(hipMemcpy(dTw, h_tws.data(), (size_t)EM * 4, hipMemcpyHostToDevice), "cTw");
    CK(hipMemcpy(dPartial, hpartial.data(), (size_t)EM * hidden * 4, hipMemcpyHostToDevice), "cPartial");

    using namespace vkernels::kernels::hip;

    auto run_quant = [&]() { mxfp4_moe_quant(dA, dPacked, dScales, M, hidden, gs); };
    auto run_sort = [&]() { mxfp4_moe_sort(dA, dSids, dA_sorted, M, hidden, top_k, EM); };
    auto run_sort_s = [&]() {
      mxfp4_moe_sort_scales(dScales, dSids, dScalesSorted, M, n_groups, top_k, EM);
    };
    auto run_scatter = [&]() {
      (void)hipMemset(dOut, 0, (size_t)M * hidden * 4);
      mxfp4_moe_scatter_reduce(dPartial, dTw, dSids, dOut, M, hidden, top_k, EM);
    };
    auto run_scatter_q = [&]() {
      (void)hipMemset(dOut, 0, (size_t)M * hidden * 4);
      mxfp4_moe_scatter_reduce_q(dPq, dPs, dTw, dSids, dOut, M, hidden, top_k,
                                 EM, gs);
    };

    // Logical bytes (read once + write once) for each op. atomicAdd traffic
    // at L2 (small outputs) is not counted -- these are the bytes that must
    // traverse HBM.
    size_t bq_in = (size_t)M * hidden * 2, bq_out = (size_t)M * hidden / 2 + (size_t)M * n_groups;
    size_t bs_in = (size_t)M * top_k * hidden * 2, bs_out = (size_t)EM * hidden * 2;
    size_t bss_in = (size_t)M * top_k * n_groups, bss_out = (size_t)EM * n_groups;
    size_t bsc_in = (size_t)EM * hidden * 4, bsc_out = (size_t)M * hidden * 4;
    size_t bsq_in = (size_t)EM * hidden / 2 + (size_t)EM * n_groups, bsq_out = (size_t)M * hidden * 4;

    auto row = [&](const char* name, const Bench& r, size_t in, size_t out) {
      double gb = hbm_gbs(in + out, r.median_us);
      std::printf("%6d %8d  %-22s %9.1f %10.1f %9.2f\n", M, EM, name,
                  r.median_us, gb, gb / 5300.0 * 100.0);
    };

    row("mxfp4_moe_quant", bench_us(run_quant, s, t), bq_in, bq_out);
    row("mxfp4_moe_sort", bench_us(run_sort, s, t), bs_in, bs_out);
    row("mxfp4_moe_sort_scales", bench_us(run_sort_s, s, t), bss_in, bss_out);
    row("mxfp4_moe_scatter_reduce", bench_us(run_scatter, s, t), bsc_in, bsc_out);
    row("mxfp4_moe_scatter_reduce_q", bench_us(run_scatter_q, s, t), bsq_in,
        bsq_out);
    if (M == 112) std::printf("\n");

    hipFree(dA); hipFree(dA_sorted); hipFree(dPbf);
    hipFree(dPacked); hipFree(dScales); hipFree(dScalesSorted);
    hipFree(dPq); hipFree(dPs); hipFree(dSids); hipFree(dTw);
    hipFree(dOut); hipFree(dPartial);
  }

  hipEventDestroy(s);
  hipEventDestroy(t);
  std::printf("Done. (%%HBM uses the published ~5.3 TB/s MI300A roof; the\n"
              "streaming copy above is the practical achievable roof.)\n");
  return 0;
}
