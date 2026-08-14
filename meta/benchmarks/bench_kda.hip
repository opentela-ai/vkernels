// meta/benchmarks/bench_kda.hip
//
// Micro-benchmark for the K3 Kimi Delta-Attention kernels (issue #21) on
// AMD gfx942 (MI300A). The forward (kda_delta_rule_fwd) is the cooperative
// per-token recurrence with the DxD state in a per-(b,h) GLOBAL scratch
// buffer (correct for any D incl. K3's 128). For each (H, S, D) shape it
// reports, against the MI300A roof (~1307 TFLOP/s bf16, ~5300 GB/s HBM3,
// ridge ~247 FLOP/byte):
//
//   * median / min latency (us),
//   * achieved TFLOP/s  = 7*S*D^2*H*B / t          (~ a=Sk + gS+b*uk + o=Sq),
//   * effective GB/s    = actual HBM bytes / t    (state: 3 read + 1 write
//                                                  per token; q,k,v,out once),
//   * arithmetic intensity (~0.44 for large D -> strongly memory-bound) and
//     the binding resource.
//
// The baseline writes the FULL DxD state to HBM every token, so for D=128 it
// moves ~16 KB/token/head of state traffic -- the dominant cost and the
// target of the LDS-resident-state and chunked-intra/inter follow-ons.
// Also reports layer_norm_gated (K3 RMSNorm+SiLU gate) and gate_chunk_cumsum
// at K3 shapes.
//
// USAGE
//   bench_kda H S D     bench a single delta_rule_fwd config (short-lived
//                       process; see bench_kda.sh for the per-config driver)
//   bench_kda           bench only the short supporting kernels
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

#include "vkernels/kernels/kda.hpp"

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
// One delta_rule_fwd config: allocates, L2-normalises keys (contract: caller
// normalises k so the recurrence state stays bounded), times the forward,
// prints one table row, and frees. The caller owns the start/stop events.
static void bench_one_fwd(const GpuInfo& info, hipEvent_t start,
                          hipEvent_t stop, int B, int H, int S, int D) {
  const size_t m = (size_t)B * H * S * D;
  const size_t ng = (size_t)B * H * S;
  const size_t nstate = (size_t)B * H * D * D;
  std::vector<float> q(m), k(m), v(m), g(ng), beta(ng), out(m, 0.0f);
  for (size_t i = 0; i < m; ++i)  { q[i] = rnd(1, (int)i); k[i] = rnd(2, (int)i); v[i] = rnd(3, (int)i); }
  for (size_t i = 0; i < ng; ++i) { g[i] = 0.05f + 0.95f * (rnd(4, (int)i) + 1.0f) * 0.5f;
                                    beta[i] = 0.05f + 0.95f * (rnd(5, (int)i) + 1.0f) * 0.5f; }
  for (int x = 0; x < B * H * S; ++x) {
    float* kk = k.data() + (size_t)x * D;
    float n = 0.0f;
    for (int d = 0; d < D; ++d) n += kk[d] * kk[d];
    n = 1.0f / std::sqrt(n + 1e-12f);
    for (int d = 0; d < D; ++d) kk[d] *= n;
  }

  float *dq, *dk, *dv, *dg, *db, *dstate, *dout;
  check_hip(hipMalloc(&dq,     m      * 4), "q");
  check_hip(hipMalloc(&dk,     m      * 4), "k");
  check_hip(hipMalloc(&dv,     m      * 4), "v");
  check_hip(hipMalloc(&dg,     ng     * 4), "g");
  check_hip(hipMalloc(&db,     ng     * 4), "b");
  check_hip(hipMalloc(&dstate, nstate * 4), "state");
  check_hip(hipMalloc(&dout,   m      * 4), "out");
  check_hip(hipMemcpy(dq, q.data(),    m  * 4, hipMemcpyHostToDevice), "cpyq");
  check_hip(hipMemcpy(dk, k.data(),    m  * 4, hipMemcpyHostToDevice), "cpyk");
  check_hip(hipMemcpy(dv, v.data(),    m  * 4, hipMemcpyHostToDevice), "cpyv");
  check_hip(hipMemcpy(dg, g.data(),    ng * 4, hipMemcpyHostToDevice), "cpyg");
  check_hip(hipMemcpy(db, beta.data(), ng * 4, hipMemcpyHostToDevice), "cpyb");

  // Each timed iter is identical: zero the state, then run the forward.
  // bench_us's hipEventSynchronize(stop) (and its warmup ordering on the
  // default stream) waits for the kernel; no internal sync needed.
  //
  // NOTE: beverin MI300A faults a default-stream workload after ~0.3-0.7 s
  // of continuous kernel time ("Memory access fault, Reason: Unknown"),
  // non-deterministically. Cap the iteration budget so the cumulative
  // work per config stays well under that ceiling (long kernels: 2 warmup
  // + <=30 timed ~= 80 ms for the 2.6 ms S=512 D=128 case). The
  // bench_kda.sh driver additionally runs each config as a separate
  // short-lived process so no single context accumulates.
  auto L = [&] {
    hipMemsetAsync(dstate, 0, nstate * 4, 0);
    vkernels::kernels::hip::kda_delta_rule_fwd_with_scratch(
        dq, dk, dv, dg, db, dstate, dout, B, H, S, D);
  };
  const bool long_k = (S >= 512);
  auto r = long_k ? bench_us(L, start, stop, 2, 30)
                  : bench_us(L, start, stop, 3, 120);

  // FLOPs ~ a=Sk(2D^2) + update gS+b*uk(~3D^2) + o=Sq(2D^2) = 7D^2 per token.
  double flops = 7.0 * S * (double)D * D * H * B;
  double tflops = flops / (r.median_us / 1e6) / 1e12;
  // HBM: state read in (1)+(3) and read+written in (2) = 4*D^2; q,k,v,out
  // once = 4*D; g,beta once = 8. Per token per head.
  double bytes = 4.0 * ((double)B * H * S * (4.0 * D * D + 4.0 * D) + 8.0 * B * H * S);
  double gbs = bytes / (r.median_us / 1e6) / 1e9;
  double ai = flops / bytes;
  const char* bound = ai < (info.tflops * 1e3 / info.bw) ? "mem" : "comp";
  std::printf("  %4d %6d %4d %9.1f %9.1f %9.3f %9.1f %5.2f  %s\n",
              H, S, D, r.min_us, r.median_us, tflops, gbs, ai, bound);

  check_hip(hipFree(dq),     "fq");
  check_hip(hipFree(dk),     "fk");
  check_hip(hipFree(dv),     "fv");
  check_hip(hipFree(dg),     "fg");
  check_hip(hipFree(db),     "fb");
  check_hip(hipFree(dstate), "fstate");
  check_hip(hipFree(dout),   "fout");
}

// ---------------------------------------------------------------------------
// Supporting kernels at K3 shapes (both are O(N) elementwise / scan).
static void bench_support(const GpuInfo& info, hipEvent_t start,
                          hipEvent_t stop) {
  (void)info;
  const int N = 8192, D = 128;
  // gate is [N, D] (the kernel reads gate[n*D + d]); weight is [D].
  std::vector<float> x((size_t)N * D), w(D), g((size_t)N * D), o((size_t)N * D);
  for (size_t i = 0; i < (size_t)N * D; ++i) { x[i] = rnd(1, (int)i); g[i] = 0.5f + 0.5f * (rnd(3, (int)i) + 1.0f) * 0.5f; }
  for (int i = 0; i < D; ++i) w[i] = rnd(2, i);
  float *dx, *dw, *dg, *dout;
  check_hip(hipMalloc(&dx,   (size_t)N * D * 4), "x");
  check_hip(hipMalloc(&dw,   (size_t)D * 4),     "w");
  check_hip(hipMalloc(&dg,   (size_t)N * D * 4), "g");
  check_hip(hipMalloc(&dout, (size_t)N * D * 4), "o");
  check_hip(hipMemcpy(dx, x.data(), (size_t)N * D * 4, hipMemcpyHostToDevice), "cpx");
  check_hip(hipMemcpy(dw, w.data(), (size_t)D * 4,     hipMemcpyHostToDevice), "cpw");
  check_hip(hipMemcpy(dg, g.data(), (size_t)N * D * 4, hipMemcpyHostToDevice), "cpg");
  auto L1 = [&] { vkernels::kernels::hip::kda_layer_norm_gated(
                      dx, dw, dg, dout, N, D, 1e-6f); };
  auto r1 = bench_us(L1, start, stop);
  // x (N*D) + out (N*D) + gate (N*D) + weight (D), all fp32.
  double b1 = 4.0 * ((size_t)N * D * 3 + (size_t)D);
  std::printf("\n=== hip::kda_layer_norm_gated  N=%d D=%d  us(min)=%.1f us(med)=%.1f"
              "  GB/s=%.1f\n", N, D, r1.min_us, r1.median_us,
              b1 / (r1.median_us / 1e6) / 1e9);
  check_hip(hipFree(dx), "fx"); check_hip(hipFree(dw), "fw");
  check_hip(hipFree(dg), "fg"); check_hip(hipFree(dout), "fo");

  const int B = 1, H = 16, cs = 64, nc = 8;
  std::vector<float> gg((size_t)B * H * nc * cs);
  for (size_t i = 0; i < gg.size(); ++i) gg[i] = 0.05f + 0.95f * (rnd(4, (int)i) + 1.0f) * 0.5f;
  float *dgg, *dintra, *dinter;
  check_hip(hipMalloc(&dgg,     (size_t)B * H * nc * cs * 4), "gg");
  check_hip(hipMalloc(&dintra,  (size_t)B * H * nc * cs * 4), "intra");
  check_hip(hipMalloc(&dinter,  (size_t)B * H * nc * 4),      "inter");
  check_hip(hipMemcpy(dgg, gg.data(), (size_t)B * H * nc * cs * 4, hipMemcpyHostToDevice), "cpgg");
  auto L2 = [&] { vkernels::kernels::hip::kda_gate_chunk_cumsum(
                      dgg, dintra, dinter, B, H, nc, cs); };
  auto r2 = bench_us(L2, start, stop);
  double b2 = 4.0 * ((size_t)B * H * nc * cs + (size_t)B * H * nc * cs + (size_t)B * H * nc);
  std::printf("=== hip::kda_gate_chunk_cumsum B=%d H=%d nc=%d cs=%d  us(min)=%.1f"
              " us(med)=%.1f  GB/s=%.1f\n", B, H, nc, cs, r2.min_us, r2.median_us,
              b2 / (r2.median_us / 1e6) / 1e9);
  check_hip(hipFree(dgg), "fgg"); check_hip(hipFree(dintra), "fintra");
  check_hip(hipFree(dinter), "finter");
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  auto info = get_gpu_info();
  if (argc == 4) {
    // Single-config delta_rule_fwd (see bench_kda.sh: each (H,S,D) runs in
    // its own short-lived process so the beverin MI300A single-context
    // watchdog never trips). Prints only the data row -- the driver owns
    // the table header and the roof line.
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    bench_one_fwd(info, start, stop, 1, std::atoi(argv[1]),
                  std::atoi(argv[2]), std::atoi(argv[3]));
    hipEventDestroy(start); hipEventDestroy(stop);
    return 0;
  }

  // No args: only the short supporting kernels (the delta_rule_fwd sweep is
  // run one (H,S,D) per process via argv -- see the bench_kda.sh driver --
  // because beverin MI300A faults a long-lived single-context sweep).
  std::printf("GPU: %s  arch=%d  bf16=%.0f TFLOP/s  HBM=%.0f GB/s\n",
              info.name.c_str(), info.arch, info.tflops, info.bw);
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  bench_support(info, start, stop);
  hipEventDestroy(start); hipEventDestroy(stop);
  return 0;
}
