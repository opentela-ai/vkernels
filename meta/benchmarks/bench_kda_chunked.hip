// meta/benchmarks/bench_kda_chunked.hip
//
// A/B micro-benchmark (#70): the committed cooperative per-token forward
// (kda_delta_rule_fwd_with_scratch) vs the chunked WY forward
// (kda_delta_rule_fwd_chunked_with_scratch) on gfx942, same inputs, same
// event timing (both entries hipDeviceSynchronize internally, like
// bench_kda). k is L2-normalised (the chunked contract; the coop kernel
// accepts it too — bench_kda normalises as well). State is re-zeroed every
// timed iteration so each sample is identical.
//
// USAGE
//   bench_kda_chunked H S D    one config: prints
//     "H=%d S=%d D=%d coop=%.1fus chunked=%.1fus speedup=%.2fx"
//   bench_kda_chunked          sweep H in {1,8,16,32,64,128} at S=512 D=128
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

#define CK(e, m)                                                          \
  do {                                                                    \
    hipError_t r = (e);                                                   \
    if (r != hipSuccess) {                                                \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));          \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

namespace {

namespace hipk = vkernels::kernels::hip;

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
      // ROCm 6.3 event quirk: rare zero-elapsed samples (see bench_kda).
      if (ms <= 0.0f) continue;
      times.push_back(ms * 1000.0f);
    }
    const int n = (int)times.size();
    if (n == 0) { iters *= 2; continue; }
    double sum = 0;
    for (double t : times) sum += t;
    double mean = sum / n;
    double sum2 = 0;
    for (double t : times) sum2 += t * t;
    double stddev = std::sqrt(std::max(0.0, sum2 / n - mean * mean));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {times[0], times[n / 2], mean};
    }
    iters *= 2;
  }
  return {};
}

static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;  // [-1, 1)
}

static float rgate(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return 0.05f + 0.95f * (float)(x % 100000u) / 100000.0f;
}

// One A/B config: allocates, normalises k (chunked contract), times BOTH
// entries on identical inputs (state re-zeroed per sample), prints one line.
void bench_ab(hipEvent_t start, hipEvent_t stop, int B, int H, int S, int D) {
  const size_t m = (size_t)B * H * S * D;
  const size_t ng = (size_t)B * H * S;
  const size_t nstate = (size_t)B * H * D * D;
  std::vector<float> q(m), k(m), v(m), g(ng * D), beta(ng);
  for (size_t i = 0; i < m; ++i) {
    q[i] = rnd(1, (int)i); k[i] = rnd(2, (int)i); v[i] = rnd(3, (int)i);
  }
  for (size_t i = 0; i < g.size(); ++i) g[i] = rgate(4, (int)i);
  for (size_t i = 0; i < beta.size(); ++i) beta[i] = rgate(5, (int)i);
  for (int x = 0; x < B * H * S; ++x) {          // L2-normalise k
    float* kk = k.data() + (size_t)x * D;
    float n = 0.0f;
    for (int d = 0; d < D; ++d) n += kk[d] * kk[d];
    n = 1.0f / std::sqrt(n + 1e-12f);
    for (int d = 0; d < D; ++d) kk[d] *= n;
  }

  float *dq, *dk, *dv, *dg, *db, *dstate, *dout, *dscratch;
  CK(hipMalloc(&dq, m * 4), "aq");
  CK(hipMalloc(&dk, m * 4), "ak");
  CK(hipMalloc(&dv, m * 4), "av");
  CK(hipMalloc(&dg, g.size() * 4), "ag");
  CK(hipMalloc(&db, ng * 4), "ab");
  CK(hipMalloc(&dstate, nstate * 4), "ast");
  CK(hipMalloc(&dout, m * 4), "ao");
  const size_t sf = hipk::kda_chunked_scratch_floats(B, H, S, D);
  CK(hipMalloc(&dscratch, sf * 4), "asc");
  CK(hipMemcpy(dq, q.data(), m * 4, hipMemcpyHostToDevice), "cq");
  CK(hipMemcpy(dk, k.data(), m * 4, hipMemcpyHostToDevice), "ck");
  CK(hipMemcpy(dv, v.data(), m * 4, hipMemcpyHostToDevice), "cv");
  CK(hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice), "cg");
  CK(hipMemcpy(db, beta.data(), ng * 4, hipMemcpyHostToDevice), "cb");

  // Watchdog-aware iteration budget (beverin faults ~0.3-0.7 s of
  // continuous default-stream kernel time; see bench_kda). The coop side
  // is the long one; chunked is expected faster.
  const bool long_k = (S >= 512);
  auto Lc = [&] {   // cooperative (committed baseline)
    hipMemsetAsync(dstate, 0, nstate * 4, 0);
    hipk::kda_delta_rule_fwd_with_scratch(dq, dk, dv, dg, db, dstate, dout,
                                          B, H, S, D);
  };
  auto Lw = [&] {   // chunked WY (#70)
    hipMemsetAsync(dstate, 0, nstate * 4, 0);
    hipk::kda_delta_rule_fwd_chunked_with_scratch(dq, dk, dv, dg, db, dstate,
                                                  dout, dscratch,
                                                  B, H, S, D, 64);
  };
  const auto rc = long_k ? bench_us(Lc, start, stop, 2, 30)
                         : bench_us(Lc, start, stop, 3, 120);
  const auto rw = long_k ? bench_us(Lw, start, stop, 2, 60)
                         : bench_us(Lw, start, stop, 3, 200);
  const double sp = (rw.median_us > 0) ? rc.median_us / rw.median_us : 0.0;
  std::printf("H=%d S=%d D=%d coop=%.1fus chunked=%.1fus speedup=%.2fx\n",
              H, S, D, rc.median_us, rw.median_us, sp);
  fflush(stdout);

  CK(hipFree(dq), "fq"); CK(hipFree(dk), "fk"); CK(hipFree(dv), "fv");
  CK(hipFree(dg), "fg"); CK(hipFree(db), "fb");
  CK(hipFree(dstate), "fst"); CK(hipFree(dout), "fo");
  CK(hipFree(dscratch), "fsc");
}

}  // namespace

int main(int argc, char** argv) {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  hipEvent_t start, stop;
  CK(hipEventCreate(&start), "ev1");
  CK(hipEventCreate(&stop), "ev2");

  if (argc >= 5 && std::strcmp(argv[1], "phases") == 0) {
    // Phase decomposition: which of cumsum / chunk-local / state dominates.
    const int H = std::atoi(argv[2]), S = std::atoi(argv[3]),
              D = std::atoi(argv[4]);
    const size_t m = (size_t)H * S * D;
    const size_t ng = (size_t)H * S;
    const size_t nstate = (size_t)H * D * D;
    std::vector<float> q(m), k(m), v(m), g(ng * D), beta(ng);
    for (size_t i = 0; i < m; ++i) {
      q[i] = rnd(1, (int)i); k[i] = rnd(2, (int)i); v[i] = rnd(3, (int)i);
    }
    for (size_t i = 0; i < g.size(); ++i) g[i] = rgate(4, (int)i);
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = rgate(5, (int)i);
    for (int x = 0; x < H * S; ++x) {
      float* kk = k.data() + (size_t)x * D;
      float n = 0.0f;
      for (int d = 0; d < D; ++d) n += kk[d] * kk[d];
      n = 1.0f / std::sqrt(n + 1e-12f);
      for (int d = 0; d < D; ++d) kk[d] *= n;
    }
    float *dq, *dk, *dv, *dg, *db, *dstate, *dout, *dscratch;
    CK(hipMalloc(&dq, m * 4), "aq");
    CK(hipMalloc(&dk, m * 4), "ak");
    CK(hipMalloc(&dv, m * 4), "av");
    CK(hipMalloc(&dg, g.size() * 4), "ag");
    CK(hipMalloc(&db, ng * 4), "ab");
    CK(hipMalloc(&dstate, nstate * 4), "ast");
    CK(hipMalloc(&dout, m * 4), "ao");
    const size_t sf = hipk::kda_chunked_scratch_floats(1, H, S, D);
    CK(hipMalloc(&dscratch, sf * 4), "asc");
    CK(hipMemcpy(dq, q.data(), m * 4, hipMemcpyHostToDevice), "cq");
    CK(hipMemcpy(dk, k.data(), m * 4, hipMemcpyHostToDevice), "ck");
    CK(hipMemcpy(dv, v.data(), m * 4, hipMemcpyHostToDevice), "cv");
    CK(hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice), "cg");
    CK(hipMemcpy(db, beta.data(), ng * 4, hipMemcpyHostToDevice), "cb");
    double us[10] = {0};
    hipk::kda_chunked_phase_times(dq, dk, dv, dg, db, dstate, dout, dscratch,
                                  1, H, S, D, 10, us);
    std::printf("phases H=%d S=%d D=%d  cumsum=%.1fus gram=%.1fus "
                "local=%.1fus state=%.1fus  (sum %.1f)\n",
                H, S, D, us[0], us[1], us[2], us[3],
                us[0] + us[1] + us[2] + us[3]);
    std::printf("  local phases: Mnload+p0=%.1f inv2a=%.1f inv2b=%.1f "
                "gemm3=%.1f gemm4=%.1f\n",
                us[4], us[5], us[6], us[7], us[8]);
    return 0;
  }

  if (argc == 4) {
    bench_ab(start, stop, 1, std::atoi(argv[1]), std::atoi(argv[2]),
             std::atoi(argv[3]));
  } else {
    for (int H : {1, 8, 16, 32, 64, 128})
      bench_ab(start, stop, 1, H, 512, 128);
    for (int S : {1024, 2048})                 // long-context scaling
      bench_ab(start, stop, 1, 32, S, 128);
  }
  return 0;
}
