// meta/benchmarks/bench_dsa_split137.hip
//
// Issue #137 focused sweep: decode dsa_sparse_fwd_split under the ambient
// VK_DSA_DB / VK_DSA_VEC / VK_DSA_FUSE / VK_DSA_VECG environment, so one
// binary run records exactly one dispatch variant and the driver script
// A/Bs variants by re-running with different env (the dispatch is
// env-selected inside dsa_sparse_fwd_split; see dsa.hip).
//
// Methodology identical to bench_dsa.hip's split_sweep (HIP events,
// variance-driven iters, same byte model), narrowed to the four decode
// shapes and the split grid the perf record sweeps, so several variants
// fit one mi300 partition slot (1 h hard limit).

// --persist [dir]: after the A/B table, write the per-shape argmin split
// into the native tuning store (core/tuning.hpp) -- the artifact
// dsa_sparse_fwd_split_for consults BEFORE its compiled-in formula, so a
// measured config replays on every gfx942 box that ships the store
// (VKERNELS_TUNING_CACHE, default ~/.cache/vkernels/tuning). The table
// times EXPLICIT split values, not split_for(), so an already-populated
// store cannot feed back into its own sweep. Refuses to persist under any
// VK_DSA_* env override: the winner would describe a diagnostic dispatch
// variant, not the one serving runs (default env = auto/fused-off).

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

#include "vkernels/core/tuning.hpp"
#include "vkernels/kernels/dsa.hpp"

static void check_hip(hipError_t err, const char* ctx) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", ctx, hipGetErrorString(err));
    std::exit(1);
  }
}

struct BenchResult { double min_us, median_us, mean_us; };

static BenchResult bench_us(const std::function<void()>& launch,
                            hipEvent_t start, hipEvent_t stop,
                            int warmup = 3, int max_iters = 60) {
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
      // ROCm 6.3 event quirk (see bench_dsa.hip): discard exact-zero
      // samples instead of poisoning min/median.
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
    const double mean = sum / n;
    double sum2 = 0;
    for (double t : times) sum2 += t * t;
    const double stddev = std::sqrt(std::max(0.0, sum2 / n - mean * mean));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {times[0], times[n / 2], mean};
    }
    iters *= 2;
  }
  return {};
}

static uint16_t f2bf_h(float v) {
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

int main(int argc, char** argv) {
  // --persist [dir]: also write the per-shape winners into the native
  // tuning store. A bare --persist uses the store's default location;
  // --persist=<dir> pins (and exports) VKERNELS_TUNING_CACHE.
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

  hipDeviceProp_t p;
  check_hip(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);
  const char* vec_s = std::getenv("VK_DSA_VEC");
  const char* fuse_s = std::getenv("VK_DSA_FUSE");
  const char* g_s = std::getenv("VK_DSA_VECG");
  const char* db_s = std::getenv("VK_DSA_DB");
  std::printf("env: VK_DSA_DB=%s VK_DSA_VEC=%s VK_DSA_FUSE=%s VK_DSA_VECG=%s\n",
              db_s ? db_s : "(unset)", vec_s ? vec_s : "(unset)",
              fuse_s ? fuse_s : "(unset)", g_s ? g_s : "(unset)");

  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);

  struct Cfg { int H, S_q, S_kv, dim, tail_dim, topk; };
  const Cfg cfgs[] = {
      {64, 1, 2048, 256, 0, 2048},  // GLM-5.3 decode, full topk
      {64, 1,  256, 256, 0,  256},  // GLM-5.3 decode, serving topk
      {64, 1,  128, 256, 0,  128},  // GLM-5.3 decode, small topk
      {16, 1,  512, 576, 64,  256}, // DeepSeek-V3 decode
  };
  const int splits[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};

  std::printf("\n=== DSA split-key sweep, issue #137 (decode shapes;"
              " split x partial [+ combine]) ===\n");
  for (const auto& c : cfgs) {
    const int H = c.H, S_q = c.S_q, S_kv = c.S_kv, dim = c.dim;
    const int tail_dim = c.tail_dim, topk = c.topk;
    const int W = dim + tail_dim, d_v = dim - tail_dim;
    const float sm_scale = (1.0f / std::sqrt((float)W)) * 1.4426950408889634f;
    const size_t nq = (size_t)S_q * H * W, nkv = (size_t)S_kv * W;
    const size_t nidx = (size_t)S_q * topk, nout = (size_t)S_q * H * d_v;
    std::vector<uint16_t> q(nq), kv(nkv), out(nout, 0);
    std::vector<int32_t> idx(nidx);
    for (size_t i = 0; i < nq; ++i)  q[i]  = f2bf_h(rnd(1, (int)i));
    for (size_t i = 0; i < nkv; ++i) kv[i] = f2bf_h(rnd(2, (int)i));
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
        S_q, H, topk, /*block_I=*/64, /*num_cu=*/p.multiProcessorCount,
        dim, tail_dim);
    std::printf("\n  H=%d S_q=%d topk=%d (dim=%d tail=%d):"
                " dsa_sparse_fwd_split_for = %d%s\n",
                H, S_q, topk, dim, tail_dim, rec,
                do_persist ? "  (store sweep armed)" : "");
    std::printf("    %5s %9s %9s %9s %8s  %s\n", "split", "us(min)",
                "us(med)", "GB/s", "speedup", "note");
    double base_us = 0.0;
    double best_us = -1.0;
    int best_split = 1;
    for (int s : splits) {
      if (s > topk) continue;
      auto L = [&] {
        vkernels::kernels::hip::dsa_sparse_fwd_split(
            S_q, S_kv, H, dim, tail_dim, topk, 1, /*block_I=*/64,
            /*inner_iter=*/1, sm_scale, /*return_lse=*/true, s,
            dq, dkv, didx, dout, dlse, dpart_out, dpart_lse);
      };
      auto r = bench_us(L, start, stop);
      if (s == 1) base_us = r.median_us;
      if (best_us < 0.0 || r.median_us < best_us) {
        best_us = r.median_us;
        best_split = s;
      }
      // Same byte model as bench_dsa.hip: kv re-read once per head.
      double bytes = 2.0 * ((double)S_q * H * W + (double)S_q * H * d_v +
                            (double)S_q * H * topk * W)
                   + 4.0 * (double)S_q * topk;
      double gbs = bytes / (r.median_us / 1e6) / 1e9;
      std::printf("    %5d %9.1f %9.1f %9.1f %7.1fx  %s\n", s, r.min_us,
                  r.median_us, gbs, base_us / r.median_us,
                  s == rec ? "<- split_for" : "");
    }
    if (do_persist) {
      vkernels::core::tuning::persist(
          "dsa_sparse_fwd_split_for", {S_q, H, topk, /*block_I=*/64,
                                       p.multiProcessorCount},
          {{"split", best_split}}, "bench_dsa_split137");
      std::printf("    -> winner split=%d (%.1f us); formula says %d %s"
                  "  [stored]\n",
                  best_split, best_us, rec,
                  best_split == rec ? "(agree)" : "(OVERRIDE)");
    }
    check_hip(hipFree(dq),   "fq");
    check_hip(hipFree(dkv),  "fkv");
    check_hip(hipFree(didx), "fidx");
    check_hip(hipFree(dout), "fout");
    check_hip(hipFree(dlse), "flse");
    check_hip(hipFree(dpart_out), "fpart_out");
    check_hip(hipFree(dpart_lse), "fpart_lse");
  }
  if (do_persist)
    std::printf("\n  Stored under %s; delete the file (or the store) to "
                "re-tune.\n", vkernels::core::tuning::store_dir().c_str());
  hipEventDestroy(start); hipEventDestroy(stop);
  return 0;
}
