// meta/benchmarks/diag_kda_fwd.hip
//
// Diagnostic: where does hip::kda_delta_rule_fwd diverge from the CPU oracle
// (kda_naive_delta_rule_fwd_cpu)? Prints the CPU ref's OWN max_abs (to tell
// "GPU-only blowup = real kernel bug" from "both blow up = test-input
// instability"), plus early-timestep outputs so we can see WHERE the
// recurrence leaves agreement.
//
// Built in build_kda via the same HIP toolchain as test_kda_correct.hip.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vkernels/kernels/kda.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) { std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r)); std::exit(1); } \
  } while (0)

namespace {
float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;  // [-1, 1)
}
float rgate(int seed, int i) { return 0.05f + 0.95f * std::fabs(rnd(seed, i)); }

void normalize_keys(std::vector<float>& k, int B, int H, int S, int D) {
  for (int x = 0; x < B * H * S; ++x) {
    float* kk = k.data() + (size_t)x * D;
    float norm = 0.0f;
    for (int d = 0; d < D; ++d) norm += kk[d] * kk[d];
    norm = 1.0f / std::sqrt(norm + 1e-12f);
    for (int d = 0; d < D; ++d) kk[d] *= norm;
  }
}
}  // namespace

int main(int argc, char** argv) {
  int D = (argc > 1) ? std::atoi(argv[1]) : 16;
  int S = (argc > 2) ? std::atoi(argv[2]) : 64;
  int H = (argc > 3) ? std::atoi(argv[3]) : 1;
  int B = 1;
  const size_t n = (size_t)B * H * S * D;
  const size_t nbs = (size_t)B * H * S;  // gate/beta count

  std::vector<float> q(n), k(n), v(n), g(nbs * D), beta(nbs);
  for (size_t i = 0; i < n; ++i) { q[i] = rnd(1, (int)i); k[i] = rnd(2, (int)i); v[i] = rnd(3, (int)i); }
  for (size_t i = 0; i < g.size(); ++i) g[i] = rgate(4, (int)i);
  for (size_t i = 0; i < beta.size(); ++i) beta[i] = rgate(5, (int)i);
  normalize_keys(k, B, H, S, D);

  // gate stats
  double gmin = 1e9, gmax = 0, gmean = 0;
  for (size_t i = 0; i < nbs; ++i) { gmin = std::min(gmin, (double)g[i]); gmax = std::max(gmax, (double)g[i]); gmean += g[i]; }
  gmean /= nbs;
  std::printf("cfg: B=%d H=%d S=%d D=%d   g[%.3f,%.3f] mean=%.3f   beta[%.3f,%.3f]\n",
              B, H, S, D, gmin, gmax, gmean, beta[0], beta[nbs-1]);

  std::vector<float> ref(n, 0.0f), host(n, 0.0f);
  vkernels::kernels::kda_naive_delta_rule_fwd_cpu(
      q.data(), k.data(), v.data(), g.data(), beta.data(), ref.data(), B, H, S, D);

  // CPU ref own magnitude
  double rmax = 0, rmin = 1e9; int rmax_i = 0;
  for (size_t i = 0; i < n; ++i) { double a = std::fabs((double)ref[i]); if (a > rmax) { rmax = a; rmax_i = (int)i; } if (a < rmin && a > 0) rmin = a; }
  std::printf("CPU ref:  max_abs=%.6g  (at i=%d, t=%d,d=%d)  min_nz=%.6g\n",
              rmax, rmax_i, rmax_i / D, rmax_i % D, rmin);
  // a few early-timestep CPU outputs (head 0, token t, all D)
  std::printf("CPU ref early tokens (h=0):\n");
  int ts[] = {0, 1, 2, 3, 5, 10, 20, 30, S-1};
  for (int t : ts) {
    if (t < 0 || t >= S) continue;
    const size_t off = (size_t)t * D;
    std::printf("  t=%-3d:", t);
    for (int d = 0; d < std::min(D, 6); ++d) std::printf(" %+.4f", ref[off + d]);
    std::printf("%s\n", D > 6 ? " ..." : "");
  }

  // GPU
  float *dq, *dk, *dv, *dg, *db, *dout;
  CK(hipMalloc(&dq, n * 4), "q"); CK(hipMalloc(&dk, n * 4), "k");
  CK(hipMalloc(&dv, n * 4), "v"); CK(hipMalloc(&dg, nbs * D * 4), "g");
  CK(hipMalloc(&db, nbs * 4), "b"); CK(hipMalloc(&dout, n * 4), "out");
  CK(hipMemcpy(dq, q.data(), n * 4, hipMemcpyHostToDevice), "q");
  CK(hipMemcpy(dk, k.data(), n * 4, hipMemcpyHostToDevice), "k");
  CK(hipMemcpy(dv, v.data(), n * 4, hipMemcpyHostToDevice), "v");
  CK(hipMemcpy(dg, g.data(), nbs * D * 4, hipMemcpyHostToDevice), "g");
  CK(hipMemcpy(db, beta.data(), nbs * 4, hipMemcpyHostToDevice), "b");
  vkernels::kernels::hip::kda_delta_rule_fwd(dq, dk, dv, dg, db, dout, B, H, S, D, S/4);
  CK(hipDeviceSynchronize(), "sync");
  CK(hipMemcpy(host.data(), dout, n * 4, hipMemcpyDeviceToHost), "rd");
  hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dg); hipFree(db); hipFree(dout);

  double hmax = 0; int hmax_i = 0;
  for (size_t i = 0; i < n; ++i) { double a = std::fabs((double)host[i]); if (a > hmax) { hmax = a; hmax_i = (int)i; } }
  std::printf("GPU host: max_abs=%.6g  (at i=%d, t=%d,d=%d)\n",
              hmax, hmax_i, hmax_i / D, hmax_i % D);
  std::printf("GPU host early tokens (h=0):\n");
  for (int t : ts) {
    if (t < 0 || t >= S) continue;
    const size_t off = (size_t)t * D;
    std::printf("  t=%-3d:", t);
    for (int d = 0; d < std::min(D, 6); ++d) std::printf(" %+.4f", host[off + d]);
    std::printf("%s\n", D > 6 ? " ..." : "");
  }

  // first token where CPU and GPU disagree by > 1e-3
  int first_bad = -1;
  for (int t = 0; t < S && first_bad < 0; ++t)
    for (int d = 0; d < D; ++d) {
      size_t off = (size_t)t * D + d;
      if (std::fabs((double)host[off] - ref[off]) > 1e-3 * std::fmax(std::fabs((double)ref[off]), 1.0)) {
        first_bad = t; break;
      }
    }
  std::printf("first disagree (rel>1e-3): token t=%d\n", first_bad);
  return 0;
}
