// meta/benchmarks/test_mla_correct.hip
//
// End-to-end correctness test for the absorbed-form MLA forward (issue #21)
// on gfx942, against the CPU reference in vkernels::kernels::mla_fwd_cpu.
//
//   hip::mla_fwd  vs  mla_fwd_cpu (two-pass softmax, fp32)
//
// Shapes: the K3 decode/prefill MLA config
//   (kv_lora_rank=512, qk_rope_head_dim=64 → Dq=576; H ∈ {1, 128})
// at serving S_q ∈ {1, 64, 8192} with S_kv = S_q and q_start=kv_start=0.
// Scale = 1/√(kv_lora_rank + qk_rope_head_dim) = 1/√576.
//
// Tolerance: the device kernel accumulates an online softmax in fp32 and the
// CPU reference uses a two-pass fp32 softmax; for K3 shapes both are exact
// fp32 reductions over the same masked keys, so they agree to within fp32
// round-off scaled by the sequence length. We report max_abs / max_rel and
// require max_rel < 2e-2 (same convention as test_gemm_bf16_correct.hip).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/mla.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

float rnd(int seed, int i) {  // deterministic pseudo-random in [-1, 1]
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

struct Stats { double max_abs, max_rel; };

Stats run_case(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
               int kv_lora_rank, int qk_rope_head_dim, float scale,
               const std::vector<float>& q,
               const std::vector<float>& k_c,
               const std::vector<float>& k_pe,
               const std::vector<float>& v_c,
               float* dq, float* dkc, float* dkp, float* dvc, float* dout) {
  const int Dq = kv_lora_rank + qk_rope_head_dim;
  std::vector<float> host(((size_t)B * H * S_q) * kv_lora_rank, 0.0f);
  std::vector<float> ref(((size_t)B * H * S_q) * kv_lora_rank, 0.0f);

  // CPU oracle.
  vkernels::kernels::mla_fwd_cpu(
      B, H, S_q, S_kv, q_start, kv_start, kv_lora_rank, qk_rope_head_dim,
      scale, q.data(), k_c.data(), k_pe.data(), v_c.data(), ref.data());

  // Device.
  const size_t nq = (size_t)B * H * S_q * Dq;
  const size_t nkc = (size_t)B * S_kv * kv_lora_rank;
  const size_t nkp = (size_t)B * S_kv * qk_rope_head_dim;
  const size_t nvc = (size_t)B * S_kv * kv_lora_rank;
  const size_t nout = (size_t)B * H * S_q * kv_lora_rank;
  CK(hipMemcpy(dq, q.data(), nq * 4, hipMemcpyHostToDevice), "cpyq");
  CK(hipMemcpy(dkc, k_c.data(), nkc * 4, hipMemcpyHostToDevice), "cpykc");
  CK(hipMemcpy(dkp, k_pe.data(), nkp * 4, hipMemcpyHostToDevice), "cpykp");
  CK(hipMemcpy(dvc, v_c.data(), nvc * 4, hipMemcpyHostToDevice), "cpyvc");
  vkernels::kernels::hip::mla_fwd(
      B, H, S_q, S_kv, q_start, kv_start, kv_lora_rank, qk_rope_head_dim,
      scale, dq, dkc, dkp, dvc, dout);
  CK(hipDeviceSynchronize(), "sync");
  CK(hipMemcpy(host.data(), dout, nout * 4, hipMemcpyDeviceToHost), "rdout");

  Stats s{0.0, 0.0};
  for (size_t i = 0; i < nout; ++i) {
    double e = std::fabs((double)host[i] - ref[i]);
    if (e > s.max_abs) s.max_abs = e;
    double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    if (e / d > s.max_rel) s.max_rel = e / d;
  }
  return s;
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  // K3 MLA config.
  const int kv_lora_rank = 512;
  const int qk_rope_head_dim = 64;
  const int Dq = kv_lora_rank + qk_rope_head_dim;
  const float scale = 1.0f / std::sqrt((float)Dq);

  // Peak buffers across all cases.
  const int Bmax = 1, Hmax = 128, Sqmax = 8192, Svmax = 8192;
  float *dq, *dkc, *dkp, *dvc, *dout;
  CK(hipMalloc(&dq, (size_t)Bmax * Hmax * Sqmax * Dq * 4), "alloc q");
  CK(hipMalloc(&dkc, (size_t)Bmax * Svmax * kv_lora_rank * 4), "alloc kc");
  CK(hipMalloc(&dkp, (size_t)Bmax * Svmax * qk_rope_head_dim * 4), "alloc kp");
  CK(hipMalloc(&dvc, (size_t)Bmax * Svmax * kv_lora_rank * 4), "alloc vc");
  CK(hipMalloc(&dout, (size_t)Bmax * Hmax * Sqmax * kv_lora_rank * 4),
     "alloc out");

  int fails = 0;
  const float THRESH = 2e-2f;

  struct Cfg { int B, H, S_q; };
  const Cfg cfgs[] = {
      {1, 1, 1},     // decode, 1 head (hand-check scale)
      {1, 1, 64},    // decode, short
      {1, 16, 64},   // decode, multi-head
      {1, 1, 8192},  // prefill (single head; H=128 × 8192 is too slow for CPU)
  };

  for (const auto& c : cfgs) {
    const int S_kv = c.S_q;
    std::vector<float> q((size_t)c.B * c.H * c.S_q * Dq);
    std::vector<float> kc((size_t)c.B * S_kv * kv_lora_rank);
    std::vector<float> kp((size_t)c.B * S_kv * qk_rope_head_dim);
    std::vector<float> vc((size_t)c.B * S_kv * kv_lora_rank);
    for (size_t i = 0; i < q.size(); ++i) q[i] = rnd(1, (int)i);
    for (size_t i = 0; i < kc.size(); ++i) kc[i] = rnd(2, (int)i);
    for (size_t i = 0; i < kp.size(); ++i) kp[i] = rnd(3, (int)i);
    for (size_t i = 0; i < vc.size(); ++i) vc[i] = rnd(4, (int)i);

    auto st = run_case(c.B, c.H, c.S_q, S_kv, 0, 0, kv_lora_rank,
                       qk_rope_head_dim, scale, q, kc, kp, vc,
                       dq, dkc, dkp, dvc, dout);
    bool ok = st.max_rel < THRESH;
    std::printf("  B=%d H=%-3d S_q=%-4d  max_abs=%.4f max_rel=%.6f  %s\n",
                c.B, c.H, c.S_q, st.max_abs, st.max_rel,
                ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  hipFree(dq); hipFree(dkc); hipFree(dkp); hipFree(dvc); hipFree(dout);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
