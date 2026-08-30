// meta/benchmarks/test_dsa_correct.hip
//
// End-to-end correctness test for the DeepseekSparseAttn sparse-MLA forward
// (issue #51) on gfx942, against the CPU reference in
// vkernels::kernels::dsa_sparse_fwd_cpu.
//
//   hip::dsa_sparse_fwd  vs  dsa_sparse_fwd_cpu (two-pass base-2 softmax)
//
// Shapes: BOTH families the issue calls out —
//   * GLM-5.3-Flash : dim=256, tail_dim=0  -> W=256, d_v=256  (the layout the
//     tilelang code path cannot compile; the kernel skips the rope-tail dot
//     via tail_tiles == 0).
//   * DeepSeek-V3   : dim=576, tail_dim=64 -> W=640, d_v=512.
//
// at serving S_q in {1 (decode), 64 (short), 8192 (prefill)}. The indexer
// selects `topk` keys per query from [0, S_kv) with ~10% masked to -1 (the
// kpool-tail path the kernel must zero).
//
// Method. The CPU oracle is fp32 end-to-end; the device kernel is bf16
// storage / fp32 accumulation (the established gfx942 convention, identical
// to gemm_bf16.hip). To isolate the kernel's correctness from bf16 input
// quantization, the fp32 source is rounded to bf16 FIRST (host RNE), then the
// CPU oracle runs on the recovered fp32 and the device on the bf16 inputs —
// exactly as test_gemm_bf16_correct.hip does. Only online-vs-two-pass
// ordering and the single RNE on store differ, so they agree to within ~1 bf16
// ULP. We report max_abs / max_rel and require max_rel < 2e-2 for the output
// and < 1e-3 for the fp32 LSE.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/dsa.hpp"

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

// Host bf16 (uint16 IEEE bit patterns) <-> fp32, RNE — identical to the
// helpers in gemm_bf16.cpp and to f32bits_to_bf16 in moe_device.hip.
uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return (uint16_t)(b >> 16);
}
float bf2f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

struct Stats { double max_abs, max_rel; };
struct LseStats { double max_abs, max_rel, worst_inf; };

void cmp_out(const std::vector<float>& got, const std::vector<float>& ref,
             Stats* s) {
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs((double)got[i] - ref[i]);
    if (e > s->max_abs) s->max_abs = e;
    double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    if (e / d > s->max_rel) s->max_rel = e / d;
  }
}

void cmp_lse(const std::vector<float>& got, const std::vector<float>& ref,
             LseStats* s) {
  for (size_t i = 0; i < ref.size(); ++i) {
    const float g = got[i], r = ref[i];
    const bool gi = std::isfinite(g), ri = std::isfinite(r);
    if (gi != ri) {                       // -inf vs finite is a hard failure
      s->worst_inf = INFINITY;
      continue;
    }
    if (!gi && !ri) continue;             // both -inf (fully masked): exact
    double e = std::fabs((double)g - r);
    if (e > s->max_abs) s->max_abs = e;
    double d = std::fmax(std::fabs((double)r), 1.0);
    if (e / d > s->max_rel) s->max_rel = e / d;
  }
}

struct CaseStats { Stats out; LseStats lse; };

CaseStats run_case(int S_q, int S_kv, int H, int dim, int tail_dim, int topk,
                   int kv_group, int block_I, int inner_iter, float sm_scale,
                   const std::vector<float>& q,    // fp32 source (S_q*H*W)
                   const std::vector<float>& kv,   // fp32 source (S_kv*W)
                   const std::vector<int32_t>& idx,// [S_q*topk]
                   uint16_t* dq, uint16_t* dkv, int32_t* didx, uint16_t* dout,
                   float* dlse) {
  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;
  const size_t nq = (size_t)S_q * H * W;
  const size_t nkv = (size_t)S_kv * W;
  const size_t nout = (size_t)S_q * H * d_v;
  const size_t nlse = (size_t)S_q * H;

  // Round source -> bf16, then recover to fp32 (the value BOTH paths see).
  std::vector<uint16_t> qb(nq), kb(nkv);
  std::vector<float> qr(nq), kr(nkv);
  for (size_t i = 0; i < nq; ++i) { qb[i] = f2bf(q[i]); qr[i] = bf2f(qb[i]); }
  for (size_t i = 0; i < nkv; ++i) { kb[i] = f2bf(kv[i]); kr[i] = bf2f(kb[i]); }

  // CPU oracle on the bf16-recovered inputs.
  std::vector<float> ref(nout, 0.0f), ref_lse(nlse, 0.0f);
  vkernels::kernels::dsa_sparse_fwd_cpu(
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I, inner_iter,
      sm_scale, /*return_lse=*/true, qr.data(), kr.data(), idx.data(),
      ref.data(), ref_lse.data());

  // Device.
  CK(hipMemcpy(dq, qb.data(), nq * 2, hipMemcpyHostToDevice), "cpyq");
  CK(hipMemcpy(dkv, kb.data(), nkv * 2, hipMemcpyHostToDevice), "cpykv");
  // indices are [1, S_q, kv_group==1, topk] -> S_q*topk int32 (the CPU
  // oracle and the kernel share the same single-head_kv selection).
  CK(hipMemcpy(didx, idx.data(), (size_t)S_q * topk * 4,
               hipMemcpyHostToDevice), "cpyidx");
  vkernels::kernels::hip::dsa_sparse_fwd(
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I, inner_iter,
      sm_scale, /*return_lse=*/true, dq, dkv, didx, dout, dlse);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<uint16_t> hout(nout);
  std::vector<float> hlse(nlse);
  CK(hipMemcpy(hout.data(), dout, nout * 2, hipMemcpyDeviceToHost), "rdout");
  CK(hipMemcpy(hlse.data(), dlse, nlse * 4, hipMemcpyDeviceToHost), "rdlse");

  std::vector<float> got(nout);
  for (size_t i = 0; i < nout; ++i) got[i] = bf2f(hout[i]);

  CaseStats cs{};
  cmp_out(got, ref, &cs.out);
  cmp_lse(hlse, ref_lse, &cs.lse);
  return cs;
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  const float LOG2E = 1.4426950408889634f;

  // Peak buffers across all cases (GLM-5.3 and DeepSeek-V3).
  const int Sqmax = 8192, Hmax = 64, Wmax = 640, Svmax = 4096;
  const int Dvmax = 512, Topkmax = 2048;
  uint16_t *dq, *dkv, *dout;
  int32_t* didx;
  float* dlse;
  CK(hipMalloc(&dq, (size_t)Sqmax * Hmax * Wmax * 2), "alloc q");
  CK(hipMalloc(&dkv, (size_t)Svmax * Wmax * 2), "alloc kv");
  CK(hipMalloc(&didx, (size_t)Sqmax * Topkmax * 4), "alloc idx");
  CK(hipMalloc(&dout, (size_t)Sqmax * Hmax * Dvmax * 2), "alloc out");
  CK(hipMalloc(&dlse, (size_t)Sqmax * Hmax * 4), "alloc lse");

  int fails = 0;
  const float OUT_THRESH = 2e-2f;
  const float LSE_THRESH = 1e-3f;

  struct Cfg { int S_q, S_kv, H, dim, tail_dim, topk; };
  const Cfg cfgs[] = {
      // GLM-5.3-Flash: tail_dim == 0 (the uncompilable tilelang case).
      {1, 128, 64, 256, 0, 128},     // decode, topk=128
      {1, 256, 64, 256, 0, 256},     // decode, topk=256
      {1, 2048, 64, 256, 0, 2048},   // decode, full topk=2048
      {64, 256, 1, 256, 0, 128},     // short
      {8192, 1024, 1, 256, 0, 128},  // prefill (H=1; CPU-tractable)
      // DeepSeek-V3: tail_dim > 0.
      {1, 256, 16, 576, 64, 256},    // decode, d_v=512
      {1, 512, 16, 576, 64, 256},    // decode, more keys
      {64, 256, 1, 576, 64, 256},    // short
      {8192, 1024, 1, 576, 64, 256}, // prefill (H=1)
      // Tiny hand-check shapes (sanity, both tail families).
      {1, 2, 1, 2, 0, 2},            // scores {1,1} -> {0.5,0.5}, lse=2
      {1, 2, 1, 2, 1, 2},            // tail>0: scores {2,3} -> 4/1.5
  };

  for (const auto& c : cfgs) {
    const int W = c.dim + c.tail_dim;
    const float sm_scale = (1.0f / std::sqrt((float)W)) * LOG2E;
    std::vector<float> q((size_t)c.S_q * c.H * W);
    std::vector<float> kv((size_t)c.S_kv * W);
    std::vector<int32_t> idx((size_t)c.S_q * c.topk);
    for (size_t i = 0; i < q.size(); ++i) q[i] = rnd(1, (int)i);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = rnd(2, (int)i);
    for (int i = 0; i < c.S_q; ++i)
      for (int k = 0; k < c.topk; ++k)
        idx[(size_t)i * c.topk + k] =
            (k % 10 == 9) ? -1 : (int32_t)(((unsigned)(i * 31 + k * 7)) %
                                           (unsigned)c.S_kv);

    auto st = run_case(c.S_q, c.S_kv, c.H, c.dim, c.tail_dim, c.topk, 1, 64,
                       1, sm_scale, q, kv, idx, dq, dkv, didx, dout, dlse);
    bool ok = st.out.max_rel < OUT_THRESH && st.lse.max_rel < LSE_THRESH &&
              st.lse.worst_inf < 1.0;
    std::printf("  S_q=%-4d S_kv=%-4d H=%-2d dim=%-3d tail=%-2d topk=%-4d  "
                "out[a=%.4f r=%.6f] lse[a=%.4f r=%.6f inf=%.1f]  %s\n",
                c.S_q, c.S_kv, c.H, c.dim, c.tail_dim, c.topk,
                st.out.max_abs, st.out.max_rel, st.lse.max_abs, st.lse.max_rel,
                st.lse.worst_inf, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  hipFree(dq); hipFree(dkv); hipFree(didx); hipFree(dout); hipFree(dlse);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
