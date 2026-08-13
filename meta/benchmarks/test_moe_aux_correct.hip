// meta/benchmarks/test_moe_aux_correct.hip
//
// HIP-vs-CPU correctness harness for the MXFP4 MoE orchestration ops
// (issue #22): mxfp4_moe_quant, mxfp4_moe_sort, mxfp4_moe_sort_scales,
// mxfp4_moe_scatter_reduce, mxfp4_moe_scatter_reduce_q.
//
// The five ops are pure data-movement / integer-quantization primitives
// (no floating accumulation across blocks), so the quant and two sort ops
// are checked bit-exact. The two scatter-reduce ops accumulate into each
// output token with atomicAdd (top_k > 1 → several sorted rows land on one
// token), so the reduction order can differ host vs device; those are
// checked with a small relative tolerance.
//
// Build (inside the vkernels:hip-full container on an MI300A node):
//   cmake --preset hip -B build-hip -DVKERNELS_BUILD_BENCHMARKS=ON
//   cmake --build build-hip --target test_moe_aux_correct
//   ./build-hip/meta/benchmarks/test_moe_aux_correct

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/moe_aux.hpp"    // CPU references (oracle)
#include "vkernels/kernels/moe_fused.hpp"  // moe_align_block_size

// The HIP entry points live in moe_aux.hip under vkernels::kernels::hip
// with identical signatures; they are intentionally not re-declared in the
// header (so the host reference names stay unique in `vkl` discovery), so
// the harness forward-declares them here.
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

// Deterministic pseudo-random in [-1, 1] (same recipe as the moe harness).
float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

float bits2f(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
uint32_t f2bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }

// bf16 round-to-nearest-even, matching moe_aux.cpp / moe.cpp.
uint16_t f2bf(float v) {
  uint32_t b = f2bits(v);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}
float bf2f(uint16_t v) { return bits2f((uint32_t)v << 16); }

int g_pass = 0, g_fail = 0;

void check_bits(const char* name, const uint8_t* cpu, const uint8_t* gpu,
                std::size_t n) {
  std::size_t bad = 0, first = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (cpu[i] != gpu[i]) {
      if (bad == 0) first = i;
      ++bad;
    }
  if (bad == 0) {
    std::printf("  %-22s bit-exact  PASS (%zu bytes)\n", name, n);
    ++g_pass;
  } else {
    std::printf("  %-22s bit-exact  FAIL (%zu/%zu differ @%zu: cpu=%02x gpu=%02x)\n",
                name, bad, n, first, cpu[first], gpu[first]);
    ++g_fail;
  }
}

void check_bits16(const char* name, const uint16_t* cpu, const uint16_t* gpu,
                  std::size_t n) {
  std::size_t bad = 0, first = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (cpu[i] != gpu[i]) {
      if (bad == 0) first = i;
      ++bad;
    }
  if (bad == 0) {
    std::printf("  %-22s bit-exact  PASS (%zu vals)\n", name, n);
    ++g_pass;
  } else {
    std::printf("  %-22s bit-exact  FAIL (%zu/%zu differ @%zu: cpu=%04x gpu=%04x)\n",
                name, bad, n, first, cpu[first], gpu[first]);
    ++g_fail;
  }
}

void check_float(const char* name, const float* cpu, const float* gpu,
                 std::size_t n, double rel_tol) {
  double max_abs = 0, max_rel = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double err = std::fabs((double)gpu[i] - (double)cpu[i]);
    if (err > max_abs) max_abs = err;
    double d = std::fmax(std::fabs((double)cpu[i]), 1.0);
    if (err / d > max_rel) { max_rel = err / d; first = i; }
  }
  if (max_rel < rel_tol) {
    std::printf("  %-22s rel<%.0e   PASS (max_abs=%.4e max_rel=%.4e)\n",
                name, rel_tol, max_abs, max_rel);
    ++g_pass;
  } else {
    std::printf("  %-22s rel<%.0e   FAIL (max_abs=%.4e max_rel=%.4e @%zu: cpu=%g gpu=%g)\n",
                name, rel_tol, max_abs, max_rel, first,
                (double)cpu[first], (double)gpu[first]);
    ++g_fail;
  }
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (gfx%d)\n", p.name, p.gcnRevision);

  // Small-but-nontrivial config: multiple M/N blocks, padding rows, and
  // multi-expert routing (top_k=2, each token to 2 distinct experts).
  constexpr int M = 8, hidden = 256, ispp = 128, top_k = 2;
  constexpr int E = 4, BLOCK_M = 16, group_size = 32;
  static_assert(hidden % group_size == 0, "hidden must divide group_size");
  const int n_groups = hidden / group_size;
  const int N = M * top_k;
  const int EM_max = E * BLOCK_M;  // every expert padded to its own block

  // --- Host inputs ---
  std::vector<uint16_t> hA(M * hidden);
  for (int i = 0; i < M * hidden; ++i) hA[i] = f2bf(rnd(1, i) * 0.1f);

  std::vector<int32_t> h_topk_ids(M * top_k);
  std::vector<float> h_topk_w(M * top_k);
  for (int i = 0; i < M; ++i) {
    h_topk_ids[i * top_k + 0] = i % E;
    h_topk_ids[i * top_k + 1] = (i + 1) % E;
    h_topk_w[i * top_k + 0] = 0.7f;
    h_topk_w[i * top_k + 1] = 0.3f;
  }

  std::vector<int32_t> h_sids(EM_max), h_eids(EM_max / BLOCK_M);
  int EM = vkernels::kernels::moe_align_block_size(
      h_topk_ids.data(), M, top_k, BLOCK_M, E, h_sids.data(), h_eids.data());
  std::printf("config: M=%d hidden=%d ispp=%d top_k=%d E=%d gs=%d "
              "EM=%d (padded to %d-blocks)\n",
              M, hidden, ispp, top_k, E, group_size, EM, EM_max);

  // Per-sorted-row routing weights: tws[r] = topk_w_flat[sorted_ids[r]].
  std::vector<float> h_tws(EM, 0.0f);
  for (int r = 0; r < EM; ++r)
    if (h_sids[r] < N) h_tws[r] = h_topk_w[h_sids[r]];

  // --- CPU oracle ---
  std::vector<uint8_t> packed_c(M * hidden / 2), scales_c(M * n_groups);
  vkernels::kernels::mxfp4_moe_quant(hA.data(), packed_c.data(),
                                     scales_c.data(), M, hidden, group_size);

  std::vector<uint16_t> A_sorted_c(EM * hidden, 0);
  vkernels::kernels::mxfp4_moe_sort(hA.data(), h_sids.data(), A_sorted_c.data(),
                                    M, hidden, top_k, EM);

  std::vector<uint8_t> scales_sorted_c(EM * n_groups, 0);
  vkernels::kernels::mxfp4_moe_sort_scales(scales_c.data(), h_sids.data(),
                                           scales_sorted_c.data(), M,
                                           n_groups, top_k, EM);

  // fp32 partials in sorted order (stand-in for the grouped-GEMM output).
  std::vector<float> partial(EM * hidden);
  for (int i = 0; i < EM * hidden; ++i) partial[i] = rnd(20, i) * 0.5f;
  std::vector<float> out_c(M * hidden, 0.0f);
  vkernels::kernels::mxfp4_moe_scatter_reduce(partial.data(), h_tws.data(),
                                              h_sids.data(), out_c.data(), M,
                                              hidden, top_k, EM);

  // Quantized partial for _q: reuse mxfp4_moe_quant on the [EM, hidden]
  // fp32-as-bf16 partial. The _q op dequantizes inline, so its oracle is the
  // CPU _q on the same (partial_q, partial_s) — not the fp32 scatter above.
  std::vector<uint16_t> partial_bf(EM * hidden);
  for (int i = 0; i < EM * hidden; ++i) partial_bf[i] = f2bf(partial[i]);
  std::vector<uint8_t> partial_q(EM * hidden / 2), partial_s(EM * n_groups);
  vkernels::kernels::mxfp4_moe_quant(partial_bf.data(), partial_q.data(),
                                     partial_s.data(), EM, hidden, group_size);
  std::vector<float> out_q_c(M * hidden, 0.0f);
  vkernels::kernels::mxfp4_moe_scatter_reduce_q(
      partial_q.data(), partial_s.data(), h_tws.data(), h_sids.data(),
      out_q_c.data(), M, hidden, top_k, EM, group_size);

  // --- Device buffers ---
  uint16_t *dA, *dA_sorted, *dPbf;
  uint8_t *dPacked, *dScales, *dScalesSorted, *dPq, *dPs;
  int32_t *dSids;
  float *dTw, *dOut, *dOut_q, *dPartial;
  CK(hipMalloc(&dA, M * hidden * 2), "mA");
  CK(hipMalloc(&dA_sorted, EM * hidden * 2), "mA_sorted");
  CK(hipMalloc(&dPbf, EM * hidden * 2), "mPbf");
  CK(hipMalloc(&dPacked, M * hidden / 2), "mPacked");
  CK(hipMalloc(&dScales, M * n_groups), "mScales");
  CK(hipMalloc(&dScalesSorted, EM * n_groups), "mScalesSorted");
  CK(hipMalloc(&dPq, EM * hidden / 2), "mPq");
  CK(hipMalloc(&dPs, EM * n_groups), "mPs");
  CK(hipMalloc(&dSids, EM * 4), "mSids");
  CK(hipMalloc(&dTw, EM * 4), "mTw");
  CK(hipMalloc(&dOut, M * hidden * 4), "mOut");
  CK(hipMalloc(&dOut_q, M * hidden * 4), "mOut_q");
  CK(hipMalloc(&dPartial, EM * hidden * 4), "mPartial");

  CK(hipMemcpy(dA, hA.data(), M * hidden * 2, hipMemcpyHostToDevice), "cA");
  CK(hipMemcpy(dPbf, partial_bf.data(), EM * hidden * 2, hipMemcpyHostToDevice), "cPbf");
  CK(hipMemcpy(dSids, h_sids.data(), EM * 4, hipMemcpyHostToDevice), "cSids");
  CK(hipMemcpy(dTw, h_tws.data(), EM * 4, hipMemcpyHostToDevice), "cTw");
  CK(hipMemcpy(dPartial, partial.data(), EM * hidden * 4, hipMemcpyHostToDevice), "cPartial");

  // --- HIP kernels ---
  std::printf("HIP vs CPU:\n");
  using namespace vkernels::kernels::hip;

  mxfp4_moe_quant(dA, dPacked, dScales, M, hidden, group_size);
  mxfp4_moe_sort(dA, dSids, dA_sorted, M, hidden, top_k, EM);
  mxfp4_moe_sort_scales(dScales, dSids, dScalesSorted, M, n_groups, top_k, EM);

  // Quantize the bf16 partial on device (same path the W4A4 down-projection
  // would produce), feeding scatter_reduce_q.
  mxfp4_moe_quant(dPbf, dPq, dPs, EM, hidden, group_size);

  CK(hipMemset(dOut, 0, M * hidden * 4), "zOut");
  mxfp4_moe_scatter_reduce(dPartial, dTw, dSids, dOut, M, hidden, top_k, EM);
  CK(hipMemset(dOut_q, 0, M * hidden * 4), "zOut_q");
  mxfp4_moe_scatter_reduce_q(dPq, dPs, dTw, dSids, dOut_q, M, hidden, top_k,
                             EM, group_size);
  CK(hipDeviceSynchronize(), "sync");

  // --- Read back & compare ---
  std::vector<uint8_t> packed_g(M * hidden / 2), scales_g(M * n_groups);
  std::vector<uint16_t> A_sorted_g(EM * hidden, 0);
  std::vector<uint8_t> scales_sorted_g(EM * n_groups, 0);
  std::vector<uint8_t> pq_g(EM * hidden / 2), ps_g(EM * n_groups);
  std::vector<float> out_g(M * hidden), out_q_g(M * hidden);
  CK(hipMemcpy(packed_g.data(), dPacked, M * hidden / 2, hipMemcpyDeviceToHost), "rPacked");
  CK(hipMemcpy(scales_g.data(), dScales, M * n_groups, hipMemcpyDeviceToHost), "rScales");
  CK(hipMemcpy(A_sorted_g.data(), dA_sorted, EM * hidden * 2, hipMemcpyDeviceToHost), "rA_sorted");
  CK(hipMemcpy(scales_sorted_g.data(), dScalesSorted, EM * n_groups, hipMemcpyDeviceToHost), "rScalesSorted");
  CK(hipMemcpy(pq_g.data(), dPq, EM * hidden / 2, hipMemcpyDeviceToHost), "rPq");
  CK(hipMemcpy(ps_g.data(), dPs, EM * n_groups, hipMemcpyDeviceToHost), "rPs");
  CK(hipMemcpy(out_g.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rOut");
  CK(hipMemcpy(out_q_g.data(), dOut_q, M * hidden * 4, hipMemcpyDeviceToHost), "rOut_q");

  check_bits("mxfp4_moe_quant.packed", packed_c.data(), packed_g.data(),
             M * hidden / 2);
  check_bits("mxfp4_moe_quant.scales", scales_c.data(), scales_g.data(),
             M * n_groups);
  check_bits16("mxfp4_moe_sort", A_sorted_c.data(), A_sorted_g.data(),
               (std::size_t)EM * hidden);
  check_bits("mxfp4_moe_sort_scales", scales_sorted_c.data(),
             scales_sorted_g.data(), (std::size_t)EM * n_groups);
  // The on-device re-quant of the bf16 partial must match the CPU quant of
  // the same bf16 partial (both feed scatter_reduce_q).
  check_bits("mxfp4_moe_quant(partial).packed", partial_q.data(), pq_g.data(),
             EM * hidden / 2);
  check_bits("mxfp4_moe_quant(partial).scales", partial_s.data(), ps_g.data(),
             EM * n_groups);
  // Scatter-reduce uses atomicAdd → tolerant comparison.
  check_float("mxfp4_moe_scatter_reduce", out_c.data(), out_g.data(),
              M * hidden, 1e-5);
  check_float("mxfp4_moe_scatter_reduce_q", out_q_c.data(), out_q_g.data(),
              M * hidden, 1e-5);

  hipFree(dA); hipFree(dA_sorted); hipFree(dPbf);
  hipFree(dPacked); hipFree(dScales); hipFree(dScalesSorted);
  hipFree(dPq); hipFree(dPs); hipFree(dSids); hipFree(dTw);
  hipFree(dOut); hipFree(dOut_q); hipFree(dPartial);

  std::printf("%s (%d passed, %d failed)\n",
              g_fail == 0 ? "ALL PASS" : "FAILURES", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
