// vkernels/kernels/dsa_topk_device.cuh
//
// The two scalar dsa_topk_logits device kernels, shared VERBATIM by the HIP
// TU (dsa.hip, gfx942) and the CUDA TU (dsa.cu, GB10 wmma port). They are
// plain scalar CUDA/HIP-C++ -- no vendor intrinsics -- so ONE definition
// compiles under hipcc and nvcc, exactly like device_numeric.cuh (which this
// header pulls in for the fp8 dequant). Both backends' launchers stage and
// index identically (lane = KV token, block = (batch, split_kv)), so the
// kernels are already byte-identical ports; keeping one copy makes the
// cross-backend bit-identical-output claim structural instead of a
// maintained coincidence.
//
// Only the two scalar GEMV variants live here. The Matrix-Core fast paths
// genuinely differ (AMD MFMA inline asm vs nvcuda::wmma fragments) and stay
// in their backend TUs. Include from a device-compiled TU, after the vendor
// runtime header (<hip/hip_runtime.h> or <cuda_runtime.h>); the kernels are
// `static`, so each including TU gets an internal-linkage copy and no
// device-link symbol can collide.
#pragma once

#include <cstdint>

#include "vkernels/kernels/device_numeric.cuh"

namespace vkernels::kernels {

// ======================================================================
//  fp32-Q scalar GEMV kernel
// ======================================================================
// One warp (B lanes) per (batch, split_kv) block stages Q (H*D fp32) and
// the per-head gate (H fp32) once, then -- per page -- one K tile (B*D
// fp8->fp32) and the B per-token scales (fp32, packed in the trailing B*4
// bytes of each KV block). Total shared = (H*D + H + B*D + B) * 4 = 49,536 B
// at the GLM-5.3 config (H=32, D=128, B=64) -- UNDER the 64 KB non-optin LDS
// cap on gfx942, so NO hipFuncSetAttribute opt-in is needed there (gfx942's
// opt-in ceiling EQUALS the non-optin cap, 65,536 B, verified on a CSCS
// beverin node -- see the KB note mi300a-dynamic-lds-no-optin; the MHC 256 KB
// bug was a *static* over-allocation, not a dynamic-LDS opt-in). On GB10 the
// launcher opts in to the 101,376 B ceiling, so this kernel also serves the
// larger shapes (e.g. H=128 -> 99,072 B) that gfx942 must refuse.
//
// Larger H (e.g. 64 -> 66,048 B) exceeds gfx942's fp32-Q cap, so the launcher
// dispatches dsa_topk_logits_kernel_fp8q below. Each lane owns ONE KV token
// (j = lane), reuses the staged K across all H heads, and writes
// out[b, i*B+lane] only when t = i*B+lane < seq_len[b] (the caller ZEROES
// the output first). The grid's split_kv slices [0, ceildiv(seq_len, B))
// across the split blocks and is perf-only (the grouped logit is
// grouping-independent, mirroring the forward's block_I/inner_iter).
//
// This is the simple, correct baseline (one lane per key, sequential H-sum,
// K staged once per page and reused across all H heads) -- the topk's
// counterpart to dsa_sparse_fwd's "one query row per wavefront, one key at
// a time". A tiled-key / MFMA-fp8 prefetch is a follow-on optimisation, not
// required for correctness.
static __global__ void dsa_topk_logits_kernel(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int H, int D, int B, int max_table_len, int max_seq_len, int split_kv) {
  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int lane = threadIdx.x;                // 0..B-1 -> KV token j

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  extern __shared__ float smem[];
  float* sQ = smem;                            // H * D
  float* sGate = sQ + (size_t)H * D;           // H
  float* sK = sGate + H;                       // B * D (reloaded per page)
  float* sKscale = sK + (size_t)B * D;         // B

  // --- cooperative load Q (H*D fp8 -> fp32) + gate (H fp32) once per block ---
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  for (int idx = lane; idx < H * D; idx += B) sQ[idx] = fp8e4m3fnuz_to_f32(qp[idx]);
  const float* gp = weight + (size_t)b * H;
  for (int h = lane; h < H; h += B) sGate[h] = gp[h];
  __syncthreads();

  for (int it = 0; it < n_iters; ++it) {
    const int i = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + i];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));
    // keys: B*D fp8 e4m3fnuz (bytes [0 : B*D]); scales: B fp32 (bytes
    // [B*D : B*(D+4)], 4-aligned). Cooperative load into shared.
    for (int idx = lane; idx < B * D; idx += B) sK[idx] = fp8e4m3fnuz_to_f32(kbase[idx]);
    for (int j = lane; j < B; j += B) sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];
    __syncthreads();

    // --- this lane owns KV token j = lane within the page ---
    const float* kj = sK + (size_t)lane * D;
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      const float* qh = sQ + (size_t)h * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot += kj[d] * qh[d];
      acc += fmaxf(dot, 0.0f) * sGate[h];
    }
    const int t = i * B + lane;
    if (t < seq_len) out[(size_t)b * max_seq_len + t] = sKscale[lane] * acc;

    __syncthreads();                           // before reloading sK next iter
  }
}

// ======================================================================
//  fp8-Q scalar GEMV kernel (the raw-fp8-Q fallback)
// ======================================================================
// Q staged as RAW fp8 e4m3fnuz (1 byte/element), dequantised on the fly in
// the dot loop with the SAME fp8e4m3fnuz_to_f32 helper the fp32-Q kernel
// uses at load -- the dequanted Q values, and therefore the fp32 dot
// accumulator, the gated H-sum and the written output, are BIT-IDENTICAL to
// dsa_topk_logits_kernel, so the harness cross-check (max_rel < 1e-3 in
// test_dsa_topk_correct.{hip,cu}) carries over unchanged.
//
// On gfx942 this is the fallback for shapes whose fp32-Q staging
// (H*D + H + B*D + B) * 4 exceeds the 64 KB non-optin dynamic-LDS cap
// (e.g. H=64 -> 66,048 B): gfx942 has NO hipFuncSetAttribute opt-in past
// 64 KB (hipFuncSetAttribute(MaxDynamicSharedMemorySize, N>65536) returns
// hipSuccess but the launch silently never runs; verified on a CSCS beverin
// node -- see the KB note mi300a-dynamic-lds-no-optin), so instead of a
// driver opt-in this variant stages Q raw. Staging drops to
// (H + B*D + B) * 4 + H*D bytes = 41,472 B at H=64 (37,248 B at H=32, though
// the fp32-Q kernel remains the fast path there -- Q dequanted once, not per
// dot). On GB10 the opt-in cap is high enough that the fp32-Q kernel runs
// instead; this kernel is ported anyway (and exercised by the harnesses) for
// parity with the HIP dispatch. Each lane owns ONE KV token (j = lane) as in
// the fp32-Q kernel; only the Q storage layout differs.
static __global__ void dsa_topk_logits_kernel_fp8q(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int H, int D, int B, int max_table_len, int max_seq_len, int split_kv) {
  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int lane = threadIdx.x;                // 0..B-1 -> KV token j

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  // fp8-Q staging: the gate, one K tile and its per-token scales as fp32
  // (naturally aligned -- they come first), then Q as RAW fp8 bytes
  // (dequantised on the fly in the dot loop below).
  extern __shared__ float smem[];
  float* sGate = smem;                         // H
  float* sK = sGate + H;                       // B * D (reloaded per page)
  float* sKscale = sK + (size_t)B * D;         // B
  uint8_t* sQ_raw = reinterpret_cast<uint8_t*>(sKscale + B);  // H * D (fp8)

  // --- cooperative load Q (raw fp8) + gate (fp32) once per block ---
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  for (int idx = lane; idx < H * D; idx += B) sQ_raw[idx] = qp[idx];
  const float* gp = weight + (size_t)b * H;
  for (int h = lane; h < H; h += B) sGate[h] = gp[h];
  __syncthreads();

  for (int it = 0; it < n_iters; ++it) {
    const int i = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + i];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));
    // keys: B*D fp8 e4m3fnuz (bytes [0 : B*D]); scales: B fp32 (bytes
    // [B*D : B*(D+4)], 4-aligned). Cooperative load into shared.
    for (int idx = lane; idx < B * D; idx += B) sK[idx] = fp8e4m3fnuz_to_f32(kbase[idx]);
    for (int j = lane; j < B; j += B) sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];
    __syncthreads();

    // --- this lane owns KV token j = lane within the page ---
    const float* kj = sK + (size_t)lane * D;
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      const uint8_t* qh_raw = sQ_raw + (size_t)h * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot += kj[d] * fp8e4m3fnuz_to_f32(qh_raw[d]);
      acc += fmaxf(dot, 0.0f) * sGate[h];
    }
    const int t = i * B + lane;
    if (t < seq_len) out[(size_t)b * max_seq_len + t] = sKscale[lane] * acc;

    __syncthreads();                           // before reloading sK next iter
  }
}

}  // namespace vkernels::kernels
