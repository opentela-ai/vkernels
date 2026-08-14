// vkernels/kernels/mla.hip
//
// Multi-head Latent Attention forward (gfx942 / MI300A), issue #21.
// Online-softmax FlashAttention-style kernel over the absorbed MLA form
// (see mla.hpp): each block owns `BQ` query rows for one (batch, head) and
// streams the shared K/V latent one key at a time, accumulating the weighted
// output in fp32 with a numerically-stable online softmax.
//
//   score[i,j] = scale · ( q_nope[i] · k_c[j] + q_rope[i] · k_pe[j] )
//   out[i]     = Σ_j softmax_causal(score)[i,j] · v_c[j]
//
// Tile & data layout. One warp (64 lanes) owns exactly one query row, so a
// block of `BQ` warps (BQ·64 threads) computes BQ rows. The head dimension
// (kv_lora_rank + qk_rope_head_dim) is laid out STRIDED across the 64 lanes:
// lane `d` owns head elements {d, d+64, d+128, ...}. At the K3 config
// (kv_lora_rank = 512, qk_rope_head_dim = 64 → Dq = 576) that is 8 no-ropes
// + 1 rope = 9 elements per lane and 8 output accumulators per lane — well
// inside the register file, and exact (no bounds-checks fire). The per-key
// score is a single scalar, formed by a warp-shuffle reduction of the
// lane-local partial dot and therefore broadcast to every lane, which keeps
// the per-row online state (max, sum, output slice) lock-step across the
// warp with no LDS synchronisation on the hot path.
//
// Memory. Each key's k_c/k_pe/v_c is read exactly once from global memory;
// the BQ query rows share a key read via L1/L2. The query rows themselves
// are staged in LDS once per block. This is the simple, correct baseline
// (one key at a time, no double-buffered key tile); a tiled key prefetch is
// a follow-on optimisation, not required for correctness against the oracle.
//
// Two-implementation model:
//   mla.cpp  -- CPU reference (oracle), always compiled, two-pass softmax.
//   mla.hip  -- this file, compiled with VKERNELS_HAS_HIP, online softmax.
//               Cross-checked against the oracle by test_mla_correct.hip.
#include <cstdint>

#include "vkernels/kernels/mla.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>

namespace vkernels::kernels::hip {

// Largest supported kv_lora_rank (→ 1024/64 = 16 accumulators per lane).
static constexpr int kMaxLr = 1024;
static constexpr int kPerLane = (kMaxLr + 63) / 64;  // head elements per lane

template <int BQ>
__global__ void mla_fwd_kernel(const float* __restrict__ q,
                               const float* __restrict__ k_c,
                               const float* __restrict__ k_pe,
                               const float* __restrict__ v_c,
                               float* __restrict__ out,
                               int B, int H, int S_q, int S_kv,
                               int q_start, int kv_start,
                               int lr, int rhd, float scale) {
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 6;          // which query row within the tile
  const int lane = tid & 63;          // strided head index this lane owns
  const int gi = q_tile * BQ + warp;  // global query row
  if (gi >= S_q) return;

  const int Dq = lr + rhd;
  const int gqi = q_start + gi;

  // --- stage the BQ query rows in LDS once (per-block working set) ---
  extern __shared__ float sQ[];       // sQ[BQ][Dq]
  for (int idx = tid; idx < BQ * Dq; idx += BQ * 64) {
    const int r = idx / Dq, c = idx % Dq;
    const int gr = q_tile * BQ + r;
    sQ[idx] = (gr < S_q) ? q[(((size_t)b * H + h) * S_q + gr) * Dq + c]
                          : 0.0f;
  }
  __syncthreads();

  const float* qrow = &sQ[warp * Dq];  // this warp's query row

  // Lane-local slices of the no-ropes output accumulator (strided by 64).
  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;          // running max (same on every lane)
  float row_sum = 0.0f;               // running denominator (same on every lane)

  const int nope_tiles = (lr + 63) / 64;
  for (int j = 0; j < S_kv; ++j) {
    const int gkj = kv_start + j;

    // --- lane-local partial score (no-ropes + rope), then warp-reduce ---
    float partial = 0.0f;
    if (gkj <= gqi) {
      const float* kc = k_c + ((size_t)b * S_kv + j) * lr;
      for (int t = 0; t < nope_tiles; ++t) {
        const int idx = lane + t * 64;
        if (idx < lr) partial += qrow[idx] * kc[idx];
      }
      if (lane < rhd) {               // one rope element per lane (rhd <= 64)
        const float* kp = k_pe + ((size_t)b * S_kv + j) * rhd;
        partial += qrow[lr + lane] * kp[lane];
      }
    }
    float score = partial;
    for (int off = 32; off > 0; off >>= 1)   // warp-wide sum → broadcast
      score += __shfl_xor(score, off);
    // Masked keys (gkj > gqi) must contribute ZERO weight, so clamp their
    // score to -inf exactly as the CPU oracle does (else exp(0-max)=1).
    score = (gkj <= gqi) ? (score * scale) : -INFINITY;

    // --- online softmax update (lock-step; identical on every lane) ---
    const float new_max = fmaxf(row_max, score);
    const float exp_old = (row_max == -INFINITY) ? 0.0f
                                                 : expf(row_max - new_max);
    const float exp_s = (score == -INFINITY) ? 0.0f : expf(score - new_max);
    row_sum = row_sum * exp_old + exp_s;
    for (int t = 0; t < nope_tiles; ++t) o[t] *= exp_old;
    if (gkj <= gqi) {
      const float* vc = v_c + ((size_t)b * S_kv + j) * lr;
      for (int t = 0; t < nope_tiles; ++t) {
        const int idx = lane + t * 64;
        if (idx < lr) o[t] += exp_s * vc[idx];
      }
    }
    row_max = new_max;
  }

  // --- normalise and write the lane-local output slice ---
  const float inv = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
  float* orow = out + (((size_t)b * H + h) * S_q + gi) * lr;
  for (int t = 0; t < nope_tiles; ++t) {
    const int idx = lane + t * 64;
    if (idx < lr) orow[idx] = o[t] * inv;
  }
}

// ---------------------------------------------------------------------------
//  Launchers
// ---------------------------------------------------------------------------
namespace {
template <int BQ>
void launch(const float* q, const float* k_c, const float* k_pe,
            const float* v_c, float* out, int B, int H, int S_q, int S_kv,
            int q_start, int kv_start, int lr, int rhd, float scale) {
  const int Dq = lr + rhd;
  const int shmem = BQ * Dq * sizeof(float);
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, B);
  mla_fwd_kernel<BQ><<<grid, block, shmem, 0>>>(
      q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start, lr, rhd,
      scale);
}
}  // namespace

void mla_fwd_with_tile(int B, int H, int S_q, int S_kv, int q_start,
                       int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                       float scale, const float* q, const float* k_c,
                       const float* k_pe, const float* v_c, float* out,
                       int bq, int bn_kv) {
  (void)bn_kv;  // baseline streams one key at a time; reserved for a tiled
                // key prefetch in a follow-on optimisation.
  if (bq <= 1)
    launch<1>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale);
  else if (bq <= 2)
    launch<2>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale);
  else if (bq <= 4)
    launch<4>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale);
  else
    launch<8>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale);
}

void mla_fwd(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
             int kv_lora_rank, int qk_rope_head_dim, float scale,
             const float* q, const float* k_c, const float* k_pe,
             const float* v_c, float* out) {
  int bq = 0, bn = 0, th = 0;
  ::vkernels::kernels::mla_config_for(S_q, kv_lora_rank, qk_rope_head_dim,
                                      &bq, &bn, &th);
  mla_fwd_with_tile(B, H, S_q, S_kv, q_start, kv_start, kv_lora_rank,
                    qk_rope_head_dim, scale, q, k_c, k_pe, v_c, out, bq, bn);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
