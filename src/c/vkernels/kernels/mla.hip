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
#include <mutex>

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
//  Split-K decode (issue #82)
//
//  A decode grid of B*H*S_q blocks (one 64-thread block per row) leaves CUs
//  idle when B*H*S_q << kMlaCus — the H=1 S_q=1 S_kv=8192 case runs on ONE
//  block: 6.5 GB/s = 0.12% of HBM. mla_fwd_split_for splits the S_kv window
//  across `split` blocks per row; each writes the per-split NORMALIZED
//  partial-softmax output (softmax_causal restricted to its visible keys:
//  Σ exp(score − row_max) · v_c / row_sum) and a per-split lse
//  (row_max + log(row_sum), −inf when its slice has no visible keys), and a
//  combine kernel folds the splits in FIXED ascending order:
//
//    out = Σ_sp exp(lse_sp − lse_g) · partial_out_sp ,
//    lse_g = m_g + log( Σ_sp exp(lse_sp − m_g) )
//
//  That sum is exactly the softmax numerator over lse_g, so no second
//  normalisation is needed and the result matches mla_fwd_cpu within the
//  same fp tolerance (the fixed ascending order keeps it deterministic).
// ---------------------------------------------------------------------------
__global__ void mla_fwd_split_kernel(const float* __restrict__ q,
                                     const float* __restrict__ k_c,
                                     const float* __restrict__ k_pe,
                                     const float* __restrict__ v_c,
                                     float* __restrict__ partial_out,
                                     float* __restrict__ partial_lse,
                                     int H, int S_q, int S_kv,
                                     int q_start, int kv_start,
                                     int lr, int rhd, float scale,
                                     int split, int kvs) {
  const int gi = blockIdx.x;          // one query row per block (decode)
  const int h = blockIdx.y;
  const int z = blockIdx.z;           // b * split + sp
  const int b = z / split;
  const int sp = z - b * split;
  const int lane = threadIdx.x;       // 64 threads: one wavefront

  const int Dq = lr + rhd;
  const int gqi = q_start + gi;

  // Stage this block's single query row in shared memory.
  extern __shared__ float sQ[];       // sQ[Dq]
  for (int c = lane; c < Dq; c += 64)
    sQ[c] = q[(((size_t)b * H + h) * S_q + gi) * Dq + c];
  __syncthreads();

  // Key slice [lo, hi) of the S_kv window owned by this split.
  const int lo = sp * kvs;
  const int hi = min(lo + kvs, S_kv);

  float o[kPerLane];                  // lane-local no-ropes output slice
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;          // running max (same on every lane)
  float row_sum = 0.0f;               // running denominator (same on every lane)

  const int nope_tiles = (lr + 63) / 64;
  for (int j = lo; j < hi; ++j) {
    const int gkj = kv_start + j;

    float partial = 0.0f;
    if (gkj <= gqi) {
      const float* kc = k_c + ((size_t)b * S_kv + j) * lr;
      for (int t = 0; t < nope_tiles; ++t) {
        const int idx = lane + t * 64;
        if (idx < lr) partial += sQ[idx] * kc[idx];
      }
      if (lane < rhd) {               // one rope element per lane (rhd <= 64)
        const float* kp = k_pe + ((size_t)b * S_kv + j) * rhd;
        partial += sQ[lr + lane] * kp[lane];
      }
    }
    float score = partial;
    for (int off = 32; off > 0; off >>= 1)   // warp-wide sum → broadcast
      score += __shfl_xor(score, off);
    // Masked keys must contribute ZERO weight, exactly as the CPU oracle
    // does: clamp their score to -inf.
    score = (gkj <= gqi) ? (score * scale) : -INFINITY;

    const float new_max = fmaxf(row_max, score);
    const float exp_old = (row_max == -INFINITY) ? 0.0f
                                                 : expf(row_max - new_max);
    const float exp_s = (score == -INFINITY) ? 0.0f : expf(score - new_max);
    row_sum = row_sum * exp_old + exp_s;
    for (int t = 0; t < kPerLane; ++t) o[t] *= exp_old;
    if (gkj <= gqi) {
      const float* vc = v_c + ((size_t)b * S_kv + j) * lr;
      for (int t = 0; t < nope_tiles; ++t) {
        const int idx = lane + t * 64;
        if (idx < lr) o[t] += exp_s * vc[idx];
      }
    }
    row_max = new_max;
  }

  // Per-split partial: normalized partial-softmax output slice (this split's
  // attention restricted to its visible keys) + lse = row_max + log(row_sum).
  // Normalizing here makes the combine exact:
  // out = Σ_sp exp(lse_sp − lse_g) · partial_out_sp.
  const size_t row = (size_t)((size_t)b * H + h) * S_q + gi;
  const float inv = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
  float* porow = partial_out + (row * split + sp) * lr;
  for (int t = 0; t < nope_tiles; ++t) {
    const int idx = lane + t * 64;
    if (idx < lr) porow[idx] = o[t] * inv;
  }
  if (lane == 0) {
    const float lse = (row_sum > 0.0f) ? row_max + logf(row_sum) : -INFINITY;
    partial_lse[row * split + sp] = lse;
  }
}

// Fixed-order (ascending split) combine over the split-K partials. One block
// per (b, h, query row); lane `d` owns the strided output slice
// {d, d+64, ...}, matching the split kernel's layout.
__global__ void mla_fwd_split_combine_kernel(
    const float* __restrict__ partial_out, const float* __restrict__ partial_lse,
    float* __restrict__ out, int S_q, int split, int lr) {
  const int gi = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int lane = threadIdx.x;
  const size_t row = (size_t)((size_t)b * gridDim.y + h) * S_q + gi;

  extern __shared__ float sLse[];     // sLse[split]
  for (int sp = lane; sp < split; sp += 64)
    sLse[sp] = partial_lse[row * split + sp];
  __syncthreads();

  const int nope_tiles = (lr + 63) / 64;
  float* orow = out + row * lr;

  // Global max over the per-split lses.
  float m_g = -INFINITY;
  for (int sp = 0; sp < split; ++sp) m_g = fmaxf(m_g, sLse[sp]);
  if (m_g == -INFINITY) {   // every key masked: zero row, as the CPU oracle
    for (int t = 0; t < nope_tiles; ++t) {
      const int idx = lane + t * 64;
      if (idx < lr) orow[idx] = 0.0f;
    }
    return;
  }

  // lse_g = m_g + log(Σ exp(lse_sp − m_g)), summed in fixed ascending order.
  float sum = 0.0f;
  for (int sp = 0; sp < split; ++sp) {
    if (sLse[sp] == -INFINITY) continue;  // split slice fully masked
    sum += expf(sLse[sp] - m_g);
  }
  const float lse_g = m_g + logf(sum);

  // out = Σ_sp exp(lse_sp − lse_g) · partial_out_sp (fixed ascending order;
  // already the softmax numerator over lse_g — no extra normalisation).
  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  for (int sp = 0; sp < split; ++sp) {
    const float w = (sLse[sp] == -INFINITY) ? 0.0f : expf(sLse[sp] - lse_g);
    if (w == 0.0f) continue;
    const float* p = partial_out + (row * split + sp) * lr;
    for (int t = 0; t < nope_tiles; ++t) {
      const int idx = lane + t * 64;
      if (idx < lr) o[t] += w * p[idx];
    }
  }
  for (int t = 0; t < nope_tiles; ++t) {
    const int idx = lane + t * 64;
    if (idx < lr) orow[idx] = o[t];
  }
}

// ---------------------------------------------------------------------------
//  Launchers
// ---------------------------------------------------------------------------
namespace {
template <int BQ>
void launch(const float* q, const float* k_c, const float* k_pe,
            const float* v_c, float* out, int B, int H, int S_q, int S_kv,
            int q_start, int kv_start, int lr, int rhd, float scale,
            hipStream_t st) {
  const int Dq = lr + rhd;
  const int shmem = BQ * Dq * sizeof(float);
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, B);
  mla_fwd_kernel<BQ><<<grid, block, shmem, st>>>(
      q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start, lr, rhd,
      scale);
}

// Device workspace for the split-K partials (issue #82): grow-only, shared
// across consecutive mla_fwd launches. Both kernels of one launch are
// enqueued under the same lock that (re)sizes the buffers, so a later
// resize's hipFree — which synchronises the device — can never race a
// not-yet-launched kernel against a freed buffer. The resize path runs only
// on first use / shape growth; steady-state decode just enqueues.
struct SplitWorkspace {
  std::mutex mu;
  float* partial_out = nullptr;
  float* partial_lse = nullptr;
  size_t cap_out = 0;
  size_t cap_lse = 0;
};
SplitWorkspace& split_ws() {
  static SplitWorkspace w;
  return w;
}

// Split-K decode launch: allocates (grow-only) the per-split partial buffers
// and enqueues the split kernel + fixed-order combine kernel on the caller
// stream (`st`; nullptr = legacy default stream 0). Returns false when the
// buffers cannot be sized; the caller then falls back to the plain
// single-block path.
//
// Capture safety (issue #45): the resize path hipFree()s, which
// device-synchronises — illegal mid-capture. While `st` is capturing we
// therefore never touch the workspace: if it is already sized (caller warmed
// the shape up before capture, the steady-state serving pattern) the split
// path enqueues on `st` exactly as eagerly; otherwise we return false and the
// caller falls back to the capture-safe non-split kernel.
bool launch_split(const float* q, const float* k_c, const float* k_pe,
                  const float* v_c, float* out, int B, int H, int S_q,
                  int S_kv, int q_start, int kv_start, int lr, int rhd,
                  float scale, int split, hipStream_t st) {
  const int Dq = lr + rhd;
  const int kvs = (S_kv + split - 1) / split;  // every split keeps >= 1 key
  const size_t rows = (size_t)B * H * S_q;
  const size_t n_out = rows * split * lr;
  const size_t n_lse = rows * split;

  auto& w = split_ws();
  std::lock_guard<std::mutex> lock(w.mu);
  hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
  const bool capturing =
      st != nullptr && hipStreamIsCapturing(st, &cap_status) == hipSuccess &&
      cap_status == hipStreamCaptureStatusActive;
  if (!capturing) {
    if (w.cap_out < n_out) {
      float* p = nullptr;
      if (hipMalloc(&p, n_out * sizeof(float)) == hipSuccess) {
        if (w.partial_out) hipFree(w.partial_out);
        w.partial_out = p;
        w.cap_out = n_out;
      }
    }
    if (w.cap_lse < n_lse) {
      float* p = nullptr;
      if (hipMalloc(&p, n_lse * sizeof(float)) == hipSuccess) {
        if (w.partial_lse) hipFree(w.partial_lse);
        w.partial_lse = p;
        w.cap_lse = n_lse;
      }
    }
  }
  if (w.cap_out < n_out || w.cap_lse < n_lse) return false;

  const dim3 block(64);
  const dim3 gridS(S_q, H, B * split);
  mla_fwd_split_kernel<<<gridS, block, Dq * sizeof(float), st>>>(
      q, k_c, k_pe, v_c, w.partial_out, w.partial_lse, H, S_q, S_kv, q_start,
      kv_start, lr, rhd, scale, split, kvs);
  const dim3 gridC(S_q, H, B);
  mla_fwd_split_combine_kernel<<<gridC, block, split * sizeof(float), st>>>(
      w.partial_out, w.partial_lse, out, S_q, split, lr);
  return true;
}
}  // namespace

void mla_fwd_with_tile(int B, int H, int S_q, int S_kv, int q_start,
                       int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                       float scale, const float* q, const float* k_c,
                       const float* k_pe, const float* v_c, float* out,
                       int bq, int bn_kv, void* stream) {
  (void)bn_kv;  // baseline streams one key at a time; reserved for a tiled
                // key prefetch in a follow-on optimisation.
  hipStream_t st = static_cast<hipStream_t>(stream);
  if (bq <= 1)
    launch<1>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale, st);
  else if (bq <= 2)
    launch<2>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale, st);
  else if (bq <= 4)
    launch<4>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale, st);
  else
    launch<8>(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
              kv_lora_rank, qk_rope_head_dim, scale, st);
}

void mla_fwd(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
             int kv_lora_rank, int qk_rope_head_dim, float scale,
             const float* q, const float* k_c, const float* k_pe,
             const float* v_c, float* out, void* stream) {
  hipStream_t st = static_cast<hipStream_t>(stream);
  // Decode shaped grids that cannot fill the CUs take the split-K path
  // (issue #82); mla_fwd_split_for returns 1 for prefill shapes and for
  // grids that already fill them, which then run exactly as before.
  const int split = ::vkernels::kernels::mla_fwd_split_for(B, H, S_q, S_kv);
  if (split > 1 &&
      launch_split(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
                   kv_lora_rank, qk_rope_head_dim, scale, split, st)) {
    return;
  }
  int bq = 0, bn = 0, th = 0;
  ::vkernels::kernels::mla_config_for(S_q, kv_lora_rank, qk_rope_head_dim,
                                      &bq, &bn, &th);
  mla_fwd_with_tile(B, H, S_q, S_kv, q_start, kv_start, kv_lora_rank,
                    qk_rope_head_dim, scale, q, k_c, k_pe, v_c, out, bq, bn,
                    stream);
}

void mla_fwd_with_split(int B, int H, int S_q, int S_kv, int q_start,
                        int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                        float scale, const float* q, const float* k_c,
                        const float* k_pe, const float* v_c, float* out,
                        int split, void* stream) {
  hipStream_t st = static_cast<hipStream_t>(stream);
  if (split > 1 &&
      launch_split(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
                   kv_lora_rank, qk_rope_head_dim, scale, split, st)) {
    return;
  }
  int bq = 0, bn = 0, th = 0;
  ::vkernels::kernels::mla_config_for(S_q, kv_lora_rank, qk_rope_head_dim,
                                      &bq, &bn, &th);
  mla_fwd_with_tile(B, H, S_q, S_kv, q_start, kv_start, kv_lora_rank,
                    qk_rope_head_dim, scale, q, k_c, k_pe, v_c, out, bq, bn,
                    stream);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
