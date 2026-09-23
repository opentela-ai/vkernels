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
// (one key at a time, no double-buffered key tile); serving prefill uses
// the fused-head kernels below (issue #151).
//
// Two-implementation model:
//   mla.cpp  -- CPU reference (oracle), always compiled, two-pass softmax.
//   mla.hip  -- this file, compiled with VKERNELS_HAS_HIP, online softmax.
//               Cross-checked against the oracle by test_mla_correct.hip.
#include <cstdint>
#include <climits>
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
//
//  Issue #151 adds the BQ>1 PREFILL variant: a block owns BQ query rows (one
//  warp per row, as in the plain kernel) over one head, each warp covering
//  the block's key slice for its row. This keeps the causal mask per row and
//  divides the KV traffic by BQ versus one-row blocks; the combine kernel is
//  unchanged (it is per-row and split-count agnostic). BQ=1 is the decode
//  path from #82 (grid.x = S_q, one row per block).
// ---------------------------------------------------------------------------
template <int BQ>
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
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int z = blockIdx.z;           // b * split + sp
  const int b = z / split;
  const int sp = z - b * split;
  const int warp = threadIdx.x >> 6;  // which query row within the tile
  const int lane = threadIdx.x & 63;  // strided head index this lane owns
  const int gi = q_tile * BQ + warp;
  // Rows past the end of S_q (last tile) keep computing — uniform control
  // flow around __syncthreads — but never store their partials.
  const bool active = gi < S_q;
  const int gqi = active ? q_start + gi : INT_MAX;
  const int Dq = lr + rhd;

  // Stage this block's BQ query rows in shared memory.
  extern __shared__ float sQ[];       // sQ[BQ][Dq]
  for (int idx = threadIdx.x; idx < BQ * Dq; idx += BQ * 64) {
    const int r = idx / Dq, c = idx - r * Dq;
    const int gr = q_tile * BQ + r;
    sQ[idx] = (gr < S_q)
                  ? q[(((size_t)b * H + h) * S_q + gr) * Dq + c]
                  : 0.0f;
  }
  __syncthreads();
  const float* qrow = &sQ[warp * Dq];  // this warp's query row

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
  if (!active) return;  // padding row of the last query tile: no partial
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
//  Fused-head prefill (issue #151)
//
//  The plain prefill grid launches (S_q/BQ)·H·B blocks and every block
//  streams the full shared KV latent one key at a time, so k_c/k_pe/v_c are
//  re-read once per (query tile, head) — ~36 GB vs ~290 MB ideal at the
//  H=128 S_q=512 K3 shape — even though the latent is head-SHARED.
//
//  This kernel batches BH heads into one block alongside its BQ query rows:
//  a block owns BQ·BH (row, head) pairs, one wavefront each, all streaming
//  the SAME key sequence. Because k_c/k_pe/v_c are identical for every head,
//  the block-wide key reads are absorbed by L1 and the KV latent is read
//  from DRAM once per (query tile, head group) instead of once per
//  (query tile, head) — a BH-fold traffic cut — while each wavefront keeps
//  the exact plain-kernel inner loop (which allocates cleanly: no LDS key
//  staging, no serial head loop; a BN-tiled LDS variant of this kernel
//  measured 57 spilled SGPRs on gfx942 and ran ~50x slower per key).
//
//  Grid: (ceil(S_q/BQ), ceil(H/BH), B) blocks of BQ·BH wavefronts; warp w
//  owns row q_tile·BQ + w/BH and head hgroup·BH + w%BH. Heads per group are
//  capped at 16/BQ so the block stays <= 1024 threads (16 wavefronts).
// ---------------------------------------------------------------------------
template <int BQ, int BH>
__global__ void mla_fwd_headfused_kernel(const float* __restrict__ q,
                                         const float* __restrict__ k_c,
                                         const float* __restrict__ k_pe,
                                         const float* __restrict__ v_c,
                                         float* __restrict__ out,
                                         int B, int H, int S_q, int S_kv,
                                         int q_start, int kv_start,
                                         int lr, int rhd, float scale) {
  const int q_tile = blockIdx.x;
  const int hgroup = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 63;          // strided head index this lane owns
  const int warp = tid >> 6;          // (row, head) pair within the block
  const int r = warp / BH;            // query row within the tile
  const int hh = warp - r * BH;       // head within the group
  const int h = hgroup * BH + hh;
  const int gi = q_tile * BQ + r;
  // Warp-uniform guards: whole wavefronts may exit before any barrier (this
  // kernel has none — every wavefront is independent).
  if (gi >= S_q || h >= H) return;

  const int Dq = lr + rhd;
  const int gqi = q_start + gi;

  // This warp's query row slice (lane-strided), loaded once. The runtime
  // nope_tiles bound keeps o[]/qr[] allocatable exactly as in the plain
  // kernel (full unrolling spilled SGPRs).
  float qr[kPerLane];
  float qrope = 0.0f;
  {
    const float* qrow = q + (((size_t)b * H + h) * S_q + gi) * Dq;
    const int nope_tiles = (lr + 63) / 64;
    for (int t = 0; t < nope_tiles; ++t) {
      const int idx = lane + t * 64;
      qr[t] = (idx < lr) ? qrow[idx] : 0.0f;
    }
    if (lane < rhd) qrope = qrow[lr + lane];  // one rope element per lane
  }

  // Lane-local slices of the no-ropes output accumulator (strided by 64).
  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;          // running max (same on every lane)
  float row_sum = 0.0f;               // running denominator (same on every lane)

  const int nope_tiles = (lr + 63) / 64;
  for (int j = 0; j < S_kv; ++j) {
    const int gkj = kv_start + j;

    // --- lane-local partial score (no-ropes + rope), then warp-reduce ---
    // k_c/k_pe rows are head-shared: all BH·BQ wavefronts of the block read
    // the same rows, so L1 serves the duplicates.
    float partial = 0.0f;
    if (gkj <= gqi) {
      const float* kc = k_c + ((size_t)b * S_kv + j) * lr;
      for (int t = 0; t < nope_tiles; ++t) {
        const int idx = lane + t * 64;
        if (idx < lr) partial += qr[t] * kc[idx];
      }
      if (lane < rhd) {               // one rope element per lane (rhd <= 64)
        const float* kp = k_pe + ((size_t)b * S_kv + j) * rhd;
        partial += qrope * kp[lane];
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
//  Fused-head prefill, wide-rank pipeline (issue #151): same block mapping
//  as mla_fwd_headfused_kernel, specialised for ranks where lr = LP·64
//  (Kimi-K3: lr=512, LP=8) and rhd <= 64. Two latency tricks:
//
//  * BN-key software pipeline: a tile's k_c/k_pe/v_c loads are issued up
//    front into registers, so the per-key chain (warp reduce → expf →
//    accumulate) runs ALU-only while the next tile's loads are in flight.
//    The unpipelined loop sits at ~1300 cycles/key on MI300A (latency, not
//    bandwidth); the pipeline amortises load latency over BN keys.
//  * Consecutive-lane ownership: lane l owns floats [l·LP, (l+1)·LP) of
//    every lr-row, so each row load is LP/4 float4s (2 instead of 8 scalar
//    loads at LP=8) and every address is 32 B aligned (Dq = lr + rhd keeps
//    rows 16 B aligned; q/out rows are lr-sized and likewise aligned).
//
//  The online softmax also takes a branch-free fast path: keys that do not
//  advance the running max skip the accumulator rescale and one expf.
//  BN=2 keeps the register tile under the occupancy cliff: a workgroup's
//  wavefronts must fit the CU's 256 KB register file.
// ---------------------------------------------------------------------------
template <int BQ, int BH, int LP>
__global__ void mla_fwd_headfused_wide_kernel(
    const float* __restrict__ q, const float* __restrict__ k_c,
    const float* __restrict__ k_pe, const float* __restrict__ v_c,
    float* __restrict__ out, int B, int H, int S_q, int S_kv,
    int q_start, int kv_start, int rhd, float scale) {
  constexpr int LR = LP * 64;          // launcher guarantees lr == LR
  constexpr int BN = 2;                // keys in flight per wavefront
  const int q_tile = blockIdx.x;
  const int hgroup = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int r = warp / BH;
  const int hh = warp - r * BH;
  const int h = hgroup * BH + hh;
  const int gi = q_tile * BQ + r;
  if (gi >= S_q || h >= H) return;
  const int gqi = q_start + gi;

  // This wavefront's query row slice (lane-consecutive float4s).
  float qr[LP];
  float qrope = 0.0f;
  {
    const float* qrow = q + (((size_t)b * H + h) * S_q + gi) * (LR + rhd);
    const float4* q4 = reinterpret_cast<const float4*>(qrow + lane * LP);
    #pragma unroll
    for (int f = 0; f < LP / 4; ++f) {
      const float4 v4 = q4[f];
      qr[f * 4 + 0] = v4.x; qr[f * 4 + 1] = v4.y;
      qr[f * 4 + 2] = v4.z; qr[f * 4 + 3] = v4.w;
    }
    if (lane < rhd) qrope = qrow[LR + lane];
  }

  float o[LP];
  #pragma unroll
  for (int t = 0; t < LP; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;
  float row_sum = 0.0f;

  for (int j0 = 0; j0 < S_kv; j0 += BN) {
    const int n = (S_kv - j0 < BN) ? S_kv - j0 : BN;
    // Stage 1: issue the tile's loads (independent → all in flight).
    float kcr[BN][LP], vcr[BN][LP], kpr[BN];
    #pragma unroll
    for (int n2 = 0; n2 < BN; ++n2) {
      if (n2 < n) {
        const int j = j0 + n2;
        const float* kc = k_c + ((size_t)b * S_kv + j) * LR;
        const float* vc = v_c + ((size_t)b * S_kv + j) * LR;
        const float4* kc4 = reinterpret_cast<const float4*>(kc + lane * LP);
        const float4* vc4 = reinterpret_cast<const float4*>(vc + lane * LP);
        #pragma unroll
        for (int f = 0; f < LP / 4; ++f) {
          const float4 k4 = kc4[f];
          kcr[n2][f * 4 + 0] = k4.x; kcr[n2][f * 4 + 1] = k4.y;
          kcr[n2][f * 4 + 2] = k4.z; kcr[n2][f * 4 + 3] = k4.w;
          const float4 v4 = vc4[f];
          vcr[n2][f * 4 + 0] = v4.x; vcr[n2][f * 4 + 1] = v4.y;
          vcr[n2][f * 4 + 2] = v4.z; vcr[n2][f * 4 + 3] = v4.w;
        }
        kpr[n2] = (lane < rhd)
                      ? k_pe[((size_t)b * S_kv + j) * rhd + lane]
                      : 0.0f;
      }
    }
    // Stage 2: ALU-only online-softmax merge over the staged keys. Keys
    // that do not advance row_max skip the accumulator rescale entirely.
    #pragma unroll
    for (int n2 = 0; n2 < BN; ++n2) {
      const int gkj = kv_start + j0 + n2;
      if (n2 >= n) break;
      float score;
      if (gkj <= gqi) {
        float partial = qrope * kpr[n2];
        #pragma unroll
        for (int t = 0; t < LP; ++t) partial += qr[t] * kcr[n2][t];
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1)
          partial += __shfl_xor(partial, off);
        score = partial * scale;
      } else {
        score = -INFINITY;             // masked key: zero weight, as oracle
      }
      if (score > row_max) {
        // New max: one expf (the new key's weight is exactly 1).
        const float e = expf(row_max - score);   // -inf → 0 (first key)
        row_sum = row_sum * e + 1.0f;
        #pragma unroll
        for (int t = 0; t < LP; ++t) o[t] = o[t] * e + vcr[n2][t];
        row_max = score;
      } else {
        const float w =
            (score == -INFINITY) ? 0.0f : expf(score - row_max);
        row_sum += w;
        #pragma unroll
        for (int t = 0; t < LP; ++t) o[t] += w * vcr[n2][t];
      }
    }
  }

  const float inv = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
  float* orow = out + (((size_t)b * H + h) * S_q + gi) * LR;
  if (LP % 4 == 0) {
    float4* o4 = reinterpret_cast<float4*>(orow + lane * LP);
    #pragma unroll
    for (int f = 0; f < LP / 4; ++f) {
      o4[f] = make_float4(o[f * 4 + 0] * inv, o[f * 4 + 1] * inv,
                          o[f * 4 + 2] * inv, o[f * 4 + 3] * inv);
    }
  } else {
    #pragma unroll
    for (int t = 0; t < LP; ++t) orow[lane * LP + t] = o[t] * inv;
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

  // Decode (S_q <= 8): one row per block, BQ=1 (issue #82, unchanged).
  // Chunked / short prefill (S_q > 8): BQ=4 rows per block (issue #151) —
  // BQ× fewer full-KV passes than one-row blocks, same combine.
  const int bq = (S_q > 8) ? 4 : 1;
  const dim3 block(bq * 64);
  const dim3 gridS((S_q + bq - 1) / bq, H, B * split);
  if (bq == 4) {
    mla_fwd_split_kernel<4><<<gridS, block, 4 * Dq * sizeof(float), st>>>(
        q, k_c, k_pe, v_c, w.partial_out, w.partial_lse, H, S_q, S_kv,
        q_start, kv_start, lr, rhd, scale, split, kvs);
  } else {
    mla_fwd_split_kernel<1><<<gridS, block, Dq * sizeof(float), st>>>(
        q, k_c, k_pe, v_c, w.partial_out, w.partial_lse, H, S_q, S_kv,
        q_start, kv_start, lr, rhd, scale, split, kvs);
  }
  // A failed split-kernel launch would leave the grow-only workspace with
  // stale contents that the combine below would fold into `out` -- the
  // wrong-but-plausible failure mode. Report failure so the caller falls
  // back to the plain (unsplit) path instead.
  if (hipGetLastError() != hipSuccess) return false;
  const dim3 gridC(S_q, H, B);
  // The combine kernel assumes 64 threads (its strided slices and the sLse
  // staging loop are hard-wired to 64); keep it at 64 regardless of BQ.
  mla_fwd_split_combine_kernel<<<gridC, 64, split * sizeof(float), st>>>(
      w.partial_out, w.partial_lse, out, S_q, split, lr);
  return true;
}

// Fused-head prefill launch (issue #151). bq selects the query tile (rows
// per block); the heads-per-block group follows as min(H, 16/BQ) rounded
// down to a power of two (<= 1024 threads per block).
void launch_fused(const float* q, const float* k_c, const float* k_pe,
                  const float* v_c, float* out, int B, int H, int S_q,
                  int S_kv, int q_start, int kv_start, int lr, int rhd,
                  float scale, int bq, int bh, hipStream_t st) {
  const int BQ = (bq < 2) ? 1 : (bq < 4) ? 2 : (bq < 8) ? 4 : 8;

  // Wide-rank pipeline (K3: lr=512 = 8 lane slices, rhd <= 64): the BN-key
  // register tile costs ~65 VGPRs per wavefront, so cap the block at 12
  // wavefronts (12 · 65 · 64 · 4 B ≈ 200 KB of the CU's 256 KB file).
  if (lr == 512 && rhd <= 64) {
    int BHw = 12 / BQ;
    while ((BHw & (BHw - 1)) != 0) BHw &= BHw - 1;  // power of two (case list)
    if (BHw > bh) BHw = bh;
    if (BHw < 1) BHw = 1;
    const dim3 block(BQ * BHw * 64);
    const dim3 grid((S_q + BQ - 1) / BQ, (H + BHw - 1) / BHw, B);
#define VK_MLA_WIDE_CASE(BQ_, BH_)                                         \
  case ((BQ_) * 32 + (BH_)):                                              \
    mla_fwd_headfused_wide_kernel<BQ_, BH_, 8><<<grid, block, 0, st>>>(      \
        q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start, rhd,   \
        scale);                                                            \
    return
    switch (BQ * 32 + BHw) {
      VK_MLA_WIDE_CASE(1, 1);
      VK_MLA_WIDE_CASE(1, 2);
      VK_MLA_WIDE_CASE(1, 4);
      VK_MLA_WIDE_CASE(1, 8);
      VK_MLA_WIDE_CASE(2, 1);
      VK_MLA_WIDE_CASE(2, 2);
      VK_MLA_WIDE_CASE(2, 4);
      VK_MLA_WIDE_CASE(4, 1);
      VK_MLA_WIDE_CASE(4, 2);
      VK_MLA_WIDE_CASE(8, 1);
      default: break;
    }
#undef VK_MLA_WIDE_CASE
  }

  const dim3 block(BQ * bh * 64);
  const dim3 grid((S_q + BQ - 1) / BQ, (H + bh - 1) / bh, B);

#define VK_MLA_FUSED_CASE(BQ_, BH_)                                        \
  case ((BQ_) * 32 + (BH_)):                                              \
    mla_fwd_headfused_kernel<BQ_, BH_><<<grid, block, 0, st>>>(              \
        q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start, lr,    \
        rhd, scale);                                                       \
    return
  switch (BQ * 32 + bh) {
    VK_MLA_FUSED_CASE(1, 1);
    VK_MLA_FUSED_CASE(1, 2);
    VK_MLA_FUSED_CASE(1, 4);
    VK_MLA_FUSED_CASE(1, 8);
    VK_MLA_FUSED_CASE(1, 16);
    VK_MLA_FUSED_CASE(2, 1);
    VK_MLA_FUSED_CASE(2, 2);
    VK_MLA_FUSED_CASE(2, 4);
    VK_MLA_FUSED_CASE(2, 8);
    VK_MLA_FUSED_CASE(4, 1);
    VK_MLA_FUSED_CASE(4, 2);
    VK_MLA_FUSED_CASE(4, 4);
    VK_MLA_FUSED_CASE(8, 1);
    VK_MLA_FUSED_CASE(8, 2);
    default: break;
  }
#undef VK_MLA_FUSED_CASE
}
}  // namespace

void mla_fwd_with_tile(int B, int H, int S_q, int S_kv, int q_start,
                       int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                       float scale, const float* q, const float* k_c,
                       const float* k_pe, const float* v_c, float* out,
                       int bq, int bn_kv, void* stream) {
  (void)bn_kv;  // decode streams one key at a time; the split-K path (#82)
                // handles occupancy for the grids that need it
  hipStream_t st = static_cast<hipStream_t>(stream);
  if (S_q > 8) {
    // Prefill (issue #151): fused-head kernel — bq selects the query tile;
    // min(H, 16/BQ) heads ride along per block (power-of-two) so the shared
    // KV latent is re-read once per head group instead of once per head.
    const int bh = ::vkernels::kernels::mla_prefill_heads_per_block(bq, H);
    launch_fused(q, k_c, k_pe, v_c, out, B, H, S_q, S_kv, q_start, kv_start,
                 kv_lora_rank, qk_rope_head_dim, scale, bq, bh, st);
    return;
  }
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
  // Grids that cannot fill the CUs take the split path (issues #82/#151:
  // decode and chunked prefill); everything else runs the plain decode
  // kernel (S_q <= 8) or the fused-head tiled prefill kernel (S_q > 8).
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
