// vkernels/kernels/dsa.hip
//
// DeepseekSparseAttn (DSA) sparse-MLA forward (gfx942 / MI300A), issue #51.
// Online-softmax FlashAttention-style kernel over the `topk` INDEXER-SELECTED
// keys (see dsa.hpp): each block owns `BQ` query rows for one head and
// streams its selected keys one at a time, accumulating the combined output in
// fp32 with a numerically-stable base-2 online softmax.
//
//   score[i,h,k] = sm_scale . ( q_main[i,h] . k_main[idx]
//                              + (tail_dim>0) q_tail[i,h] . k_tail[idx] )
//   out[i,h]     = Sum_k softmax(score)[i,h,k] . v[idx]            (d_v wide)
//   lse[i,h]     = log2( Sum_k 2^score[i,h,k] )   (base-2, when return_lse)
//
// `sm_scale` already folds log2(e), so 2^score is the natural-exp weight and
// the base-2 LSE is the standard FlashAttention log-sum-exp.
//
// Storage. Following the established gfx942 bf16 convention (gemm_bf16.hip,
// moe_device.hip), bf16 is carried as raw uint16 IEEE-754 bit patterns and
// converted to/from fp32 with bf16_to_f32 (zero-extend of the top 16 bits)
// and f32bits_to_bf16 (round-to-nearest-even, the SAME helper the CPU oracle
// uses). The kernel is therefore bf16-tolerant against the oracle exactly as
// gemm_bf16 is.
//
// Tile & data layout. One wavefront (64 lanes) owns exactly one query row, so
// a block of `BQ` warps computes BQ rows. The head dimension (dim + tail_dim)
// is laid out STRIDED across the 64 lanes: lane `d` owns head elements
// {d, d+64, d+128, ...}. At the GLM-5.3 config (dim=256, tail_dim=0 -> W=256)
// that is 4 elements per lane and 4 output accumulators per lane; at the
// DeepSeek-V3 config (dim=576, tail_dim=64 -> W=640) it is 10 main + 1 tail +
// 8 output — well inside the register file, and exact (no bounds-checks fire).
// The per-key score is a single scalar, formed by a warp-shuffle reduction of
// the lane-local partial dot and therefore broadcast to every lane, which
// keeps the per-row online state (max, sum, output slice) lock-step across
// the warp with no LDS synchronisation on the hot path.
//
// tail_dim == 0 (GLM-5.3-Flash): the rope-tail dot loop runs zero iterations
// (the tile count `tail_tiles` is 0), so NO zero-size GEMM is ever issued —
// the exact case the tilelang code path cannot compile. The guard is a
// runtime branch on `tail_dim`, not a template specialization, so one binary
// serves both shapes.
//
// Memory. Each selected key's kv vector is read exactly once from global
// memory (bf16 -> fp32 on load); the BQ query rows are staged in LDS once per
// block (bf16 -> fp32). This is the simple, correct baseline (one key at a
// time, no double-buffered key tile); a tiled key prefetch is a follow-on
// optimisation, not required for correctness against the oracle. The combined
// result is grouping-independent, so each block streams all of its `topk`
// keys directly (no separate partial_o / partial_lse + combine kernel).
//
// Two-implementation model:
//   dsa.cpp  -- CPU reference (oracle), always compiled, two-pass softmax.
//   dsa.hip  -- this file, compiled with VKERNELS_HAS_HIP, online softmax.
//               Cross-checked against the oracle by test_dsa_correct.hip.
#include <cstdint>

#include "vkernels/kernels/dsa.hpp"

#if VKERNELS_HAS_HIP
#  include "vkernels/kernels/moe_device.hip"  // f32bits_to_bf16 (shared RNE)
#  include <hip/hip_runtime.h>

namespace vkernels::kernels::hip {

// Largest supported head width (dim + tail_dim) -> 1024/64 = 16 per lane.
static constexpr int kMaxW = 1024;
static constexpr int kPerLane = (kMaxW + 63) / 64;  // head elements per lane

// bf16 -> float (top 16 bits of fp32, low 17 mantissa bits zero). Identical
// to the helper in gemm_bf16.hip / moe_aux.hip and to the CPU oracle.
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  float f;
  uint32_t u = static_cast<uint32_t>(b) << 16;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}
// float -> bf16 (RNE), via the shared f32bits_to_bf16 (moe_device.hip).
__device__ __forceinline__ uint16_t f2bf(float v) {
  uint32_t b;
  __builtin_memcpy(&b, &v, sizeof(float));
  return f32bits_to_bf16(b);
}

template <int BQ>
__global__ void dsa_sparse_fwd_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kv,
    const int32_t* __restrict__ indices,
    uint16_t* __restrict__ out, float* __restrict__ lse_out,
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk, int kv_group,
    float sm_scale, bool return_lse) {
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;            // kv_group == 1: one shared head_kv
  const int tid = threadIdx.x;
  const int warp = tid >> 6;           // which query row within the tile
  const int lane = tid & 63;           // strided head index this lane owns
  const int gi = q_tile * BQ + warp;   // global query row
  if (gi >= S_q) return;

  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;
  const int main_tiles = (dim + 63) / 64;
  const int tail_tiles = (tail_dim + 63) / 64;  // 0 when tail_dim == 0
  const int v_tiles = (d_v + 63) / 64;

  // --- stage the BQ query rows in LDS once (bf16 -> fp32, per-block) ---
  extern __shared__ float sQ[];       // sQ[BQ][W]
  for (int idx = tid; idx < BQ * W; idx += BQ * 64) {
    const int r = idx / W, c = idx % W;
    const int gr = q_tile * BQ + r;
    sQ[idx] = (gr < S_q) ? bf16_to_f32(
                  q[((size_t)gr * H + h) * W + c])
                         : 0.0f;
  }
  __syncthreads();

  const float* qrow = &sQ[warp * W];  // this warp's query row (fp32)
  const int32_t* idx_i =
      indices + (size_t)gi * kv_group * topk;  // kv_group == 1 -> gi*topk

  // Lane-local slices of the output accumulator (strided by 64 over d_v).
  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;          // running max (same on every lane)
  float row_sum = 0.0f;               // running denominator (same on every lane)

  for (int k = 0; k < topk; ++k) {
    const int32_t idx = idx_i[k];
    const bool valid = (idx >= 0) && (idx < S_kv);

    // --- lane-local partial score (main + tail), then warp-reduce ---
    float partial = 0.0f;
    if (valid) {
      const uint16_t* kv_p = kv + (size_t)idx * W;  // kv_group==1
      for (int t = 0; t < main_tiles; ++t) {
        const int d = lane + t * 64;
        if (d < dim) partial += qrow[d] * bf16_to_f32(kv_p[d]);
      }
      for (int t = 0; t < tail_tiles; ++t) {  // 0 iters when tail_dim == 0
        const int d = lane + t * 64;
        if (d < tail_dim)
          partial += qrow[dim + d] * bf16_to_f32(kv_p[dim + d]);
      }
    }
    float score = partial;
    for (int off = 32; off > 0; off >>= 1)  // warp-wide sum -> broadcast
      score += __shfl_xor(score, off);
    // Masked keys must contribute ZERO weight, so clamp their score to -inf
    // exactly as the CPU oracle does (else exp2(0-max)=1).
    score = valid ? (score * sm_scale) : -INFINITY;

    // --- online softmax update (lock-step; identical on every lane) ---
    const float new_max = fmaxf(row_max, score);
    const float exp_old = (row_max == -INFINITY) ? 0.0f : exp2f(row_max - new_max);
    const float exp_s = (score == -INFINITY) ? 0.0f : exp2f(score - new_max);
    row_sum = row_sum * exp_old + exp_s;
    for (int t = 0; t < v_tiles; ++t) o[t] *= exp_old;
    if (valid) {
      const uint16_t* vp = kv + (size_t)idx * W;  // v = kv[idx][0:d_v]
      for (int t = 0; t < v_tiles; ++t) {
        const int d = lane + t * 64;
        if (d < d_v) o[t] += exp_s * bf16_to_f32(vp[d]);
      }
    }
    row_max = new_max;
  }

  // --- normalise and write the lane-local output slice (fp32 -> bf16) ---
  const float inv = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
  uint16_t* orow = out + (((size_t)gi * H + h) * d_v);
  for (int t = 0; t < v_tiles; ++t) {
    const int d = lane + t * 64;
    if (d < d_v) orow[d] = f2bf(o[t] * inv);
  }
  if (return_lse) {
    const float l = (row_sum > 0.0f) ? (row_max + log2f(row_sum)) : -INFINITY;
    if (lane == 0) lse_out[(size_t)gi * H + h] = l;
  }
}

// ---------------------------------------------------------------------------
//  Launchers
// ---------------------------------------------------------------------------
namespace {
template <int BQ>
void launch(const void* q, const void* kv, const void* indices, void* out,
            void* lse, int S_q, int S_kv, int H, int dim, int tail_dim,
            int topk, int kv_group, float sm_scale, bool return_lse) {
  const int W = dim + tail_dim;
  const int shmem = BQ * W * sizeof(float);
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, 1);
  dsa_sparse_fwd_kernel<BQ><<<grid, block, shmem, 0>>>(
      reinterpret_cast<const uint16_t*>(q),
      reinterpret_cast<const uint16_t*>(kv),
      reinterpret_cast<const int32_t*>(indices),
      reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(lse),
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, sm_scale, return_lse);
}
}  // namespace

void dsa_sparse_fwd_with_tile(int S_q, int S_kv, int H, int dim, int tail_dim,
                              int topk, int kv_group, float sm_scale,
                              bool return_lse,
                              const void* q, const void* kv,
                              const void* indices, void* out, void* lse,
                              int bq, int block_I, int inner_iter, int bn_kv) {
  (void)block_I;     // baseline streams one key at a time; reserved for a
  (void)inner_iter;  // future tiled-key prefetch (group-then-combine merge).
  (void)bn_kv;
  if (bq <= 1)
    launch<1>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse);
  else if (bq <= 2)
    launch<2>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse);
  else if (bq <= 4)
    launch<4>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse);
  else
    launch<8>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse);
}

void dsa_sparse_fwd(int S_q, int S_kv, int H, int dim, int tail_dim,
                    int topk, int kv_group, int block_I, int inner_iter,
                    float sm_scale, bool return_lse,
                    const void* q, const void* kv, const void* indices,
                    void* out, void* lse) {
  int bq = 0, th = 0, bi = 0, ii = 0;
  ::vkernels::kernels::dsa_config_for(S_q, H, dim, topk, &bq, &th, &bi, &ii);
  (void)th;
  (void)block_I;
  (void)inner_iter;
  dsa_sparse_fwd_with_tile(S_q, S_kv, H, dim, tail_dim, topk, kv_group,
                           sm_scale, return_lse, q, kv, indices, out, lse,
                           bq, bi, ii, 0);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
