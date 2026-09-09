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
#include <cstdio>

#include "vkernels/kernels/dsa.hpp"
#include "vkernels/util/error.hpp"  // VK_EXPECTS (host: throws std::invalid_argument)

#if VKERNELS_HAS_HIP
#  include "vkernels/kernels/moe_device.hip"  // shared bf16<->f32 + fp4 helpers
#  include "vkernels/kernels/dsa_topk_device.cuh"  // shared scalar dsa_topk_logits kernels
#  include <hip/hip_runtime.h>

namespace vkernels::kernels::hip {

// MFMA helpers for the bf16 gated-logit kernel below. Mirrors the verified
// gemm_bf16.hip declarations verbatim; kept in an anonymous namespace so the
// identical typedef/constant here never collides with gemm_bf16.hip's copy
// (no linker-visible symbol either way, but this is the clean form).
namespace {
typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v4s __attribute__((ext_vector_type(4)));
static constexpr int kMfmaK = 16;   // one bf16 MFMA reduces K=16
static constexpr int kBK     = 64;  // K-tile (fixed); D=128 -> 2 K-tiles
}  // namespace

// Largest supported head width (dim + tail_dim) -> 1024/64 = 16 per lane.
static constexpr int kMaxW = 1024;
static constexpr int kPerLane = (kMaxW + 63) / 64;  // head elements per lane


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
            int topk, int kv_group, float sm_scale, bool return_lse,
            hipStream_t stream = nullptr) {  // #69: caller stream
  const int W = dim + tail_dim;
  const int shmem = BQ * W * sizeof(float);
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, 1);
  dsa_sparse_fwd_kernel<BQ><<<grid, block, shmem, stream>>>(
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
                              int bq, int block_I, int inner_iter, int bn_kv,
                              void* stream) {
  // Host-side preconditions (mirror dsa_sparse_fwd_cpu, issue #57): a
  // misconfigured caller must raise a named error, not a silent hang or
  // out-of-bounds launch. The two-pass softmax writes a row into `out`
  // then reads it back, so `out` must not alias any input or `lse`.
  const int d_v = dim - tail_dim;
  VK_EXPECTS(dim > 0 && tail_dim >= 0 && d_v > 0,
             "dim > tail_dim >= 0 required");
  VK_EXPECTS(topk > 0, "topk must be positive");
  VK_EXPECTS(kv_group == 1, "kv_group must be 1 (single shared head_kv)");
  VK_EXPECTS(block_I > 0 && inner_iter > 0,
             "block_I and inner_iter must be positive");
  VK_EXPECTS(topk % (block_I * inner_iter) == 0,
             "topk must be a multiple of block_I*inner_iter");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 || q != nullptr,
             "q must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || S_kv == 0 || kv != nullptr,
             "kv must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 || indices != nullptr,
             "indices must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 || out != nullptr,
             "out must not be null");
  VK_EXPECTS(!return_lse || lse != nullptr, "lse must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 ||
                 (out != q && out != kv && out != indices &&
                  (!return_lse || out != lse)),
             "out must not alias q, kv, indices, or lse");
  (void)block_I;     // baseline streams one key at a time; reserved for a
  (void)inner_iter;  // future tiled-key prefetch (group-then-combine merge).
  (void)bn_kv;
  if (bq <= 1)
    launch<1>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse, static_cast<hipStream_t>(stream));
  else if (bq <= 2)
    launch<2>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse, static_cast<hipStream_t>(stream));
  else if (bq <= 4)
    launch<4>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse, static_cast<hipStream_t>(stream));
  else
    launch<8>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse, static_cast<hipStream_t>(stream));
}

void dsa_sparse_fwd(int S_q, int S_kv, int H, int dim, int tail_dim,
                    int topk, int kv_group, int block_I, int inner_iter,
                    float sm_scale, bool return_lse,
                    const void* q, const void* kv, const void* indices,
                    void* out, void* lse, void* stream) {
  // Validate the caller's block_I/inner_iter (issue #57) even though the
  // launch tile comes from dsa_config_for (below): a misconfigured caller
  // gets a named error instead of a silently-wrong launch.
  const int d_v0 = dim - tail_dim;
  VK_EXPECTS(dim > 0 && tail_dim >= 0 && d_v0 > 0,
             "dim > tail_dim >= 0 required");
  VK_EXPECTS(topk > 0, "topk must be positive");
  VK_EXPECTS(kv_group == 1, "kv_group must be 1 (single shared head_kv)");
  VK_EXPECTS(block_I > 0 && inner_iter > 0,
             "block_I and inner_iter must be positive");
  VK_EXPECTS(topk % (block_I * inner_iter) == 0,
             "topk must be a multiple of block_I*inner_iter");
  int bq = 0, th = 0, bi = 0, ii = 0;
  ::vkernels::kernels::dsa_config_for(S_q, H, dim, topk, &bq, &th, &bi, &ii);
  (void)th;
  (void)block_I;
  (void)inner_iter;
  dsa_sparse_fwd_with_tile(S_q, S_kv, H, dim, tail_dim, topk, kv_group,
                           sm_scale, return_lse, q, kv, indices, out, lse,
                           bq, bi, ii, 0, stream);
}

// ---------------------------------------------------------------------------
//  Split-key forward (decode occupancy fix): the plain forward's grid is
//  (ceil(S_q/BQ), H) -- 64 blocks on MI300A's 228 CUs at the GLM-5.3-Flash
//  decode shape, each streaming its `topk` keys serially (measured 0.4% of
//  the HBM roof; docs/performance/dsa/gfx942.md). The partial kernel below
//  adds a blockIdx.z split of the per-query index range: each block runs the
//  SAME online softmax over its [k0, k1) chunk but writes its NORMALIZED
//  fp32 partial output (numerator / local row-sum -- the flash-decoding
//  convention; the raw numerator would need a 2^(row_max_s - m) weight, not
//  2^(lse_s - m)) plus the per-split lse, and a combine kernel merges the
//  splits in the log2 domain (the exact online-softmax merge:
//  w_s = 2^(lse_s - max_lse), out = sum_s w_s*part_s / sum_w,
//  lse = max + log2(sum_w); masked/empty splits carry lse = -inf / weight 0
//  exactly like masked keys above). PERF ONLY -- the merged result matches
//  dsa_sparse_fwd_cpu to bf16 tolerance (test_dsa_correct.hip split rows,
//  including uneven splits); any 1 <= split_kv <= topk is legal.
// ---------------------------------------------------------------------------
template <int BQ>
__global__ void dsa_sparse_fwd_partial_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv,
    const int32_t* __restrict__ indices, float* __restrict__ part_out,
    float* __restrict__ part_lse,
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk, int kv_group,
    float sm_scale, int split_kv) {
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;            // kv_group == 1: one shared head_kv
  const int s = blockIdx.z;            // which [k0,k1) chunk of the topk range
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

  // This split's contiguous chunk of the per-query index range. Uneven
  // splits are legal (the chunk boundaries need not align to block_I: the
  // kernel streams one key at a time).
  const int k0 = (int)(((long)topk * s) / split_kv);
  const int k1 = (int)(((long)topk * (s + 1)) / split_kv);

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

  for (int k = k0; k < k1; ++k) {
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

  // --- write the NORMALIZED per-split output (fp32, no bf16 rounding here
  // -- the combine re-weights in fp32 and rounds once) + per-split lse ---
  const float inv = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
  float* porow = part_out + (((size_t)gi * H + h) * split_kv + s) * d_v;
  for (int t = 0; t < v_tiles; ++t) {
    const int d = lane + t * 64;
    if (d < d_v) porow[d] = o[t] * inv;
  }
  if (lane == 0)
    part_lse[((size_t)gi * H + h) * split_kv + s] =
        (row_sum > 0.0f) ? (row_max + log2f(row_sum)) : -INFINITY;
}

__global__ void dsa_sparse_fwd_combine_kernel(
    const float* __restrict__ part_out, const float* __restrict__ part_lse,
    uint16_t* __restrict__ out, float* __restrict__ lse_out,
    int S_q, int H, int d_v, int split_kv, bool return_lse) {
  // part_out rows are the per-split NORMALIZED outputs (num / row_sum); the
  // merge re-weights them by w_s = 2^(lse_s - m) and divides by sum_w -- the
  // exact log2-domain online-softmax merge.
  const int gi = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x;        // 64 lanes over d_v, like the forward
  const size_t base = ((size_t)gi * H + h) * split_kv;

  // Every lane redundantly reduces the per-split lse (split_kv scalar reads,
  // L2-broadcast) so the weights below are lock-step without LDS traffic.
  float m = -INFINITY;
  for (int s = 0; s < split_kv; ++s) m = fmaxf(m, part_lse[base + s]);

  if (m == -INFINITY) {
    // Every selected key of every split is masked -> zero row, lse = -inf
    // (same contract as the plain forward's fully-masked row).
    uint16_t* orow = out + ((size_t)gi * H + h) * d_v;
    for (int d = lane; d < d_v; d += 64) orow[d] = 0;
    if (lane == 0 && return_lse) lse_out[(size_t)gi * H + h] = -INFINITY;
    return;
  }

  // Lane-local slices of the merged numerator (strided by 64 over d_v).
  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float denom = 0.0f;                  // same on every lane
  for (int s = 0; s < split_kv; ++s) {
    const float l = part_lse[base + s];
    const float w = (l == -INFINITY) ? 0.0f : exp2f(l - m);
    denom += w;
    const float* po = part_out + (base + s) * d_v;
    for (int t = 0; t < kPerLane; ++t) {
      const int d = lane + t * 64;
      if (d < d_v) o[t] += w * po[d];
    }
  }

  const float inv = 1.0f / denom;      // denom >= w_max_split = 1 > 0 here
  uint16_t* orow = out + ((size_t)gi * H + h) * d_v;
  for (int t = 0; t < kPerLane; ++t) {
    const int d = lane + t * 64;
    if (d < d_v) orow[d] = f2bf(o[t] * inv);
  }
  if (lane == 0 && return_lse)
    lse_out[(size_t)gi * H + h] = m + log2f(denom);
}

namespace {
template <int BQ>
void launch_partial(const void* q, const void* kv, const void* indices,
                    void* part_out, void* part_lse, int S_q, int S_kv, int H,
                    int dim, int tail_dim, int topk, int kv_group,
                    float sm_scale, int split_kv,
                    hipStream_t stream = nullptr) {  // #69: caller stream
  const int W = dim + tail_dim;
  const int shmem = BQ * W * sizeof(float);
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, split_kv);
  dsa_sparse_fwd_partial_kernel<BQ><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const uint16_t*>(q),
      reinterpret_cast<const uint16_t*>(kv),
      reinterpret_cast<const int32_t*>(indices),
      reinterpret_cast<float*>(part_out), reinterpret_cast<float*>(part_lse),
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, sm_scale, split_kv);
}
}  // namespace

void dsa_sparse_fwd_split(int S_q, int S_kv, int H, int dim, int tail_dim,
                          int topk, int kv_group, int block_I, int inner_iter,
                          float sm_scale, bool return_lse, int split_kv,
                          const void* q, const void* kv, const void* indices,
                          void* out, void* lse,
                          void* partial_out, void* partial_lse,
                          void* stream) {
  // Host-side preconditions: the same contract as dsa_sparse_fwd_with_tile
  // (issue #57) plus the split bounds; the combine writes `out` from the
  // partials, so `out` must not alias them either.
  const int d_v = dim - tail_dim;
  VK_EXPECTS(dim > 0 && tail_dim >= 0 && d_v > 0,
             "dim > tail_dim >= 0 required");
  VK_EXPECTS(topk > 0, "topk must be positive");
  VK_EXPECTS(kv_group == 1, "kv_group must be 1 (single shared head_kv)");
  VK_EXPECTS(block_I > 0 && inner_iter > 0,
             "block_I and inner_iter must be positive");
  VK_EXPECTS(topk % (block_I * inner_iter) == 0,
             "topk must be a multiple of block_I*inner_iter");
  VK_EXPECTS(split_kv >= 1, "split_kv must be positive");
  VK_EXPECTS(split_kv <= topk,
             "split_kv must not exceed topk (every split needs >= 1 key)");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 || q != nullptr,
             "q must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || S_kv == 0 || kv != nullptr,
             "kv must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 || indices != nullptr,
             "indices must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 || out != nullptr,
             "out must not be null");
  VK_EXPECTS(!return_lse || lse != nullptr, "lse must not be null");
  if (split_kv == 1) {
    // Degenerate split: identical to the plain forward, no scratch needed.
    dsa_sparse_fwd(S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I,
                   inner_iter, sm_scale, return_lse, q, kv, indices, out,
                   lse);
    return;
  }
  VK_EXPECTS(partial_out != nullptr && partial_lse != nullptr,
             "partial_out and partial_lse scratch required when split_kv > 1");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 ||
                 (out != partial_out && out != partial_lse &&
                  (!return_lse || lse != partial_out &&
                                 lse != partial_lse)),
             "out/lse must not alias the partial scratch");
  int bq = 0, th = 0, bi = 0, ii = 0;
  ::vkernels::kernels::dsa_config_for(S_q, H, dim, topk, &bq, &th, &bi, &ii);
  (void)th;
  (void)block_I;     // the partial kernel streams one key at a time, like
  (void)inner_iter;  // the plain forward (tiled-key prefetch is a follow-on)
  if (bq <= 1)
    launch_partial<1>(q, kv, indices, partial_out, partial_lse, S_q, S_kv, H,
                      dim, tail_dim, topk, kv_group, sm_scale, split_kv,
                      static_cast<hipStream_t>(stream));
  else if (bq <= 2)
    launch_partial<2>(q, kv, indices, partial_out, partial_lse, S_q, S_kv, H,
                      dim, tail_dim, topk, kv_group, sm_scale, split_kv,
                      static_cast<hipStream_t>(stream));
  else if (bq <= 4)
    launch_partial<4>(q, kv, indices, partial_out, partial_lse, S_q, S_kv, H,
                      dim, tail_dim, topk, kv_group, sm_scale, split_kv,
                      static_cast<hipStream_t>(stream));
  else
    launch_partial<8>(q, kv, indices, partial_out, partial_lse, S_q, S_kv, H,
                      dim, tail_dim, topk, kv_group, sm_scale, split_kv,
                      static_cast<hipStream_t>(stream));
  dim3 cgrid(S_q, H);
  dsa_sparse_fwd_combine_kernel<<<cgrid, 64, 0, static_cast<hipStream_t>(stream)>>>(
      reinterpret_cast<const float*>(partial_out),
      reinterpret_cast<const float*>(partial_lse),
      reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(lse),
      S_q, H, d_v, split_kv, return_lse);
}

// ---------------------------------------------------------------------------
//  DSA paged-MQA gated top-k logits (gfx942), issue #51. See dsa.hpp for the
//  gated-logit formula and the tensor layout; this kernel is the plain
//  gfx942 replacement for the tilelang path (JIT-compiles but the launched
//  kernel NEVER returns for num_heads in {32,64}) and deep_gemm's CUDA-only
//  fp8_paged_mqa_logits.
//
//  The two scalar variants -- dsa_topk_logits_kernel (fp32-Q) and
//  dsa_topk_logits_kernel_fp8q (raw-fp8-Q fallback) -- are the shared
//  definitions in dsa_topk_device.cuh, byte-identical to the ones the CUDA
//  TU (dsa.cu) launches. Only the MFMA fast path below is gfx942-specific.
// ---------------------------------------------------------------------------

// bf16-MFMA variant of dsa_topk_logits_kernel -- the launcher's FAST path
// for the GLM-5.3 indexer family (H a multiple of 16, D a multiple of 64,
// B a multiple of 16; see dsa_topk_logits_fits_lds_mfma). gfx942 has NO
// hipFuncSetAttribute opt-in past 64 KB (hipFuncSetAttribute(
// MaxDynamicSharedMemorySize, N>65536) returns hipSuccess but the launch
// silently never runs; verified on a CSCS beverin node -- see the KB note
// mi300a-dynamic-lds-no-optin), so instead of a driver opt-in this variant
// factors the per-page [H, B] = Q[H,D] @ K[B,D]^T dot matrix through the
// verified bf16 MFMA (__builtin_amdgcn_mfma_f32_16x16x16bf16_1k; see
// gemm_bf16.hip for the empirically-confirmed fragment layout), then does
// the ReLU + gated H-sum + per-token k_scale in fp32. The binding
// constraint at decode (one block, occupancy) therefore moves from SCALAR
// D-dots (one wavefront, H sequential dots per token) to MATRIX CORE: each
// of the B/16 wavefronts issues D/kBK K-tiles of kBK/kMfmaK=4 MFMAs over
// kNF=H/16 column fragments -- the whole [B,H] dot matrix per page in one
// pass.
//
// Storage (the SMALLEST of the three variants at every admitted width):
//   sQt[D][H]   bf16  -- Q transposed ONCE per block (reused across every
//                        page in the split). fp8->bf16 is LOSSLESS (every
//                        fp8 e4m3fnuz finite value is exactly representable
//                        in bf16; verified by check_fp8_bf16_exact before
//                        this kernel existed), and bf16*bf16->fp32 holds the
//                        14-bit product exactly in 23, so the ONLY residual
//                        vs the CPU oracle is fp32 reassociation across the
//                        D/kBK K-tiles -- well within the 1e-3 cross-check.
//   sK[B][kBK]  bf16  -- one K-tile, reloaded per K-tile per page
//   sGate[H]    fp32  -- the per-head gate, loaded once per block
//   sKscale[B]  fp32  -- per-token scales, loaded once per page
//   shmem = (D*H + B*kBK)*2 + (H + B)*4   bytes   (kBK = 64, fixed)
// At the GLM-5.3 widths: 16,768 B (H=32) / 25,088 B (H=64) / 41,728 B
// (H=128) -- vs the fp8-Q kernel's 37,248 / 41,472 / 49,920 B and the
// fp32-Q kernel's 49,536 / 66,048 B+ (the latter refused). BM = B, BN = H;
// THREADS = (B/16)*64 (one wavefront per 16-row fragment), so at B=64 the
// block is 256 threads = 4 wavefronts (vs the other two variants' 1).
//
// Per-token epilogue. acc[nf][i] holds dot[j, h] for j = (tid>>4)*4 + i and
// h = nf*16 + (tid&15) (the verified 16x16 C fragment: row = warp*16 +
// (lane>>4)*4 + i, col = lane&15). Each of the 4 rows a lane owns needs the
// gated sum over ALL H heads; the 16 lanes sharing (tid>>4) each hold a
// different head (nf*16 + (tid&15)), so a 4-step __shfl_xor (off 1,2,4,8)
// collapses them to the full sum on every lane. Only (tid&15)==0 writes the
// 4 outputs (one per j-group) to avoid redundant stores.
__global__ void dsa_topk_logits_kernel_mfma(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int H, int D, int B, int max_table_len, int max_seq_len, int split_kv) {
  const int kNF    = H / 16;           // column / head fragments
  const int kWarps = B / 16;           // one wavefront per 16-row fragment
  const int kTh    = kWarps * 64;      // threads per block (= B*4)

  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int tid  = threadIdx.x;        // 0..kTh-1
  const int warp = tid >> 6;           // 0..kWarps-1
  const int m    = tid & 15;           // row within the 16-row fragment
  const int kq   = (tid & 63) >> 4;    // 0..3 -> which 4-K group
  const int row_base = warp * 16;      // this warp's 16-row fragment (j)

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  extern __shared__ float smem[];
  uint16_t* sQt = reinterpret_cast<uint16_t*>(smem);             // D * H
  float*    sGate = reinterpret_cast<float*>(sQt + (size_t)D * H);      // H
  uint16_t* sK = reinterpret_cast<uint16_t*>(sGate + H);         // B * kBK
  float*    sKscale = reinterpret_cast<float*>(sK + (size_t)B * kBK);   // B

  // --- cooperative load Q transposed as bf16 sQt[D][H] once per block ---
  // Q is laid out [H][D] (q[h*D + d] = Q[h,d]); sQt is the TRANSPOSE
  // [D][H] so the MFMA's B fragment (sQt[K-row=d][N-col=h]) reads Q[h,d]
  // in one bf16 load. fp8 -> fp32 -> bf16 is lossless (verified before
  // this kernel existed).
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  for (int idx = tid; idx < H * D; idx += kTh) {
    const int h = idx / D;
    const int d = idx % D;
    sQt[(size_t)d * H + h] = f2bf(fp8e4m3fnuz_to_f32(qp[(size_t)h * D + d]));
  }
  const float* gp = weight + (size_t)b * H;
  for (int h = tid; h < H; h += kTh) sGate[h] = gp[h];
  __syncthreads();

  // kMaxNF covers every shape dsa_topk_logits_fits_lds_mfma admits (the
  // loosest is D=64, B=16 -> H<=480 -> kNF=30; kMaxNF=32 is a safe upper
  // bound, used [0,kNF) only). The accumulators are ZEROED PER PAGE --
  // each page is an independent [H,B] = Q@K^T dot matrix; carrying acc
  // across pages would sum prior pages' K into the current output.
  v4f acc[32];

  for (int it = 0; it < n_iters; ++it) {
    const int pg = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + pg];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));

    // Per-token scales: load ONCE per page (the trailing B*4 bytes; they do
    // not change across K-tiles).
    for (int j = tid; j < B; j += kTh)
      sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];

    for (int nf = 0; nf < kNF; ++nf)
      acc[nf] = (v4f){0.0f, 0.0f, 0.0f, 0.0f};

    // K-tiles: D/kBK of them, each kBK wide. A K-tile is consumed by
    // kBK/kMfmaK = 4 MFMAs; per MFMA, kq in [0,4) selects this lane's 4 K.
    for (int kt = 0; kt < D / kBK; ++kt) {
      // cooperative load sK[B][kBK] for THIS K-tile (bf16). col is the LOCAL
      // K-column; gcol = kt*kBK + col is the global D-column. (D is a
      // multiple of kBK so no D-bound check.)
      for (int idx = tid; idx < B * kBK; idx += kTh) {
        const int row = idx / kBK;          // 0..B-1 (KV token j)
        const int col = idx % kBK;          // 0..kBK-1 (local K-col)
        const int gcol = kt * kBK + col;
        sK[(size_t)row * kBK + col] =
            f2bf(fp8e4m3fnuz_to_f32(kbase[(size_t)row * D + gcol]));
      }
      __syncthreads();

      // --- kBK/kMfmaK = 4 MFMAs consume the K-tile; acc is preserved across
      //     K-tiles -> the full D-dot per (j, h). ---
      // A fragment (sK, the M=B side): a[i] = sK[row_base+m][k0+i].
      // B fragment (sQt, the N=H side): b[i] = sQt[gd0+i][nb],
      //   nb = nf*16 + m (head h), gd0 = kt*kBK + k0 (global D-row).
      #pragma unroll
      for (int mf = 0; mf < kBK / kMfmaK; ++mf) {
        const int k0 = mf * kMfmaK + kq * 4;       // 4 local K-cols
        v4s a = {(short)sK[(size_t)(row_base + m) * kBK + k0],
                 (short)sK[(size_t)(row_base + m) * kBK + k0 + 1],
                 (short)sK[(size_t)(row_base + m) * kBK + k0 + 2],
                 (short)sK[(size_t)(row_base + m) * kBK + k0 + 3]};
        const int gd0 = kt * kBK + k0;
        #pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
          const int nb = nf * 16 + m;
          v4s b = {(short)sQt[(size_t)(gd0)     * H + nb],
                   (short)sQt[(size_t)(gd0 + 1) * H + nb],
                   (short)sQt[(size_t)(gd0 + 2) * H + nb],
                   (short)sQt[(size_t)(gd0 + 3) * H + nb]};
          acc[nf] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[nf], 0, 0, 0);
        }
      }
      __syncthreads();                       // before reloading sK next tile
    }

    // --- per-token gated reduce (fp32) + k_scale, then write ---
    // acc[nf][i] = dot[j=(tid>>4)*4+i, h=nf*16+(tid&15)]. The 16 lanes
    // sharing (tid>>4) each hold a different head; __shfl_xor(1/2/4/8)
    // collapses them to sum_h gate[h]*max(0,dot) on every lane. Only
    // (tid&15)==0 writes the 4 outputs (one per j-group).
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float partial = 0.0f;
      #pragma unroll
      for (int nf = 0; nf < kNF; ++nf) {
        float d = acc[nf][i];
        if (d < 0.0f) d = 0.0f;             // ReLU (max(0, raw))
        partial += sGate[nf * 16 + m] * d;
      }
      for (int off = 1; off < 16; off <<= 1)
        partial += __shfl_xor(partial, off);
      if ((tid & 15) == 0) {
        const int j = (tid >> 4) * 4 + i;   // 0..B-1 (kWarps*4 groups)
        const int t = pg * B + j;
        if (t < seq_len)
          out[(size_t)b * max_seq_len + t] = sKscale[j] * partial;
      }
    }
  }
}

// Dispatch shim shared by dsa_topk_logits (auto) and
// dsa_topk_logits_with_variant (explicit). `variant`: 0 = auto
// (mfma -> fp32q -> fp8q -> refuse); 1 = fp32-Q; 2 = fp8-Q; 3 = mfma.
// An explicit variant that does NOT fit its LDS cap is refused with a
// stderr diagnostic + no-op (the caller zeroed `out`, which is what
// stays) rather than launch a block gfx942's driver would drop -- mirrors
// the auto path's contract.
namespace {
void dsa_topk_logits_dispatch(int batch_size, int num_heads, int head_dim,
                              int block, int max_table_len, int max_seq_len,
                              int split_kv, const void* q_fp8,
                              const void* kvcache_u8, const void* weight,
                              const void* seq_lens, const void* page_table,
                              void* out, int variant) {
  if (batch_size <= 0 || max_table_len <= 0) return;  // nothing to write
  if (num_heads <= 0 || head_dim <= 0 || block <= 0) return;  // bad dims
  // One wavefront (B=64 lanes), one KV token per lane. Grid splits the
  // pages across split_kv blocks per batch (perf only; grouping-independent,
  // mirrors the forward's block_I/inner_iter). Stream 0 matches dsa_sparse_fwd;
  // the caller synchronises (see vkernels_dsa.tilelang_fp8_paged_mqa_logits).
  dim3 grid(batch_size, split_kv, 1);

  const bool want_mfma  = (variant == 3) || (variant == 0);
  const bool want_fp32q = (variant == 1) || (variant == 0);
  const bool want_fp8q  = (variant == 2) || (variant == 0);

  // Fast path: the bf16-MFMA kernel (smallest footprint, Matrix-Core).
  // Shapes need H%16==0, D%64==0, B%16==0 (the verified 16x16x16bf16_1k
  // fragment); the block runs (B/16)*64 threads (one wavefront per 16-row
  // fragment). See dsa.hpp::dsa_topk_logits_fits_lds_mfma.
  if (want_mfma && ::vkernels::kernels::dsa_topk_logits_fits_lds_mfma(
                       num_heads, head_dim, block)) {
    const int shmem = (head_dim * num_heads + block * kBK) * 2 +
                      (num_heads + block) * sizeof(float);
    dim3 blockdim((block / 16) * 64);
    dsa_topk_logits_kernel_mfma<<<grid, blockdim, shmem, 0>>>(
        reinterpret_cast<const uint8_t*>(q_fp8),
        reinterpret_cast<const uint8_t*>(kvcache_u8),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const int32_t*>(seq_lens),
        reinterpret_cast<const int32_t*>(page_table),
        reinterpret_cast<float*>(out),
        num_heads, head_dim, block, max_table_len, max_seq_len, split_kv);
    return;
  }

  // fp32-Q kernel (Q dequanted once into shared) for shapes that fit the
  // larger fp32-Q footprint -- the GLM-5.3 indexer (H=32, D=128, B=64
  // stages 49,536 B). See dsa.hpp::dsa_topk_logits_fits_lds.
  if (want_fp32q && ::vkernels::kernels::dsa_topk_logits_fits_lds(
                        num_heads, head_dim, block)) {
    const int shmem = (num_heads * head_dim + num_heads +
                       block * head_dim + block) * sizeof(float);
    dim3 blockdim(block);
    dsa_topk_logits_kernel<<<grid, blockdim, shmem, 0>>>(
        reinterpret_cast<const uint8_t*>(q_fp8),
        reinterpret_cast<const uint8_t*>(kvcache_u8),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const int32_t*>(seq_lens),
        reinterpret_cast<const int32_t*>(page_table),
        reinterpret_cast<float*>(out),
        num_heads, head_dim, block, max_table_len, max_seq_len, split_kv);
    return;
  }

  // fp8-Q kernel (Q staged raw, dequantised on the fly in the dot loop with
  // the SAME helper -- bit-identical output to the fp32-Q kernel above) for
  // shapes that fit the smaller fp8-Q footprint (H=64, D=128, B=64 stages
  // 41,472 B). See dsa.hpp::dsa_topk_logits_fits_lds_fp8q.
  if (want_fp8q && ::vkernels::kernels::dsa_topk_logits_fits_lds_fp8q(
                       num_heads, head_dim, block)) {
    const int shmem = (num_heads + block * head_dim + block) * sizeof(float) +
                      num_heads * head_dim;
    dim3 blockdim(block);
    dsa_topk_logits_kernel_fp8q<<<grid, blockdim, shmem, 0>>>(
        reinterpret_cast<const uint8_t*>(q_fp8),
        reinterpret_cast<const uint8_t*>(kvcache_u8),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const int32_t*>(seq_lens),
        reinterpret_cast<const int32_t*>(page_table),
        reinterpret_cast<float*>(out),
        num_heads, head_dim, block, max_table_len, max_seq_len, split_kv);
    return;
  }

  // No admitted variant. REFUSE the shape rather than launch a block the
  // driver silently drops; the caller zeroed `out`, which is what stays
  // (logged, not a silent wrong all-zero top-k).
  std::fprintf(stderr,
      "vk_hip_dsa_topk_logits: indexer (H=%d D=%d B=%d) exceeds gfx942's 64 KB "
      "non-optin dynamic-LDS cap under ALL of mfma (%d B), fp32-Q (%d B) and "
      "fp8-Q (%d B). Refusing (output left as the caller provided). See "
      "dsa.hpp::dsa_topk_logits_fits_lds{,_fp8q,_mfma}.\n",
      num_heads, head_dim, block,
      (head_dim * num_heads + block * kBK) * 2 + (num_heads + block) * 4,
      (num_heads * head_dim + num_heads + block * head_dim + block) * 4,
      (num_heads + block * head_dim + block) * 4 + num_heads * head_dim);
}
}  // namespace

void dsa_topk_logits_with_variant(int batch_size, int num_heads, int head_dim,
                                  int block, int max_table_len,
                                  int max_seq_len, int split_kv,
                                  const void* q_fp8, const void* kvcache_u8,
                                  const void* weight, const void* seq_lens,
                                  const void* page_table, void* out,
                                  int variant) {
  if (variant != 0 && variant != 1 && variant != 2 && variant != 3) {
    std::fprintf(stderr,
        "vk_hip_dsa_topk_logits_with_variant: bad variant=%d "
        "(want 0=auto, 1=fp32q, 2=fp8q, 3=mfma). Refusing.\n", variant);
    return;
  }
  dsa_topk_logits_dispatch(batch_size, num_heads, head_dim, block,
                           max_table_len, max_seq_len, split_kv, q_fp8,
                           kvcache_u8, weight, seq_lens, page_table, out,
                           variant);
}

void dsa_topk_logits(int batch_size, int num_heads, int head_dim, int block,
                     int max_table_len, int max_seq_len, int split_kv,
                     const void* q_fp8, const void* kvcache_u8,
                     const void* weight, const void* seq_lens,
                     const void* page_table, void* out) {
  // Auto: mfma (smallest, Matrix-Core) -> fp32-Q -> fp8-Q -> refuse.
  dsa_topk_logits_dispatch(batch_size, num_heads, head_dim, block,
                           max_table_len, max_seq_len, split_kv, q_fp8,
                           kvcache_u8, weight, seq_lens, page_table, out, 0);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
