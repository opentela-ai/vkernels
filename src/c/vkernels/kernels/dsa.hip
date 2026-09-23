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
// block (bf16 -> fp32). Issue #149 gives the plain path two composable
// speedups: (1) the kernel is width-specialized (compile-time tile counts at
// the serving widths) so the output accumulators stay register-resident, and
// (2) a staged variant double-buffers key tiles in LDS on the prefill path
// (launch_stage / VK_DSA_STAGE), keeping kti key rows in flight per warp
// instead of one. Decode shapes keep the serial behavior. The combined
// result is grouping-independent, so each block streams all of its `topk`
// keys directly (no separate partial_o / partial_lse + combine kernel).
//
// Two-implementation model:
//   dsa.cpp  -- CPU reference (oracle), always compiled, two-pass softmax.
//   dsa.hip  -- this file, compiled with VKERNELS_HAS_HIP, online softmax.
//               Cross-checked against the oracle by test_dsa_correct.hip.
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "vkernels/kernels/dsa.hpp"
#include "vkernels/util/error.hpp"  // VK_EXPECTS (host: throws std::invalid_argument)

#if VKERNELS_HAS_HIP
#  include "vkernels/core/tuning.hpp"
#  include "vkernels/kernels/moe_device.hip"  // shared bf16<->f32 + fp4 helpers
#  include "vkernels/kernels/dsa_topk_device.cuh"  // shared scalar dsa_topk_logits kernels
#  include <hip/hip_runtime.h>

namespace vkernels::kernels::hip {

// Tuning-store tile override (docs/tuning-cache.md): consulted by BOTH
// dsa_sparse_fwd and dsa_sparse_fwd_split ahead of dsa_config_for, so a
// measured per-shape sweep on this device arch pins the launch tile.
// Implemented in dsa.cpp (host TU, always compiled -- the split-K selector
// consults it too) and declared in dsa.hpp; only `bq` is live today
// (block_I / inner_iter are reserved by the baseline kernels -- see
// with_tile's (void) casts); they are persisted anyway so a future
// tiled-key kernel inherits the store untouched. A malformed or stale
// record falls back to the formula: a store must never break serving.

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


// Width specialization (issue #149 step 1). MAIN_TILES/TAIL_TILES/V_TILES
// are compile-time tile counts; 0 keeps the dynamic-width fallback (the tile
// count is then derived from the runtime dim/tail_dim). At the serving
// widths -- GLM-5.3 (dim=256, tail_dim=0) and DeepSeek-V3 (dim=576,
// tail_dim=64) -- constant bounds let every t-loop fully unroll, so the
// `o[kPerLane]` output accumulators index STATICALLY and live in registers
// instead of scratch (a runtime-bound loop over a local array cannot be
// unrolled, so the dynamic-width lowering spills it to per-wave scratch and
// puts ~16 scratch accesses per key on the dependent chain).
template <int BQ, int MAIN_TILES = 0, int TAIL_TILES = 0, int V_TILES = 0>
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

  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;
  // The ternaries fold to constants in the specialized instantiations, so
  // every loop below trips a compile-time bound there.
  const int main_tiles =
      (MAIN_TILES > 0) ? MAIN_TILES : (dim + 63) / 64;
  const int tail_tiles =
      (TAIL_TILES > 0) ? TAIL_TILES : (tail_dim + 63) / 64;  // 0 when tail_dim == 0
  const int v_tiles = (V_TILES > 0) ? V_TILES : (d_v + 63) / 64;

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

  // Out-of-range rows exit AFTER the staging barrier: the staging loop
  // above distributes all BQ rows across every warp (r = idx / W spans
  // warps whenever W > 64), so every warp must help stage sQ even when
  // S_q < BQ (a store-pinned oversized tile on a decode shape). This
  // kernel has no later barriers, so exiting here cannot deadlock.
  if (gi >= S_q) return;
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
//  Register-prefetch plain kernel (issue #149 step 2, low-risk variant).
//  Software-pipelines the serial key chain IN REGISTERS: key k+1's row and
//  index are loaded into per-lane registers BEFORE the online-softmax update
//  over key k consumes the register-held key k, so the global loads of the
//  next key issue ~one key of compute earlier -- 2 key rows in flight per
//  warp with ZERO LDS and therefore NO residency loss (the LDS-staged kernel
//  below trades in-flight depth for blocks/CU, which the kti sweep shows it
//  cannot afford at the DSv3 width or the H=64 grid). Per-key ARITHMETIC is
//  identical to the serial kernel: the strided main/tail score loops are the
//  flat loop over WT = main_tiles + tail_tiles tiles in the same ascending-d
//  order, and the V accumulate reads the same bf16 values, so every
//  correctness row is bit-comparable to the serial kernel. Specialized
//  widths only (WT compile-time); dynamic widths keep the serial kernel.
template <int BQ, int WT, int V_TILES>
__global__ void dsa_sparse_fwd_pf_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv,
    const int32_t* __restrict__ indices, uint16_t* __restrict__ out,
    float* __restrict__ lse_out, int S_q, int S_kv, int H, int dim,
    int tail_dim, int topk, int kv_group, float sm_scale, bool return_lse) {
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int gi = q_tile * BQ + warp;
  if (gi >= S_q) return;

  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;

  extern __shared__ float sQ[];       // sQ[BQ][W]
  for (int idx = tid; idx < BQ * W; idx += BQ * 64) {
    const int r = idx / W, c = idx % W;
    const int gr = q_tile * BQ + r;
    sQ[idx] = (gr < S_q) ? bf16_to_f32(
                  q[((size_t)gr * H + h) * W + c])
                         : 0.0f;
  }
  __syncthreads();

  const float* qrow = &sQ[warp * W];
  const int32_t* idx_i = indices + (size_t)gi * kv_group * topk;

  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;
  float row_sum = 0.0f;

  // Load key `k`'s row + index into the given register set (masked keys
  // load as zero rows; validity is re-checked at consume time, exactly
  // like the staged kernels' zero-filled rows).
#define VK_DSA_PF_LOAD(ridx_, row_, k_)                                       \
  do {                                                                        \
    const int32_t ki_ = idx_i[k_];                                            \
    (ridx_) = ki_;                                                            \
    if (ki_ >= 0 && ki_ < S_kv) {                                             \
      const uint16_t* p_ = kv + (size_t)ki_ * W;                              \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < WT; ++t) {                                          \
        const int d = lane + t * 64;                                          \
        (row_)[t] = (d < W) ? p_[d] : (uint16_t)0;                            \
      }                                                                       \
    } else {                                                                  \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < WT; ++t) (row_)[t] = (uint16_t)0;                   \
    }                                                                         \
  } while (0)

  int32_t cur_idx;
  uint16_t cur[WT];
  VK_DSA_PF_LOAD(cur_idx, cur, 0);

  for (int k = 0; k < topk; ++k) {
    // Issue key k+1's loads FIRST: they are independent of everything
    // below, so they fill the in-order issue slots while the warp works.
    int32_t nxt_idx = -1;
    uint16_t nxt[WT];
    if (k + 1 < topk) VK_DSA_PF_LOAD(nxt_idx, nxt, k + 1);

    const bool valid = (cur_idx >= 0) && (cur_idx < S_kv);

    // --- lane-local partial score (main + tail, ascending d), then
    //     warp-reduce (identical order/tree to the serial kernel) ---
    float partial = 0.0f;
    if (valid) {
      _Pragma("unroll")
      for (int t = 0; t < WT; ++t) {
        const int d = lane + t * 64;
        if (d < W) partial += qrow[d] * bf16_to_f32(cur[t]);
      }
    }
    float score = partial;
    for (int off = 32; off > 0; off >>= 1)
      score += __shfl_xor(score, off);
    score = valid ? (score * sm_scale) : -INFINITY;

    // --- online softmax update (lock-step; identical on every lane) ---
    const float new_max = fmaxf(row_max, score);
    const float exp_old =
        (row_max == -INFINITY) ? 0.0f : exp2f(row_max - new_max);
    const float exp_s =
        (score == -INFINITY) ? 0.0f : exp2f(score - new_max);
    row_sum = row_sum * exp_old + exp_s;
    for (int t = 0; t < V_TILES; ++t) o[t] *= exp_old;
    if (valid) {
      _Pragma("unroll")
      for (int t = 0; t < V_TILES; ++t) {
        const int d = lane + t * 64;
        if (d < d_v) o[t] += exp_s * bf16_to_f32(cur[t]);
      }
    }
    row_max = new_max;

    // Rotate: key k+1 becomes key k (its loads are in flight / done).
    cur_idx = nxt_idx;
    _Pragma("unroll")
    for (int t = 0; t < WT; ++t) cur[t] = nxt[t];
  }
#undef VK_DSA_PF_LOAD

  const float inv = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
  uint16_t* orow = out + (((size_t)gi * H + h) * d_v);
  for (int t = 0; t < V_TILES; ++t) {
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
template <int BQ, int MAIN_TILES, int TAIL_TILES, int V_TILES>
void launch_impl(const void* q, const void* kv, const void* indices, void* out,
                 void* lse, int S_q, int S_kv, int H, int dim, int tail_dim,
                 int topk, int kv_group, float sm_scale, bool return_lse,
                 hipStream_t stream) {
  const int W = dim + tail_dim;
  const int shmem = BQ * W * sizeof(float);
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, 1);
  dsa_sparse_fwd_kernel<BQ, MAIN_TILES, TAIL_TILES, V_TILES>
      <<<grid, block, shmem, stream>>>(
          reinterpret_cast<const uint16_t*>(q),
          reinterpret_cast<const uint16_t*>(kv),
          reinterpret_cast<const int32_t*>(indices),
          reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(lse),
          S_q, S_kv, H, dim, tail_dim, topk, kv_group, sm_scale, return_lse);
}
// Width dispatch (issue #149 step 1): the two serving widths get
// compile-time tile counts (register-resident accumulators, unrolled t-loops);
// every other shape keeps the dynamic-width fallback.
template <int BQ>
void launch(const void* q, const void* kv, const void* indices, void* out,
            void* lse, int S_q, int S_kv, int H, int dim, int tail_dim,
            int topk, int kv_group, float sm_scale, bool return_lse,
            hipStream_t stream = nullptr) {  // #69: caller stream
  if (dim == 256 && tail_dim == 0) {
    launch_impl<BQ, 4, 0, 4>(q, kv, indices, out, lse, S_q, S_kv, H, dim,
                             tail_dim, topk, kv_group, sm_scale, return_lse,
                             stream);
  } else if (dim == 576 && tail_dim == 64) {
    launch_impl<BQ, 9, 1, 8>(q, kv, indices, out, lse, S_q, S_kv, H, dim,
                             tail_dim, topk, kv_group, sm_scale, return_lse,
                             stream);
  } else {
    launch_impl<BQ, 0, 0, 0>(q, kv, indices, out, lse, S_q, S_kv, H, dim,
                             tail_dim, topk, kv_group, sm_scale, return_lse,
                             stream);
  }
}

// ---------------------------------------------------------------------------
//  Plain-path staged kernel (issue #149 step 2). The serial kernel above
//  keeps ~1 key row in flight per warp (load -> use dependency, gfx942
//  issues in-order), which caps it at ~21-28% of the HBM roof on prefill.
//  This variant stages key tiles into LDS (the issue-#80 double-buffer
//  machinery, minus the split/combine) so tile i+1's global loads issue
//  while the warp runs the online-softmax update over tile i: `kti` key
//  rows in flight per warp instead of 1. Per-key ARITHMETIC is identical
//  to the serial kernel (same multiply order, same shfl reduction tree,
//  same fmaxf/exp2 update sequence), so every correctness row is
//  unchanged; masked keys stage as zero rows exactly like the db kernel.
//  Shared layout: sQ[BQ*W f32] | sKey[2*BQ*kti*W u16] | sIdx[2*BQ*kti i32],
//  one private sKey/sIdx slice per warp, lane-strided staging (issue #80
//  audit fix), warp-synchronous (no block barriers on the hot path).
template <int BQ, int MAIN_TILES = 0, int TAIL_TILES = 0, int V_TILES = 0>
__global__ void dsa_sparse_fwd_stage_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv,
    const int32_t* __restrict__ indices, uint16_t* __restrict__ out,
    float* __restrict__ lse_out, int S_q, int S_kv, int H, int dim,
    int tail_dim, int topk, int kv_group, float sm_scale, bool return_lse,
    int kti) {
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;            // kv_group == 1: one shared head_kv
  const int tid = threadIdx.x;
  const int warp = tid >> 6;           // which query row within the tile
  const int lane = tid & 63;           // strided head index this lane owns
  const int gi = q_tile * BQ + warp;   // global query row
  if (gi >= S_q) return;               // whole warp exits: no cross-warp LDS deps

  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;
  const int main_tiles =
      (MAIN_TILES > 0) ? MAIN_TILES : (dim + 63) / 64;
  const int tail_tiles =
      (TAIL_TILES > 0) ? TAIL_TILES : (tail_dim + 63) / 64;
  const int v_tiles = (V_TILES > 0) ? V_TILES : (d_v + 63) / 64;

  extern __shared__ float sQ[];       // sQ[BQ][W]
  for (int idx = tid; idx < BQ * W; idx += BQ * 64) {
    const int r = idx / W, c = idx % W;
    const int gr = q_tile * BQ + r;
    sQ[idx] = (gr < S_q) ? bf16_to_f32(
                  q[((size_t)gr * H + h) * W + c])
                         : 0.0f;
  }
  __syncthreads();

  uint16_t* sKey = reinterpret_cast<uint16_t*>(sQ + BQ * W);
  int32_t* sIdx = reinterpret_cast<int32_t*>(sKey + (size_t)2 * BQ * kti * W);

  const float* qrow = &sQ[warp * W];  // this warp's query row (fp32)
  const int32_t* idx_i =
      indices + (size_t)gi * kv_group * topk;  // kv_group == 1 -> gi*topk

  // Lane-local slices of the output accumulator (strided by 64 over d_v).
  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;          // running max (same on every lane)
  float row_sum = 0.0f;               // running denominator (same on every lane)

// gfx942 has no __syncwarp in this HIP toolchain (see the db partial kernel
// below for the full rationale): one wavefront per warp, LDS executes wave
// instructions in issue order, so the empty asm is a compiler fence only.
#if defined(__HIP__)
#define VK_DSA_PSTAGE_SYNC() asm volatile("" ::: "memory")
#else
#define VK_DSA_PSTAGE_SYNC() __syncwarp()
#endif

// Stage keys [tb, tb + tlen) of this warp's index range into buffer `sbuf`,
// then a warp barrier. Same body as the db partial kernel's VK_DSA_STAGE_TILE
// (uint4-vectorized when W % 8 == 0, zero-filled masked rows).
#define VK_DSA_PSTAGE_TILE(sbuf_, tb_, tlen_)                                 \
  do {                                                                        \
    uint16_t* sdst =                                                          \
        sKey + (size_t)(((sbuf_) * BQ + warp) * kti) * W;                     \
    int32_t* idst = sIdx + ((sbuf_) * BQ + warp) * kti;                       \
    if (W % 8 == 0) {                                                         \
      const int rpp = W / 8; /* chunks (8 elements) per key row */            \
      for (int ch = lane; ch < (tlen_) * rpp; ch += 64) {                     \
        const int r = ch / rpp;                                               \
        const int c = (ch - r * rpp) * 8;                                     \
        const int32_t kv_idx = idx_i[(tb_) + r];                              \
        idst[r] = kv_idx;                                                     \
        uint4 v = {0u, 0u, 0u, 0u};                                           \
        if (kv_idx >= 0 && kv_idx < S_kv)                                     \
          v = *reinterpret_cast<const uint4*>(                                \
              kv + (size_t)kv_idx * W + c);                                   \
        *reinterpret_cast<uint4*>(sdst + (size_t)r * W + c) = v;              \
      }                                                                       \
    } else {                                                                  \
      for (int e = lane; e < (tlen_) * W; e += 64) {                          \
        const int r = e / W;                                                  \
        const int c = e - r * W;                                              \
        const int32_t kv_idx = idx_i[(tb_) + r];                              \
        idst[r] = kv_idx;                                                     \
        sdst[(size_t)r * W + c] =                                             \
            (kv_idx >= 0 && kv_idx < S_kv)                                    \
                ? kv[(size_t)kv_idx * W + c]                                  \
                : (uint16_t)0;                                                \
      }                                                                       \
    }                                                                         \
    VK_DSA_PSTAGE_SYNC(); /* staged tile visible before it is consumed */     \
  } while (0)

  // Prologue: stage the first tile into buffer 0.
  VK_DSA_PSTAGE_TILE(0, 0, (topk < kti) ? topk : kti);

  int buf = 0;
  for (int tb = 0; tb < topk; tb += kti) {
    const int te = (topk - tb < kti) ? topk : (tb + kti);
    if (te < topk) {  // prefetch the next tile into buf ^ 1
      VK_DSA_PSTAGE_TILE(buf ^ 1, te, (topk - te < kti) ? (topk - te) : kti);
    }
    for (int k = tb; k < te; ++k) {
      const int tk = k - tb;
      const int32_t idx = sIdx[(buf * BQ + warp) * kti + tk];
      const uint16_t* rowp =
          sKey + (size_t)((buf * BQ + warp) * kti + tk) * W;
      const bool valid = (idx >= 0) && (idx < S_kv);

      // --- lane-local partial score (main + tail), then warp-reduce ---
      float partial = 0.0f;
      if (valid) {
        for (int t = 0; t < main_tiles; ++t) {
          const int d = lane + t * 64;
          if (d < dim) partial += qrow[d] * bf16_to_f32(rowp[d]);
        }
        for (int t = 0; t < tail_tiles; ++t) {  // 0 iters when tail_dim == 0
          const int d = lane + t * 64;
          if (d < tail_dim)
            partial += qrow[dim + d] * bf16_to_f32(rowp[dim + d]);
        }
      }
      float score = partial;
      for (int off = 32; off > 0; off >>= 1)  // warp-wide sum -> broadcast
        score += __shfl_xor(score, off);
      score = valid ? (score * sm_scale) : -INFINITY;

      // --- online softmax update (lock-step; identical on every lane) ---
      const float new_max = fmaxf(row_max, score);
      const float exp_old =
          (row_max == -INFINITY) ? 0.0f : exp2f(row_max - new_max);
      const float exp_s =
          (score == -INFINITY) ? 0.0f : exp2f(score - new_max);
      row_sum = row_sum * exp_old + exp_s;
      for (int t = 0; t < v_tiles; ++t) o[t] *= exp_old;
      if (valid) {
        for (int t = 0; t < v_tiles; ++t) {
          const int d = lane + t * 64;
          if (d < d_v) o[t] += exp_s * bf16_to_f32(rowp[d]);
        }
      }
      row_max = new_max;
    }
    buf ^= 1;
    VK_DSA_PSTAGE_SYNC(); /* buffer fully consumed before its next restaging */
  }
#undef VK_DSA_PSTAGE_TILE
#undef VK_DSA_PSTAGE_SYNC

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

template <int BQ, int MAIN_TILES, int TAIL_TILES, int V_TILES>
bool launch_stage_impl(const void* q, const void* kv, const void* indices,
                       void* out, void* lse, int S_q, int S_kv, int H, int dim,
                       int tail_dim, int topk, int kv_group, float sm_scale,
                       bool return_lse, int kti, hipStream_t stream) {
  const int W = dim + tail_dim;
  const size_t shmem = (size_t)BQ * W * 4u +
                       (size_t)kti * (2u * BQ * ((size_t)W * 2u + 4u));
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, 1);
  dsa_sparse_fwd_stage_kernel<BQ, MAIN_TILES, TAIL_TILES, V_TILES>
      <<<grid, block, shmem, stream>>>(
          reinterpret_cast<const uint16_t*>(q),
          reinterpret_cast<const uint16_t*>(kv),
          reinterpret_cast<const int32_t*>(indices),
          reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(lse),
          S_q, S_kv, H, dim, tail_dim, topk, kv_group, sm_scale, return_lse,
          kti);
  return true;
}

// Prefill-aware staging gate for the plain forward. Engaged ONLY where the
// issue-#80 data says staging wins -- LONG key blocks (prefill), the opposite
// regime from the split path's thousands-of-tiny-blocks loss. VK_DSA_STAGE:
// 0 forces the serial kernel, 1 (default) applies the gate, 2 forces staging
// at every shape (correctness coverage). VK_DSA_KTI=<n> overrides the
// keys-per-tile (0 = auto: 4, ~21 KB LDS at BQ=4/W=256 -> 3 blocks/CU, 4 key
// rows in flight per warp).
template <int BQ>
bool launch_stage(const void* q, const void* kv, const void* indices, void* out,
                  void* lse, int S_q, int S_kv, int H, int dim, int tail_dim,
                  int topk, int kv_group, float sm_scale, bool return_lse,
                  hipStream_t stream) {
  static const int stage_mode = [] {
    const char* e = std::getenv("VK_DSA_STAGE");
    return e ? std::atoi(e) : 1;
  }();
  static const int kti_ovr = [] {
    const char* e = std::getenv("VK_DSA_KTI");
    return e ? std::atoi(e) : 0;
  }();
  if (stage_mode == 0) return false;
  if (stage_mode == 1 && !(S_q > 8 && topk >= 64)) return false;
  const int W = dim + tail_dim;
  const size_t shq = (size_t)BQ * (size_t)W * 4u;                // sQ
  const size_t per_key = (size_t)2 * BQ * ((size_t)W * 2u + 4u); // 2 key bufs + idx
  constexpr size_t kCap = 44u * 1024u;  // gfx942 48 KB no-attribute cap, margin
  if (shq >= kCap) return false;
  const int budget = (int)((kCap - shq) / per_key);
  int kti = (kti_ovr > 0) ? kti_ovr : 4;
  if (kti > budget) kti = budget;
  if (kti < 1) return false;
  if (dim == 256 && tail_dim == 0)
    return launch_stage_impl<BQ, 4, 0, 4>(q, kv, indices, out, lse, S_q, S_kv,
                                          H, dim, tail_dim, topk, kv_group,
                                          sm_scale, return_lse, kti, stream);
  if (dim == 576 && tail_dim == 64)
    return launch_stage_impl<BQ, 9, 1, 8>(q, kv, indices, out, lse, S_q, S_kv,
                                          H, dim, tail_dim, topk, kv_group,
                                          sm_scale, return_lse, kti, stream);
  return launch_stage_impl<BQ, 0, 0, 0>(q, kv, indices, out, lse, S_q, S_kv, H,
                                        dim, tail_dim, topk, kv_group,
                                        sm_scale, return_lse, kti, stream);
}

// Register-prefetch dispatch (issue #149). VK_DSA_PF: 0 forces the plain
// serial kernel, 1 (default) engages the pf kernel in the prefill regime
// (S_q > 8, any topk -- the loads issue one key early regardless of list
// length), 2 forces it at every shape (correctness coverage).
template <int BQ>
bool launch_pf(const void* q, const void* kv, const void* indices, void* out,
               void* lse, int S_q, int S_kv, int H, int dim, int tail_dim,
               int topk, int kv_group, float sm_scale, bool return_lse,
               hipStream_t stream) {
  static const int pf_mode = [] {
    const char* e = std::getenv("VK_DSA_PF");
    return e ? std::atoi(e) : 1;
  }();
  if (pf_mode == 0) return false;
  if (pf_mode == 1 && !(S_q > 8)) return false;
  const int W = dim + tail_dim;
  const int shmem = BQ * W * sizeof(float);
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, 1);
  if (dim == 256 && tail_dim == 0) {
    dsa_sparse_fwd_pf_kernel<BQ, 4, 4><<<grid, block, shmem, stream>>>(
        reinterpret_cast<const uint16_t*>(q),
        reinterpret_cast<const uint16_t*>(kv),
        reinterpret_cast<const int32_t*>(indices),
        reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(lse),
        S_q, S_kv, H, dim, tail_dim, topk, kv_group, sm_scale, return_lse);
    return true;
  }
  if (dim == 576 && tail_dim == 64) {
    dsa_sparse_fwd_pf_kernel<BQ, 10, 8><<<grid, block, shmem, stream>>>(
        reinterpret_cast<const uint16_t*>(q),
        reinterpret_cast<const uint16_t*>(kv),
        reinterpret_cast<const int32_t*>(indices),
        reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(lse),
        S_q, S_kv, H, dim, tail_dim, topk, kv_group, sm_scale, return_lse);
    return true;
  }
  return false;  // dynamic widths keep the serial kernel
}

// Plain-forward dispatch (issue #149): staged kernel when its gate admits
// it AND VK_DSA_PF does not claim the shape first (the pf kernel hides the
// same latency with no LDS residency cost, so it wins the priority), then
// the pf kernel, then the serial kernel. Decode shapes keep the measured
// serial behavior under both gates.
template <int BQ>
void launch_auto(const void* q, const void* kv, const void* indices, void* out,
                 void* lse, int S_q, int S_kv, int H, int dim, int tail_dim,
                 int topk, int kv_group, float sm_scale, bool return_lse,
                 hipStream_t stream = nullptr) {  // #69: caller stream
  if (launch_pf<BQ>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim,
                    topk, kv_group, sm_scale, return_lse, stream))
    return;
  if (launch_stage<BQ>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim,
                       topk, kv_group, sm_scale, return_lse, stream))
    return;
  launch<BQ>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
             kv_group, sm_scale, return_lse, stream);
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
    launch_auto<1>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse, static_cast<hipStream_t>(stream));
  else if (bq <= 2)
    launch_auto<2>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse, static_cast<hipStream_t>(stream));
  else if (bq <= 4)
    launch_auto<4>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
              kv_group, sm_scale, return_lse, static_cast<hipStream_t>(stream));
  else
    launch_auto<8>(q, kv, indices, out, lse, S_q, S_kv, H, dim, tail_dim, topk,
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
  if (!dsa_tile_from_store(S_q, H, dim, tail_dim, topk, &bq, &bi, &ii))
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

  // Out-of-range rows exit AFTER the staging barrier: the staging loop
  // above distributes all BQ rows across every warp (r = idx / W spans
  // warps whenever W > 64), so every warp must help stage sQ even when
  // S_q < BQ (a store-pinned oversized tile on a decode shape). This
  // kernel has no later barriers, so exiting here cannot deadlock.
  if (gi >= S_q) return;
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

// gfx942 has no __syncwarp in this HIP toolchain (see the dsa_kpool.hip
// note on HIP wavefront intrinsics). A warp here is one full 64-lane
// wavefront and the LDS unit executes wave instructions in issue order,
// so the staging stores are globally ordered before any later staged read
// at the hardware level; the empty inline asm is only a compiler fence to
// stop instruction scheduling from hoisting the staged LDS reads above
// the staging stores. On a CUDA backend, __syncwarp() is the exact form.
#if defined(__HIP__)
#define VK_DSA_WARP_SYNC() asm volatile("" ::: "memory")
#else
#define VK_DSA_WARP_SYNC() __syncwarp()
#endif

template <int BQ>
__global__ void dsa_sparse_fwd_partial_db_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv,
    const int32_t* __restrict__ indices, float* __restrict__ part_out,
    float* __restrict__ part_lse,
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk, int kv_group,
    float sm_scale, int split_kv, int kti, int pipeline) {
  // Issue #80: double-buffered variant of dsa_sparse_fwd_partial_kernel.
  // Key tiles are staged into LDS so the global loads of tile i+1 issue
  // while the warp runs the online-softmax update over tile i (gfx942 has
  // no cp.async, so latency hiding comes from issuing a tile's worth of
  // independent loads in bulk, uint2-vectorized when W % 8 == 0).
  // Per-key ARITHMETIC is identical to the serial kernel (same multiply
  // order, same shfl reduction tree, same fmaxf/exp2 update sequence; the
  // loads only move from L2 to staged LDS), so every correctness row is
  // unchanged. `kti` (keys per tile) is sized by the host so sQ + both
  // staging buffers stay under gfx942's 48 KB pre-attribute LDS cap.
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;            // kv_group == 1: one shared head_kv
  const int s = blockIdx.z;            // which [k0,k1) chunk of the topk range
  const int tid = threadIdx.x;
  const int warp = tid >> 6;           // which query row within the tile
  const int lane = tid & 63;           // strided head index this lane owns
  const int gi = q_tile * BQ + warp;   // global query row

  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;
  const int main_tiles = (dim + 63) / 64;
  const int tail_tiles = (tail_dim + 63) / 64;  // 0 when tail_dim == 0
  const int v_tiles = (d_v + 63) / 64;

  // This split's contiguous chunk of the per-query index range. Uneven
  // splits stay legal: chunk boundaries need not align to anything and a
  // tile is simply the next `kti` keys of the chunk.
  const int k0 = (int)(((long)topk * s) / split_kv);
  const int k1 = (int)(((long)topk * (s + 1)) / split_kv);

  // Shared layout: sQ[BQ*W floats] | sKey[2*BQ*kti*W u16] | sIdx[2*BQ*kti i32],
  // one private sKey/sIdx slice per warp (each warp streams its own query
  // row's keys, so warp-synchronous __syncwarp() suffices, no barriers).
  extern __shared__ float sQ[];
  for (int idx = tid; idx < BQ * W; idx += BQ * 64) {
    const int r = idx / W, c = idx % W;
    const int gr = q_tile * BQ + r;
    sQ[idx] = (gr < S_q) ? bf16_to_f32(
                  q[((size_t)gr * H + h) * W + c])
                         : 0.0f;
  }
  __syncthreads();

  // Out-of-range rows exit AFTER the staging barrier: the staging loop
  // above distributes all BQ rows across every warp (r = idx / W spans
  // warps whenever W > 64), so every warp must help stage sQ even when
  // S_q < BQ (a store-pinned oversized tile on a decode shape). This
  // kernel has no later barriers, so exiting here cannot deadlock.
  if (gi >= S_q) return;
  uint16_t* sKey = reinterpret_cast<uint16_t*>(sQ + BQ * W);
  int32_t* sIdx = reinterpret_cast<int32_t*>(sKey + (size_t)2 * BQ * kti * W);

  const float* qrow = &sQ[warp * W];  // this warp's query row (fp32)
  const int32_t* idx_i =
      indices + (size_t)gi * kv_group * topk;  // kv_group == 1 -> gi*topk

  // Lane-local slices of the output accumulator (strided by 64 over d_v).
  float o[kPerLane];
  for (int t = 0; t < kPerLane; ++t) o[t] = 0.0f;
  float row_max = -INFINITY;          // running max (same on every lane)
  float row_sum = 0.0f;               // running denominator (same on every lane)

  // Issue the global loads for key tile [tb, tb + tlen) into staging
  // buffer `sbuf`, then a warp barrier. Written as a code block (repeated
  // for the prologue tile and the in-loop prefetch) to keep device-lambda
  // support assumptions out of the picture. Staging is lane-strided (NOT
  // tid-strided): each warp stages its OWN slice, so the chunk index must
  // restart at 0 per warp or slices of warps w > 0 would leave their first
  // 64*w chunks unstaged (issue #80 audit fix; invisible at BQ == 1, wrong
  // for prefill S_q > 8).
#define VK_DSA_STAGE_TILE(sbuf_, tb_, tlen_)                                  \
  do {                                                                        \
    uint16_t* sdst =                                                          \
        sKey + (size_t)(((sbuf_) * BQ + warp) * kti) * W;                     \
    int32_t* idst = sIdx + ((sbuf_) * BQ + warp) * kti;                        \
    if (W % 8 == 0) {                                                         \
      const int rpp = W / 8; /* chunks (8 elements) per key row */            \
      for (int ch = lane; ch < (tlen_) * rpp; ch += 64) {                     \
        const int r = ch / rpp;                                               \
        const int c = (ch - r * rpp) * 8;                                     \
        const int32_t kv_idx = idx_i[(tb_) + r];                              \
        idst[r] = kv_idx;                                                     \
        /* one chunk = 8 bf16 = 16 B = uint4. The original uint2 copy moved */\
        /* only the FIRST 4 of every 8-element chunk, leaving the odd half */ \
        /* of every staged key row as uninitialized LDS garbage -- the     */ \
        /* nondeterministic split-LSE corruption of issue #80 (present in */  \
        /* both VK_DSA_DBPIPE modes; the overlap was never the culprit).   */  \
        uint4 v = {0u, 0u, 0u, 0u};                                           \
        if (kv_idx >= 0 && kv_idx < S_kv)                                     \
          v = *reinterpret_cast<const uint4*>(                                \
              kv + (size_t)kv_idx * W + c);                                   \
        *reinterpret_cast<uint4*>(sdst + (size_t)r * W + c) = v;              \
      }                                                                       \
    } else {                                                                  \
      for (int e = lane; e < (tlen_) * W; e += 64) {                          \
        const int r = e / W;                                                  \
        const int c = e - r * W;                                              \
        const int32_t kv_idx = idx_i[(tb_) + r];                              \
        idst[r] = kv_idx;                                                     \
        sdst[(size_t)r * W + c] =                                             \
            (kv_idx >= 0 && kv_idx < S_kv)                                    \
                ? kv[(size_t)kv_idx * W + c]                                   \
                : (uint16_t)0;                                                \
      }                                                                       \
    }                                                                         \
    VK_DSA_WARP_SYNC(); /* staged tile visible before it is consumed */       \
  } while (0)

  // Prologue: stage the first tile into buffer 0. In the no-pipeline
  // diagnostic mode (VK_DSA_DBPIPE=0) every tile is staged at the top of
  // its own iteration instead — stage-then-consume with zero overlap —
  // so a correctness difference between the modes isolates the prefetch
  // overlap from the staging/index arithmetic.
  if (pipeline) VK_DSA_STAGE_TILE(0, k0, (k1 - k0 < kti) ? (k1 - k0) : kti);

  int buf = 0;
  for (int tb = k0; tb < k1; tb += kti) {
    const int te = (k1 - tb < kti) ? k1 : (tb + kti);
    if (!pipeline) {
      VK_DSA_STAGE_TILE(0, tb, te - tb);
    } else if (te < k1) {  // uniform per warp; prefetch next tile into buf^1
      VK_DSA_STAGE_TILE(buf ^ 1, te, (k1 - te < kti) ? (k1 - te) : kti);
    }
    for (int k = tb; k < te; ++k) {
      const int tk = k - tb;
      const int32_t idx = sIdx[(buf * BQ + warp) * kti + tk];
      const uint16_t* rowp =
          sKey + (size_t)((buf * BQ + warp) * kti + tk) * W;
      const bool valid = (idx >= 0) && (idx < S_kv);

      // --- lane-local partial score (main + tail), then warp-reduce ---
      float partial = 0.0f;
      if (valid) {
        for (int t = 0; t < main_tiles; ++t) {
          const int d = lane + t * 64;
          if (d < dim) partial += qrow[d] * bf16_to_f32(rowp[d]);
        }
        for (int t = 0; t < tail_tiles; ++t) {  // 0 iters when tail_dim == 0
          const int d = lane + t * 64;
          if (d < tail_dim)
            partial += qrow[dim + d] * bf16_to_f32(rowp[dim + d]);
        }
      }
      float score = partial;
      for (int off = 32; off > 0; off >>= 1)  // warp-wide sum -> broadcast
        score += __shfl_xor(score, off);
      // Masked keys must contribute ZERO weight, so clamp their score to
      // -inf exactly as the CPU oracle does (else exp2(0-max)=1).
      score = valid ? (score * sm_scale) : -INFINITY;

      // --- online softmax update (lock-step; identical on every lane) ---
      const float new_max = fmaxf(row_max, score);
      const float exp_old =
          (row_max == -INFINITY) ? 0.0f : exp2f(row_max - new_max);
      const float exp_s =
          (score == -INFINITY) ? 0.0f : exp2f(score - new_max);
      row_sum = row_sum * exp_old + exp_s;
      for (int t = 0; t < v_tiles; ++t) o[t] *= exp_old;
      if (valid) {
        for (int t = 0; t < v_tiles; ++t) {
          const int d = lane + t * 64;
          if (d < d_v) o[t] += exp_s * bf16_to_f32(rowp[d]);
        }
      }
      row_max = new_max;
    }
    if (pipeline) buf ^= 1;
    VK_DSA_WARP_SYNC(); /* buffer fully consumed before its next restaging */
  }
#undef VK_DSA_STAGE_TILE
#undef VK_DSA_WARP_SYNC

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

// Issue #137: vectorized + batched serial partial (the two levers the
// perf record called for at full topk, where the db kernel's LDS staging
// strictly loses to TLP):
//
//   (a) Wider per-key loads. The serial kernel reads each 256-wide bf16 key
//       row as four 2-byte lane-strided loads (score) and re-reads them for
//       the V update. With tail_dim == 0 the K and V regions COINCIDE
//       ([0, dim) of the same kv row), so one 8-byte uint2 per lane (4 bf16,
//       lane-contiguous instead of lane-strided: 64 lanes x 4 elements
//       cover the row) serves score AND V -- one load instruction instead
//       of eight, perfectly coalesced (512 B/wavefront).
//   (b) Key batching in REGISTERS (no LDS -- that is the db kernel's job
//       and its occupancy tax). The serial kernel's per-key chain is two
//       dependent global round trips (index load -> address -> data load
//       -> use). This kernel hoists G keys' index + data loads into
//       registers BEFORE consuming them sequentially, so the round-trip
//       latency amortizes over G keys while the online-softmax update
//       sequence stays per-key sequential (identical masked-key semantics:
//       out-of-range/invalid group entries run the same -inf update).
//
// Per-key arithmetic is otherwise identical to the serial kernel (same
// exp2/fmax update sequence, same shfl reduction tree); the only numeric
// difference is the dot-product association across lanes (4 CONTIGUOUS
// elements per lane instead of 4 strided) -- bf16-tolerant vs the oracle
// exactly like every other lane mapping here.
//
// FUSE mode (lever (b) of the issue, the combine merge): instead of
// returning after the partial write, the warp fences its partial_out /
// partial_lse stores device-wide, atomically bumps a per-(gi,h) arrival
// counter (host memsets it to 0 on the stream before the launch), and the
// LAST arriving warp for that (gi,h) runs the exact log2-domain merge
// inline (same weights/order as dsa_sparse_fwd_combine_kernel: w_s =
// 2^(lse_s - m), fixed s = 0..split_kv-1 order) and writes out/lse
// directly. One launch + one part_out round trip saved at the split counts
// where the partial is now fast enough that launch overhead shows.
// threadFenceReduction pattern (CUDA samples): writer side
// __threadfence() before the device-scope atomicAdd; the last arriver's
// reads are safe because every other writer has already fenced.
template <int BQ, int G>
__global__ void dsa_sparse_fwd_partial_vec_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv,
    const int32_t* __restrict__ indices, float* __restrict__ part_out,
    float* __restrict__ part_lse, uint16_t* __restrict__ fused_out,
    float* __restrict__ fused_lse, int* __restrict__ counters,
    int S_q, int S_kv, int H, int dim, int topk, float sm_scale,
    int split_kv, int return_lse, int fuse) {
  static_assert(G >= 1 && (G & (G - 1)) == 0, "G must be a power of two");
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int s = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int gi = q_tile * BQ + warp;
  if (gi >= S_q) return;  // whole warp exits (no cross-warp state)

  const int W = dim;                    // tail_dim == 0 in this kernel
  const int d_v = dim;                  // tail_dim == 0: V = K region

  // Admitted shapes: tail_dim == 0 and dim a multiple of 4 with
  // dim <= 4*64, so lane `l` owns the CONTIGUOUS element group
  // [4l, 4l+4) of every 256-wide row and either the whole group is in
  // range or none of it is (no per-element masks on the hot path).
  const bool active = 4 * lane < dim;

  const int k0 = (int)(((long)topk * s) / split_kv);
  const int k1 = (int)(((long)topk * (s + 1)) / split_kv);

  extern __shared__ float sQ[];
  for (int idx = tid; idx < BQ * W; idx += BQ * 64) {
    const int r = idx / W, c = idx % W;
    const int gr = q_tile * BQ + r;
    sQ[idx] = (gr < S_q) ? bf16_to_f32(
                  q[((size_t)gr * H + h) * W + c])
                         : 0.0f;
  }
  __syncthreads();

  const float* qrow = &sQ[warp * W];
  const int32_t* idx_i = indices + (size_t)gi * topk;  // kv_group == 1

  // The query group this lane dots against is fixed across keys: hoist it.
  float qv[4];
  for (int j = 0; j < 4; ++j)
    qv[j] = active ? qrow[4 * lane + j] : 0.0f;

  float o[4];
  for (int j = 0; j < 4; ++j) o[j] = 0.0f;
  float row_max = -INFINITY;
  float row_sum = 0.0f;

  for (int g0 = k0; g0 < k1; g0 += G) {
    int32_t idxv[G];
    uint2 kd[G];
#pragma unroll
    for (int j = 0; j < G; ++j) {
      const int k = g0 + j;
      // Uniform per warp; the tail group treats missing entries as masked
      // keys (score -inf, zero weight) -- the serial kernel's semantics.
      idxv[j] = (k < k1) ? idx_i[k] : -1;
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
      kd[j] = make_uint2(0u, 0u);
      if (active && idxv[j] >= 0 && idxv[j] < S_kv)
        kd[j] = *reinterpret_cast<const uint2*>(
            kv + (size_t)idxv[j] * W + (size_t)4 * lane);  // 4 bf16 = 8 B
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
      if (g0 + j >= k1) break;  // tail group: nothing staged to consume
      const int32_t idx = idxv[j];
      const bool valid = (idx >= 0) && (idx < S_kv);

      // --- lane-local partial score (4 contiguous products), warp-reduce
      // --- (same xor tree as the serial kernel) ---
      const float k_a = bf16_to_f32((uint16_t)kd[j].x);
      const float k_b = bf16_to_f32((uint16_t)(kd[j].x >> 16));
      const float k_c = bf16_to_f32((uint16_t)kd[j].y);
      const float k_d = bf16_to_f32((uint16_t)(kd[j].y >> 16));
      float partial = 0.0f;
      if (valid && active) {
        partial = qv[0] * k_a;
        partial += qv[1] * k_b;
        partial += qv[2] * k_c;
        partial += qv[3] * k_d;
      }
      float score = partial;
      for (int off = 32; off > 0; off >>= 1)
        score += __shfl_xor(score, off);
      score = valid ? (score * sm_scale) : -INFINITY;

      // --- online softmax update (lock-step; identical to serial) ---
      const float new_max = fmaxf(row_max, score);
      const float exp_old =
          (row_max == -INFINITY) ? 0.0f : exp2f(row_max - new_max);
      const float exp_s =
          (score == -INFINITY) ? 0.0f : exp2f(score - new_max);
      row_sum = row_sum * exp_old + exp_s;
      for (int t = 0; t < 4; ++t) o[t] *= exp_old;
      if (valid && active) {  // V region == K region (tail_dim == 0)
        o[0] += exp_s * k_a;
        o[1] += exp_s * k_b;
        o[2] += exp_s * k_c;
        o[3] += exp_s * k_d;
      }
      row_max = new_max;
    }
  }

  // --- write the NORMALIZED per-split output + per-split lse (same
  // --- contract as the serial/db kernels) ---
  const float inv = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
  float* porow = part_out + (((size_t)gi * H + h) * split_kv + s) * d_v;
  for (int j = 0; j < 4; ++j) {
    const int d = 4 * lane + j;
    if (d < d_v) porow[d] = o[j] * inv;
  }
  if (lane == 0)
    part_lse[((size_t)gi * H + h) * split_kv + s] =
        (row_sum > 0.0f) ? (row_max + log2f(row_sum)) : -INFINITY;

  if (!fuse) return;

  // --- FUSE: last warp for this (gi, h) merges the splits inline ---
  __threadfence();  // partial stores visible device-wide before the count
  int arrivals = 0;
  if (lane == 0) arrivals = atomicAdd(&counters[(size_t)gi * H + h], 1);
  // Broadcast lane 0's value with the xor tree (gfx942 HIP: no __shfl/
  // __syncwarp; summing 63 zero contributions reproduces the broadcast).
  for (int off = 32; off > 0; off >>= 1)
    arrivals += __shfl_xor(arrivals, off);
  if (arrivals != split_kv - 1) return;  // uniform per warp

  __threadfence();  // every other writer fenced before us: reads are safe
  const size_t base = ((size_t)gi * H + h) * split_kv;
  float m = -INFINITY;
  for (int sp = 0; sp < split_kv; ++sp) m = fmaxf(m, part_lse[base + sp]);
  uint16_t* orow = fused_out + ((size_t)gi * H + h) * d_v;
  if (m == -INFINITY) {
    // Every selected key of every split is masked -> zero row, lse = -inf
    // (same contract as dsa_sparse_fwd_combine_kernel).
    for (int j = 0; j < 4; ++j) {
      const int d = 4 * lane + j;
      if (d < d_v) orow[d] = 0;
    }
    if (lane == 0 && return_lse) fused_lse[(size_t)gi * H + h] = -INFINITY;
    return;
  }
  // Exact log2-domain merge, same weights and fixed s-order as the combine
  // kernel; per element d the accumulation sequence over s is identical.
  float fo[4];
  for (int j = 0; j < 4; ++j) fo[j] = 0.0f;
  float denom = 0.0f;
  for (int sp = 0; sp < split_kv; ++sp) {
    const float l = part_lse[base + sp];
    const float w = (l == -INFINITY) ? 0.0f : exp2f(l - m);
    denom += w;
    const float* po = part_out + (base + sp) * d_v;
    for (int j = 0; j < 4; ++j) {
      const int d = 4 * lane + j;
      if (d < d_v) fo[j] += w * po[d];
    }
  }
  const float finv = 1.0f / denom;  // denom >= 1 (max-weight split)
  for (int j = 0; j < 4; ++j) {
    const int d = 4 * lane + j;
    if (d < d_v) orow[d] = f2bf(fo[j] * finv);
  }
  if (lane == 0 && return_lse)
    fused_lse[(size_t)gi * H + h] = m + log2f(denom);
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

// Issue #80: double-buffered partial launch. Never launched directly by
// callers: use launch_partial_auto below so the serial kernel stays the
// fallback for shapes that cannot stage even one key.
template <int BQ>
bool launch_partial_db(const void* q, const void* kv, const void* indices,
                       void* part_out, void* part_lse, int S_q, int S_kv,
                       int H, int dim, int tail_dim, int topk, int kv_group,
                       float sm_scale, int split_kv, int& kti,
                       hipStream_t stream) {
  // Issue #80 A/B diagnostics (bench-only, same pattern as VK_MOE_KTIME):
  // VK_DSA_DB=0 forces the serial kernel (this returns false, so
  // launch_partial_auto falls back); VK_DSA_DB=2 forces the db kernel,
  // bypassing the occupancy gate below (correctness coverage for both
  // kernels at every shape); VK_DSA_KTI=<n> overrides the auto-sized
  // keys-per-tile (0 = auto). Read once per process.
  static const int db_mode = [] {
    const char* e = std::getenv("VK_DSA_DB");
    return e ? std::atoi(e) : 1;
  }();
  static const int kti_ovr = [] {
    const char* e = std::getenv("VK_DSA_KTI");
    return e ? std::atoi(e) : 0;
  }();
  static const int dbpipe = [] {
    const char* e = std::getenv("VK_DSA_DBPIPE");
    return e ? std::atoi(e) : 1;
  }();
  if (db_mode == 0) return false;
  const int W = dim + tail_dim;
  const size_t shq = (size_t)BQ * (size_t)W * 4u;                // sQ
  const size_t per_key = (size_t)2 * BQ * ((size_t)W * 2u + 4u); // 2 key bufs + idx
  // gfx942 caps no-attribute dynamic shmem at 48 KB/block; budget 44 KB
  // with margin. Auto picks kti <= 8 (more adds little once the chain is
  // pipelined) but VK_DSA_KTI can go deeper for A/B (up to the budget).
  constexpr size_t kCap = 44u * 1024u;
  if (shq >= kCap) return false;
  const int budget = (int)((kCap - shq) / per_key);
  int new_kti = (kti_ovr > 0) ? kti_ovr : 8;
  if (new_kti > budget) new_kti = budget;
  if (new_kti < 1) return false;
  kti = new_kti;
  const size_t shmem = shq + (size_t)kti * per_key;
  // Issue #80 occupancy gate. The staged kernel buys ILP with LDS: at the
  // GLM decode width (BQ=1, W=256, kti=8) a block wants 9.3 KB, capping
  // residency at 7 blocks/CU (64 KB LDS/CU on gfx942) vs the serial
  // kernel's 1 KB (wave-limited, ~32 blocks/CU). Measured crossover
  // (docs/performance/dsa/gfx942.md, 4 decode shapes x split sweep): the
  // db kernel wins while the WHOLE grid fits one LDS-limited residency
  // wave (all four shapes at split<=16, -5..-13%) and strictly loses once
  // it spills (split>=32 rows, up to +43% at topk=2048 split=64) -- with
  // thousands of tiny blocks, TLP already hides the serial chain and
  // staging is pure overhead. Engage the db kernel only in its win region;
  // VK_DSA_DB=2 bypasses this gate for diagnostics.
  static const int num_cu = [] {
    int dev = 0, n = 0;
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount,
                              dev) != hipSuccess || n <= 0)
      return 228;  // gfx942 CU count, same default as dsa_sparse_fwd_split_for
    return n;
  }();
  if (db_mode == 1) {
    const int blocks_per_cu = (int)(64u * 1024u / shmem);  // LDS 64 KB/CU
    const long total_blocks =
        (long)((S_q + BQ - 1) / BQ) * (long)H * (long)split_kv;
    if ((long)blocks_per_cu * num_cu < total_blocks) return false;
  }
  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, split_kv);
  dsa_sparse_fwd_partial_db_kernel<BQ><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const uint16_t*>(q),
      reinterpret_cast<const uint16_t*>(kv),
      reinterpret_cast<const int32_t*>(indices),
      reinterpret_cast<float*>(part_out), reinterpret_cast<float*>(part_lse),
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, sm_scale, split_kv, kti,
      dbpipe);
  return true;
}

// Issue #137: arrival counters for the vec kernel's FUSED combine (lever
// (b)): one int per (gi, h), memset to 0 on the launch stream before every
// fused launch (so no cross-launch state and no correctness dependence on
// the allocator's contents). Grown on demand and never shrunk; retired
// generations are intentionally NOT hipFree'd (a pending kernel could still
// reference an old buffer; the leak is bounded by the number of grow steps
// and is tiny: S_q*H*4 B each).
namespace {
int* vec_combine_counters(size_t n, hipStream_t stream) {
  static int* buf = nullptr;
  static size_t cap = 0;
  if (n == 0) return nullptr;
  if (buf == nullptr || n > cap) {
    int* nb = nullptr;
    if (hipMalloc(&nb, n * sizeof(int)) != hipSuccess) return nullptr;
    if (hipMemsetAsync(nb, 0, n * sizeof(int), stream) != hipSuccess) {
      hipFree(nb);
      return nullptr;
    }
    buf = nb;
    cap = n;
  } else {
    hipMemsetAsync(buf, 0, n * sizeof(int), stream);
  }
  return buf;
}
}  // namespace

// Issue #137: vectorized + register-batched serial partial (lever (a)),
// with the optional FUSED combine (lever (b)). Returns:
//   0 -- not launched (shape not admitted / disabled): caller falls back,
//   1 -- launched, combine still required (separate kernel or caller math),
//   2 -- launched with the combine FUSED in: no combine launch needed.
// Admission: tail_dim == 0 (K and V regions coincide, the uint2 load serves
// both), dim % 4 == 0 and dim <= 256 (one contiguous 4-element group per
// lane). Everything else (DSv3 widths, plain forward, db kernel) unchanged.
template <int BQ>
int launch_partial_vec(const void* q, const void* kv, const void* indices,
                       void* part_out, void* part_lse, void* out, void* lse,
                       int S_q, int S_kv, int H, int dim, int tail_dim,
                       int topk, float sm_scale, bool return_lse,
                       int split_kv, hipStream_t stream) {
  // Issue #137 A/B diagnostics (same pattern as VK_DSA_DB):
  //   VK_DSA_VEC=0  never (legacy serial/db dispatch)
  //   VK_DSA_VEC=1  auto: engage wherever the shape is admitted
  //   VK_DSA_VEC=2  force (identical to auto since the gate was removed;
  //                 kept for backward compat with the 641060/641089 runs)
  // VK_DSA_FUSE=1 selects the fused combine (default OFF: the unfused
  // separate-combine launch measured 20-25% FASTER at the optimum and 2x
  // faster at high split -- the fence + atomic + last-warp merge tail
  // costs more than a launch); VK_DSA_VECG overrides the register batch
  // size G (1/2/4/8; default 1 = vectorization only, no batching -- the
  // A/B that isolates the two effects; G>1 measured slower).
  static const int vec_mode = [] {
    const char* e = std::getenv("VK_DSA_VEC");
    return e ? std::atoi(e) : 1;
  }();
  static const int fuse_mode = [] {
    const char* e = std::getenv("VK_DSA_FUSE");
    return e ? std::atoi(e) : 0;
  }();
  static const int g_ovr = [] {
    const char* e = std::getenv("VK_DSA_VECG");
    return e ? std::atoi(e) : 1;
  }();
  if (vec_mode == 0) return 0;
  if (tail_dim != 0 || dim <= 0 || dim % 4 != 0 || dim > 4 * 64) return 0;
  // No occupancy/TLP gate in auto mode: the UNFUSED vectorized kernel
  // (G=1, separate combine launch) measured >= the serial AND db kernels
  // at EVERY admitted (shape, split) point across the issue-#137 sweeps
  // (jobs 641060 + 641089, MI300A): topk=2048 split=32 -> 99.0 us /
  // 679 GB/s vs serial's best 168.4 (split=64) and db's 208+; topk=256
  // split=8 -> 40.0 vs db 55.1; topk=128 split=8 -> 31.2 vs db 43.5;
  // even at 1 key/split it ties serial (182.6 vs 183.1 at topk=128
  // split=128). The FUSED variant (VK_DSA_FUSE=1) is the exception: its
  // fence + atomic + last-warp merge tail costs 20-25% at the optimum
  // (132.2 vs 107.4 us at topk=2048 split=16) and 2x at high split
  // (243.3 vs 127.5 at split=64), so it ships OFF.
  const int g = (g_ovr == 1 || g_ovr == 2 || g_ovr == 4 || g_ovr == 8)
                    ? g_ovr
                    : 1;

  int fuse = fuse_mode ? 1 : 0;
  int* counters = nullptr;
  if (fuse) {
    counters = vec_combine_counters((size_t)S_q * H, stream);
    if (counters == nullptr) fuse = 0;  // alloc failure: plain partial path
  }

  dim3 block(BQ * 64);
  dim3 grid((S_q + BQ - 1) / BQ, H, split_kv);
  const size_t shmem = (size_t)BQ * dim * sizeof(float);
#define VK_LAUNCH_VEC(GG)                                                      \
  do {                                                                         \
    dsa_sparse_fwd_partial_vec_kernel<BQ, GG><<<grid, block, shmem, stream>>>(  \
        reinterpret_cast<const uint16_t*>(q),                                  \
        reinterpret_cast<const uint16_t*>(kv),                                 \
        reinterpret_cast<const int32_t*>(indices),                             \
        reinterpret_cast<float*>(part_out),                                    \
        reinterpret_cast<float*>(part_lse),                                    \
        reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(lse),       \
        counters, S_q, S_kv, H, dim, topk, sm_scale, split_kv,                 \
        return_lse ? 1 : 0, fuse);                                             \
  } while (0)
  switch (g) {
    case 1: VK_LAUNCH_VEC(1); break;
    case 2: VK_LAUNCH_VEC(2); break;
    case 8: VK_LAUNCH_VEC(8); break;
    default: VK_LAUNCH_VEC(4); break;
  }
#undef VK_LAUNCH_VEC
  return fuse ? 2 : 1;
}

// Issue #80/#137: dispatch the split partial: the vectorized kernel first
// wherever the shape is admitted (it measured >= db + serial at every
// admitted point, jobs 641060 + 641089), then the double-buffered kernel
// (its measured win region: DSv3 / tail_dim != 0 widths at low split),
// then the serial dsa_sparse_fwd_partial_kernel as the fallback. Returns
// true when a separate combine launch is still required.
template <int BQ>
bool launch_partial_auto(const void* q, const void* kv, const void* indices,
                         void* part_out, void* part_lse, int S_q, int S_kv,
                         int H, int dim, int tail_dim, int topk, int kv_group,
                         float sm_scale, int split_kv, bool return_lse,
                         void* out, void* lse,
                         hipStream_t stream = nullptr) {
  const int vec = launch_partial_vec<BQ>(q, kv, indices, part_out, part_lse,
                                         out, lse, S_q, S_kv, H, dim,
                                         tail_dim, topk, sm_scale,
                                         return_lse, split_kv, stream);
  if (vec == 2) return false;  // fused: no combine launch
  if (vec == 1) return true;   // partial only: combine still needed
  int kti = 0;
  if (launch_partial_db<BQ>(q, kv, indices, part_out, part_lse, S_q, S_kv, H,
                            dim, tail_dim, topk, kv_group, sm_scale,
                            split_kv, kti, stream))
    return true;
  launch_partial<BQ>(q, kv, indices, part_out, part_lse, S_q, S_kv, H, dim,
                     tail_dim, topk, kv_group, sm_scale, split_kv, stream);
  return true;
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
  if (!dsa_tile_from_store(S_q, H, dim, tail_dim, topk, &bq, &bi, &ii))
    ::vkernels::kernels::dsa_config_for(S_q, H, dim, topk, &bq, &th, &bi, &ii);
  (void)th;
  // The partial kernel now double-buffers its key tiles in LDS (issue
  // #80) or runs the vectorized register-batched serial variant with the
  // optional fused combine (issue #137); block_I/inner_iter remain
  // cached-pipeline params. launch_partial_auto reports whether the
  // separate combine launch is still required (the fused vec path already
  // wrote out/lse).
  (void)block_I;
  (void)inner_iter;
  bool need_combine = true;
  if (bq <= 1)
    need_combine = launch_partial_auto<1>(
        q, kv, indices, partial_out, partial_lse, S_q, S_kv, H, dim,
        tail_dim, topk, kv_group, sm_scale, split_kv, return_lse, out, lse,
        static_cast<hipStream_t>(stream));
  else if (bq <= 2)
    need_combine = launch_partial_auto<2>(
        q, kv, indices, partial_out, partial_lse, S_q, S_kv, H, dim,
        tail_dim, topk, kv_group, sm_scale, split_kv, return_lse, out, lse,
        static_cast<hipStream_t>(stream));
  else if (bq <= 4)
    need_combine = launch_partial_auto<4>(
        q, kv, indices, partial_out, partial_lse, S_q, S_kv, H, dim,
        tail_dim, topk, kv_group, sm_scale, split_kv, return_lse, out, lse,
        static_cast<hipStream_t>(stream));
  else
    need_combine = launch_partial_auto<8>(
        q, kv, indices, partial_out, partial_lse, S_q, S_kv, H, dim,
        tail_dim, topk, kv_group, sm_scale, split_kv, return_lse, out, lse,
        static_cast<hipStream_t>(stream));
  if (!need_combine) return;
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

// ---------------------------------------------------------------------------
// fp8-MFMA variant (issue #81, "tiled-key / MFMA-fp8 prefetch"). The bf16
// MFMA variant above still stages every K-tile through LDS (sK[B][kBK]
// reloaded + __syncthreads per K-tile) and every Q through a transposed
// bf16 sQt[D][H], because the bf16 16x16x16 fragment carries fp8 values
// only after an fp8->fp32->bf16 conversion. gfx942's Matrix Core consumes
// fp8 e4m3fnuz DIRECTLY: __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8
// reduces K=32 per instruction -- 2x fewer MFMAs per D than the bf16
// variant -- and its A/B fragments are 8 BYTES per lane, so the fragment
// IS one 8-byte global load of raw cache bytes. That removes the ENTIRE
// staging path: no sQt transform, no sK tile, no LDS round-trips, no
// __syncthreads on the hot loop. Shared memory collapses to the per-head
// gate alone (H*4 bytes -- 256 B at H=64 vs the bf16 variant's 25,088 B;
// the per-token scales are read straight from each page tail as ONE 16B
// load per lane -- j = row_base+kq*4 .. +3 is contiguous), so the 64 KB
// cap is a non-constraint and shapes the other variants refuse (H=128 at
// D=192) are admitted.
//
// Fragment layouts (empirically decoded on gfx942, issue #81 probe
// record; confirmed end-to-end by test_dsa_topk_correct.hip variant 4):
//   A/B fragments: lane l byte j = X[l & 15][(l >> 4) * 8 + j]
//   C fragment:    c[i] = C[4 * (l >> 4) + i][l & 15]
// i.e. the SAME C-fragment geometry as the bf16 16x16x16 instruction, so
// the gated epilogue below is byte-identical to the bf16 variant's.
// The 4-operand inline-asm form of this instruction assembles with
// garbage cbsz/blgp modifier defaults on ROCm 6.3 (silently scrambled
// operand semantics -- the probe record), so the 7-operand builtin with
// explicit zeros is REQUIRED.
//
// Numerics: fp8 e4m3fnuz products have an 8-bit significand, exact in
// fp32; the instruction accumulates its K=32 reduction in fp32 and the
// per-page dots are fp32-added over D/32 chunk results -- the same
// residual class as the bf16 variant (fp32 reassociation only), within
// the 1e-3 cross-check. The 0x7F/0xFF NaN byte encodings never occur in
// real cache data and the harness excludes them.
//
// Per wavefront w (one per 16 KV tokens), per page, per head fragment nf:
// D/32 MFMAs, each with two independent 8-byte loads (K from the page,
// Q from the per-batch tile -- the SAME Q addresses every page, L1/L2
// resident after the first). The loads are independent of the MFMA chain,
// so the compiler issues the next page's/chunk's loads while MFMAs are in
// flight (the "prefetch" of the issue title -- measured, not assumed:
// see docs/performance/dsa-topk/gfx942.md).
__global__ void dsa_topk_logits_kernel_mfma_fp8(
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
  const int lane = tid & 63;           // in-wavefront lane
  const int m    = lane & 15;          // row within the 16-row fragment
  const int kq   = lane >> 4;          // 0..3 -> which 8-K byte chunk
  const int row_base = warp * 16;      // this warp's 16-row fragment (j)

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  extern __shared__ float smem[];
  float* sGate = smem;                 // H (written once per block)

  const float* gp = weight + (size_t)b * H;
  for (int h = tid; h < H; h += kTh) sGate[h] = gp[h];
  __syncthreads();

  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  const int kt_n = D >> 5;             // D/32 fp8 MFMAs per page per fragment

  // Accumulators for [0, kNF) fragments. kMaxNF = 8 covers every admitted
  // shape (fits_lds_mfma_fp8 requires H <= 128); the predicated full unroll
  // keeps every acc[nf] index compile-time so the accumulators stay in
  // register (a runtime-bound loop here would spill them to scratch).
  constexpr int kMaxNF = 8;
  v4f acc[kMaxNF];

  // Per-page scale slot: the 4 token scales THIS lane's accumulator rows
  // need (j = row_base + kq*4 .. +3 -- contiguous, 16B-aligned because
  // B*D is a multiple of 32 and each j-group starts at a multiple of 4).
  // Read straight from the page tail -- NO sKscale staging, so the page
  // loop runs with ZERO barriers (the bf16 variant needs one per K-tile).
  const int j0 = row_base + kq * 4;

  // One-entry-deep page-table prefetch: the page POINTER of page pg+1 is
  // loaded while page pg computes, collapsing the serial
  // table->kbase->K-load dependency chain to one L2 hop per page.
  int32_t page = page_table[(size_t)b * max_table_len + i_start];

  for (int it = 0; it < n_iters; ++it) {
    const int pg = i_start + it;
    // Issue next page's table load NOW -- it is independent of everything
    // below, so it flies during this page's MFMA chain.
    int32_t page_next = 0;
    if (it + 1 < n_iters)
      page_next = page_table[(size_t)b * max_table_len + pg + 1];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));
    // This lane's 4 token scales: j = row_base + kq*4 .. +3, contiguous and
    // 16B-aligned (B*D is a multiple of 32). Read straight from the page
    // tail -- no sKscale staging, so the page loop runs with ZERO barriers.
    const v4f kscale4 = *reinterpret_cast<const v4f*>(kbase + B * D +
                                                      (size_t)j0 * 4);

    #pragma unroll
    for (int nf = 0; nf < kMaxNF; ++nf) acc[nf] = (v4f){0.0f, 0.0f, 0.0f, 0.0f};

    // --- the whole page's [16 tokens] x [H heads] dot matrix: kt_n x kNF
    // MFMAs, no LDS, no barriers. The A fragment (K, the M side) is this
    // lane's 8 bytes of token row row_base+m at D-offset t*32+kq*8; the B
    // fragment (Q, the N side) is head nf*16+m at the SAME offset, so the
    // two loads share their address form. acc[nf][i] accumulates
    // dot[j = row_base + kq*4 + i, h = nf*16 + m].
    for (int t = 0; t < kt_n; ++t) {
      const size_t doff = (size_t)t * 32 + (size_t)kq * 8;
      long a;
      __builtin_memcpy(&a, kbase + (size_t)(row_base + m) * D + doff, 8);
      #pragma unroll
      for (int nf = 0; nf < kMaxNF; ++nf) {
        if (nf < kNF) {
          long bq;
          __builtin_memcpy(&bq, qp + (size_t)(nf * 16 + m) * D + doff, 8);
          acc[nf] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
              a, bq, acc[nf], 0, 0, 0);
        }
      }
    }

    // --- per-token gated reduce (fp32) + k_scale, then write -- identical
    // geometry to the bf16-MFMA variant (same C fragment).
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float partial = 0.0f;
      #pragma unroll
      for (int nf = 0; nf < kMaxNF; ++nf) {
        if (nf < kNF) {
          float d = acc[nf][i];
          if (d < 0.0f) d = 0.0f;             // ReLU (max(0, raw))
          partial += sGate[nf * 16 + m] * d;
        }
      }
      for (int off = 1; off < 16; off <<= 1)
        partial += __shfl_xor(partial, off);
      if ((tid & 15) == 0) {
        const int j = j0 + i;               // == row_base + kq*4 + i
        const int tk = pg * B + j;
        if (tk < seq_len)
          out[(size_t)b * max_seq_len + tk] = kscale4[i] * partial;
      }
    }
    page = page_next;
  }
}

// dsa_topk_logits_with_variant (explicit). `variant`: 0 = auto
// (mfma-fp8 -> mfma -> fp32q -> fp8q -> refuse); 1 = fp32-Q; 2 = fp8-Q;
// 3 = mfma; 4 = mfma-fp8. An explicit variant that does NOT fit its LDS
// cap is refused with a stderr diagnostic + no-op (the caller zeroed
// `out`, which is what stays) rather than launch a block gfx942's driver
// would drop -- mirrors the auto path's contract.
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
  const bool want_mfma8 = (variant == 4) || (variant == 0);

  // FASTEST path: the fp8-MFMA kernel (issue #81) -- raw-fp8 fragments
  // straight from global (no staging, no barriers, K=32 per instruction).
  // Admits every bf16-MFMA shape (D%64 => D%32) plus more (H=128 at
  // D=192), and is picked FIRST by auto. See
  // dsa.hpp::dsa_topk_logits_fits_lds_mfma_fp8.
  if (want_mfma8 && dsa_topk_logits_fits_lds_mfma_fp8(num_heads, head_dim,
                                                      block)) {
    const int shmem = num_heads * (int)sizeof(float);  // sGate only
    dim3 blockdim((block / 16) * 64);
    dsa_topk_logits_kernel_mfma_fp8<<<grid, blockdim, shmem, 0>>>(
        reinterpret_cast<const uint8_t*>(q_fp8),
        reinterpret_cast<const uint8_t*>(kvcache_u8),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const int32_t*>(seq_lens),
        reinterpret_cast<const int32_t*>(page_table),
        reinterpret_cast<float*>(out),
        num_heads, head_dim, block, max_table_len, max_seq_len, split_kv);
    return;
  }

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
      "non-optin dynamic-LDS cap under ALL of mfma-fp8 (needs H%%16==0, H<=128, "
      "D%%32==0, B%%16==0, B<=256), mfma (%d B), fp32-Q (%d B) and fp8-Q (%d B). "
      "Refusing (output left as the caller provided). See "
      "dsa.hpp::dsa_topk_logits_fits_lds{,_fp8q,_mfma,_mfma_fp8}.\n",
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
  if (variant != 0 && variant != 1 && variant != 2 && variant != 3 &&
      variant != 4) {
    std::fprintf(stderr,
        "vk_hip_dsa_topk_logits_with_variant: bad variant=%d "
        "(want 0=auto, 1=fp32q, 2=fp8q, 3=mfma, 4=mfma-fp8). Refusing.\n",
        variant);
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
