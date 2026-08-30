// vkernels/kernels/moe_fused.hip
//
// MXFP4 fused-MoE grouped GEMM — HIP kernel for AMD gfx942/gfx90a.
//
// Uses the clang MFMA builtin __builtin_amdgcn_mfma_f32_16x16x16bf16_1k,
// which handles the VGPR fragment layout correctly across ROCm versions.
//
// Verified fragment layout (empirically confirmed on gfx90a):
//   A: m = lane % 16, k0 = (lane/16)*4, a[i] = A[m][k0+i]
//   B: n = lane % 16, k0 = (lane/16)*4, b[i] = B[k0+i][n]
//   C: col = lane % 16, row = (lane/16)*4 + i, c[i] = C[row][col]
//
// Strategy (matches xkernels 2-stage fused-MoE):
//   Stage 0: gate_up GEMM + SwiGLU → act [EM, ispp] bf16
//            (gate and up accumulators held in fp32; single bf16 rounding)
//   Stage 1: down GEMM + routed combine → out [M, hidden] fp32
//
// Each block is 64 threads (1 wavefront); the N dimension is split across
// blockIdx.z (4 z-blocks, each a 16-column slice).

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/moe.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#  include "vkernels/kernels/moe_device.hip"  // shared bf16<->f32 + fp4 helpers

namespace vkernels::kernels::hip {

typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v4s __attribute__((ext_vector_type(4)));

static constexpr int kBM = 16,  kBN = 64, kBK = 64;
static constexpr int kGroupSize = 32;
static constexpr int kMfmaK    = 16;
static constexpr int kWarps    = 4;
static constexpr int kWhreads  = 64;

// Prefill tile config (M > 32): 64x64 tiles, 256 threads (4 wavefronts).
// A 64-row M-tile reuses each dequantized B column across 64 rows instead of
// 16.  BN is 64, not 128: BN=128 (24 KB LDS + 64 accumulator VGPRs) caps the
// kernel at 2 blocks/CU, while BN=64 (16 KB LDS + 32 accumulator VGPRs) lifts
// it to 3 blocks/CU — measured ~1.5x faster despite halving the B-tile reuse.
static constexpr int kBM_pf = 64, kBN_pf = 64, kBK_pf = 64;
static constexpr int kThreads_pf = 256;

// ======================================================================
//  Dequant helpers
// ======================================================================
__device__ __forceinline__ float ue8m0_decode(uint8_t s) {
  if (s == 0xFF) return 0.0f;
  float f;
  uint32_t bits = static_cast<uint32_t>(s) << 23;
  __builtin_memcpy(&f, &bits, sizeof(float));
  return f;
}

// Apply the gate/up epilogue. `g`/`u` are the bias-adjusted accumulators;
// `activation` selects SwiGLU (clamp + silu) or Kimi-K3 SiTU (situ_and_mul:
// tanh softcaps, no swiglu_limit clamp).
__device__ __forceinline__ float epilogue_act(
    int activation, float g, float u, float beta, float linear_beta,
    float limit) {
  if (activation == kSiTU) {
    float sig = 1.0f / (1.0f + expf(-g));
    float gate_out = beta * tanhf(g / beta) * sig;
    float up_out = (linear_beta > 0.0f) ? linear_beta * tanhf(u / linear_beta)
                                        : u;
    return gate_out * up_out;
  }
  if (limit > 0.0f) {
    if (g > limit) g = limit;
    if (u > limit) u = limit;
    if (u < -limit) u = -limit;
  }
  float silu_g = g / (1.0f + expf(-g));
  return silu_g * u;
}

// Load the 16x64 A tile into shared memory (stage 0: input activations).
// 64 threads; each thread loads 16 bf16 = 4 uint2 reads.
// sorted_ids holds FLAT topk indices (token*top_k + sel).
// `k_base` is the column offset into each A row where the calling rank's
// weight shard starts (0 for the single-rank fused call; r*hidden/tp for a
// TP rank consuming a w13 shard of width hidden/tp).
__device__ __forceinline__ void load_a_tile(
    uint16_t sA[kBM][kBK], const uint16_t* __restrict__ A,
    const int32_t* __restrict__ sorted_ids, int tk_base,
    int k_start, int stride, int M, int top_k, int lane, int k_base = 0) {
  int row = lane % kBM;
  int base_col = (lane / kBM) * (kBK / (kWhreads / kBM));
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  if (flat < N) {
    int token = flat / top_k;
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      const uint32_t* src = (const uint32_t*)(&A[(size_t)token * stride + k_base + k_start + col_off]);
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      uint2 v = *(const uint2*)(src);
      dst[0] = v.x; dst[1] = v.y;
    }
  } else {
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      dst[0] = 0; dst[1] = 0;
    }
  }
}

// Load the act tile [EM, ispp] into shared memory, indexed by SORTED ROW
// (tk_base + row).  Each expert's intermediate output occupies its own
// sorted row, so top_k>1 does not overwrite.
// `k_base` is the column offset into each act row where the calling rank's
// w2 shard starts (0 for the fused call; r*ispp/tp for a TP rank).
__device__ __forceinline__ void load_act_tile(
    uint16_t sA[kBM][kBK], const uint16_t* __restrict__ act,
    const int32_t* __restrict__ sorted_ids, int tk_base,
    int k_start, int ispp, int M, int top_k, int lane, int k_base = 0) {
  int row = lane % kBM;
  int base_col = (lane / kBM) * (kBK / (kWhreads / kBM));
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  if (flat < N) {
    const uint16_t* src_row = &act[(size_t)(tk_base + row) * ispp + k_base + k_start];
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      const uint32_t* src = (const uint32_t*)(src_row + col_off);
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      uint2 v = *(const uint2*)(src);
      dst[0] = v.x; dst[1] = v.y;
    }
  } else {
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      dst[0] = 0; dst[1] = 0;
    }
  }
}

// Dequant a packed fp4 weight column slice → sB[kBK][16].
// packed/scale point at the [n_global, 0] origin; k_start selects the K tile.
__device__ __forceinline__ void dequant_b_16cols(
    uint16_t sB[kBK][16],
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
    int k_start, int stride_packed, int stride_scale_n, int lane) {
  int n = lane % 16;
  int kp_base = lane / 16;
  constexpr int kPackedK = kBK/2, kKpStep = kWhreads/16;
  constexpr int kBytesPerThread = (kPackedK * 16) / kWhreads;
  int kp_off = k_start / 2;
  int sc_off = k_start / kGroupSize;
  for (int it = 0; it < kBytesPerThread; ++it) {
    int kp = kp_base + it * kKpStep;
    if (kp >= kPackedK) break;
    uint8_t pb = packed[n * stride_packed + kp_off + kp];
    int gi = (kp * 2) / kGroupSize;
    uint8_t sval = scale[n * stride_scale_n + sc_off + gi];
    float s = ue8m0_decode(sval);
    uint32_t flo = fp4nib_to_f32bits(pb & 0x0F);
    float vlo; __builtin_memcpy(&vlo, &flo, sizeof(float));
    vlo *= s; __builtin_memcpy(&flo, &vlo, sizeof(float));
    sB[kp*2][n] = f32bits_to_bf16(flo);
    uint32_t fhi = fp4nib_to_f32bits((pb>>4) & 0x0F);
    float vhi; __builtin_memcpy(&vhi, &fhi, sizeof(float));
    vhi *= s; __builtin_memcpy(&fhi, &vhi, sizeof(float));
    sB[kp*2+1][n] = f32bits_to_bf16(fhi);
  }
}

// Dequant a packed fp4 weight column slice → sB[kBK][16], K-MAJOR layout.
// packed is [E, 2, hidden/2, ispp] and scale is [E, 2, hidden/kGroupSize,
// ispp]: the N (column) dimension is the FAST, contiguous one, so the 16
// columns owned by this block (lane % 16) read ONE coalesced 16-byte cache
// line per (kp) instead of 16 scattered cache lines in the N-major layout
// above (issue #41, item 4).  `stride_k` = ispp (the number of columns).
__device__ __forceinline__ void dequant_b_16cols_kmajor(
    uint16_t sB[kBK][16],
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
    int k_start, int stride_k, int stride_scale_k, int lane) {
  int n = lane % 16;
  int kp_base = lane / 16;
  constexpr int kPackedK = kBK/2, kKpStep = kWhreads/16;
  constexpr int kBytesPerThread = (kPackedK * 16) / kWhreads;
  int kp_off = k_start / 2;
  int sc_off = k_start / kGroupSize;
  for (int it = 0; it < kBytesPerThread; ++it) {
    int kp = kp_base + it * kKpStep;
    if (kp >= kPackedK) break;
    uint8_t pb = packed[(kp_off + kp) * stride_k + n];
    int gi = (kp * 2) / kGroupSize;
    uint8_t sval = scale[(sc_off + gi) * stride_scale_k + n];
    float s = ue8m0_decode(sval);
    uint32_t flo = fp4nib_to_f32bits(pb & 0x0F);
    float vlo; __builtin_memcpy(&vlo, &flo, sizeof(float));
    vlo *= s; __builtin_memcpy(&flo, &vlo, sizeof(float));
    sB[kp*2][n] = f32bits_to_bf16(flo);
    uint32_t fhi = fp4nib_to_f32bits((pb>>4) & 0x0F);
    float vhi; __builtin_memcpy(&vhi, &fhi, sizeof(float));
    vhi *= s; __builtin_memcpy(&fhi, &vhi, sizeof(float));
    sB[kp*2+1][n] = f32bits_to_bf16(fhi);
  }
}

// Run the K=64 MFMA loop (4 steps of K=16), accumulating into acc.
__device__ __forceinline__ void mfma_k64(
    v4f& acc, const uint16_t sA[kBM][kBK], const uint16_t sB[kBK][16],
    int lane) {
  for (int mf = 0; mf < kBK / kMfmaK; ++mf) {
    int m = lane % 16;
    int k0 = mf * kMfmaK + (lane / 16) * 4;
    int n = lane % 16;
    v4s a = { (short)sA[m][k0], (short)sA[m][k0+1],
              (short)sA[m][k0+2], (short)sA[m][k0+3] };
    v4s b = { (short)sB[k0][n], (short)sB[k0+1][n],
              (short)sB[k0+2][n], (short)sB[k0+3][n] };
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
  }
}

// ======================================================================
//  Prefill helpers (BM=64, BN=64, 256 threads)
// ======================================================================

// Load the 64x64 A/act tile. 256 threads; each thread loads 16 bf16 (4 uint2).
__device__ __forceinline__ void load_a_tile_pf(
    uint16_t sA[kBM_pf][kBK_pf], const uint16_t* __restrict__ src,
    const int32_t* __restrict__ sorted_ids, int tk_base,
    int k_start, int stride, int M, int top_k, int tid) {
  int row = tid >> 2;                 // tid / 4 (0..63)
  int base_col = (tid & 3) * 16;      // 4 threads per row, 16 cols each
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  if (flat < N) {
    int token = flat / top_k;
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      const uint32_t* s = (const uint32_t*)(&src[(size_t)token * stride + k_start + col_off]);
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      uint2 v = *(const uint2*)(s);
      dst[0] = v.x; dst[1] = v.y;
    }
  } else {
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      dst[0] = 0; dst[1] = 0;
    }
  }
}

// Load the 64x64 act tile (stage 1).  act [EM, ispp] is indexed by SORTED ROW
// (tk_base + row), unlike A which is indexed by token.
__device__ __forceinline__ void load_act_tile_pf(
    uint16_t sA[kBM_pf][kBK_pf], const uint16_t* __restrict__ act,
    const int32_t* __restrict__ sorted_ids, int tk_base,
    int k_start, int ispp, int M, int top_k, int tid) {
  int row = tid >> 2;
  int base_col = (tid & 3) * 16;
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  if (flat < N) {
    const uint16_t* src_row = &act[(size_t)(tk_base + row) * ispp + k_start];
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      const uint32_t* s = (const uint32_t*)(src_row + col_off);
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      uint2 v = *(const uint2*)(s);
      dst[0] = v.x; dst[1] = v.y;
    }
  } else {
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      dst[0] = 0; dst[1] = 0;
    }
  }
}

// Dequant a packed fp4 weight tile into sB[kBK_pf][kBN] (64x64).
// packed: [N][K/2] row-major, N=kBN, K/2=32. 256 threads.
//   n = tid % kBN, kp_grp = tid / kBN; kp = kp_grp + (256/kBN)*it.
// Consecutive threads write consecutive n (coalesced shared stores); the
// global read is strided by stride_packed (uncoalesced, as in the decode
// dequant) but each element is reused kBM_pf times by the MFMA loop.
template <int kBN>
__device__ __forceinline__ void dequant_b_pf(
    uint16_t sB[kBK_pf][kBN],
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
    int k_start, int stride_packed, int stride_scale_n, int tid) {
  int n = tid & (kBN - 1);
  int kp_grp = tid / kBN;
  constexpr int kPackedK = kBK_pf / 2;                     // 32
  constexpr int kBytesPerThread = (kPackedK * kBN) / kThreads_pf;
  constexpr int kKpStep = kThreads_pf / kBN;
  int kp_off = k_start / 2;
  int sc_off = k_start / kGroupSize;
  for (int it = 0; it < kBytesPerThread; ++it) {
    int kp = kp_grp + it * kKpStep;
    uint8_t pb = packed[n * stride_packed + kp_off + kp];
    int gi = (kp * 2) / kGroupSize;
    float s = ue8m0_decode(scale[n * stride_scale_n + sc_off + gi]);
    uint32_t flo = fp4nib_to_f32bits(pb & 0x0F);
    float vlo; __builtin_memcpy(&vlo, &flo, sizeof(float));
    vlo *= s; __builtin_memcpy(&flo, &vlo, sizeof(float));
    sB[kp*2][n] = f32bits_to_bf16(flo);
    uint32_t fhi = fp4nib_to_f32bits((pb >> 4) & 0x0F);
    float vhi; __builtin_memcpy(&vhi, &fhi, sizeof(float));
    vhi *= s; __builtin_memcpy(&fhi, &vhi, sizeof(float));
    sB[kp*2+1][n] = f32bits_to_bf16(fhi);
  }
}

// Run the K=64 MFMA loop over kBN/16 column fragments for one 16-row
// fragment.  Each wavefront (warp) owns row fragment warp (rows warp*16..+16)
// and accumulates acc[nf] += sA[row_frag] x sB[:, nf*16..+16].
template <int kBN>
__device__ __forceinline__ void mfma_k64_pf(
    v4f acc[kBN / 16], const uint16_t sA[kBM_pf][kBK_pf],
    const uint16_t sB[kBK_pf][kBN], int warp, int lane) {
  constexpr int kNF = kBN / 16;
  int m = lane & 15;          // row within 16-row fragment
  int n = lane & 15;          // col within 16-col fragment
  int kq = lane >> 4;         // 0..3 -> which 4-K group
  int row_base = warp * 16;
  for (int mf = 0; mf < kBK_pf / kMfmaK; ++mf) {
    int k0 = mf * kMfmaK + kq * 4;
    v4s a = { (short)sA[row_base + m][k0],     (short)sA[row_base + m][k0 + 1],
              (short)sA[row_base + m][k0 + 2], (short)sA[row_base + m][k0 + 3] };
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) {
      int nb = nf * 16 + n;
      v4s b = { (short)sB[k0][nb],     (short)sB[k0 + 1][nb],
                (short)sB[k0 + 2][nb], (short)sB[k0 + 3][nb] };
      acc[nf] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[nf], 0, 0, 0);
    }
  }
}

// ======================================================================
//  Stage 0: gate_up GEMM + SwiGLU → act [EM, ispp] bf16
// ======================================================================
// kKM selects the B-weight packing: N-major [E,2,ispp,hidden/2] (default,
// stride hidden/2 per column, scattered reads) or K-major
// [E,2,hidden/2,ispp] (issue #41, item 4, coalesced 16-byte column reads).
template <bool kKM = false>
__global__ __launch_bounds__(kWhreads, 10)
void gateup_swiglu_kernel(
    const uint16_t* __restrict__ A,
    const uint8_t*  __restrict__ w13,
    const uint8_t*  __restrict__ w13_scale,
    const float*    __restrict__ b13,
    uint16_t*       __restrict__ act,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int hidden, int ispp, int M, int top_k, float swiglu_limit,
    int activation, float beta, float linear_beta, int has_bias) {

  __shared__ uint16_t sA[kBM][kBK];
  __shared__ uint16_t sB_gate[kBK][16];
  __shared__ uint16_t sB_up[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp_gate;
  const uint8_t* sc_gate;
  const uint8_t* bp_up;
  const uint8_t* sc_up;
  if constexpr (kKM) {
    // K-major [E,2,hidden/2,ispp]: gate/up are dim-1, N is fast/contiguous.
    int slab = (hidden / 2) * ispp;
    int slab_s = (hidden / kGroupSize) * ispp;
    bp_gate = w13 + (size_t)(expert * 2 + 0) * slab + n_global;
    sc_gate = w13_scale + (size_t)(expert * 2 + 0) * slab_s + n_global;
    bp_up   = w13 + (size_t)(expert * 2 + 1) * slab + n_global;
    sc_up   = w13_scale + (size_t)(expert * 2 + 1) * slab_s + n_global;
  } else {
    int w13_bytes = 2 * ispp * (hidden / 2);
    int w13s_bytes = 2 * ispp * (hidden / kGroupSize);
    bp_gate = w13 + expert * w13_bytes + n_global * (hidden / 2);
    sc_gate = w13_scale + expert * w13s_bytes + n_global * (hidden / kGroupSize);
    bp_up   = w13 + expert * w13_bytes + (n_global + ispp) * (hidden / 2);
    sc_up   = w13_scale + expert * w13s_bytes + (n_global + ispp) * (hidden / kGroupSize);
  }

  v4f acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
  v4f acc_u = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kb = 0; kb < hidden / kBK; ++kb) {
    int ks = kb * kBK;
    // sA load and sB dequant write different LDS buffers, so no barrier is
    // needed between them; the barrier before MFMA orders both.
    load_a_tile(sA, A, sorted_ids, tk_base, ks, hidden, M, top_k, lane);
    if constexpr (kKM) {
      dequant_b_16cols_kmajor(sB_gate, bp_gate, sc_gate, ks, ispp, ispp, lane);
      dequant_b_16cols_kmajor(sB_up,   bp_up,   sc_up,   ks, ispp, ispp, lane);
    } else {
      dequant_b_16cols(sB_gate, bp_gate, sc_gate, ks,
                       hidden / 2, hidden / kGroupSize, lane);
      dequant_b_16cols(sB_up, bp_up, sc_up, ks,
                       hidden / 2, hidden / kGroupSize, lane);
    }
    __syncthreads();

    mfma_k64(acc_g, sA, sB_gate, lane);
    mfma_k64(acc_u, sA, sB_up, lane);
    __syncthreads();
  }

  // Epilogue: bias + activation → act [EM, ispp] bf16 (single bf16 rounding).
  // C layout: col = lane % 16, row = (lane/16)*4 + i.
  int col = lane % 16;
  int row_base = (lane / 16) * 4;
  for (int i = 0; i < 4; ++i) {
    int flat = sorted_ids[tk_base + row_base + i];
    if (flat < M * top_k) {
      float g = acc_g[i];
      if (has_bias)
        g += b13[(size_t)expert*2*ispp + n_global + col];
      float u = acc_u[i];
      if (has_bias)
        u += b13[(size_t)expert*2*ispp + n_global + col + ispp];

      act[(size_t)(tk_base + row_base + i)*ispp + n_global + col] =
          f2bf(epilogue_act(activation, g, u, beta, linear_beta, swiglu_limit));
    }
  }
}

// ======================================================================
//  Stage 1: down GEMM + routed combine → out
// ======================================================================
__global__ __launch_bounds__(kWhreads, 16)
void down_combine_kernel(
    const uint16_t* __restrict__ act,
    const uint8_t*  __restrict__ w2,
    const uint8_t*  __restrict__ w2_scale,
    const float*    __restrict__ b2,
    float*          __restrict__ out,
    const float*    __restrict__ topk_ws,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int ispp, int hidden, int M, int top_k, int has_bias) {

  __shared__ uint16_t sA[kBM][kBK];
  __shared__ uint16_t sB[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  int w2_bytes = hidden * (ispp / 2);
  int w2s_bytes = hidden * (ispp / kGroupSize);
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp = w2 + expert * w2_bytes
      + n_global * (ispp / 2);
  const uint8_t* sc = w2_scale + expert * w2s_bytes
      + n_global * (ispp / kGroupSize);

  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kb = 0; kb < ispp / kBK; ++kb) {
    int ks = kb * kBK;
    load_act_tile(sA, act, sorted_ids, tk_base, ks, ispp, M, top_k, lane);
    dequant_b_16cols(sB, bp, sc, ks, ispp / 2, ispp / kGroupSize, lane);
    __syncthreads();

    mfma_k64(acc, sA, sB, lane);
    __syncthreads();
  }

  // Epilogue: bias + weight + scatter-add into out.
  int col = lane % 16;
  int row_base = (lane / 16) * 4;
  for (int i = 0; i < 4; ++i) {
    int flat = sorted_ids[tk_base + row_base + i];
    if (flat < M * top_k) {
      int token = flat / top_k;
      float v = acc[i];
      if (has_bias)
        v += b2[(size_t)expert*hidden + n_global + col];
      v *= topk_ws[flat];
      int oc = n_global + col;
      if (oc < hidden)
        atomicAdd(&out[(size_t)token * hidden + oc], v);
    }
  }
}

// ======================================================================
//  Prefill: Stage 0 — gate_up GEMM + SwiGLU → act [EM, ispp] bf16
//  (BM=64, BN=64, 256 threads). One expert per 64-row block (the
//  caller must align with block_size=64).
// ======================================================================
template <int kBN>
__global__ __launch_bounds__(kThreads_pf, 4)
void gateup_swiglu_kernel_prefill(
    const uint16_t* __restrict__ A,
    const uint8_t*  __restrict__ w13,
    const uint8_t*  __restrict__ w13_scale,
    const float*    __restrict__ b13,
    uint16_t*       __restrict__ act,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int hidden, int ispp, int M, int top_k, float swiglu_limit,
    int activation, float beta, float linear_beta, int has_bias) {

  constexpr int kNF = kBN / 16;
  __shared__ uint16_t sA[kBM_pf][kBK_pf];
  __shared__ uint16_t sB[kBK_pf][kBN];  // reused for gate then up

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM_pf;
  int w13_bytes = 2 * ispp * (hidden / 2);
  int w13s_bytes = 2 * ispp * (hidden / kGroupSize);
  int tid = threadIdx.x;
  int lane = tid & 63;
  int warp = tid >> 6;
  int n_global = pid_n * kBN;

  const uint8_t* bp_gate = w13 + expert * w13_bytes
      + n_global * (hidden / 2);
  const uint8_t* sc_gate = w13_scale + expert * w13s_bytes
      + n_global * (hidden / kGroupSize);
  const uint8_t* bp_up = w13 + expert * w13_bytes
      + (n_global + ispp) * (hidden / 2);
  const uint8_t* sc_up = w13_scale + expert * w13s_bytes
      + (n_global + ispp) * (hidden / kGroupSize);

  v4f acc_g[kNF], acc_u[kNF];
#pragma unroll
  for (int nf = 0; nf < kNF; ++nf) {
    acc_g[nf] = {0.0f, 0.0f, 0.0f, 0.0f};
    acc_u[nf] = {0.0f, 0.0f, 0.0f, 0.0f};
  }

  for (int kb = 0; kb < hidden / kBK_pf; ++kb) {
    int ks = kb * kBK_pf;
    // sA load and sB dequant are independent (different LDS buffers) — no
    // barrier between them; the single barrier before each MFMA suffices.
    load_a_tile_pf(sA, A, sorted_ids, tk_base, ks, hidden, M, top_k, tid);
    dequant_b_pf(sB, bp_gate, sc_gate, ks, hidden / 2, hidden / kGroupSize, tid);
    __syncthreads();
    mfma_k64_pf(acc_g, sA, sB, warp, lane);
    __syncthreads();

    dequant_b_pf(sB, bp_up, sc_up, ks, hidden / 2, hidden / kGroupSize, tid);
    __syncthreads();
    mfma_k64_pf(acc_u, sA, sB, warp, lane);
    __syncthreads();
  }

  // Epilogue: C col = lane%16, row = warp*16 + (lane/16)*4 + i.
  int col = lane & 15;
  int row_base = warp * 16 + (lane >> 4) * 4;
#pragma unroll
  for (int nf = 0; nf < kNF; ++nf) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int flat = sorted_ids[tk_base + row_base + i];
      if (flat < M * top_k) {
        float g = acc_g[nf][i];
        if (has_bias)
          g += b13[(size_t)expert * 2 * ispp + n_global + nf * 16 + col];
        float u = acc_u[nf][i];
        if (has_bias)
          u += b13[(size_t)expert * 2 * ispp + n_global + nf * 16 + col + ispp];

        act[(size_t)(tk_base + row_base + i) * ispp + n_global + nf * 16 + col] =
            f2bf(epilogue_act(activation, g, u, beta, linear_beta, swiglu_limit));
      }
    }
  }
}

// ======================================================================
//  Prefill: Stage 1 — down GEMM + routed combine → out (BM=64, BN=64)
// ======================================================================
template <int kBN>
__global__ __launch_bounds__(kThreads_pf, 4)
void down_combine_kernel_prefill(
    const uint16_t* __restrict__ act,
    const uint8_t*  __restrict__ w2,
    const uint8_t*  __restrict__ w2_scale,
    const float*    __restrict__ b2,
    float*          __restrict__ out,
    const float*    __restrict__ topk_ws,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int ispp, int hidden, int M, int top_k, int has_bias) {

  constexpr int kNF = kBN / 16;
  __shared__ uint16_t sA[kBM_pf][kBK_pf];
  __shared__ uint16_t sB[kBK_pf][kBN];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM_pf;
  int w2_bytes = hidden * (ispp / 2);
  int w2s_bytes = hidden * (ispp / kGroupSize);
  int tid = threadIdx.x;
  int lane = tid & 63;
  int warp = tid >> 6;
  int n_global = pid_n * kBN;

  const uint8_t* bp = w2 + expert * w2_bytes
      + n_global * (ispp / 2);
  const uint8_t* sc = w2_scale + expert * w2s_bytes
      + n_global * (ispp / kGroupSize);

  v4f acc[kNF];
#pragma unroll
  for (int nf = 0; nf < kNF; ++nf) acc[nf] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kb = 0; kb < ispp / kBK_pf; ++kb) {
    int ks = kb * kBK_pf;
    load_act_tile_pf(sA, act, sorted_ids, tk_base, ks, ispp, M, top_k, tid);
    dequant_b_pf(sB, bp, sc, ks, ispp / 2, ispp / kGroupSize, tid);
    __syncthreads();

    mfma_k64_pf(acc, sA, sB, warp, lane);
    __syncthreads();
  }

  int col = lane & 15;
  int row_base = warp * 16 + (lane >> 4) * 4;
#pragma unroll
  for (int nf = 0; nf < kNF; ++nf) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int flat = sorted_ids[tk_base + row_base + i];
      if (flat < M * top_k) {
        int token = flat / top_k;
        float v = acc[nf][i];
        if (has_bias)
          v += b2[(size_t)expert * hidden + n_global + nf * 16 + col];
        v *= topk_ws[flat];
        int oc = n_global + nf * 16 + col;
        if (oc < hidden)
          atomicAdd(&out[(size_t)token * hidden + oc], v);
      }
    }
  }
}

// ======================================================================
//  Gather topk weights into sorted order
// ======================================================================
// The former `gather_weights_kernel` (reorder topk_w → sorted_w[EM]) has
// been removed: the down-combine epilogue reads `topk_w[flat]` directly,
// so no separate sorted_w buffer or gather launch is needed (issue #41,
// items 1 + 2).
// ======================================================================
//  Host launcher
// ======================================================================
void fused_moe_mxfp4(
    const uint16_t* A,
    const uint8_t*  w13,  const uint8_t*  w13_scale,
    const uint8_t*  w2,   const uint8_t*  w2_scale,
    const int32_t*  topk_ids,
    const float*    topk_w,
    uint16_t*       act_scratch,
    float*          out,
    int M, int hidden, int ispp, int top_k,
    const int32_t* sorted_ids, const int32_t* expert_ids,
    int EM, float swiglu_limit,
    int activation, float beta, float linear_beta,
    const float* b13, const float* b2, int block_size, bool kmajor, void* stream) {

  // The device kernels below launch on `stream` (the caller's current
  // stream, e.g. PyTorch's compute stream) rather than the HIP default
  // stream. This orders the MoE writes with respect to the buffers the
  // caller allocated (act_scratch, out) and will read (out -> bf16 copy)
  // on that same stream, removing the device-wide synchronize the
  // caller previously needed (issue: per-step MoE became a TP barrier
  // because a device-wide drain serialised every all-to-all).
  // `stream == nullptr` reproduces the legacy default-stream launch.
  hipStream_t hipstream = static_cast<hipStream_t>(stream);

  // The combine epilogue reads `topk_w[flat]` directly (where
  // `flat = sorted_ids[...]` is already guarded by `flat < M*top_k`),
  // so no separate `sorted_w` buffer or gather kernel is needed: the
  // launcher performs no device allocation of its own and launches one
  // fewer kernel per forward (issue #41, items 1 + 2).
  //
  // Optional per-kernel timing (VK_MOE_KTIME=1) emits gateup/down
  // elapsed microseconds to stderr.  The hipEventSynchronize after each
  // launch serialises the two stages, so the bench's end-to-end timing is
  // only meaningful when this env var is unset.
  const bool ktime = (std::getenv("VK_MOE_KTIME") != nullptr);
  hipEvent_t ks = nullptr, ke = nullptr;
  if (ktime) { hipEventCreate(&ks); hipEventCreate(&ke); }
#define VK_TBEGIN() do { if (ktime) hipEventRecord(ks, hipstream); } while (0)
#define VK_TEND(name) do {                                                     \
    if (ktime) {                                                               \
      hipEventRecord(ke, hipstream);                                                      \
      hipEventSynchronize(ke);                                                 \
      float ms = 0.f; hipEventElapsedTime(&ms, ks, ke);                        \
      std::fprintf(stderr, "[ktime] %-8s %8.3f us\n", name, ms * 1000.f);      \
    }                                                                          \
  } while (0)

  if (block_size >= kBM_pf) {
    // Prefill regime: 64x64 tiles, 256 threads.  The caller must have
    // aligned expert_ids per 64-row block (EM % 64 == 0).
    {
      dim3 g(EM / kBM_pf, ispp / kBN_pf);
      VK_TBEGIN();
      gateup_swiglu_kernel_prefill<kBN_pf><<<g, kThreads_pf, 0, hipstream>>>(
          A, w13, w13_scale, b13, act_scratch,
          sorted_ids, expert_ids, hidden, ispp, M, top_k, swiglu_limit,
          activation, beta, linear_beta, b13 != nullptr);
      VK_TEND("gateup");
    }
    {
      dim3 g(EM / kBM_pf, hidden / kBN_pf);
      VK_TBEGIN();
      down_combine_kernel_prefill<kBN_pf><<<g, kThreads_pf, 0, hipstream>>>(
          act_scratch, w2, w2_scale, b2, out, topk_w,
          sorted_ids, expert_ids, ispp, hidden, M, top_k, b2 != nullptr);
      VK_TEND("down");
    }
  } else {
    // Decode regime: 16x64 tiles, 64 threads (1 wavefront), blockIdx.z
    // splits the N dimension into 4 warps of 16 columns.
    {
      dim3 g(EM / kBM, ispp / kBN, kWarps);
      VK_TBEGIN();
      if (kmajor)
        gateup_swiglu_kernel<true><<<g, kWhreads, 0, hipstream>>>(
            A, w13, w13_scale, b13, act_scratch,
            sorted_ids, expert_ids, hidden, ispp, M, top_k, swiglu_limit,
            activation, beta, linear_beta, b13 != nullptr);
      else
        gateup_swiglu_kernel<false><<<g, kWhreads, 0, hipstream>>>(
            A, w13, w13_scale, b13, act_scratch,
            sorted_ids, expert_ids, hidden, ispp, M, top_k, swiglu_limit,
            activation, beta, linear_beta, b13 != nullptr);
      VK_TEND("gateup");
    }
    {
      dim3 g(EM / kBM, hidden / kBN, kWarps);
      VK_TBEGIN();
      down_combine_kernel<<<g, kWhreads, 0, hipstream>>>(
          act_scratch, w2, w2_scale, b2, out, topk_w,
          sorted_ids, expert_ids, ispp, hidden, M, top_k, b2 != nullptr);
      VK_TEND("down");
    }
  }
#undef VK_TBEGIN
#undef VK_TEND
  if (ktime) { hipEventDestroy(ks); hipEventDestroy(ke); }

  // No device allocation is performed or freed by this launcher; the
  // caller owns `act_scratch` and reuses it across forward passes.
}

// ======================================================================
//  Distributed (TP) stage kernels
// ======================================================================
// The single-rank fused path fuses the nonlinear epilogues into the two
// GEMM kernels, which blocks tensor parallelism: a TP rank owns a K-slice of
// hidden/ispp, so the linear accumulators of all ranks must be summed BEFORE
// the activation.  These four kernels expose the stages with an explicit
// boundary (moe_fused.hpp / dist_moe.hpp), so the caller can run its own
// all-reduce in between:
//
//   moe_gateup_preact  gate_up[EM, 2*ispp] fp32  (raw gate|up accumulators)
//   moe_act_epilogue   act[EM, ispp] bf16        (bias + activation)
//   moe_down_preact    partial[EM, hidden] fp32  (raw down accumulators)
//   moe_combine        out[M, hidden] fp32       (bias + weight + scatter)
//
// Each consumes the per-rank shard layouts produced by dist_moe.hpp (w13
// [E, 2*ispp, hidden_k/2], w2 [E, hidden, ispp_k/2]); `k_base` is the column
// offset into A / act where the shard starts.  Decode regime (16x64 tiles,
// 64 threads); prefill follows with the 256-thread templates.

// Stage 0a: gate/up GEMM (pre-activation, pre-bias) → gate_up [EM, 2*ispp].
__global__ void gateup_preact_kernel(
    const uint16_t* __restrict__ A,
    const uint8_t*  __restrict__ w13,
    const uint8_t*  __restrict__ w13_scale,
    float*          __restrict__ gate_up,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int a_stride, int k_base, int hidden_k, int ispp, int M, int top_k) {

  __shared__ uint16_t sA[kBM][kBK];
  __shared__ uint16_t sB_gate[kBK][16];
  __shared__ uint16_t sB_up[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  int w13_bytes = 2 * ispp * (hidden_k / 2);
  int w13s_bytes = 2 * ispp * (hidden_k / kGroupSize);
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp_gate = w13 + expert * w13_bytes
      + n_global * (hidden_k / 2);
  const uint8_t* sc_gate = w13_scale + expert * w13s_bytes
      + n_global * (hidden_k / kGroupSize);
  const uint8_t* bp_up = w13 + expert * w13_bytes
      + (n_global + ispp) * (hidden_k / 2);
  const uint8_t* sc_up = w13_scale + expert * w13s_bytes
      + (n_global + ispp) * (hidden_k / kGroupSize);

  v4f acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
  v4f acc_u = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kb = 0; kb < hidden_k / kBK; ++kb) {
    int ks = kb * kBK;
    load_a_tile(sA, A, sorted_ids, tk_base, ks, a_stride, M, top_k, lane,
                k_base);
    dequant_b_16cols(sB_gate, bp_gate, sc_gate, ks,
                     hidden_k / 2, hidden_k / kGroupSize, lane);
    dequant_b_16cols(sB_up, bp_up, sc_up, ks,
                     hidden_k / 2, hidden_k / kGroupSize, lane);
    __syncthreads();

    mfma_k64(acc_g, sA, sB_gate, lane);
    mfma_k64(acc_u, sA, sB_up, lane);
    __syncthreads();
  }

  // Raw accumulators → gate_up (padding rows hold the zeroed-tile result).
  int col = lane % 16;
  int row_base = (lane / 16) * 4;
  for (int i = 0; i < 4; ++i) {
    int row = tk_base + row_base + i;
    gate_up[(size_t)row * 2 * ispp + n_global + col] = acc_g[i];
    gate_up[(size_t)row * 2 * ispp + ispp + n_global + col] = acc_u[i];
  }
}

// Stage 0b: bias + activation epilogue → act [EM, ispp] bf16.  One thread
// per (sorted row, ispp col); padding rows are skipped.
__global__ void act_epilogue_kernel(
    const float*    __restrict__ gate_up,
    const float*    __restrict__ b13,
    uint16_t*       __restrict__ act,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int M, int top_k, int EM, int ispp, float swiglu_limit,
    int activation, float beta, float linear_beta, int has_bias) {

  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)EM * ispp) return;
  int row = (int)(idx / ispp);
  int col = (int)(idx % ispp);
  int flat = sorted_ids[row];
  if (flat >= M * top_k) return;  // padding row
  int expert = expert_ids[row / kBM];
  if (expert < 0) return;         // padding block

  float g = gate_up[(size_t)row * 2 * ispp + col];
  float u = gate_up[(size_t)row * 2 * ispp + ispp + col];
  if (has_bias) {
    g += b13[(size_t)expert * 2 * ispp + col];
    u += b13[(size_t)expert * 2 * ispp + ispp + col];
  }
  act[(size_t)row * ispp + col] =
      f2bf(epilogue_act(activation, g, u, beta, linear_beta, swiglu_limit));
}

// Stage 1a: down GEMM (pre-bias, pre-combine) → partial [EM, hidden].
__global__ void down_preact_kernel(
    const uint16_t* __restrict__ act,
    const uint8_t*  __restrict__ w2,
    const uint8_t*  __restrict__ w2_scale,
    float*          __restrict__ partial,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int a_stride, int k_base, int ispp_k, int hidden, int M, int top_k) {

  __shared__ uint16_t sA[kBM][kBK];
  __shared__ uint16_t sB[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  int w2_bytes = hidden * (ispp_k / 2);
  int w2s_bytes = hidden * (ispp_k / kGroupSize);
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp = w2 + expert * w2_bytes
      + n_global * (ispp_k / 2);
  const uint8_t* sc = w2_scale + expert * w2s_bytes
      + n_global * (ispp_k / kGroupSize);

  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kb = 0; kb < ispp_k / kBK; ++kb) {
    int ks = kb * kBK;
    load_act_tile(sA, act, sorted_ids, tk_base, ks, a_stride, M, top_k, lane,
                  k_base);
    dequant_b_16cols(sB, bp, sc, ks, ispp_k / 2, ispp_k / kGroupSize, lane);
    __syncthreads();

    mfma_k64(acc, sA, sB, lane);
    __syncthreads();
  }

  // Raw accumulators → partial (padding rows hold the zeroed-tile result).
  int col = lane % 16;
  int row_base = (lane / 16) * 4;
  for (int i = 0; i < 4; ++i) {
    int row = tk_base + row_base + i;
    partial[(size_t)row * hidden + n_global + col] = acc[i];
  }
}

// Stage 1b: routed combine → out[M, hidden] += (partial + b2) * topk_w.
__global__ void combine_kernel(
    const float*    __restrict__ partial,
    const float*    __restrict__ b2,
    const float*    __restrict__ topk_ws,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    float*          __restrict__ out,
    int M, int hidden, int top_k, int EM, int has_bias) {

  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)EM * hidden) return;
  int row = (int)(idx / hidden);
  int col = (int)(idx % hidden);
  int flat = sorted_ids[row];
  if (flat >= M * top_k) return;  // padding row
  int token = flat / top_k;
  int expert = expert_ids[row / kBM];
  if (expert < 0) return;         // padding block

  float v = partial[(size_t)row * hidden + col];
  if (has_bias) v += b2[(size_t)expert * hidden + col];
  v *= topk_ws[row];
  atomicAdd(&out[(size_t)token * hidden + col], v);
}

// --------------------------------------------------------------------------
//  Host launchers for the TP stages
// --------------------------------------------------------------------------

void moe_gateup_preact(
    const uint16_t* A, const uint8_t* w13, const uint8_t* w13_scale,
    const int32_t* sorted_ids, const int32_t* expert_ids, float* gate_up,
    int M, int hidden, int ispp, int top_k, int EM,
    int k_base, int hidden_k) {
  dim3 g(EM / kBM, ispp / kBN, kWarps);
  gateup_preact_kernel<<<g, kWhreads>>>(
      A, w13, w13_scale, gate_up, sorted_ids, expert_ids,
      hidden, k_base, hidden_k, ispp, M, top_k);
}

void moe_act_epilogue(
    const float* gate_up, const float* b13, uint16_t* act,
    const int32_t* sorted_ids, const int32_t* expert_ids,
    int M, int top_k, int EM, int ispp, float swiglu_limit,
    int activation, float beta, float linear_beta) {
  long total = (long)EM * ispp;
  int nb = (int)((total + 255) / 256);
  act_epilogue_kernel<<<nb, 256>>>(
      gate_up, b13, act, sorted_ids, expert_ids, M, top_k, EM, ispp,
      swiglu_limit, activation, beta, linear_beta, b13 != nullptr);
}

void moe_down_preact(
    const uint16_t* act, const uint8_t* w2, const uint8_t* w2_scale,
    const int32_t* sorted_ids, const int32_t* expert_ids, float* partial,
    int M, int ispp, int hidden, int top_k, int EM,
    int k_base, int ispp_k) {
  dim3 g(EM / kBM, hidden / kBN, kWarps);
  down_preact_kernel<<<g, kWhreads>>>(
      act, w2, w2_scale, partial, sorted_ids, expert_ids,
      ispp, k_base, ispp_k, hidden, M, top_k);
}

void moe_combine(
    const float* partial, const float* b2, const float* topk_w_sorted,
    const int32_t* sorted_ids, const int32_t* expert_ids, float* out,
    int M, int hidden, int top_k, int EM) {
  long total = (long)EM * hidden;
  int nb = (int)((total + 255) / 256);
  combine_kernel<<<nb, 256>>>(
      partial, b2, topk_w_sorted, sorted_ids, expert_ids, out,
      M, hidden, top_k, EM, b2 != nullptr);
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
