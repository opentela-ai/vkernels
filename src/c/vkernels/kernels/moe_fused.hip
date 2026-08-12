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

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/moe.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#  include "vkernels/kernels/moe_device.hip"

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

__device__ __forceinline__ uint16_t f2bf(float v) {
  uint32_t b; __builtin_memcpy(&b, &v, sizeof(float));
  return f32bits_to_bf16(b);
}

// Load the 16x64 A tile into shared memory (stage 0: input activations).
// 64 threads; each thread loads 16 bf16 = 4 uint2 reads.
// sorted_ids holds FLAT topk indices (token*top_k + sel).
__device__ __forceinline__ void load_a_tile(
    uint16_t sA[kBM][kBK], const uint16_t* __restrict__ A,
    const int32_t* __restrict__ sorted_ids, int tk_base,
    int k_start, int stride, int M, int top_k, int lane) {
  int row = lane % kBM;
  int base_col = (lane / kBM) * (kBK / (kWhreads / kBM));
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  if (flat < N) {
    int token = flat / top_k;
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      const uint32_t* src = (const uint32_t*)(&A[(size_t)token * stride + k_start + col_off]);
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
__device__ __forceinline__ void load_act_tile(
    uint16_t sA[kBM][kBK], const uint16_t* __restrict__ act,
    const int32_t* __restrict__ sorted_ids, int tk_base,
    int k_start, int ispp, int M, int top_k, int lane) {
  int row = lane % kBM;
  int base_col = (lane / kBM) * (kBK / (kWhreads / kBM));
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  if (flat < N) {
    const uint16_t* src_row = &act[(size_t)(tk_base + row) * ispp + k_start];
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
__global__ void gateup_swiglu_kernel(
    const uint16_t* __restrict__ A,
    const uint8_t*  __restrict__ w13,
    const uint8_t*  __restrict__ w13_scale,
    const float*    __restrict__ b13,
    uint16_t*       __restrict__ act,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int hidden, int ispp, int M, int top_k, float swiglu_limit, int has_bias) {

  __shared__ uint16_t sA[kBM][kBK];
  __shared__ uint16_t sB_gate[kBK][16];
  __shared__ uint16_t sB_up[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  int w13_bytes = 2 * ispp * (hidden / 2);
  int w13s_bytes = 2 * ispp * (hidden / kGroupSize);
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp_gate = w13 + expert * w13_bytes
      + n_global * (hidden / 2);
  const uint8_t* sc_gate = w13_scale + expert * w13s_bytes
      + n_global * (hidden / kGroupSize);
  const uint8_t* bp_up = w13 + expert * w13_bytes
      + (n_global + ispp) * (hidden / 2);
  const uint8_t* sc_up = w13_scale + expert * w13s_bytes
      + (n_global + ispp) * (hidden / kGroupSize);

  v4f acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
  v4f acc_u = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kb = 0; kb < hidden / kBK; ++kb) {
    int ks = kb * kBK;
    // sA load and sB dequant write different LDS buffers, so no barrier is
    // needed between them; the barrier before MFMA orders both.
    load_a_tile(sA, A, sorted_ids, tk_base, ks, hidden, M, top_k, lane);
    dequant_b_16cols(sB_gate, bp_gate, sc_gate, ks,
                     hidden / 2, hidden / kGroupSize, lane);
    dequant_b_16cols(sB_up, bp_up, sc_up, ks,
                     hidden / 2, hidden / kGroupSize, lane);
    __syncthreads();

    mfma_k64(acc_g, sA, sB_gate, lane);
    mfma_k64(acc_u, sA, sB_up, lane);
    __syncthreads();
  }

  // Epilogue: act = silu(clamp(gate)) * clamp(up) → single bf16 rounding.
  // C layout: col = lane % 16, row = (lane/16)*4 + i.
  int col = lane % 16;
  int row_base = (lane / 16) * 4;
  float lim = swiglu_limit;
  for (int i = 0; i < 4; ++i) {
    int flat = sorted_ids[tk_base + row_base + i];
    if (flat < M * top_k) {
      float g = acc_g[i];
      if (has_bias)
        g += b13[(size_t)expert*2*ispp + n_global + col];
      if (lim > 0.0f && g > lim) g = lim;
      float silu_g = g / (1.0f + expf(-g));

      float u = acc_u[i];
      if (has_bias)
        u += b13[(size_t)expert*2*ispp + n_global + col + ispp];
      if (lim > 0.0f) { if (u > lim) u = lim; if (u < -lim) u = -lim; }

      act[(size_t)(tk_base + row_base + i)*ispp + n_global + col] = f2bf(silu_g * u);
    }
  }
}

// ======================================================================
//  Stage 1: down GEMM + routed combine → out
// ======================================================================
__global__ void down_combine_kernel(
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
      v *= topk_ws[tk_base + row_base + i];
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
__global__ void gateup_swiglu_kernel_prefill(
    const uint16_t* __restrict__ A,
    const uint8_t*  __restrict__ w13,
    const uint8_t*  __restrict__ w13_scale,
    const float*    __restrict__ b13,
    uint16_t*       __restrict__ act,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int hidden, int ispp, int M, int top_k, float swiglu_limit, int has_bias) {

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
  float lim = swiglu_limit;
#pragma unroll
  for (int nf = 0; nf < kNF; ++nf) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int flat = sorted_ids[tk_base + row_base + i];
      if (flat < M * top_k) {
        float g = acc_g[nf][i];
        if (has_bias)
          g += b13[(size_t)expert * 2 * ispp + n_global + nf * 16 + col];
        if (lim > 0.0f && g > lim) g = lim;
        float silu_g = g / (1.0f + expf(-g));

        float u = acc_u[nf][i];
        if (has_bias)
          u += b13[(size_t)expert * 2 * ispp + n_global + nf * 16 + col + ispp];
        if (lim > 0.0f) { if (u > lim) u = lim; if (u < -lim) u = -lim; }

        act[(size_t)(tk_base + row_base + i) * ispp + n_global + nf * 16 + col] =
            f2bf(silu_g * u);
      }
    }
  }
}

// ======================================================================
//  Prefill: Stage 1 — down GEMM + routed combine → out (BM=64, BN=64)
// ======================================================================
template <int kBN>
__global__ void down_combine_kernel_prefill(
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
        v *= topk_ws[tk_base + row_base + i];
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
__global__ void gather_weights_kernel(
    float* __restrict__ sw, const float* __restrict__ tw,
    const int32_t* __restrict__ sid, int n, int nflat) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int f = sid[i];
    sw[i] = (f >= 0 && f < nflat) ? tw[f] : 0.0f;
  }
}

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
    const float* b13, const float* b2, int block_size) {

  float* sorted_w = nullptr;
  (void)hipMalloc(&sorted_w, EM * sizeof(float));
  {
    int nb = (EM + 255) / 256;
    gather_weights_kernel<<<nb, 256>>>(sorted_w, topk_w, sorted_ids, EM, M * top_k);
  }

  if (block_size >= kBM_pf) {
    // Prefill regime: 64x64 tiles, 256 threads.  The caller must have
    // aligned expert_ids per 64-row block (EM % 64 == 0).
    {
      dim3 g(EM / kBM_pf, ispp / kBN_pf);
      gateup_swiglu_kernel_prefill<kBN_pf><<<g, kThreads_pf>>>(
          A, w13, w13_scale, b13, act_scratch,
          sorted_ids, expert_ids, hidden, ispp, M, top_k, swiglu_limit,
          b13 != nullptr);
    }
    {
      dim3 g(EM / kBM_pf, hidden / kBN_pf);
      down_combine_kernel_prefill<kBN_pf><<<g, kThreads_pf>>>(
          act_scratch, w2, w2_scale, b2, out, sorted_w,
          sorted_ids, expert_ids, ispp, hidden, M, top_k, b2 != nullptr);
    }
  } else {
    // Decode regime: 16x64 tiles, 64 threads (1 wavefront), blockIdx.z
    // splits the N dimension into 4 warps of 16 columns.
    {
      dim3 g(EM / kBM, ispp / kBN, kWarps);
      gateup_swiglu_kernel<<<g, kWhreads>>>(
          A, w13, w13_scale, b13, act_scratch,
          sorted_ids, expert_ids, hidden, ispp, M, top_k, swiglu_limit,
          b13 != nullptr);
    }
    {
      dim3 g(EM / kBM, hidden / kBN, kWarps);
      down_combine_kernel<<<g, kWhreads>>>(
          act_scratch, w2, w2_scale, b2, out, sorted_w,
          sorted_ids, expert_ids, ispp, hidden, M, top_k, b2 != nullptr);
    }
  }

  (void)hipFree(sorted_w);
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
