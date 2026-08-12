// vkernels/kernels/moe_fused.hip
//
// MXFP4 fused-MoE grouped GEMM — HIP kernel for AMD gfx942/gfx90a.
#include <cstdint>
#include <cmath>

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/moe.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#  include "vkernels/kernels/moe_device.hip"

namespace vkernels::kernels::hip {

static constexpr int kBM = 16, kBN = 64, kBK = 64;
static constexpr int kGroupSize = 32;
static constexpr int kMfmaK    = 16;
static constexpr int kWarps    = 4;
static constexpr int kWhreads  = kWarps * 64;

static_assert(kBM * kBK == 1024, "A tile");
static_assert(kBK * kBN == 4096, "B tile");
static_assert(kWhreads == 256,   "threads");

// ======================================================================
//  ue8m0 scale decode
// ======================================================================
__device__ __forceinline__ float ue8m0_decode(uint8_t s) {
  if (s == 0xFF) return 0.0f;
  int unbiased = static_cast<int>(s) - 127;
  uint32_t bits;
  if (unbiased >= -126) {
    bits = static_cast<uint32_t>(s) << 23;
  } else {
    int shift = -(unbiased + 126);
    bits = (shift < 32) ? (0x00800000u >> shift) : 0u;
  }
  float f;
  __builtin_memcpy(&f, &bits, sizeof(float));
  return f;
}

// ======================================================================
//  Dequantize BLOCK_K × BLOCK_N packed weights → smem (bf16)
// ======================================================================
__device__ void dequant_b_tile(
    const uint8_t* b_packed, const uint8_t* b_scale,
    int stride_n, int stride_scale_n, int stride_scale_k,
    uint16_t* smem_b) {

  constexpr int kN = kBN, kK = kBK;
  constexpr int kPackedK = kK / 2;
  constexpr int kBytesPerThread = (kPackedK * kN) / kWhreads;  // 8
  constexpr int kKpStep = kWhreads / kN;  // 4

  int n = threadIdx.x % kN;
  int kp_base = threadIdx.x / kN;

  for (int it = 0; it < kBytesPerThread; ++it) {
    int kp = kp_base + it * kKpStep;
    if (kp >= kPackedK) break;

    uint8_t pb = b_packed[n * stride_n + kp];
    int gi = (kp * 2) / kGroupSize;
    uint8_t sc = b_scale[n * stride_scale_n + gi * stride_scale_k];

    uint32_t flo = fp4nib_to_f32bits(pb & 0x0F);
    float vlo; __builtin_memcpy(&vlo, &flo, sizeof(float));
    vlo *= ue8m0_decode(sc);
    __builtin_memcpy(&flo, &vlo, sizeof(float));

    uint32_t fhi = fp4nib_to_f32bits((pb >> 4) & 0x0F);
    float vhi; __builtin_memcpy(&vhi, &fhi, sizeof(float));
    vhi *= ue8m0_decode(sc);
    __builtin_memcpy(&fhi, &vhi, sizeof(float));

    smem_b[(kp * 2)     * kN + n] = f32bits_to_bf16(flo);
    smem_b[(kp * 2 + 1) * kN + n] = f32bits_to_bf16(fhi);
  }
}

// ======================================================================
//  MFMA fragment loaders
// ======================================================================
__device__ __forceinline__ void load_a(uint32_t a[2],
    const uint16_t smem[][kBK], int k_off, int lane) {
  int m = lane % 16;
  int k0 = k_off + (lane / 16) * 4;
  a[0] = (static_cast<uint32_t>(smem[m][k0+1])<<16) | smem[m][k0];
  a[1] = (static_cast<uint32_t>(smem[m][k0+3])<<16) | smem[m][k0+2];
}

__device__ __forceinline__ void load_b(uint32_t b[2],
    const uint16_t smem[][kBN], int k_off, int n_base, int lane) {
  int n = n_base + (lane % 16);
  int k0 = k_off + (lane / 16) * 4;
  b[0] = (static_cast<uint32_t>(smem[k0+1][n])<<16) | smem[k0][n];
  b[1] = (static_cast<uint32_t>(smem[k0+3][n])<<16) | smem[k0+2][n];
}

__device__ __forceinline__ int out_row(int lane) { return lane / 4; }
__device__ __forceinline__ int out_col(int lane) { return (lane % 4) * 4; }

__device__ __forceinline__ uint16_t f2bf(float v) {
  uint32_t b; __builtin_memcpy(&b, &v, sizeof(float));
  return f32bits_to_bf16(b);
}

// ======================================================================
//  Stage 0: gate_up + SwiGLU
// ======================================================================
__global__ void gateup_swiglu_kernel(
    const uint16_t* __restrict__ A,
    const uint8_t*  __restrict__ w13,
    const uint8_t*  __restrict__ w13_scale,
    const float*    __restrict__ b13,
    uint16_t*       __restrict__ act,
    const int32_t*  __restrict__ sorted_ids,
    const int32_t*  __restrict__ expert_ids,
    int hidden, int ispp, float swiglu_limit, int has_bias) {

  __shared__ uint16_t sA[kBM][kBK];
  __shared__ uint16_t sB[kBK][kBN];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  int w13_bytes = 2 * ispp * (hidden / 2);
  int w13s_bytes = 2 * ispp * (hidden / kGroupSize);

  int warp = threadIdx.x / 64, lane = threadIdx.x % 64;
  float cg[4]={0}, cu[4]={0};

  for (int kb = 0; kb < hidden / kBK; ++kb) {
    int ks = kb * kBK;

    // Load A tile: vectorized uint2 (4 bf16/thread)
    {
      int row = threadIdx.x % kBM;
      int col_off = (threadIdx.x / kBM) * 4;
      int token = sorted_ids[tk_base + row];
      const uint32_t* src = (const uint32_t*)(
          &A[static_cast<std::size_t>(token) * hidden + ks + col_off]);
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      uint2 v = *(const uint2*)(src);
      dst[0] = v.x; dst[1] = v.y;
    }
    __syncthreads();

    // Gate
    {
      int kp_off = ks / 2, sc_off = ks / kGroupSize;
      const uint8_t* bp = w13 + expert * w13_bytes
                          + pid_n * kBN * (hidden / 2) + kp_off;
      const uint8_t* sc = w13_scale + expert * w13s_bytes
                          + pid_n * kBN * (hidden / kGroupSize) + sc_off;
      dequant_b_tile(bp, sc, hidden/2, hidden/kGroupSize, 1,
                     (uint16_t*)(sB));
    }
    __syncthreads();

    for (int mf = 0; mf < kBK / kMfmaK; ++mf) {
      uint32_t ar[2], br[2];
      load_a(ar, sA, mf * kMfmaK, lane);
      load_b(br, sB, mf * kMfmaK, warp * 16, lane);
      mfma_f32_16x16x16bf16(cg, ar, br);
    }

    // Up
    {
      int kp_off = ks / 2, sc_off = ks / kGroupSize;
      const uint8_t* bp = w13 + expert * w13_bytes
                          + (pid_n * kBN + ispp) * (hidden / 2) + kp_off;
      const uint8_t* sc = w13_scale + expert * w13s_bytes
                          + (pid_n * kBN + ispp) * (hidden / kGroupSize) + sc_off;
      dequant_b_tile(bp, sc, hidden/2, hidden/kGroupSize, 1,
                     (uint16_t*)(sB));
    }
    __syncthreads();

    for (int mf = 0; mf < kBK / kMfmaK; ++mf) {
      uint32_t ar[2], br[2];
      load_a(ar, sA, mf * kMfmaK, lane);
      load_b(br, sB, mf * kMfmaK, warp * 16, lane);
      mfma_f32_16x16x16bf16(cu, ar, br);
    }
    __syncthreads();
  }

  // SwiGLU epilogue
  int row = out_row(lane), cb = out_col(lane);
  float lim = swiglu_limit;
  for (int i = 0; i < 4; ++i) {
    float g = cg[i], u = cu[i];
    if (has_bias) {
      g += b13[static_cast<std::size_t>(expert)*2*ispp + pid_n*kBN + warp*16 + cb + i];
      u += b13[static_cast<std::size_t>(expert)*2*ispp + pid_n*kBN + warp*16 + cb + i + ispp];
    }
    if (lim > 0) { if(g>lim) g=lim; if(u>lim) u=lim; if(u<-lim) u=-lim; }
    float sg = g / (1.0f + expf(-g));
    int tok = sorted_ids[tk_base + row];
    act[static_cast<std::size_t>(tok)*ispp + pid_n*kBN + warp*16 + cb + i] = f2bf(sg * u);
  }
}

// ======================================================================
//  Stage 1: down + combine
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
    int ispp, int hidden, int has_bias) {

  __shared__ uint16_t sA[kBM][kBK];
  __shared__ uint16_t sB[kBK][kBN];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  int w2_bytes = hidden * (ispp / 2);
  int w2s_bytes = hidden * (ispp / kGroupSize);

  int warp = threadIdx.x / 64, lane = threadIdx.x % 64;
  float c[4] = {0};

  for (int kb = 0; kb < ispp / kBK; ++kb) {
    int ks = kb * kBK;

    {
      int row = threadIdx.x % kBM;
      int col_off = (threadIdx.x / kBM) * 4;
      int token = sorted_ids[tk_base + row];
      const uint32_t* src = (const uint32_t*)(
          &act[static_cast<std::size_t>(token) * ispp + ks + col_off]);
      uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
      uint2 v = *(const uint2*)(src);
      dst[0] = v.x; dst[1] = v.y;
    }
    __syncthreads();

    {
      int kp_off = ks / 2, sc_off = ks / kGroupSize;
      const uint8_t* bp = w2 + expert * w2_bytes
                          + pid_n * kBN * (ispp / 2) + kp_off;
      const uint8_t* sc = w2_scale + expert * w2s_bytes
                          + pid_n * kBN * (ispp / kGroupSize) + sc_off;
      dequant_b_tile(bp, sc, ispp/2, ispp/kGroupSize, 1,
                     (uint16_t*)(sB));
    }
    __syncthreads();

    for (int mf = 0; mf < kBK / kMfmaK; ++mf) {
      uint32_t ar[2], br[2];
      load_a(ar, sA, mf * kMfmaK, lane);
      load_b(br, sB, mf * kMfmaK, warp * 16, lane);
      mfma_f32_16x16x16bf16(c, ar, br);
    }
    __syncthreads();
  }

  int row = out_row(lane), cb = out_col(lane);
  int tok = sorted_ids[tk_base + row];
  float w = topk_ws[tk_base + row];

  for (int i = 0; i < 4; ++i) {
    float v = c[i];
    if (has_bias)
      v += b2[static_cast<std::size_t>(expert)*hidden + pid_n*kBN + warp*16 + cb + i];
    v *= w;
    int oc = pid_n * kBN + warp * 16 + cb + i;
    if (oc < hidden)
      atomicAdd(&out[static_cast<std::size_t>(tok) * hidden + oc], v);
  }
}

// ======================================================================
//  Gather topk weights into sorted order
// ======================================================================
__global__ void gather_weights_kernel(
    float* __restrict__ sw, const float* __restrict__ tw,
    const int32_t* __restrict__ sid, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) sw[i] = tw[sid[i]];
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
    const float* b13, const float* b2) {

  float* sorted_w = nullptr;
  (void)hipMalloc(&sorted_w, EM * sizeof(float));
  {
    int nb = (EM + 255) / 256;
    gather_weights_kernel<<<nb, 256>>>(sorted_w, topk_w, sorted_ids, EM);
  }

  {
    dim3 g(EM / kBM, ispp / kBN);
    gateup_swiglu_kernel<<<g, kWhreads>>>(
        A, w13, w13_scale, b13, act_scratch,
        sorted_ids, expert_ids, hidden, ispp, swiglu_limit,
        b13 != nullptr);
  }

  {
    dim3 g(EM / kBM, hidden / kBN);
    down_combine_kernel<<<g, kWhreads>>>(
        act_scratch, w2, w2_scale, b2, out, sorted_w,
        sorted_ids, expert_ids, ispp, hidden, b2 != nullptr);
  }

  (void)hipFree(sorted_w);
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
