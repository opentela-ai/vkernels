// vkernels/kernels/glm_moe.hip
//
// GLM block-FP8 expert primitives on gfx942 (issue #64). Device
// counterpart of glm_moe.cpp — see glm_moe.hpp for the format contract.
//
// glm_fp8_block_gemv — decode-fused E4M3FN GEMV (M = 1/2), split-K:
//   * partial kernel: grid (N/32 row-blocks, SK k-splits). Each 256-
//     thread block owns 32 output rows (8 warps x 4); the block's K slice
//     is staged in LDS as fp32 once (<= 8 KB) and reused by every row.
//     Each lane streams one uint32 = 4 E4M3FN codes per 128-wide scale
//     segment (coalesced 128 B per warp per segment), decodes with bit
//     assembly (normals/NaN) or direct arithmetic (subnormals), and
//     accumulates in fp32; one shuffle reduction per row writes one fp32
//     partial to [M, N, SK].
//   * reduce kernel: fixed-order fp32 sum over SK, one round-to-nearest-
//     even bfloat16 store.
// SK exists because the row-parallel grid alone underfills MI300A: N=4096
// gives 128 blocks on 228 CUs with ~14% occupancy and ~175 GB/s (a
// measured 30x gap to the 5.3 TB/s roof); SK=4..8 restores ~500+ blocks.
// The op stays memory-bound on the weight bytes at M=1 (AI ~ 2 FLOP/byte)
// and nothing is ever materialized in BF16/FP32.
#include <cstdint>
#include <cstdio>

#include <hip/hip_runtime.h>

#include "vkernels/kernels/glm_moe.hpp"

#if VKERNELS_HAS_HIP

namespace vkernels::kernels::hip {

namespace {

constexpr int kTh = 256;
constexpr int kRowsPerWarp = 4;
constexpr int kMaxK = 4096;              // activation staging bound
constexpr int kMaxSeg = kMaxK / 128;     // 32 scale segments

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
  return __uint_as_float((uint32_t)v << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float v) {
  const uint32_t u = __float_as_uint(v);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Branchless E4M3FN decode (the branchy version serialises SIMT lanes and
// cost ~2.5x): placing (e<<23 | m<<20) into an f32 and scaling by 2^120
// maps BOTH normals and subnormals in one multiply — an f32 with exponent
// field e has value 2^(e-127)*(1+m/8), and the true code value is
// 2^(e-7)*(1+m/8) = that * 2^120; the e=0 subnormal case telescopes to the
// same multiplier. Zero decodes to zero. Divergence from the CPU oracle:
// the reserved NaN encodings 0x7F/0xFF decode to +-480 here (the oracle
// yields NaN) — weight tensors never carry them (contract in glm_moe.hpp).
__device__ __forceinline__ float e4m3_f32(uint32_t code) {
  return __uint_as_float(((code & 0x80u) << 24) | ((code & 0x7Fu) << 20)) *
         0x1p120f;
}

// Partial: rows [row0, row0+32) of out, k-segments [seg0, seg0+segs).
__global__ void glm_fp8_gemv_partial_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    const float* __restrict__ scales, float* __restrict__ part,
    int M, int N, int K, int SK, int segs) {
  const int seg0 = blockIdx.y * segs;   // this split's k-segment range
  __shared__ float sx[2 * kMaxK];        // M x slice (slice <= K <= 4096)
  const int tid = threadIdx.x;
  const int slice = segs * 128;
  for (int i = tid; i < M * slice; i += kTh)
    sx[i] = bf16_to_f32(x[(size_t)((i / slice) * K) + seg0 * 128 + (i % slice)]);
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int split = blockIdx.y;
  const int row0 = (blockIdx.x * 8 + warp) * kRowsPerWarp;
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int row = row0 + r;
    if (row >= N) break;
    const uint8_t* wrow = w + (size_t)row * K;
    const float* srow = scales + (size_t)(row >> 7) * (K >> 7);
    // One pass over the weight row serves BOTH tokens (two accumulators,
    // one x-load pair per segment) — M=2 must not re-read the fp8 stream.
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int s = 0; s < segs; ++s) {
      const int seg = seg0 + s;
      const float sc = srow[seg];
      const uint32_t code =
          *(const uint32_t*)(wrow + (size_t)seg * 128 + lane * 4);
      const float d0 = e4m3_f32(code & 0xFFu);
      const float d1 = e4m3_f32((code >> 8) & 0xFFu);
      const float d2 = e4m3_f32((code >> 16) & 0xFFu);
      const float d3 = e4m3_f32(code >> 24);
      const float4 xv0 = *(const float4*)(sx + s * 128 + lane * 4);
      acc0 += sc * (d0 * xv0.x + d1 * xv0.y + d2 * xv0.z + d3 * xv0.w);
      if (M > 1) {
        const float4 xv1 =
            *(const float4*)(sx + slice + s * 128 + lane * 4);
        acc1 += sc * (d0 * xv1.x + d1 * xv1.y + d2 * xv1.z + d3 * xv1.w);
      }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc0 += __shfl_xor(acc0, off);
      acc1 += __shfl_xor(acc1, off);
    }
    if (lane == 0) {
      part[((size_t)0 * N + row) * SK + split] = acc0;
      if (M > 1) part[((size_t)1 * N + row) * SK + split] = acc1;
    }
  }
}

__global__ void glm_fp8_gemv_reduce_kernel(const float* __restrict__ part,
                                           uint16_t* __restrict__ out,
                                           int MN, int SK) {
  const int idx = blockIdx.x * kTh + threadIdx.x;
  if (idx >= MN) return;
  const float* p = part + (size_t)idx * SK;
  float acc = 0.0f;
  for (int s = 0; s < SK; ++s) acc += p[s];
  out[idx] = f32_to_bf16(acc);
}

int gemv_pick_sk(int N, int K) {
  // Fill MI300A's 228 CUs with >= ~2 blocks each, within {1,2,4,8} and
  // the segment budget (K/128 splits at most).
  int sk = (456 * 32 + N - 1) / N;
  if (sk < 1) sk = 1;
  if (sk > 8) sk = 8;
  while (sk > 1 && (K / 128) % sk != 0) sk >>= 1;   // must divide K/128
  return sk;
}

}  // namespace

void glm_fp8_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w,
                                     const float* scales, uint16_t* out,
                                     float* part, int M, int N, int K,
                                     int sk) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (M > 2 || K > kMaxK || K % 128 != 0 || N % 128 != 0 || sk < 1 ||
      sk > 8 || (K / 128) % sk != 0) {
    std::fprintf(stderr,
                 "glm_fp8_block_gemv: requires M<=2, K<=4096, N%%128==0, "
                 "K%%128==0, sk in {1,2,4,8} dividing K/128 (got M=%d N=%d "
                 "K=%d sk=%d)\n", M, N, K, sk);
    return;
  }
  const int segs = (K / 128) / sk;
  const dim3 grid((N + 31) / 32, sk);
  glm_fp8_gemv_partial_kernel<<<grid, kTh>>>(x, w, scales, part, M, N, K, sk,
                                             segs);
  const int mn = M * N;
  glm_fp8_gemv_reduce_kernel<<<(mn + kTh - 1) / kTh, kTh>>>(part, out, mn, sk);
}

int glm_fp8_gemv_pick_sk(int N, int K) { return gemv_pick_sk(N, K); }

void glm_fp8_block_gemv(const uint16_t* x, const uint8_t* w,
                        const float* scales, uint16_t* out,
                        int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  const int sk = gemv_pick_sk(N, K);
  float* part = nullptr;
  hipMalloc(&part, (size_t)M * N * sk * sizeof(float));
  glm_fp8_block_gemv_with_scratch(x, w, scales, out, part, M, N, K, sk);
  hipFree(part);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
