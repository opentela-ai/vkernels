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
// SEGS > 0 is a template constant so the segment loop fully unrolls and
// per-segment loads batch ahead of their consumers — with a runtime bound
// the in-order warp issue serialises load->use per segment and the kernel
// stalls on L2 latency (measured ~2x slower). SEGS == 0 is the generic
// runtime-bound fallback for odd segment counts (e.g. K=1152 -> 9).
template <int SEGS>
__global__ void glm_fp8_gemv_partial_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    const float* __restrict__ scales, float* __restrict__ part,
    int M, int N, int K, int SK, int rt_segs) {
  __shared__ float sx[2 * kMaxK];        // M x slice (slice <= K <= 4096)
  const int tid = threadIdx.x;
  const int segs = (SEGS > 0) ? SEGS : rt_segs;
  const int slice = segs * 128;
  for (int i = tid; i < M * slice; i += kTh)
    sx[i] = bf16_to_f32(x[(size_t)((i / slice) * K) +
                          (size_t)blockIdx.y * slice + (i % slice)]);
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int split = blockIdx.y;
  const int row0 = (blockIdx.x * 8 + warp) * kRowsPerWarp;
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int row = row0 + r;
    if (row >= N) break;
    const uint8_t* wrow = w + (size_t)row * K;
    const float* srow = scales + (size_t)(row >> 7) * (K >> 7);
    // One pass over the weight row serves BOTH tokens; the decode's 2^120
    // scale is folded into the per-segment scale (sc120), leaving one FMA
    // per code plus byte extraction (at roof speed the per-code arithmetic
    // otherwise exceeds the CU's vector throughput).
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (int s = 0; s < segs; ++s) {
      const int seg = blockIdx.y * segs + s;
      const float sc120 = srow[seg] * 0x1p120f;
      const uint32_t code =
          *(const uint32_t*)(wrow + (size_t)seg * 128 + lane * 4);
      const uint32_t b0 = code & 0xFFu, b1 = (code >> 8) & 0xFFu;
      const uint32_t b2 = (code >> 16) & 0xFFu, b3 = code >> 24;
      const float d0 =
          __uint_as_float(((b0 & 0x80u) << 24) | ((b0 & 0x7Fu) << 20));
      const float d1 =
          __uint_as_float(((b1 & 0x80u) << 24) | ((b1 & 0x7Fu) << 20));
      const float d2 =
          __uint_as_float(((b2 & 0x80u) << 24) | ((b2 & 0x7Fu) << 20));
      const float d3 =
          __uint_as_float(((b3 & 0x80u) << 24) | ((b3 & 0x7Fu) << 20));
      const float4 xv0 = *(const float4*)(sx + s * 128 + lane * 4);
      acc0 += sc120 * (d0 * xv0.x + d1 * xv0.y + d2 * xv0.z + d3 * xv0.w);
      if (M > 1) {
        const float4 xv1 =
            *(const float4*)(sx + slice + s * 128 + lane * 4);
        acc1 += sc120 * (d0 * xv1.x + d1 * xv1.y + d2 * xv1.z + d3 * xv1.w);
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

// FNUZ-storage variant (issue #148): w holds e4m3fnuz bytes (the #71
// in-place conversion halves every payload) and scales are the DOUBLED
// fp32 scales, so value = fnuz(code) * s2 exactly reproduces the original
// e4m3fn weight. The decode itself is the HARDWARE fnuz converter —
// v_cvt_pk_f32_fp8 turns two fnuz codes into two fp32 in one instruction
// (exact: normals, subnormals and the fnuz zero; NaN 0x80 never occurs
// under the storage contract) — replacing the manual decode's ~4 ALU ops
// per code. The #64 kernel's own comment flags that per-code arithmetic
// "exceeds the CU's vector throughput" at roof speed, i.e. the manual
// decode is issue-rate-bound (~659 GB/s measured); the cvt path cuts the
// per-code instruction count ~2.5x so the stream, not the ALU, sets the
// rate. Launch structure, x staging and split-K reduce are byte-for-byte
// the #64 design.
template<int SEGS>
__global__ void glm_fp8_fnuz_gemv_partial_kernel(const uint16_t* __restrict__ x,
                                                 const uint8_t* __restrict__ w,
                                                 const float* __restrict__ scales,
                                                 float* __restrict__ part,
                                                 int M,
                                                 int N,
                                                 int K,
                                                 int SK,
                                                 int rt_segs) {
  __shared__ float sx[2 * kMaxK]; // M x slice (slice <= K <= 4096)
  const int tid = threadIdx.x;
  const int segs = (SEGS > 0) ? SEGS : rt_segs;
  const int slice = segs * 128;
  for (int i = tid; i < M * slice; i += kTh)
    sx[i] = bf16_to_f32(x[(size_t)((i / slice) * K) + (size_t)blockIdx.y * slice + (i % slice)]);
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int split = blockIdx.y;
  const int row0 = (blockIdx.x * 8 + warp) * kRowsPerWarp;
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int row = row0 + r;
    if (row >= N) break;
    const uint8_t* wrow = w + (size_t)row * K;
    const float* srow = scales + (size_t)(row >> 7) * (K >> 7);
    float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
    for (int s = 0; s < segs; ++s) {
      const int seg = blockIdx.y * segs + s;
      const float sc = srow[seg]; // fnuz convention: scale is DOUBLED
      const uint32_t code = *(const uint32_t*)(wrow + (size_t)seg * 128 + lane * 4);
      // the builtin returns a clang float2 vector (not HIP's float2)
      const auto d01 = __builtin_amdgcn_cvt_pk_f32_fp8((uint16_t)(code & 0xFFFFu), 0);
      const auto d23 = __builtin_amdgcn_cvt_pk_f32_fp8((uint16_t)(code >> 16), 0);
      const float4 xv0 = *(const float4*)(sx + s * 128 + lane * 4);
      acc0 += sc * (d01[0] * xv0.x + d01[1] * xv0.y + d23[0] * xv0.z + d23[1] * xv0.w);
      if (M > 1) {
        const float4 xv1 = *(const float4*)(sx + slice + s * 128 + lane * 4);
        acc1 += sc * (d01[0] * xv1.x + d01[1] * xv1.y + d23[0] * xv1.z + d23[1] * xv1.w);
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

// MFMA variant (issue #148): the fp8 dot product itself moves to the
// CDNA3 tensor core — v_mfma_f32_16x16x32_fp8_fp8 consumes RAW fnuz
// fragments, so the per-code cvt of the fnuz-cvt kernel disappears too:
// per 32 k-elements x 16 rows the ALU cost is ONE wave instruction. The
// bottleneck hypothesis: the cvt path is still issue-rate-bound
// (~590 GB/s vs the 2 TB/s the same structure reaches on bf16, which
// needs half the arithmetic per byte). Structure: 256 threads = 4 wave64
// warps, each warp owns 16 weight rows (N/64 row-blocks x SK k-splits);
// x is staged + per-128-group fnuz-quantized in LDS once per block (the
// fused-quantize trick from the triton decode path), then per (segment,
// 32-k chunk) each lane feeds its 8-byte A/B fragments. Wave64 fragment
// layout (decoded empirically by meta/benchmarks/probe_mfma_fp8.hip —
// the C operand is TRANSPOSED relative to A): A byte (lane l, j) =
// A[l%16][8*(l/16)+j], B byte = B[8*(l/16)+j][l%16], C slot (l, s) =
// C[4*(l/16)+s][l%16]; results scale by s2a[token, k-group] *
// s2w[row, k-group] (both scales are per 128-k block).
// the mfma builtins use clang's OpenCL-style vectors, not HIP's float4
typedef float v4f __attribute__((vector_size(16)));
static __device__ constexpr v4f vzero{0.f, 0.f, 0.f, 0.f};

template<int SEGS>
__global__ void glm_fp8_fnuz_mfma_gemv_kernel(const uint16_t* __restrict__ x,
                                              const uint8_t* __restrict__ w,
                                              const float* __restrict__ scales,
                                              float* __restrict__ part,
                                              int M,
                                              int N,
                                              int K,
                                              int SK,
                                              int rt_segs,
                                              uint32_t* dbg_sxa = nullptr,
                                              float* dbg_s2a = nullptr) {
  __shared__ float sx[2 * kMaxK];         // M x slice fp32
  __shared__ uint32_t sxa[2 * kMaxK / 4]; // M x slice fp8 payload
  __shared__ float ss2a[2 * kMaxSeg];     // [M][segs] DOUBLED x scales
  const int tid = threadIdx.x;
  const int segs = (SEGS > 0) ? SEGS : rt_segs;
  const int slice = segs * 128;
  for (int i = tid; i < M * slice; i += kTh)
    sx[i] = bf16_to_f32(x[(size_t)((i / slice) * K) + (size_t)blockIdx.y * slice + (i % slice)]);
  __syncthreads();

  // per-(token, 128-group) amax -> DOUBLED scale s2a = amax / 224.
  // wave64: each lane must cover TWO of the 128 group elements (lane and
  // lane+64); a single read reduces only half the group and silently
  // under-scales the quantized payload
  const int lane = tid & 63, warp = tid >> 6; // wave64
  for (int g = warp; g < M * segs; g += 4) {
    const int base = g * 128;
    float v = fmaxf(__builtin_fabsf(sx[base + lane]), __builtin_fabsf(sx[base + 64 + lane]));
#pragma unroll
    for (int off = 32; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor(v, off));
    if (lane == 0) ss2a[g] = fmaxf(v, 1e-4f) / 224.0f;
  }
  __syncthreads();

  // quantize the staged activations to fnuz payload bytes (RNE, matching
  // the python path): two f32 -> fp8x2 per builtin call; every 4-byte word
  // lies inside one 128-group (slice and 128 are multiples of 4)
  for (int i = tid; i < M * slice / 4; i += kTh) {
    const int base = i * 4;
    const int mt = base / slice, col = base % slice;
    const float inv = 1.0f / ss2a[(size_t)mt * segs + col / 128];
    uint32_t u = __builtin_amdgcn_cvt_pk_fp8_f32(sx[base] * inv, sx[base + 1] * inv, 0u, false);
    u = __builtin_amdgcn_cvt_pk_fp8_f32(sx[base + 2] * inv, sx[base + 3] * inv, u, true);
    sxa[i] = u;
  }
  __syncthreads();
  if (blockIdx.x == 0 && blockIdx.y == 0) {
    if (dbg_sxa != nullptr)
      for (int i = tid; i < M * slice / 4; i += kTh)
        dbg_sxa[i] = sxa[i];
    if (dbg_s2a != nullptr)
      for (int i = tid; i < M * segs; i += kTh)
        dbg_s2a[i] = ss2a[i];
  }
  __syncthreads();

  // wave64 fragment layout (probe_mfma_fp8-decoded):
  //   A byte (lane l, j)  = A[l % 16][8*(l/16) + j]   (row = token)
  //   B byte (lane l, j)  = B[8*(l/16) + j][l % 16]   (col = weight row)
  //   C slot (lane l, s)  = C[4*(l/16) + s][l % 16]   -- TRANSPOSED vs A!
  // so a lane's A bytes feed token l%16 while its four C slots cover
  // tokens 4*(l/16)+0..3 of ONE weight row.
  const int row0 = (blockIdx.x * 4 + warp) * 16;
  const int at = lane % 16;       // A row (token) this lane feeds
  const int n_tile = lane % 16;   // C/B column (weight-row offset)
  const int kg = 8 * (lane / 16); // k offset of this lane's bytes
  // per-lane weight scale: this lane's C slots all cover the ONE weight
  // row row0 + n_tile, and each mfma call's 8 b bytes lie inside exactly
  // one 128-k scale block, so s2w is per (row, k-segment)
  const int row_c = row0 + n_tile;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const bool real = at < M;
  for (int s = 0; s < segs; ++s) {
    const int seg = blockIdx.y * segs + s; // global 128-segment
    const float s2w_seg = scales[(size_t)(row_c >> 7) * (K >> 7) + seg];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      // slice-local and W-global k of this lane's 8 bytes
      const int kl = s * 128 + c * 32 + kg;
      const int kw = seg * 128 + c * 32 + kg;
      const uint64_t a = real ? *(const uint64_t*)(sxa + (size_t)at * slice / 4 + kl / 4) : 0ull;
      const uint64_t b = *(const uint64_t*)(w + (size_t)(row0 + n_tile) * K + kw);
      const v4f r =
          __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8((long long)a, (long long)b, vzero, 0, 0, 0);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int tok = 4 * (lane / 16) + t; // C row = token
        if (tok < M) acc[t] += r[t] * ss2a[(size_t)tok * segs + s] * s2w_seg;
      }
    }
  }
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int tok = 4 * (lane / 16) + t;
    if (tok < M) part[((size_t)tok * N + row0 + n_tile) * SK + blockIdx.y] = acc[t];
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
  switch (segs) {
    case 2:  glm_fp8_gemv_partial_kernel<2><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 4:  glm_fp8_gemv_partial_kernel<4><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 8:  glm_fp8_gemv_partial_kernel<8><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 16: glm_fp8_gemv_partial_kernel<16><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 32: glm_fp8_gemv_partial_kernel<32><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    default: glm_fp8_gemv_partial_kernel<0><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
  }
  const int mn = M * N;
  glm_fp8_gemv_reduce_kernel<<<(mn + kTh - 1) / kTh, kTh>>>(part, out, mn, sk);
}

void glm_fp8_fnuz_block_gemv_with_scratch(const uint16_t* x,
                                          const uint8_t* w,
                                          const float* scales,
                                          uint16_t* out,
                                          float* part,
                                          int M,
                                          int N,
                                          int K,
                                          int sk) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (M > 2 || K > kMaxK || K % 128 != 0 || N % 128 != 0 || sk < 1 || sk > 8 ||
      (K / 128) % sk != 0) {
    std::fprintf(stderr,
                 "glm_fp8_fnuz_block_gemv: requires M<=2, K<=4096, N%%128==0, "
                 "K%%128==0, sk in {1,2,4,8} dividing K/128 (got M=%d N=%d "
                 "K=%d sk=%d)\n",
                 M,
                 N,
                 K,
                 sk);
    return;
  }
  const int segs = (K / 128) / sk;
  const dim3 grid((N + 31) / 32, sk);
  switch (segs) {
  case 2:
    glm_fp8_fnuz_gemv_partial_kernel<2><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs);
    break;
  case 4:
    glm_fp8_fnuz_gemv_partial_kernel<4><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs);
    break;
  case 8:
    glm_fp8_fnuz_gemv_partial_kernel<8><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs);
    break;
  case 16:
    glm_fp8_fnuz_gemv_partial_kernel<16><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs);
    break;
  case 32:
    glm_fp8_fnuz_gemv_partial_kernel<32><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs);
    break;
  default:
    glm_fp8_fnuz_gemv_partial_kernel<0><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs);
    break;
  }
  const int mn = M * N;
  glm_fp8_gemv_reduce_kernel<<<(mn + kTh - 1) / kTh, kTh>>>(part, out, mn, sk);
}

void glm_fp8_fnuz_block_gemv(
    const uint16_t* x, const uint8_t* w, const float* scales, uint16_t* out, int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  const int sk = gemv_pick_sk(N, K);
  float* part = nullptr;
  hipMalloc(&part, (size_t)M * N * sk * sizeof(float));
  glm_fp8_fnuz_block_gemv_with_scratch(x, w, scales, out, part, M, N, K, sk);
  hipFree(part);
}

void glm_fp8_fnuz_mfma_block_gemv_with_scratch(const uint16_t* x,
                                               const uint8_t* w,
                                               const float* scales,
                                               uint16_t* out,
                                               float* part,
                                               int M,
                                               int N,
                                               int K,
                                               int sk) {
  glm_fp8_fnuz_mfma_block_gemv_debug(x, w, scales, out, part, M, N, K, sk, nullptr, nullptr);
}

void glm_fp8_fnuz_mfma_block_gemv_debug(const uint16_t* x,
                                        const uint8_t* w,
                                        const float* scales,
                                        uint16_t* out,
                                        float* part,
                                        int M,
                                        int N,
                                        int K,
                                        int sk,
                                        uint32_t* dbg_sxa,
                                        float* dbg_s2a) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (M > 2 || K > kMaxK || K % 128 != 0 || N % 128 != 0 || sk < 1 || sk > 8 ||
      (K / 128) % sk != 0) {
    std::fprintf(stderr,
                 "glm_fp8_fnuz_mfma_block_gemv: requires M<=2, K<=4096, "
                 "N%%128==0, K%%128==0, sk in {1,2,4,8} dividing K/128 "
                 "(got M=%d N=%d K=%d sk=%d)\n",
                 M,
                 N,
                 K,
                 sk);
    return;
  }
  const int segs = (K / 128) / sk;
  const dim3 grid(N / 64, sk); // 64 rows per 4-wave64-warp block
  switch (segs) {
  case 2:
    glm_fp8_fnuz_mfma_gemv_kernel<2>
        <<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a);
    break;
  case 4:
    glm_fp8_fnuz_mfma_gemv_kernel<4>
        <<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a);
    break;
  case 8:
    glm_fp8_fnuz_mfma_gemv_kernel<8>
        <<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a);
    break;
  case 16:
    glm_fp8_fnuz_mfma_gemv_kernel<16>
        <<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a);
    break;
  case 32:
    glm_fp8_fnuz_mfma_gemv_kernel<32>
        <<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a);
    break;
  default:
    glm_fp8_fnuz_mfma_gemv_kernel<0>
        <<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a);
    break;
  }
  const int mn = M * N;
  glm_fp8_gemv_reduce_kernel<<<(mn + kTh - 1) / kTh, kTh>>>(part, out, mn, sk);
}

void glm_fp8_fnuz_mfma_block_gemv(
    const uint16_t* x, const uint8_t* w, const float* scales, uint16_t* out, int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  const int sk = gemv_pick_sk(N, K);
  float* part = nullptr;
  hipMalloc(&part, (size_t)M * N * sk * sizeof(float));
  glm_fp8_fnuz_mfma_block_gemv_with_scratch(x, w, scales, out, part, M, N, K, sk);
  hipFree(part);
}

void glm_fp8_fnuz_block_gemv_with_scratch(const uint16_t* x,
                                          const uint8_t* w,
                                          const float* scales, uint16_t* out,
                                          float* part, int M, int N, int K,
                                          int sk) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (M > 2 || K > kMaxK || K % 128 != 0 || N % 128 != 0 || sk < 1 ||
      sk > 8 || (K / 128) % sk != 0) {
    std::fprintf(stderr,
                 "glm_fp8_fnuz_block_gemv: requires M<=2, K<=4096, N%%128==0, "
                 "K%%128==0, sk in {1,2,4,8} dividing K/128 (got M=%d N=%d "
                 "K=%d sk=%d)\n", M, N, K, sk);
    return;
  }
  const int segs = (K / 128) / sk;
  const dim3 grid((N + 31) / 32, sk);
  switch (segs) {
    case 2:  glm_fp8_fnuz_gemv_partial_kernel<2><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 4:  glm_fp8_fnuz_gemv_partial_kernel<4><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 8:  glm_fp8_fnuz_gemv_partial_kernel<8><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 16: glm_fp8_fnuz_gemv_partial_kernel<16><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    case 32: glm_fp8_fnuz_gemv_partial_kernel<32><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
    default: glm_fp8_fnuz_gemv_partial_kernel<0><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs); break;
  }
  const int mn = M * N;
  glm_fp8_gemv_reduce_kernel<<<(mn + kTh - 1) / kTh, kTh>>>(part, out, mn, sk);
}

void glm_fp8_fnuz_block_gemv(const uint16_t* x, const uint8_t* w,
                             const float* scales, uint16_t* out,
                             int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  const int sk = gemv_pick_sk(N, K);
  float* part = nullptr;
  hipMalloc(&part, (size_t)M * N * sk * sizeof(float));
  glm_fp8_fnuz_block_gemv_with_scratch(x, w, scales, out, part, M, N, K, sk);
  hipFree(part);
}

void glm_fp8_fnuz_mfma_block_gemv_with_scratch(const uint16_t* x,
                                               const uint8_t* w,
                                               const float* scales,
                                               uint16_t* out, float* part,
                                               int M, int N, int K, int sk) {
  glm_fp8_fnuz_mfma_block_gemv_debug(x, w, scales, out, part, M, N, K, sk,
                                     nullptr, nullptr);
}

void glm_fp8_fnuz_mfma_block_gemv_debug(const uint16_t* x,
                                        const uint8_t* w,
                                        const float* scales,
                                        uint16_t* out, float* part,
                                        int M, int N, int K, int sk,
                                        uint32_t* dbg_sxa, float* dbg_s2a) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (M > 2 || K > kMaxK || K % 128 != 0 || N % 128 != 0 || sk < 1 ||
      sk > 8 || (K / 128) % sk != 0) {
    std::fprintf(stderr,
                 "glm_fp8_fnuz_mfma_block_gemv: requires M<=2, K<=4096, "
                 "N%%128==0, K%%128==0, sk in {1,2,4,8} dividing K/128 "
                 "(got M=%d N=%d K=%d sk=%d)\n", M, N, K, sk);
    return;
  }
  const int segs = (K / 128) / sk;
  const dim3 grid(N / 64, sk);          // 64 rows per 4-wave64-warp block
  switch (segs) {
    case 2:  glm_fp8_fnuz_mfma_gemv_kernel<2><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a); break;
    case 4:  glm_fp8_fnuz_mfma_gemv_kernel<4><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a); break;
    case 8:  glm_fp8_fnuz_mfma_gemv_kernel<8><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a); break;
    case 16: glm_fp8_fnuz_mfma_gemv_kernel<16><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a); break;
    case 32: glm_fp8_fnuz_mfma_gemv_kernel<32><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a); break;
    default: glm_fp8_fnuz_mfma_gemv_kernel<0><<<grid, kTh>>>(x, w, scales, part, M, N, K, sk, segs, dbg_sxa, dbg_s2a); break;
  }
  const int mn = M * N;
  glm_fp8_gemv_reduce_kernel<<<(mn + kTh - 1) / kTh, kTh>>>(part, out, mn, sk);
}

void glm_fp8_fnuz_mfma_block_gemv(const uint16_t* x, const uint8_t* w,
                                  const float* scales, uint16_t* out,
                                  int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  const int sk = gemv_pick_sk(N, K);
  float* part = nullptr;
  hipMalloc(&part, (size_t)M * N * sk * sizeof(float));
  glm_fp8_fnuz_mfma_block_gemv_with_scratch(x, w, scales, out, part,
                                            M, N, K, sk);
  hipFree(part);
}

void glm_fp8_block_gemv(const uint16_t* x, const uint8_t* w,
                        const float* scales, uint16_t* out,
                        int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  const int sk = gemv_pick_sk(N, K);  // vkernels::kernels, store-aware
  float* part = nullptr;
  hipMalloc(&part, (size_t)M * N * sk * sizeof(float));
  glm_fp8_block_gemv_with_scratch(x, w, scales, out, part, M, N, K, sk);
  hipFree(part);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
