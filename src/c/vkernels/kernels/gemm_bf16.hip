// vkernels/kernels/gemm_bf16.hip
//
// bf16 GEMM on AMD gfx942 (CDNA3 / MI300A) via the K16 bf16 MFMA
// (__builtin_amdgcn_mfma_f32_16x16x16bf16_1k), for the Kimi-K3 projection
// shapes (issue #29). Compiled only with VKERNELS_HAS_HIP.
//
//   C[M, N] = alpha * A[M, K] @ B[K, N] + beta * C[M, N]   (bf16 in/out,
//                                                            fp32 accumulate)
//
// Layout (see gemm_bf16.hpp): A is [M, K], B is [K, N], C is [M, N], all
// dense row-major bf16 (uint16_t bit patterns). The serving recipe passes
// its [N, K] weight transposed as B.
//
// Kernel structure (mirrors the validated prefill helpers in moe_fused.hip):
//   * Each block owns an output tile [BM, BN]; one wavefront per 16-row
//     fragment (THREADS = (BM/16) * 64), so each wavefront computes
//     (BM/16) * (BN/16) MFMA fragments.
//   * Per K-tile (BK = 64), every thread cooperatively loads 4 bf16 (uint2)
//     of A and B into LDS with M/N/K bounds-checking (M, N, K may not be exact
//     multiples of the tile). The sB source row carries the K-tile
//     offset k_off (B's K-tile changes every iteration); A's M-tile is
//     fixed for the block, so only A's K column carries k_off.
//   * Four K=16 MFMAs consume the BK=64 K-tile, accumulating into per-
//     wavefront v4f accumulators (BN/16 of them, 4 VGPR each).
//   * Epilogue: col = lane % 16, row = warp*16 + (lane/16)*4 + i (the
//     verified 16x16x16bf16 C fragment layout), with M/N bounds-checks.
//
// Verified fragment layout (empirically confirmed on gfx90a, moe_fused.hip):
//   A: m = lane % 16, k0 = (lane/16)*4, a[i] = A[m][k0+i]
//   B: n = lane % 16, k0 = (lane/16)*4, b[i] = B[k0+i][n]
//   C: col = lane % 16, row = (lane/16)*4 + i, c[i] = C[row][col]
//
// Tuning. hip::gemm_bf16 selects a tile via gemm_bf16_config_for; the
// explicit-tile hip::gemm_bf16_with_config is the offline-autotuner hook
// used by meta/benchmarks/bench_gemm_bf16.hip to sweep candidate tiles.
#include <cstdint>

#include "vkernels/kernels/gemm_bf16.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#  include "vkernels/kernels/moe_device.hip"  // f32bits_to_bf16

namespace vkernels::kernels::hip {

typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v4s __attribute__((ext_vector_type(4)));

static constexpr int kMfmaK = 16;   // one bf16 MFMA reduces K=16
static constexpr int BK     = 64;   // K-tile (fixed); every K3 K is a multiple of 64

// bf16 -> float (top 16 bits of fp32, low 17 mantissa bits zero).
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  float f;
  uint32_t u = static_cast<uint32_t>(b) << 16;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}
// float -> bf16 (RNE, identical to f32bits_to_bf16 in moe_device.hip and to
// the host oracle in gemm_bf16.cpp).
__device__ __forceinline__ uint16_t f2bf(float v) {
  uint32_t b;
  __builtin_memcpy(&b, &v, sizeof(float));
  return f32bits_to_bf16(b);
}

// ======================================================================
//  Tiled bf16 MFMA GEMM kernel
// ======================================================================
// Each block computes C[m_tile*BM : +BM, n_tile*BN : +BN]. THREADS is
// (BM/16)*64: one wavefront per 16-row output fragment; that wavefront
// owns rows [warp*16, warp*16+16) and accumulates BN/16 column fragments.
template <int BM, int BN>
__global__ void gemm_bf16_kernel(const uint16_t* __restrict__ A,
                                  const uint16_t* __restrict__ B,
                                  uint16_t* __restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
  static_assert(BM % 16 == 0 && BN % 16 == 0, "tile must be a multiple of 16");
  static_assert(BK % kMfmaK == 0, "BK must be a multiple of the MFMA K");

  constexpr int kNF    = BN / 16;        // column fragments per wavefront
  constexpr int kWarps = BM / 16;        // one wavefront per 16-row fragment
  constexpr int kTh    = kWarps * 64;

  __shared__ uint16_t sA[BM][BK];
  __shared__ uint16_t sB[BK][BN];

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid    = threadIdx.x;
  const int warp   = tid >> 6;
  const int lane   = tid & 63;
  const int m      = lane & 15;     // row within the 16-row fragment
  const int n      = lane & 15;     // col within the 16-col fragment
  const int kq     = lane >> 4;     // 0..3 -> which 4-K group
  const int row_base = warp * 16;

  v4f acc[kNF];
  for (int nf = 0; nf < kNF; ++nf) acc[nf] = (v4f){0.0f, 0.0f, 0.0f, 0.0f};

  const int n_ktiles = (K + BK - 1) / BK;
  for (int kt = 0; kt < n_ktiles; ++kt) {
    const int k_off = kt * BK;

    // --- cooperative load of sA[BM][BK] (4 bf16 = 1 uint2 per chunk) ---
    constexpr int a_chunks = (BM * BK) / 4;
    for (int idx = tid; idx < a_chunks; idx += kTh) {
      const int row = idx / (BK / 4);
      const int col = (idx % (BK / 4)) * 4;
      const int gr = m_tile * BM + row;
      uint32_t* dst = (uint32_t*)(&sA[row][col]);
      if (gr < M && k_off + col + 3 < K) {
        const uint32_t* src =
            (const uint32_t*)(&A[(size_t)gr * K + k_off + col]);
        uint2 v = *(const uint2*)(src);
        dst[0] = v.x; dst[1] = v.y;
      } else {
        for (int t = 0; t < 4; ++t) {
          const int kc = k_off + col + t;
          sA[row][col + t] = (gr < M && kc < K) ? A[(size_t)gr * K + kc]
                                                 : (uint16_t)0;
        }
      }
    }

    // --- cooperative load of sB[BK][BN] (bounds-checked on K and N) ---
    constexpr int b_chunks = (BK * BN) / 4;
    for (int idx = tid; idx < b_chunks; idx += kTh) {
      const int row = idx / (BN / 4);
      const int col = (idx % (BN / 4)) * 4;
      const int gk = k_off + row;                  // global K row for this row
      const int gc = n_tile * BN + col;            // global N column (chunk)
      if (gk < K && gc + 3 < N) {
        const uint32_t* src = (const uint32_t*)(&B[(size_t)gk * N + gc]);
        uint32_t* dst = (uint32_t*)(&sB[row][col]);
        uint2 v = *(const uint2*)(src);
        dst[0] = v.x; dst[1] = v.y;
      } else {
        for (int t = 0; t < 4; ++t) {
          const int gcc = gc + t;
          sB[row][col + t] = (gk < K && gcc < N) ? B[(size_t)gk * N + gcc] : (uint16_t)0;
        }
      }
    }
    __syncthreads();

    // K-loop: BK/kMfmaK (= 4) MFMAs consume the K-tile.
    for (int mf = 0; mf < BK / kMfmaK; ++mf) {
      const int k0 = mf * kMfmaK + kq * 4;
      v4s a = {(short)sA[row_base + m][k0],     (short)sA[row_base + m][k0 + 1],
               (short)sA[row_base + m][k0 + 2], (short)sA[row_base + m][k0 + 3]};
#pragma unroll
      for (int nf = 0; nf < kNF; ++nf) {
        const int nb = nf * 16 + n;
        v4s b = {(short)sB[k0][nb],     (short)sB[k0 + 1][nb],
                 (short)sB[k0 + 2][nb], (short)sB[k0 + 3][nb]};
        acc[nf] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[nf], 0, 0, 0);
      }
    }
    __syncthreads();
  }

  // Epilogue. col = lane%16, row = warp*16 + (lane/16)*4 + i, per nf.
  const int col = lane & 15;
  const int row_off = (lane >> 4) * 4;
  for (int nf = 0; nf < kNF; ++nf) {
    const int gcol = n_tile * BN + nf * 16 + col;
    if (gcol >= N) break;              // gcol increases monotonically with nf
    for (int i = 0; i < 4; ++i) {
      const int grow = m_tile * BM + row_base + row_off + i;
      if (grow >= M) break;            // grow increases monotonically with i
      float v = alpha * acc[nf][i];
      if (beta != 0.0f) v += beta * bf16_to_f32(C[(size_t)grow * N + gcol]);
      C[(size_t)grow * N + gcol] = f2bf(v);
    }
  }
}

namespace {
template <int BM, int BN>
void launch(const uint16_t* A, const uint16_t* B, uint16_t* C,
            int M, int N, int K, float alpha, float beta) {
  constexpr int kTh = (BM / 16) * 64;
  dim3 block(kTh);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);  // x = M-tiles, y = N-tiles (moe_fused)
  gemm_bf16_kernel<BM, BN><<<grid, block, 0, 0>>>(
      A, B, C, M, N, K, alpha, beta);
}
}  // namespace

// Explicit-tile entry point (offline autotuner). BK is fixed at 64 (every
// K3 K is a multiple of 64); `threads` is derived per tile as (bm/16)*64.
// Unknown (bm, bn) fall back to the serving default (16, 64).
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads) {
  (void)bk;      // BK is fixed at 64 inside the kernel.
  (void)threads; // derived as (bm/16)*64 per tile.

  const int Mi = static_cast<int>(M);
  const int Ni = static_cast<int>(N);
  const int Ki = static_cast<int>(K);

  if (bm == 16 && bn == 16)        launch<16, 16>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 16 && bn == 64)   launch<16, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 16 && bn == 128)  launch<16, 128>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 32 && bn == 64)   launch<32, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 32 && bn == 128)  launch<32, 128>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 64 && bn == 64)   launch<64, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 64 && bn == 128)  launch<64, 128>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else                             launch<16, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
}

// Shape-selected entry point. Picks a tuned tile via gemm_bf16_config_for
// (the host-visible selector in vkernels::kernels) and dispatches.
void gemm_bf16(std::size_t M, std::size_t N, std::size_t K, float alpha,
               const uint16_t* A, const uint16_t* B, float beta,
               uint16_t* C) {
  int bm = 0, bn = 0, bk = 0, threads = 0;
  ::vkernels::kernels::gemm_bf16_config_for(M, N, K, &bm, &bn, &bk, &threads);
  gemm_bf16_with_config(M, N, K, alpha, A, B, beta, C, bm, bn, bk, threads);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
