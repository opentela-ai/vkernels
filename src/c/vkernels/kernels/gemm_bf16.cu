// vkernels/kernels/gemm_bf16.cu -- CUDA wmma implementation (NVIDIA), a port
// of the gfx942 MFMA kernel in gemm_bf16.hip (issue #29) for an NVIDIA
// deployment track (verified on GB10 / Grace Blackwell, sm_121).
//
//   C[M, N] = alpha * A[M, K] @ B[K, N] + beta * C[M, N]   (bf16 in/out,
//                                                            fp32 accumulate)
//
// Uses nvcuda::wmma 16x16x16 bf16 tensor cores (supported sm_70+, legacy path
// on Blackwell). Each block owns an [BM, BN] output tile; one warp (32
// threads) per 16x16 output fragment, so blockDim = (BM/16)*(BN/16)*32.
// K is tiled at BK = 64 (every K3 K is a multiple of 64): each K-tile stages
// sA[BM][64] and sB[64][BN] in shared memory, then BK/16 = 4 wmma mma_syncs
// (K = 16 each) consume it. The epilogue stages the fp32 accumulator through
// a per-warp shared buffer, applies alpha/beta with the SAME round-to-
// nearest-even bf16 store as gemm_bf16_cpu (the oracle), and writes global C
// with M/N bounds-checking.
//
// Same contract / config as the HIP kernel (see gemm_bf16.hpp): cuda::gemm_bf16
// selects a tile via gemm_bf16_config_for; cuda::gemm_bf16_with_config is the
// explicit-tile autotuner hook (forward-declared by the bench, not in the
// public header -- same convention as hip::gemm_bf16_with_config).
#include "vkernels/kernels/gemm_bf16.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>
#  include <mma.h>

namespace vkernels::kernels::cuda {

using namespace nvcuda::wmma;

static constexpr int kMmaK = 16;   // one bf16 wmma reduces K = 16
static constexpr int BK    = 64;   // K-tile (fixed); every K3 K is a multiple of 64

// bf16 <-> f32, bit-exact with gemm_bf16_cpu (the oracle). Replicated here so
// the .cu TU is self-contained on a host+device build (the HIP helpers in
// moe_device.hip are unavailable under VKERNELS_HAS_CUDA).
static __device__ __forceinline__ uint16_t f2bf(float f) {
  union { float f; uint32_t u; } x;
  x.f = f;
  uint32_t b = x.u;
  uint32_t lsb = (b >> 16) & 1u;
  b += 0x7FFFu + lsb;
  return static_cast<uint16_t>(b >> 16);
}
static __device__ __forceinline__ float bf2f(uint16_t v) {
  union { uint32_t u; float f; } x;
  x.u = static_cast<uint32_t>(v) << 16;
  return x.f;
}

// ======================================================================
//  Tiled bf16 wmma GEMM kernel
// ======================================================================
// Each block computes C[m_tile*BM : +BM, n_tile*BN : +BN]. One warp per
// 16x16 output fragment (kWarps = (BM/16)*(BN/16), kTh = kWarps*32); warp
// (warp_r, warp_c) owns rows [warp_r*16, +16) and cols [warp_c*16, +16).
template <int BM, int BN>
__global__ void gemm_bf16_kernel(const uint16_t* __restrict__ A,
                                  const uint16_t* __restrict__ B,
                                  uint16_t* __restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
  static_assert(BM % 16 == 0 && BN % 16 == 0, "tile must be a multiple of 16");
  static_assert(BK % kMmaK == 0, "BK must be a multiple of the wmma K");

  constexpr int kFragsM = BM / 16;
  constexpr int kFragsN = BN / 16;
  constexpr int kWarps  = kFragsM * kFragsN;
  constexpr int kTh     = kWarps * 32;

  __shared__ uint16_t sA[BM][BK];
  __shared__ uint16_t sB[BK][BN];
  // Per-warp fp32 accumulator staging. Lives in DYNAMIC shared memory (see
  // launch<>): at (BM,BN)=(64,128) it is 32 warps*256 = 32 KB, which with the
  // static sA/sB (24 KB) would bust the 48 KB *static* cap. Dynamic shared
  // has its own (opt-in) cap, so the total 56 KB fits on GB10 (~228 KB/SM).
  extern __shared__ float sc[];  // per-warp fp32 accumulator staging

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;        // 0 .. kWarps-1
  const int warp_r = warp / kFragsN;  // which 16-row fragment
  const int warp_c = warp % kFragsN;  // which 16-col fragment
  const int lane   = tid & 31;

  fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  const int n_ktiles = (K + BK - 1) / BK;
  for (int kt = 0; kt < n_ktiles; ++kt) {
    const int k_off = kt * BK;

    // --- cooperative load of sA[BM][BK] (bounds-checked on M and K) ---
    for (int idx = tid; idx < BM * BK; idx += kTh) {
      const int r = idx / BK, c = idx % BK;
      const int gr = m_tile * BM + r, gk = k_off + c;
      sA[r][c] = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : (uint16_t)0;
    }
    // --- cooperative load of sB[BK][BN] (bounds-checked on K and N) ---
    for (int idx = tid; idx < BK * BN; idx += kTh) {
      const int r = idx / BN, c = idx % BN;
      const int gk = k_off + r, gc = n_tile * BN + c;
      sB[r][c] = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : (uint16_t)0;
    }
    __syncthreads();

    // K-loop: BK/kMmaK (= 4) wmma mma_syncs consume the K-tile.
#pragma unroll
    for (int kk = 0; kk < BK; kk += kMmaK) {
      load_matrix_sync(a_frag, (const __nv_bfloat16*)&sA[warp_r * 16][kk], BK);
      load_matrix_sync(b_frag, (const __nv_bfloat16*)&sB[kk][warp_c * 16], BN);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
  }

  // Epilogue. Stage this warp's fp32 accumulator, apply alpha/beta with the
  // oracle's RNE bf16 store, and write global C with M/N bounds-checks.
  store_matrix_sync(&sc[warp * 256], c_frag, 16, mem_row_major);
  for (int i = lane; i < 256; i += 32) {
    const int r = i >> 4, c = i & 15;
    const int gr = m_tile * BM + warp_r * 16 + r;
    const int gc = n_tile * BN + warp_c * 16 + c;
    if (gr < M && gc < N) {
      float v = alpha * sc[warp * 256 + i];
      if (beta != 0.0f) v += beta * bf2f(C[(size_t)gr * N + gc]);
      C[(size_t)gr * N + gc] = f2bf(v);
    }
  }
}

namespace {
template <int BM, int BN>
void launch(const uint16_t* A, const uint16_t* B, uint16_t* C,
            int M, int N, int K, float alpha, float beta) {
  constexpr int kWarps = (BM / 16) * (BN / 16);
  constexpr int kTh = kWarps * 32;
  // Per-warp fp32 accumulator staging: one 16x16 (=256) float per warp.
  constexpr int dynamic_bytes = kWarps * 16 * 16 * (int)sizeof(float);
  // Opt in to the larger-than-default dynamic shared region ONCE per kernel
  // instantiation. cudaFuncAttributeMaxDynamicSharedMemorySize is a
  // per-context, per-function attribute that persists, so setting it on
  // every launch (as in a timing loop) perturbs the event clock and can
  // make cudaEventElapsedTime return 0. Needed only when static sA/sB +
  // dynamic sc exceed the 48 KB default per-block cap (e.g. the 64x128
  // tile: 24 KB static + 32 KB dynamic = 56 KB); harmless for smaller tiles
  // (all stay under sharedMemPerBlockOptin, ~99 KB on GB10).
  static const bool inited = [] {
    cudaFuncSetAttribute((const void*)gemm_bf16_kernel<BM, BN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dynamic_bytes);
    return true;
  }();
  (void)inited;
  dim3 block(kTh);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);  // x = M-tiles, y = N-tiles
  gemm_bf16_kernel<BM, BN><<<grid, block, dynamic_bytes, 0>>>(
      A, B, C, M, N, K, alpha, beta);
}
}  // namespace

// Explicit-tile entry point (offline autotuner). BK is fixed at 64 inside the
// kernel; `threads` is derived as (bm/16)*(bn/16)*32 for CUDA. Unknown
// (bm, bn) fall back to the serving default (16, 16).
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads) {
  (void)bk;       // BK is fixed at 64 inside the kernel.
  (void)threads;  // derived as (bm/16)*(bn/16)*32 per tile.

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
  else                             launch<16, 16>(A, B, C, Mi, Ni, Ki, alpha, beta);
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

}  // namespace vkernels::kernels::cuda

#endif  // VKERNELS_HAS_CUDA
