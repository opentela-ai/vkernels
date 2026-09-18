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
//
// Issue #77 adds the next two rungs of the hip-gemm-tuning ladder for the
// warmup (M > 64) regime, mirroring the validated GB10 (CUDA) port of
// commit 1fa1f2e:
//   * gemm_bf16_kernel_reuse<BM, BN, RM> -- cross-tile B reuse: each block
//     owns RM consecutive M-tiles for ONE N-tile, so sB is loaded once per
//     K-tile and consumed by all RM M-tiles (global B traffic / RM vs RM
//     separate blocks), and sA/sB live in a two-deep LDS ring so the load
//     of K-tile kt+1 overlaps the MFMAs of tile kt (double buffering).
//   * The b fragment (sB -> registers) is read once per (K-group, column
//     fragment) per warp and amortised over RM MFMA accumulate updates.
// On gfx942 the on-device M=8192 autotuner measured this schedule ~2x
// SLOWER than the flat (64,64) tile (the 24 KB double-buffered LDS ring at
// 128 threads caps occupancy at ~25%; see gfx942.md), so hip::gemm_bf16
// keeps the flat warmup tile and the reuse kernel is exposed only through
// gemm_bf16_reuse_with_config for the autotuner and the correctness sweep.
//
// Semantics are unchanged: same fp32-accumulate / RNE-store contract as
// gemm_bf16_cpu, verified by test_gemm_bf16_correct.hip.
#include <cstdint>

#include <mutex>

#include "vkernels/kernels/gemm_bf16.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#  include "vkernels/kernels/moe_device.hip"  // shared bf16<->f32 + fp4 helpers

namespace vkernels::kernels::hip {

typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v4s __attribute__((ext_vector_type(4)));

static constexpr int kMfmaK = 16;   // one bf16 MFMA reduces K=16
static constexpr int BK     = 64;   // K-tile (fixed); every K3 K is a multiple of 64


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

// ======================================================================
//  Split-K kernel (issue #146) -- serving / tiny-M regime
// ======================================================================
// The two-phase kernel above is latency-bound at serving M <= 64: the grid
// is (ceil(M/BM), ceil(N/BN)) and every block walks ALL n_ktiles = K/64
// K-tiles serially, so at M=5 the QKV shape runs 393 blocks x 64 threads
// (~1.7 blocks/CU on 228 CUs) with the per-K-tile global-load latency +
// barrier fully exposed (~20x above the ~17 us DRAM floor; measured 346 us).
//
// Split-K adds the missing K-parallel axis (the documented rung 3 of the
// hip-gemm-tuning ladder, gfx942.md "Future work"). blockIdx.z = s in
// [0, S) owns the K-tile range [kt0, kt1); the body is the two-phase tile
// UNCHANGED (same 4 KB LDS, same fragment layout -- no #77-style occupancy
// risk), and the epilogue stores the RAW fp32 accumulator (no alpha/beta)
// to the workspace plane P[s][M][N] instead of bf16 C. A separate combine
// kernel folds the S planes in FIXED ascending order (deterministic, fp32
// partials; same contract as the mla_fwd split-K, issue #82), then applies
// alpha / beta*C / the single RNE bf16 store.
//
// Grid (ceil(M/BM), ceil(N/BN), S); with (BM, BN) = (16, 16) and S = 8 the
// M=5 QKV grid grows 393 -> 3144 blocks with 8x shorter serial K-chains.
template <int BM, int BN>
__global__ void gemm_bf16_splitk_kernel(const uint16_t* __restrict__ A,
                                        const uint16_t* __restrict__ B,
                                        float* __restrict__ P,
                                        int M, int N, int K,
                                        int kts_per_split) {
  static_assert(BM % 16 == 0 && BN % 16 == 0, "tile must be a multiple of 16");

  constexpr int kNF    = BN / 16;
  constexpr int kWarps = BM / 16;
  constexpr int kTh    = kWarps * 64;

  __shared__ uint16_t sA[BM][BK];
  __shared__ uint16_t sB[BK][BN];

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int s      = blockIdx.z;      // split index
  const int tid    = threadIdx.x;
  const int warp   = tid >> 6;
  const int lane   = tid & 63;
  const int m      = lane & 15;
  const int n      = lane & 15;
  const int kq     = lane >> 4;
  const int row_base = warp * 16;

  const int n_ktiles = (K + BK - 1) / BK;
  // Whole K-tiles per split; the last split absorbs the remainder. Splits
  // past the last tile keep their (zero) accumulators so the combine can
  // fold all S planes unconditionally.
  const int kt0 = s * kts_per_split;
  const int kt1 = min(kt0 + kts_per_split, n_ktiles);

  v4f acc[kNF];
  for (int nf = 0; nf < kNF; ++nf) acc[nf] = (v4f){0.0f, 0.0f, 0.0f, 0.0f};

  for (int kt = kt0; kt < kt1; ++kt) {
    const int k_off = kt * BK;

    // --- cooperative load of sA[BM][BK] (identical to the two-phase kernel)
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

    // --- cooperative load of sB[BK][BN] (identical to the two-phase kernel)
    constexpr int b_chunks = (BK * BN) / 4;
    for (int idx = tid; idx < b_chunks; idx += kTh) {
      const int row = idx / (BN / 4);
      const int col = (idx % (BN / 4)) * 4;
      const int gk = k_off + row;
      const int gc = n_tile * BN + col;
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

  // Epilogue: RAW fp32 partials (no alpha/beta) to this split's plane.
  // Same fragment layout as the two-phase kernel: col = lane%16,
  // row = warp*16 + (lane/16)*4 + i.
  float* plane = P + (size_t)s * M * N;
  const int col = lane & 15;
  const int row_off = (lane >> 4) * 4;
  for (int nf = 0; nf < kNF; ++nf) {
    const int gcol = n_tile * BN + nf * 16 + col;
    if (gcol >= N) break;
    for (int i = 0; i < 4; ++i) {
      const int grow = m_tile * BM + row_base + row_off + i;
      if (grow >= M) break;
      plane[(size_t)grow * N + gcol] = acc[nf][i];
    }
  }
}

// Combine kernel (issue #146): one thread per output element, blocks tile
// 256 columns of one row. Folds the S workspace planes in FIXED ascending
// order (deterministic fp32 partial sum, same contract as the mla_fwd
// split-K combine), then applies alpha / beta*C and the single RNE bf16
// store. Planes are contiguous ([S][M][N]) so each fixed-s read sweep is
// coalesced; the workspace round-trip is ~2x MN fp32 (<= ~26 MB at the
// largest serving shape, ~5 us at roof) against a 20x latency win.
__global__ void gemm_bf16_splitk_combine_kernel(
    const float* __restrict__ P, const uint16_t* __restrict__ C_in,
    uint16_t* __restrict__ C, int M, int N, int S,
    float alpha, float beta) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float sum = 0.0f;
  const float* p0 = P + (size_t)row * N + col;
  const size_t plane = (size_t)M * N;
  for (int s = 0; s < S; ++s) sum += p0[(size_t)s * plane];

  float v = alpha * sum;
  if (beta != 0.0f) v += beta * bf16_to_f32(C_in[(size_t)row * N + col]);
  C[(size_t)row * N + col] = f2bf(v);
}

// ======================================================================
//  Cross-tile B-reuse + LDS double-buffer kernel (gfx942, issue #77)
// ======================================================================
// Each block owns RM consecutive M-tiles for ONE N-tile: for a given
// K-tile, sB[BK][BN] is loaded once from global memory and consumed by all
// RM M-tiles, so global B traffic drops RM-fold vs RM separate blocks and
// the b fragment (sB -> register) load is amortised over RM MFMA updates.
//
// LDS double-buffering: sA and sB live in a two-deep ring; the K-tile
// kt+1 loads are issued (as plain global loads, which the memory system
// keeps in flight) BEFORE the MFMAs of tile kt execute, so the global->LDS
// transfer of tile kt+1 overlaps the matrix-core work of tile kt. One
// __syncthreads() per K-tile (vs two in the two-phase kernel) both orders
// the ring and publishes the prefetch; the writes target the buffer that
// iteration kt-1 finished reading (or, at kt = 0, an untouched slot), so
// there is no WAR hazard on it.
//
// Warp mapping (cf. the two-phase kernel above): kWarps = (BM/16)*(BN/16)
// wavefronts; wavefront (warp_r, warp_c) owns rows [warp_r*16, +16) of
// EVERY M-panel and columns [warp_c*16, +16), accumulating RM v4f
// fragments (RM * kNF registers per thread -- the register cost of the
// reuse). Requires RM * kNF * 4 VGPRs to stay well under the 256-reg
// budget: the compiled configs are (16,64,R4), (32,64,R2), (32,64,R4).
template <int BM, int BN, int RM>
__global__ void gemm_bf16_kernel_reuse(const uint16_t* __restrict__ A,
                                       const uint16_t* __restrict__ B,
                                       uint16_t* __restrict__ C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
  static_assert(BM % 16 == 0 && BN % 16 == 0, "tile must be a multiple of 16");
  static_assert(BK % kMfmaK == 0, "BK must be a multiple of the MFMA K");

  constexpr int kNF    = BN / 16;                 // col fragments per warp
  constexpr int kWarps = (BM / 16) * kNF;
  constexpr int kTh    = kWarps * 64;

  __shared__ uint16_t sA[2][RM][BM][BK];  // two-deep ring, RM M-panels
  __shared__ uint16_t sB[2][BK][BN];      // two-deep ring, shared by panels

  const int m_base = blockIdx.x * (BM * RM);  // first global row of block
  const int n_tile = blockIdx.y;
  const int tid    = threadIdx.x;
  const int warp   = tid >> 6;
  const int lane   = tid & 63;
  const int warp_r = warp / kNF;
  const int warp_c = warp % kNF;
  const int m      = lane & 15;     // row within the 16-row fragment
  const int n      = lane & 15;     // col within the 16-col fragment
  const int kq     = lane >> 4;     // 0..3 -> which 4-K group
  const int row_off = (lane >> 4) * 4;  // epilogue row within the fragment

  const int n_ktiles = (K + BK - 1) / BK;

  // Cooperative uint2 load of K-tile kt into ring slot buf (bounds-checked
  // to 0 on M, N and K -- same convention as the two-phase kernel).
  auto load_tile = [&](int kt, int buf) {
    const int k_off = kt * BK;

    // sA[RM][BM][BK]: 4 bf16 = 1 uint2 per chunk.
    constexpr int a_chunks = RM * BM * BK / 4;
    for (int idx = tid; idx < a_chunks; idx += kTh) {
      const int mm  = idx / (BM * (BK / 4));
      const int rem = idx % (BM * (BK / 4));
      const int row = rem / (BK / 4);
      const int col = (rem % (BK / 4)) * 4;
      const int gr  = m_base + mm * BM + row;
      uint32_t* dst = (uint32_t*)(&sA[buf][mm][row][col]);
      if (gr < M && k_off + col + 3 < K) {
        const uint32_t* src =
            (const uint32_t*)(&A[(size_t)gr * K + k_off + col]);
        uint2 v = *(const uint2*)(src);
        dst[0] = v.x; dst[1] = v.y;
      } else {
        for (int t = 0; t < 4; ++t) {
          const int kc = k_off + col + t;
          sA[buf][mm][row][col + t] =
              (gr < M && kc < K) ? A[(size_t)gr * K + kc] : (uint16_t)0;
        }
      }
    }

    // sB[BK][BN]: bounds-checked on K and N.
    constexpr int b_chunks = BK * BN / 4;
    for (int idx = tid; idx < b_chunks; idx += kTh) {
      const int row = idx / (BN / 4);
      const int col = (idx % (BN / 4)) * 4;
      const int gk  = k_off + row;                 // global K row
      const int gc  = n_tile * BN + col;           // global N column (chunk)
      if (gk < K && gc + 3 < N) {
        const uint32_t* src = (const uint32_t*)(&B[(size_t)gk * N + gc]);
        uint32_t* dst = (uint32_t*)(&sB[buf][row][col]);
        uint2 v = *(const uint2*)(src);
        dst[0] = v.x; dst[1] = v.y;
      } else {
        for (int t = 0; t < 4; ++t) {
          const int gcc = gc + t;
          sB[buf][row][col + t] =
              (gk < K && gcc < N) ? B[(size_t)gk * N + gcc] : (uint16_t)0;
        }
      }
    }
  };

  load_tile(0, 0);
  __syncthreads();

  v4f acc[RM][kNF];
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) acc[r][nf] = (v4f){0.0f, 0.0f, 0.0f, 0.0f};

  for (int kt = 0; kt < n_ktiles; ++kt) {
    const int buf = kt & 1;
    if (kt + 1 < n_ktiles) load_tile(kt + 1, buf ^ 1);  // prefetch, in flight

    // K-loop: BK/kMfmaK (= 4) MFMAs per (panel, column fragment); the b
    // fragment is loaded once per K-group and consumed by all RM panels.
    const int nb = warp_c * 16 + n;
    for (int mf = 0; mf < BK / kMfmaK; ++mf) {
      const int k0 = mf * kMfmaK + kq * 4;
      v4s b = {(short)sB[buf][k0][nb],     (short)sB[buf][k0 + 1][nb],
               (short)sB[buf][k0 + 2][nb], (short)sB[buf][k0 + 3][nb]};
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        const int ar = warp_r * 16 + m;
        v4s a = {(short)sA[buf][r][ar][k0],     (short)sA[buf][r][ar][k0 + 1],
                 (short)sA[buf][r][ar][k0 + 2], (short)sA[buf][r][ar][k0 + 3]};
        acc[r][warp_c] =
            __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[r][warp_c],
                                                      0, 0, 0);
      }
    }
    // Orders the ring: all reads of `buf` done, all prefetch writes into
    // buf^1 (issued this iteration) complete before anyone moves on.
    __syncthreads();
  }

  // Epilogue (same fragment layout as the two-phase kernel): col = lane%16,
  // row = warp_r*16 + (lane/16)*4 + i within each panel, bounds-checked.
  const int col = lane & 15;
  const int gcol = n_tile * BN + warp_c * 16 + col;
  if (gcol >= N) return;  // uniform per lane across the RM panels
  for (int r = 0; r < RM; ++r) {
    for (int i = 0; i < 4; ++i) {
      const int grow = m_base + r * BM + warp_r * 16 + row_off + i;
      if (grow >= M) break;          // grow increases monotonically with i
      float v = alpha * acc[r][warp_c][i];
      if (beta != 0.0f) v += beta * bf16_to_f32(C[(size_t)grow * N + gcol]);
      C[(size_t)grow * N + gcol] = f2bf(v);
    }
  }
}

// Device workspace for the split-K fp32 partial planes (issue #146):
// grow-only, shared across consecutive gemm_bf16 launches -- the same
// pattern as the mla_fwd split workspace (issue #82). Both kernels of one
// launch are enqueued under the lock that (re)sizes the buffer, so a later
// resize's hipFree (which synchronises the device) can never race a
// not-yet-launched kernel against a freed buffer.
struct SplitKWorkspace {
  std::mutex mu;
  float* partials = nullptr;
  size_t cap = 0;  // elements
};
SplitKWorkspace& splitk_ws() {
  static SplitKWorkspace w;
  return w;
}

namespace {

// Split-K launch: sizes the grow-only workspace, then enqueues the split
// kernel (grid (m-tiles, n-tiles, S)) + fixed-order combine on the default
// stream. S is the caller-chosen split count (whole K-tiles per split are
// guaranteed by the caller: S <= ceil(K/BK)); (bm, bn) must be a compiled
// tile. Returns false when the workspace cannot be sized; the public entry
// then falls back to the unsplit two-phase kernel.
template <int BM, int BN>
bool launch_splitk(const uint16_t* A, const uint16_t* B, uint16_t* C,
                   int M, int N, int K, float alpha, float beta, int S) {
  const size_t n = (size_t)S * M * N;
  auto& w = splitk_ws();
  std::lock_guard<std::mutex> lock(w.mu);
  if (w.cap < n) {
    float* p = nullptr;
    if (hipMalloc(&p, n * sizeof(float)) == hipSuccess) {
      if (w.partials) hipFree(w.partials);
      w.partials = p;
      w.cap = n;
    }
  }
  if (w.cap < n) return false;

  const int n_ktiles = (K + BK - 1) / BK;
  const int kts = (n_ktiles + S - 1) / S;  // whole tiles per split

  constexpr int kTh = (BM / 16) * 64;
  dim3 block(kTh);
  dim3 gridS((M + BM - 1) / BM, (N + BN - 1) / BN, S);
  gemm_bf16_splitk_kernel<BM, BN><<<gridS, block, 0, 0>>>(
      A, B, w.partials, M, N, K, kts);
  const int cols = 256;
  dim3 gridC(M, (N + cols - 1) / cols);
  gemm_bf16_splitk_combine_kernel<<<gridC, cols, 0, 0>>>(
      w.partials, C, C, M, N, S, alpha, beta);
  return true;
}

template <int BM, int BN>
void launch(const uint16_t* A, const uint16_t* B, uint16_t* C,
            int M, int N, int K, float alpha, float beta) {
  constexpr int kTh = (BM / 16) * 64;
  dim3 block(kTh);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);  // x = M-tiles, y = N-tiles (moe_fused)
  gemm_bf16_kernel<BM, BN><<<grid, block, 0, 0>>>(
      A, B, C, M, N, K, alpha, beta);
}

template <int BM, int BN, int RM>
void launch_reuse(const uint16_t* A, const uint16_t* B, uint16_t* C,
                  int M, int N, int K, float alpha, float beta) {
  constexpr int kWarps = (BM / 16) * (BN / 16);
  constexpr int kTh    = kWarps * 64;
  dim3 block(kTh);
  // M grid shrinks by RM: each block covers RM consecutive M-tiles.
  dim3 grid((M + BM * RM - 1) / (BM * RM), (N + BN - 1) / BN);
  gemm_bf16_kernel_reuse<BM, BN, RM><<<grid, block, 0, 0>>>(
      A, B, C, M, N, K, alpha, beta);
}
}  // namespace

// Defined below (the explicit-tile autotuner entry); used by the split-K
// dispatcher's fallbacks.
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads);

// Explicit split-K entry point (issue #146): (bm, bn) is a compiled tile,
// S the split count (clamped to [1, ceil(K/BK)]). Unknown tiles fall back
// to the unsplit (16, 64) two-phase kernel; an unsizeable workspace falls
// back to the unsplit selected tile. alpha/beta semantics are the public
// contract (applied once, in the combine).
void gemm_bf16_splitk_with_config(std::size_t M, std::size_t N, std::size_t K,
                                  float alpha, const uint16_t* A,
                                  const uint16_t* B, float beta, uint16_t* C,
                                  int bm, int bn, int S) {
  const int Mi = static_cast<int>(M);
  const int Ni = static_cast<int>(N);
  const int Ki = static_cast<int>(K);
  int s = S;
  const int max_s = (Ki + BK - 1) / BK;
  if (s < 1) s = 1;
  if (s > max_s) s = max_s;

  bool ok;
  if (bm == 16 && bn == 16)       ok = launch_splitk<16, 16>(A, B, C, Mi, Ni, Ki, alpha, beta, s);
  else if (bm == 16 && bn == 64)  ok = launch_splitk<16, 64>(A, B, C, Mi, Ni, Ki, alpha, beta, s);
  else if (bm == 32 && bn == 64)  ok = launch_splitk<32, 64>(A, B, C, Mi, Ni, Ki, alpha, beta, s);
  else if (bm == 64 && bn == 64)  ok = launch_splitk<64, 64>(A, B, C, Mi, Ni, Ki, alpha, beta, s);
  else {
    // Uncompiled tile: unsplit fallback on the selected tile.
    gemm_bf16_with_config(M, N, K, alpha, A, B, beta, C, bm, bn, 64, (bm / 16) * 64);
    return;
  }
  if (!ok) gemm_bf16_with_config(M, N, K, alpha, A, B, beta, C, bm, bn, 64, (bm / 16) * 64);
}

// Cross-tile B-reuse entry point (offline autotuner, issue #77). (bm, bn)
// is the per-M-tile tile shape and rm the M-tiles per block (the block's
// effective output footprint is (rm*bm, bn)); the benchmark pairs them.
// Unknown (bm, bn, rm) fall back to the two-phase kernel (16, 64).
void gemm_bf16_reuse_with_config(std::size_t M, std::size_t N, std::size_t K,
                                 float alpha, const uint16_t* A,
                                 const uint16_t* B, float beta, uint16_t* C,
                                 int bm, int bn, int rm) {
  const int Mi = static_cast<int>(M);
  const int Ni = static_cast<int>(N);
  const int Ki = static_cast<int>(K);
  if (rm == 4 && bm == 16 && bn == 64)
    launch_reuse<16, 64, 4>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (rm == 2 && bm == 32 && bn == 64)
    launch_reuse<32, 64, 2>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (rm == 4 && bm == 32 && bn == 64)
    launch_reuse<32, 64, 4>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else
    launch<16, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
}

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
  // Serving / tiny-M regime (issue #146): the two-phase kernel is
  // latency-bound here -- the grid leaves ~1.7 blocks/CU and every block
  // walks the full K/64 K-tiles serially with the per-tile load latency +
  // barrier exposed (measured 346 us at M=5 on the QKV shape, ~20x above
  // the ~17 us DRAM floor). Route M <= 64 to the split-K kernel (rung 3 of
  // the hip-gemm-tuning ladder, untried before this change): same 4 KB-LDS
  // tile, so the #77 occupancy-collapse failure mode does not apply. S = 8
  // gives 8x shorter serial chains and 8x the blocks; gated on K carrying
  // at least 4 whole K-tiles so each split still does meaningful MFMA work.
  if (M <= 64 && K >= 4 * BK) {
    const int s = (static_cast<int>(K) / BK) < 8 ? (static_cast<int>(K) / BK) : 8;
    gemm_bf16_splitk_with_config(M, N, K, alpha, A, B, beta, C, bm, bn, s);
    return;
  }
  // The per-arch selector owns the warmup schedule. On GB10 it reports the
  // (64,64) footprint realised as the cross-tile B-reuse + cp.async kernel
  // (issue #77); on gfx942 the on-device M=8192 reuse sweep measured that
  // schedule ~2x SLOWER than the flat (64,64) tile (occupancy collapse from
  // the double-buffered LDS ring -- see gemm_bf16.cpp and gfx942.md), so the
  // AMD path stays on the flat tile. gemm_bf16_reuse_with_config remains
  // available here for the autotuner and the correctness sweep.
  gemm_bf16_with_config(M, N, K, alpha, A, B, beta, C, bm, bn, bk, threads);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
