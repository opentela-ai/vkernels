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
#  include "vkernels/kernels/device_numeric.cuh"  // bf16<->f32 (bit-exact with the oracle)

namespace vkernels::kernels::cuda {

using namespace nvcuda::wmma;

static constexpr int kMmaK = 16;   // one bf16 wmma reduces K = 16
static constexpr int BK    = 64;   // K-tile (fixed); every K3 K is a multiple of 64

// ---------------------------------------------------------------------------
// cp.async helpers (sm_80+). 16-byte shared<-global copies that do not occupy
// a register and complete asynchronously, so the LSU can fetch the next K-tile
// while the tensor cores chew the current one. `valid == false` zero-fills the
// 16 bytes instead of reading (the `src-size = 0` form); the source pointer is
// then never dereferenced, so we pass the tensor base as a safe dummy.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem,
                                            bool valid) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const unsigned saddr = (unsigned)__cvta_generic_to_shared(smem);
  if (valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr),
                 "l"(gmem));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(
                     saddr),
                 "l"(gmem));
  }
#else
  const uint4 z = make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(smem) =
      valid ? *reinterpret_cast<const uint4*>(gmem) : z;
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait_1() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_all;\n" ::);
#endif
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
      if (beta != 0.0f) v += beta * bf16_to_f32(C[(size_t)gr * N + gc]);
      C[(size_t)gr * N + gc] = f2bf(v);
    }
  }
}

namespace {

// ======================================================================
//  Double-buffered / cp.async tiled kernel (experimental, sm_80+)
// ======================================================================
// Same tile mapping and epilogue as gemm_bf16_kernel, but the K-tile loads
// are issued with cp.async into a two-deep shared ring and only awaited one
// tile ahead, so the global->shared transfer of tile kt+1 overlaps the wmma
// of tile kt. This targets the measured GB10 bottleneck (latency/serialisation
// bound: ncu shows SM 33-47%, L1TEX 62-79%, DRAM only 48-58%), not DRAM
// traffic. Requires K % 8 == 0 and N % 8 == 0 (every K3 shape satisfies both)
// so the 16-byte chunks never straddle a row boundary; callers fall back to
// the synchronous kernel otherwise.
template <int BM, int BN>
__global__ void gemm_bf16_kernel_db(const uint16_t* __restrict__ A,
                                    const uint16_t* __restrict__ B,
                                    uint16_t* __restrict__ C,
                                    int M, int N, int K,
                                    float alpha, float beta) {
  static_assert(BM % 16 == 0 && BN % 16 == 0, "tile must be a multiple of 16");
  constexpr int kFragsM = BM / 16;
  constexpr int kFragsN = BN / 16;
  constexpr int kWarps  = kFragsM * kFragsN;
  constexpr int kTh     = kWarps * 32;
  constexpr int kAChunks = BM * BK / 8;  // 16-byte chunks, 8 bf16 each
  constexpr int kBChunks = BK * BN / 8;
  constexpr int kAChunksPerRow = BK / 8;
  constexpr int kBChunksPerRow = BN / 8;

  __shared__ uint16_t sA[2][BM][BK];  // two-deep ring
  __shared__ uint16_t sB[2][BK][BN];
  extern __shared__ float sc[];  // per-warp fp32 accumulator staging

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_r = warp / kFragsN;
  const int warp_c = warp % kFragsN;
  const int lane   = tid & 31;

  const int n_ktiles = (K + BK - 1) / BK;

  // Issue the async copy of K-tile `kt` into ring slot `buf`.
  auto issue = [&](int kt, int buf) {
    const int k_off = kt * BK;
    for (int c = tid; c < kAChunks; c += kTh) {
      const int r = c / kAChunksPerRow;
      const int col = (c % kAChunksPerRow) * 8;
      const int gr = m_tile * BM + r, gk = k_off + col;
      const bool valid = (gr < M) && (gk + 8 <= K);
      cp_async_16(&sA[buf][r][col], valid ? &A[(size_t)gr * K + gk] : A,
                  valid);
    }
    for (int c = tid; c < kBChunks; c += kTh) {
      const int r = c / kBChunksPerRow;
      const int col = (c % kBChunksPerRow) * 8;
      const int gk = k_off + r, gc = n_tile * BN + col;
      const bool valid = (gk < K) && (gc + 8 <= N);
      cp_async_16(&sB[buf][r][col], valid ? &B[(size_t)gk * N + gc] : B,
                  valid);
    }
  };

  issue(0, 0);
  cp_async_commit();

  fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  for (int kt = 0; kt < n_ktiles; ++kt) {
    const int buf  = kt & 1;
    const bool more = (kt + 1 < n_ktiles);
    if (more) {
      issue(kt + 1, buf ^ 1);
      cp_async_commit();
      cp_async_wait_1();  // wait for tile kt while kt+1 stays in flight
    } else {
      cp_async_wait_all();
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += kMmaK) {
      load_matrix_sync(a_frag, (const __nv_bfloat16*)&sA[buf][warp_r * 16][kk],
                       BK);
      load_matrix_sync(b_frag, (const __nv_bfloat16*)&sB[buf][kk][warp_c * 16],
                       BN);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();  // ring slot buf is free for the kt+2 issue
  }

  store_matrix_sync(&sc[warp * 256], c_frag, 16, mem_row_major);
  for (int i = lane; i < 256; i += 32) {
    const int r = i >> 4, c = i & 15;
    const int gr = m_tile * BM + warp_r * 16 + r;
    const int gc = n_tile * BN + warp_c * 16 + c;
    if (gr < M && gc < N) {
      float v = alpha * sc[warp * 256 + i];
      if (beta != 0.0f) v += beta * bf16_to_f32(C[(size_t)gr * N + gc]);
      C[(size_t)gr * N + gc] = f2bf(v);
    }
  }
}

// ======================================================================
//  Cross-tile B-reuse + cp.async kernel (experimental)
// ======================================================================
// Each block owns RM consecutive M-tiles for ONE N-tile. For a given K-tile
// sB[BK][BN] is loaded once and consumed by all RM M-tiles, so global B
// traffic drops by RM x and the per-warp b_frag (shared->register) load is
// amortised over RM mma_syncs. sA is RM panels, double-buffered alongside
// sB, so the cp.async pipeline of the db kernel is preserved. The RM
// accumulators live in registers (RM * 8 floats per thread).
//
// Same alignment contract as the db kernel (K % 8 == 0, N % 8 == 0).
template <int BM, int BN, int RM>
__global__ void gemm_bf16_kernel_reuse(const uint16_t* __restrict__ A,
                                       const uint16_t* __restrict__ B,
                                       uint16_t* __restrict__ C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
  static_assert(BM % 16 == 0 && BN % 16 == 0, "tile must be a multiple of 16");
  constexpr int kFragsM = BM / 16;
  constexpr int kFragsN = BN / 16;
  constexpr int kWarps  = kFragsM * kFragsN;
  constexpr int kTh     = kWarps * 32;
  constexpr int kAChunks = RM * BM * BK / 8;
  constexpr int kBChunks = BK * BN / 8;
  constexpr int kAChunksPerRow = BK / 8;
  constexpr int kBChunksPerRow = BN / 8;

  __shared__ uint16_t sA[2][RM][BM][BK];  // two-deep ring
  __shared__ uint16_t sB[2][BK][BN];
  extern __shared__ float sc[];           // RM * kWarps fp32 fragments

  const int m_base = blockIdx.x * (BM * RM);  // first global row of the block
  const int n_tile = blockIdx.y;
  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_r = warp / kFragsN;
  const int warp_c = warp % kFragsN;
  const int lane   = tid & 31;
  const int n_ktiles = (K + BK - 1) / BK;

  auto issue = [&](int kt, int buf) {
    const int k_off = kt * BK;
    for (int c = tid; c < kAChunks; c += kTh) {
      const int mm   = c / (BM * kAChunksPerRow);
      const int rem  = c % (BM * kAChunksPerRow);
      const int r    = rem / kAChunksPerRow;
      const int col  = (rem % kAChunksPerRow) * 8;
      const int gr = m_base + mm * BM + r, gk = k_off + col;
      const bool valid = (gr < M) && (gk + 8 <= K);
      cp_async_16(&sA[buf][mm][r][col], valid ? &A[(size_t)gr * K + gk] : A,
                  valid);
    }
    for (int c = tid; c < kBChunks; c += kTh) {
      const int r = c / kBChunksPerRow;
      const int col = (c % kBChunksPerRow) * 8;
      const int gk = k_off + r, gc = n_tile * BN + col;
      const bool valid = (gk < K) && (gc + 8 <= N);
      cp_async_16(&sB[buf][r][col], valid ? &B[(size_t)gk * N + gc] : B,
                  valid);
    }
  };

  issue(0, 0);
  cp_async_commit();

  fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) fill_fragment(c_frag[r], 0.0f);

  for (int kt = 0; kt < n_ktiles; ++kt) {
    const int buf  = kt & 1;
    const bool more = (kt + 1 < n_ktiles);
    if (more) {
      issue(kt + 1, buf ^ 1);
      cp_async_commit();
      cp_async_wait_1();
    } else {
      cp_async_wait_all();
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += kMmaK) {
      load_matrix_sync(b_frag, (const __nv_bfloat16*)&sB[buf][kk][warp_c * 16],
                       BN);
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        load_matrix_sync(
            a_frag, (const __nv_bfloat16*)&sA[buf][r][warp_r * 16][kk], BK);
        mma_sync(c_frag[r], a_frag, b_frag, c_frag[r]);
      }
    }
    __syncthreads();  // ring slot buf is free for the kt+2 issue
  }

#pragma unroll
  for (int r = 0; r < RM; ++r) {
    store_matrix_sync(&sc[(r * kWarps + warp) * 256], c_frag[r], 16,
                      mem_row_major);
  }
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    for (int i = lane; i < 256; i += 32) {
      const int rr = i >> 4, cc = i & 15;
      const int gr = m_base + r * BM + warp_r * 16 + rr;
      const int gc = n_tile * BN + warp_c * 16 + cc;
      if (gr < M && gc < N) {
        float v = alpha * sc[(r * kWarps + warp) * 256 + i];
        if (beta != 0.0f) v += beta * bf16_to_f32(C[(size_t)gr * N + gc]);
        C[(size_t)gr * N + gc] = f2bf(v);
      }
    }
  }
}

// CUDA's default per-block shared budget (static + dynamic). Requests above
// it must opt in per function; see launch<> below.
constexpr int kSharedMemDefault = 48 * 1024;

template <int BM, int BN>
void launch(const uint16_t* A, const uint16_t* B, uint16_t* C,
            int M, int N, int K, float alpha, float beta) {
  constexpr int kWarps = (BM / 16) * (BN / 16);
  constexpr int kTh = kWarps * 32;
  // Per-warp fp32 accumulator staging: one 16x16 (=256) float per warp.
  constexpr int dynamic_bytes = kWarps * 16 * 16 * (int)sizeof(float);
  constexpr int static_bytes = (BM * BK + BK * BN) * (int)sizeof(uint16_t);
  // Opt in only when the tile's static sA/sB + dynamic sc request exceeds
  // the 48 KB default per-block budget (the 64x128 tile: 24 KB static +
  // 32 KB dynamic = 56 KB). The attribute is PER-DEVICE state, so a
  // process-lifetime set -- the static-init this replaces -- silently
  // leaves every other device in a multi-GPU process unconfigured; setting
  // it per launch is a cheap host-side call and small tiles never pay it.
  if (static_bytes + dynamic_bytes > kSharedMemDefault) {
    cudaFuncSetAttribute((const void*)gemm_bf16_kernel<BM, BN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dynamic_bytes);
  }
  dim3 block(kTh);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);  // x = M-tiles, y = N-tiles
  gemm_bf16_kernel<BM, BN><<<grid, block, dynamic_bytes, 0>>>(
      A, B, C, M, N, K, alpha, beta);
}

template <int BM, int BN>
void launch_db(const uint16_t* A, const uint16_t* B, uint16_t* C,
               int M, int N, int K, float alpha, float beta) {
  constexpr int kWarps = (BM / 16) * (BN / 16);
  constexpr int kTh = kWarps * 32;
  constexpr int dynamic_bytes = kWarps * 16 * 16 * (int)sizeof(float);
  constexpr int static_bytes = 2 * (BM * BK + BK * BN) * (int)sizeof(uint16_t);
  if (static_bytes + dynamic_bytes > kSharedMemDefault) {
    cudaFuncSetAttribute((const void*)gemm_bf16_kernel_db<BM, BN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dynamic_bytes);
  }
  dim3 block(kTh);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  gemm_bf16_kernel_db<BM, BN><<<grid, block, dynamic_bytes, 0>>>(
      A, B, C, M, N, K, alpha, beta);
}

template <int BM, int BN, int RM>
void launch_reuse(const uint16_t* A, const uint16_t* B, uint16_t* C,
                  int M, int N, int K, float alpha, float beta) {
  constexpr int kWarps = (BM / 16) * (BN / 16);
  constexpr int kTh = kWarps * 32;
  constexpr int dynamic_bytes = RM * kWarps * 16 * 16 * (int)sizeof(float);
  constexpr int static_bytes =
      2 * (RM * BM * BK + BK * BN) * (int)sizeof(uint16_t);
  if (static_bytes + dynamic_bytes > kSharedMemDefault) {
    cudaFuncSetAttribute((const void*)gemm_bf16_kernel_reuse<BM, BN, RM>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dynamic_bytes);
  }
  dim3 block(kTh);
  // M grid shrinks by RM: each block covers RM consecutive M-tiles.
  dim3 grid((M + BM * RM - 1) / (BM * RM), (N + BN - 1) / BN);
  gemm_bf16_kernel_reuse<BM, BN, RM><<<grid, block, dynamic_bytes, 0>>>(
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

// Double-buffered / cp.async variant of the explicit-tile entry point, for the
// A/B in the benchmark (not part of the public API). Falls back to the
// synchronous kernel when a shape would break the 16-byte-chunk alignment the
// async path needs (none of the K3 shapes do).
void gemm_bf16_db_with_config(std::size_t M, std::size_t N, std::size_t K,
                              float alpha, const uint16_t* A,
                              const uint16_t* B, float beta, uint16_t* C,
                              int bm, int bn) {
  const int Mi = static_cast<int>(M);
  const int Ni = static_cast<int>(N);
  const int Ki = static_cast<int>(K);
  if ((Ki % 8) != 0 || (Ni % 8) != 0) {
    gemm_bf16_with_config(M, N, K, alpha, A, B, beta, C, bm, bn, BK,
                          (bm / 16) * (bn / 16) * 32);
    return;
  }
  if (bm == 16 && bn == 16)        launch_db<16, 16>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 16 && bn == 64)   launch_db<16, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 16 && bn == 128)  launch_db<16, 128>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 32 && bn == 64)   launch_db<32, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 32 && bn == 128)  launch_db<32, 128>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 64 && bn == 64)   launch_db<64, 64>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (bm == 64 && bn == 128)  launch_db<64, 128>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else                             launch_db<16, 16>(A, B, C, Mi, Ni, Ki, alpha, beta);
}

// Cross-tile B-reuse variant (cp.async + RM M-tiles per block). The (bm, bn)
// is the per-M-tile shape; the benchmark pairs it with an RM. Falls back to
// the db kernel when the shape breaks chunk alignment.
void gemm_bf16_reuse_with_config(std::size_t M, std::size_t N, std::size_t K,
                                 float alpha, const uint16_t* A,
                                 const uint16_t* B, float beta, uint16_t* C,
                                 int bm, int bn, int rm) {
  const int Mi = static_cast<int>(M);
  const int Ni = static_cast<int>(N);
  const int Ki = static_cast<int>(K);
  if ((Ki % 8) != 0 || (Ni % 8) != 0) {
    gemm_bf16_db_with_config(M, N, K, alpha, A, B, beta, C, bm, bn);
    return;
  }
  if (rm == 2 && bm == 32 && bn == 64)
    launch_reuse<32, 64, 2>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (rm == 2 && bm == 64 && bn == 64)
    launch_reuse<64, 64, 2>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (rm == 4 && bm == 16 && bn == 64)
    launch_reuse<16, 64, 4>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else if (rm == 4 && bm == 32 && bn == 64)
    launch_reuse<32, 64, 4>(A, B, C, Mi, Ni, Ki, alpha, beta);
  else
    gemm_bf16_db_with_config(M, N, K, alpha, A, B, beta, C, bm, bn);
}

// Shape-selected entry point. Picks a tuned tile via gemm_bf16_config_for
// (the host-visible selector in vkernels::kernels) and dispatches.
void gemm_bf16(std::size_t M, std::size_t N, std::size_t K, float alpha,
               const uint16_t* A, const uint16_t* B, float beta,
               uint16_t* C) {
  int bm = 0, bn = 0, bk = 0, threads = 0;
  ::vkernels::kernels::gemm_bf16_config_for(M, N, K, &bm, &bn, &bk, &threads);
  // Warmup / prefill (M > 64): the config's (64,64) tile is implemented as a
  // 4-way M-grouped (16,64) reuse kernel. It covers the same 64x64 output
  // footprint with the same B-reuse factor as a flat (64,64) tile, but the
  // per-warp b_frag (shared->register) load is amortised over 4
  // M-fragments and the (16,64) async copy has a full row to work on;
  // measured 2-6% faster than the flat cp.async tile on the large-K K3
  // shapes and 26-52% on the small-K ones (K = 128..1536), where B reuse
  // dominates. Serving stays on the plain double-buffered kernel below.
  if (M > 64) {
    gemm_bf16_reuse_with_config(M, N, K, alpha, A, B, beta, C, 16, 64, 4);
    return;
  }
  // Serving / decode (M <= 64): cp.async double-buffered kernel (1.3-1.8x
  // over the synchronous one on GB10); it falls back to the synchronous
  // kernel for shapes that break the 16-byte-chunk alignment.
  gemm_bf16_db_with_config(M, N, K, alpha, A, B, beta, C, bm, bn);
}

}  // namespace vkernels::kernels::cuda

#endif  // VKERNELS_HAS_CUDA
