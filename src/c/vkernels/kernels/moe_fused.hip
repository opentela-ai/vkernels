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
// sA row-pad: gfx942 LDS has 32 banks; with the natural 64-u16 row stride
// the whole sA row group maps to one bank (row*32 = 0 mod 32), so every
// A-tile LDS store and mfma A-fragment read runs ~16-way conflicted
// (rocprof: ~446 bank conflicts per K-iteration per block, 5.48M per
// gateup launch).  A 66-u16 stride (33 words, odd) spreads rows across
// banks (row*33 mod 32 = row).  Costs 64 B of LDS per tile; the padded
// variant that also padded sB tripped the rocm 6.3 register allocator
// into spilling the prefetch ring (scr=332 B, 3x slower) — sB is left
// unpadded and the launch_bounds hints untouched.
static constexpr int kSA_P = kBK + 2;   // sA rows are [kSA_P]
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

// Load the act tile [EM, ispp] into shared memory, indexed by SORTED ROW
// (tk_base + row).  Each expert's intermediate output occupies its own
// sorted row, so top_k>1 does not overwrite.
// `k_base` is the column offset into each act row where the calling rank's
// w2 shard starts (0 for the fused call; r*ispp/tp for a TP rank).

// ---- A-tile staging (split form of load_a_tile / load_act_tile) ------
// Same row/col ownership as load_a_tile; the global loads land in
// registers so the next K-tile's A reads issue before the current
// tile's LDS stores + MFMA (A is L2-hot at small M but still ~hundreds
// of cycles on the critical path).
struct PrefATile { uint2 v[4]; };

__device__ __forceinline__ PrefATile prefetch_a_tile(
    const uint16_t* __restrict__ A, const int32_t* __restrict__ sorted_ids,
    int tk_base, int k_start, int stride, int M, int top_k, int lane,
    int k_base = 0) {
  int row = lane % kBM;
  int base_col = (lane / kBM) * (kBK / (kWhreads / kBM));
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  PrefATile p;
  if (flat < N) {
    int token = flat / top_k;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      const uint32_t* src =
          (const uint32_t*)(&A[(size_t)token * stride + k_base + k_start + col_off]);
      p.v[g] = *(const uint2*)(src);
    }
  } else {
#pragma unroll
    for (int g = 0; g < 4; ++g) p.v[g] = {};
  }
  return p;
}

__device__ __forceinline__ void store_a_tile_from_regs(
    uint16_t sA[kBM][kSA_P], const PrefATile& p, int lane) {
  int row = lane % kBM;
  int base_col = (lane / kBM) * (kBK / (kWhreads / kBM));
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    int col_off = base_col + g * 4;
    uint32_t* dst = (uint32_t*)(&sA[row][col_off]);
    dst[0] = p.v[g].x;
    dst[1] = p.v[g].y;
  }
}

__device__ __forceinline__ PrefATile prefetch_act_tile(
    const uint16_t* __restrict__ act, const int32_t* __restrict__ sorted_ids,
    int tk_base, int k_start, int ispp, int M, int top_k, int lane,
    int k_base = 0) {
  int row = lane % kBM;
  int base_col = (lane / kBM) * (kBK / (kWhreads / kBM));
  int flat = sorted_ids[tk_base + row];
  int N = M * top_k;
  PrefATile p;
  if (flat < N) {
    const uint16_t* src_row =
        &act[(size_t)(tk_base + row) * ispp + k_base + k_start];
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      int col_off = base_col + g * 4;
      p.v[g] = *(const uint2*)((const uint32_t*)(src_row + col_off));
    }
  } else {
#pragma unroll
    for (int g = 0; g < 4; ++g) p.v[g] = {};
  }
  return p;
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

// ======================================================================
//  Decode software pipeline: register prefetch of the next K-tile's B
//  bytes (issue #41, items 3/5 — latency-bound decode).
//
//  The decode K-loop is fully serial per block: 64 threads = one
//  wavefront, so every __syncthreads() stalls the only wavefront, and at
//  M<=8 there are too few blocks to hide the DRAM latency cross-block
//  (the default harness runs 32 gateup blocks on 228 CUs).  Each
//  iteration exposed the full global-read latency on the critical path:
//  load → dequant → barrier → MFMA → barrier → (next load).
//
//  Fix: stage iter kb+1's packed fp4 bytes + ue8m0 scales in REGISTERS
//  (the cold DRAM stream — A is L2-hot at small M, re-read by every
//  N-tile) while iter kb's dequant + MFMA execute.  Zero extra LDS
//  (the full sA+sB double-buffer was tried before and reverted: it
//  halved occupancy), ~4 VGPRs per staged tile.
//
//  The (kp, n) each thread owns is layout-independent — only the byte
//  addresses differ between N-major and K-major — so prefetch comes in
//  two flavors and the register→LDS decode is shared.  Index math is
//  identical to dequant_b_16cols / dequant_b_16cols_kmajor, so outputs
//  are bit-identical; the old functions stay for the prefill kernels.
// ======================================================================
struct PrefBTile {
  // 8 packed bytes (4 K-nibbles each) + 8 ue8m0 scale bytes, one per
  // (it -> kp = lane/16 + it*4) pair; kBytesPerThread*kWhreads covers the
  // full kBK x 16 tile (kp max = 3 + 7*4 = 31 < kPackedK, so no guard).
  uint8_t pb[8];
  uint8_t sc[8];
};

__device__ __forceinline__ PrefBTile prefetch_b16_nmajor(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
    int k_start, int stride_packed, int stride_scale_n, int lane) {
  constexpr int kPackedK = kBK / 2, kKpStep = kWhreads / 16;
  constexpr int kBytesPerThread = (kPackedK * 16) / kWhreads;
  int n = lane % 16;
  int kp_base = lane / 16;
  int kp_off = k_start / 2;
  int sc_off = k_start / kGroupSize;
  PrefBTile p;
#pragma unroll
  for (int it = 0; it < kBytesPerThread; ++it) {
    int kp = kp_base + it * kKpStep;
    p.pb[it] = packed[n * stride_packed + kp_off + kp];
    int gi = (kp * 2) / kGroupSize;
    p.sc[it] = scale[n * stride_scale_n + sc_off + gi];
  }
  return p;
}

__device__ __forceinline__ PrefBTile prefetch_b16_kmajor(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
    int k_start, int stride_k, int stride_scale_k, int lane) {
  constexpr int kPackedK = kBK / 2, kKpStep = kWhreads / 16;
  constexpr int kBytesPerThread = (kPackedK * 16) / kWhreads;
  int n = lane % 16;
  int kp_base = lane / 16;
  int kp_off = k_start / 2;
  int sc_off = k_start / kGroupSize;
  PrefBTile p;
#pragma unroll
  for (int it = 0; it < kBytesPerThread; ++it) {
    int kp = kp_base + it * kKpStep;
    p.pb[it] = packed[(kp_off + kp) * stride_k + n];
    int gi = (kp * 2) / kGroupSize;
    p.sc[it] = scale[(sc_off + gi) * stride_scale_k + n];
  }
  return p;
}

// Layout-dispatching flavor used by the gate/up kernel (template mirrors
// the kKM switch of gateup_swiglu_kernel): kmajor uses stride_k for both
// packed and scale (the N-fast slabs), nmajor uses the per-column strides.
template <bool kKM>
__device__ __forceinline__ PrefBTile prefetch_b16(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
    int k_start, int stride_k, int stride_packed, int stride_scale_n,
    int lane) {
  if constexpr (kKM)
    return prefetch_b16_kmajor(packed, scale, k_start, stride_k, stride_k,
                               lane);
  else
    return prefetch_b16_nmajor(packed, scale, k_start, stride_packed,
                               stride_scale_n, lane);
}

// Decode a staged tile into sB[kBK][16].  Pure ALU + LDS; identical math
// (and identical per-(kp,n) mapping) to dequant_b_16cols.
__device__ __forceinline__ void dequant_b16_from_regs(
    uint16_t sB[kBK][16], const PrefBTile& p, int lane) {
  constexpr int kPackedK = kBK / 2, kKpStep = kWhreads / 16;
  constexpr int kBytesPerThread = (kPackedK * 16) / kWhreads;
  int n = lane % 16;
  int kp_base = lane / 16;
#pragma unroll
  for (int it = 0; it < kBytesPerThread; ++it) {
    int kp = kp_base + it * kKpStep;
    uint8_t pb = p.pb[it];
    float s = ue8m0_decode(p.sc[it]);
    uint32_t flo = fp4nib_to_f32bits(pb & 0x0F);
    float vlo; __builtin_memcpy(&vlo, &flo, sizeof(float));
    vlo *= s; __builtin_memcpy(&flo, &vlo, sizeof(float));
    sB[kp * 2][n] = f32bits_to_bf16(flo);
    uint32_t fhi = fp4nib_to_f32bits((pb >> 4) & 0x0F);
    float vhi; __builtin_memcpy(&vhi, &fhi, sizeof(float));
    vhi *= s; __builtin_memcpy(&fhi, &vhi, sizeof(float));
    sB[kp * 2 + 1][n] = f32bits_to_bf16(fhi);
  }
}

// Run the K=64 MFMA loop (4 steps of K=16), accumulating into acc.
__device__ __forceinline__ void mfma_k64(
    v4f& acc, const uint16_t sA[kBM][kSA_P], const uint16_t sB[kBK][16],
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

  __shared__ uint16_t sA[kBM][kSA_P];
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
    // 64-bit byte offsets (issue #41): expert * w13_bytes exceeds INT32 at
    // large shapes (E=256, hidden=7168, ispp=4096: 255 * 29.4 MB = 7.5 GB >
    // 2^31) and wraps to a negative offset -> memory access fault. The
    // K-major branch above is safe ((size_t) cast on the slab index); the
    // down GEMM and the prefill/dist kernels had the same defect.
    const size_t w13_bytes  = 2 * (size_t)ispp * (hidden / 2);
    const size_t w13s_bytes = 2 * (size_t)ispp * (hidden / kGroupSize);
    bp_gate = w13 + expert * w13_bytes
                  + (size_t)n_global * (hidden / 2);
    sc_gate = w13_scale + expert * w13s_bytes
                  + (size_t)n_global * (hidden / kGroupSize);
    bp_up   = w13 + expert * w13_bytes
                  + (size_t)(n_global + ispp) * (hidden / 2);
    sc_up   = w13_scale + expert * w13s_bytes
                  + (size_t)(n_global + ispp) * (hidden / kGroupSize);
  }

  v4f acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
  v4f acc_u = {0.0f, 0.0f, 0.0f, 0.0f};

  // Software pipeline, depth-2 (see PrefBTile note above): the loads for
  // kb+2 issue while kb's dequant + MFMA run, so up to two iterations of
  // DRAM latency hide behind compute.  A-tiles stage one deep (L2-hot).
  // Ring validity: pbn0/pbu0/pa0 = tile kb (always valid in-loop); the
  // kb+1 slots shift in only when the loop continues past kb; kb+2 slots
  // are read only under the same condition they were written.
  const int nkb = hidden / kBK;
  PrefATile pa0 = prefetch_a_tile(A, sorted_ids, tk_base, 0, hidden, M,
                                  top_k, lane);
  PrefBTile pbn0 = prefetch_b16<kKM>(bp_gate, sc_gate, 0, ispp, hidden / 2,
                                     hidden / kGroupSize, lane);
  PrefBTile pbu0 = prefetch_b16<kKM>(bp_up, sc_up, 0, ispp, hidden / 2,
                                     hidden / kGroupSize, lane);
  PrefATile pa1;
  PrefBTile pbn1, pbu1;
  if (nkb > 1) {
    pa1 = prefetch_a_tile(A, sorted_ids, tk_base, kBK, hidden, M, top_k, lane);
    pbn1 = prefetch_b16<kKM>(bp_gate, sc_gate, kBK, ispp, hidden / 2,
                             hidden / kGroupSize, lane);
    pbu1 = prefetch_b16<kKM>(bp_up, sc_up, kBK, ispp, hidden / 2,
                             hidden / kGroupSize, lane);
  }
  for (int kb = 0; kb < nkb; ++kb) {
    int ks = kb * kBK;
    PrefATile pa2;
    PrefBTile pbn2, pbu2;
    if (kb + 2 < nkb) {
      pa2 = prefetch_a_tile(A, sorted_ids, tk_base, ks + 2 * kBK, hidden, M,
                            top_k, lane);
      pbn2 = prefetch_b16<kKM>(bp_gate, sc_gate, ks + 2 * kBK, ispp,
                               hidden / 2, hidden / kGroupSize, lane);
      pbu2 = prefetch_b16<kKM>(bp_up, sc_up, ks + 2 * kBK, ispp,
                               hidden / 2, hidden / kGroupSize, lane);
    }
    // sA store and sB decode write different LDS buffers, so no barrier
    // is needed between them; the barrier before MFMA orders both.
    store_a_tile_from_regs(sA, pa0, lane);
    dequant_b16_from_regs(sB_gate, pbn0, lane);
    dequant_b16_from_regs(sB_up, pbu0, lane);
    __syncthreads();

    mfma_k64(acc_g, sA, sB_gate, lane);
    mfma_k64(acc_u, sA, sB_up, lane);
    __syncthreads();
    // shift the ring (pa1/pbn1/pbu1 valid iff kb+1 < nkb, i.e. whenever
    // the next iteration consumes them)
    pa0 = pa1; pbn0 = pbn1; pbu0 = pbu1;
    pa1 = pa2; pbn1 = pbn2; pbu1 = pbu2;
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

  __shared__ uint16_t sA[kBM][kSA_P];
  __shared__ uint16_t sB[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  // 64-bit byte offsets (issue #41): expert * w2_bytes = e * hidden * ispp/2
  // exceeds INT32 at large shapes (E=32, hidden=7168, ispp=33792: 31 * 121 MB
  // = 3.75 GB > 2^31) and wraps negative -> memory access fault. This kernel
  // is shared by the N-major and K-major gate/up paths, which is why the
  // fault was --kmajor-independent.
  const size_t w2_bytes  = (size_t)hidden * (ispp / 2);
  const size_t w2s_bytes = (size_t)hidden * (ispp / kGroupSize);
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp = w2 + expert * w2_bytes
      + (size_t)n_global * (ispp / 2);
  const uint8_t* sc = w2_scale + expert * w2s_bytes
      + (size_t)n_global * (ispp / kGroupSize);

  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

  // Software pipeline, depth-2 (see gateup_swiglu_kernel).
  const int nkb = ispp / kBK;
  PrefATile pa0 = prefetch_act_tile(act, sorted_ids, tk_base, 0, ispp, M,
                                    top_k, lane);
  PrefBTile pb0 = prefetch_b16_nmajor(bp, sc, 0, ispp / 2, ispp / kGroupSize,
                                      lane);
  PrefATile pa1;
  PrefBTile pb1;
  if (nkb > 1) {
    pa1 = prefetch_act_tile(act, sorted_ids, tk_base, kBK, ispp, M, top_k,
                            lane);
    pb1 = prefetch_b16_nmajor(bp, sc, kBK, ispp / 2, ispp / kGroupSize, lane);
  }
  for (int kb = 0; kb < nkb; ++kb) {
    int ks = kb * kBK;
    PrefATile pa2;
    PrefBTile pb2;
    if (kb + 2 < nkb) {
      pa2 = prefetch_act_tile(act, sorted_ids, tk_base, ks + 2 * kBK, ispp,
                              M, top_k, lane);
      pb2 = prefetch_b16_nmajor(bp, sc, ks + 2 * kBK, ispp / 2,
                                ispp / kGroupSize, lane);
    }
    store_a_tile_from_regs(sA, pa0, lane);
    dequant_b16_from_regs(sB, pb0, lane);
    __syncthreads();

    mfma_k64(acc, sA, sB, lane);
    __syncthreads();
    pa0 = pa1; pb0 = pb1;
    pa1 = pa2; pb1 = pb2;
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
  // 64-bit byte offsets (issue #41): see gateup_swiglu_kernel — same
  // INT32 wrap of expert * w13_bytes in the prefill (64x64-tile) regime.
  const size_t w13_bytes  = 2 * (size_t)ispp * (hidden / 2);
  const size_t w13s_bytes = 2 * (size_t)ispp * (hidden / kGroupSize);
  int tid = threadIdx.x;
  int lane = tid & 63;
  int warp = tid >> 6;
  int n_global = pid_n * kBN;

  const uint8_t* bp_gate = w13 + expert * w13_bytes
      + (size_t)n_global * (hidden / 2);
  const uint8_t* sc_gate = w13_scale + expert * w13s_bytes
      + (size_t)n_global * (hidden / kGroupSize);
  const uint8_t* bp_up = w13 + expert * w13_bytes
      + (size_t)(n_global + ispp) * (hidden / 2);
  const uint8_t* sc_up = w13_scale + expert * w13s_bytes
      + (size_t)(n_global + ispp) * (hidden / kGroupSize);

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
  // 64-bit byte offsets (issue #41): see down_combine_kernel — same INT32
  // wrap of expert * w2_bytes in the prefill (64x64-tile) regime.
  const size_t w2_bytes  = (size_t)hidden * (ispp / 2);
  const size_t w2s_bytes = (size_t)hidden * (ispp / kGroupSize);
  int tid = threadIdx.x;
  int lane = tid & 63;
  int warp = tid >> 6;
  int n_global = pid_n * kBN;

  const uint8_t* bp = w2 + expert * w2_bytes
      + (size_t)n_global * (ispp / 2);
  const uint8_t* sc = w2_scale + expert * w2s_bytes
      + (size_t)n_global * (ispp / kGroupSize);

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

  __shared__ uint16_t sA[kBM][kSA_P];
  __shared__ uint16_t sB_gate[kBK][16];
  __shared__ uint16_t sB_up[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  // 64-bit byte offsets (issue #41): see gateup_swiglu_kernel — same INT32
  // wrap in the TP-stage (distributed) gate/up kernel.
  const size_t w13_bytes  = 2 * (size_t)ispp * (hidden_k / 2);
  const size_t w13s_bytes = 2 * (size_t)ispp * (hidden_k / kGroupSize);
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp_gate = w13 + expert * w13_bytes
      + (size_t)n_global * (hidden_k / 2);
  const uint8_t* sc_gate = w13_scale + expert * w13s_bytes
      + (size_t)n_global * (hidden_k / kGroupSize);
  const uint8_t* bp_up = w13 + expert * w13_bytes
      + (size_t)(n_global + ispp) * (hidden_k / 2);
  const uint8_t* sc_up = w13_scale + expert * w13s_bytes
      + (size_t)(n_global + ispp) * (hidden_k / kGroupSize);

  v4f acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
  v4f acc_u = {0.0f, 0.0f, 0.0f, 0.0f};

  // Software pipeline, depth-2 (see gateup_swiglu_kernel).
  const int nkb = hidden_k / kBK;
  PrefATile pa0 = prefetch_a_tile(A, sorted_ids, tk_base, 0, a_stride, M,
                                  top_k, lane, k_base);
  PrefBTile pbn0 = prefetch_b16_nmajor(bp_gate, sc_gate, 0, hidden_k / 2,
                                       hidden_k / kGroupSize, lane);
  PrefBTile pbu0 = prefetch_b16_nmajor(bp_up, sc_up, 0, hidden_k / 2,
                                       hidden_k / kGroupSize, lane);
  PrefATile pa1;
  PrefBTile pbn1, pbu1;
  if (nkb > 1) {
    pa1 = prefetch_a_tile(A, sorted_ids, tk_base, kBK, a_stride, M, top_k,
                          lane, k_base);
    pbn1 = prefetch_b16_nmajor(bp_gate, sc_gate, kBK, hidden_k / 2,
                               hidden_k / kGroupSize, lane);
    pbu1 = prefetch_b16_nmajor(bp_up, sc_up, kBK, hidden_k / 2,
                               hidden_k / kGroupSize, lane);
  }
  for (int kb = 0; kb < nkb; ++kb) {
    int ks = kb * kBK;
    PrefATile pa2;
    PrefBTile pbn2, pbu2;
    if (kb + 2 < nkb) {
      pa2 = prefetch_a_tile(A, sorted_ids, tk_base, ks + 2 * kBK, a_stride,
                            M, top_k, lane, k_base);
      pbn2 = prefetch_b16_nmajor(bp_gate, sc_gate, ks + 2 * kBK,
                                 hidden_k / 2, hidden_k / kGroupSize, lane);
      pbu2 = prefetch_b16_nmajor(bp_up, sc_up, ks + 2 * kBK,
                                 hidden_k / 2, hidden_k / kGroupSize, lane);
    }
    store_a_tile_from_regs(sA, pa0, lane);
    dequant_b16_from_regs(sB_gate, pbn0, lane);
    dequant_b16_from_regs(sB_up, pbu0, lane);
    __syncthreads();

    mfma_k64(acc_g, sA, sB_gate, lane);
    mfma_k64(acc_u, sA, sB_up, lane);
    __syncthreads();
    pa0 = pa1; pbn0 = pbn1; pbu0 = pbu1;
    pa1 = pa2; pbn1 = pbn2; pbu1 = pbu2;
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

  __shared__ uint16_t sA[kBM][kSA_P];
  __shared__ uint16_t sB[kBK][16];

  int pid_m = blockIdx.x, pid_n = blockIdx.y;
  int warp_n = blockIdx.z;
  int expert = expert_ids[pid_m];
  if (expert < 0) return;

  int tk_base = pid_m * kBM;
  // 64-bit byte offsets (issue #41): see down_combine_kernel — same INT32
  // wrap in the TP-stage (distributed) down kernel.
  const size_t w2_bytes  = (size_t)hidden * (ispp_k / 2);
  const size_t w2s_bytes = (size_t)hidden * (ispp_k / kGroupSize);
  int lane = threadIdx.x;
  int n_global = pid_n * kBN + warp_n * 16;

  const uint8_t* bp = w2 + expert * w2_bytes
      + (size_t)n_global * (ispp_k / 2);
  const uint8_t* sc = w2_scale + expert * w2s_bytes
      + (size_t)n_global * (ispp_k / kGroupSize);

  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

  // Software pipeline, depth-2 (see gateup_swiglu_kernel).
  const int nkb = ispp_k / kBK;
  PrefATile pa0 = prefetch_act_tile(act, sorted_ids, tk_base, 0, a_stride, M,
                                    top_k, lane, k_base);
  PrefBTile pb0 = prefetch_b16_nmajor(bp, sc, 0, ispp_k / 2,
                                      ispp_k / kGroupSize, lane);
  PrefATile pa1;
  PrefBTile pb1;
  if (nkb > 1) {
    pa1 = prefetch_act_tile(act, sorted_ids, tk_base, kBK, a_stride, M,
                            top_k, lane, k_base);
    pb1 = prefetch_b16_nmajor(bp, sc, kBK, ispp_k / 2, ispp_k / kGroupSize,
                              lane);
  }
  for (int kb = 0; kb < nkb; ++kb) {
    int ks = kb * kBK;
    PrefATile pa2;
    PrefBTile pb2;
    if (kb + 2 < nkb) {
      pa2 = prefetch_act_tile(act, sorted_ids, tk_base, ks + 2 * kBK,
                              a_stride, M, top_k, lane, k_base);
      pb2 = prefetch_b16_nmajor(bp, sc, ks + 2 * kBK, ispp_k / 2,
                                ispp_k / kGroupSize, lane);
    }
    store_a_tile_from_regs(sA, pa0, lane);
    dequant_b16_from_regs(sB, pb0, lane);
    __syncthreads();

    mfma_k64(acc, sA, sB, lane);
    __syncthreads();
    pa0 = pa1; pb0 = pb1;
    pa1 = pa2; pb1 = pb2;
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

// ======================================================================
// moe_align_block_size (device) — issue #46 follow-up
//
// The GPU counterpart of moe_align_block_size (CPU, moe_fused.cpp) and the
// Python moe_align_block_size_with_map (vllm_experts.py). Reads topk_ids
// *on-device* on `stream` (ordered after the expert-dispatch all-to-all
// that produced it) and writes the block-aligned sorted_ids / expert_ids
// layout consumed by fused_moe_mxfp4 — WITHOUT the topk_ids.cpu() host
// round-trip that dominated PP0's MoE (97-100% of its per-call
// moe:vkernel_apply; see docs/performance/moe-fused/gfx942-pp-pipeline.md).
//
// Single block, N = M*top_k <= 1024 (decode). The histogram is parallel
// (one thread per flat index, atomicAdd into shared s_count[local_e]);
// the padded-count prefix + ordered scatter are serial in thread 0 to
// match the CPU/Python reference exactly (experts ascending, flat indices
// ascending within each expert). For N <= ~64 (the K3 decode case) the
// serial tail is a handful of microseconds — negligible next to the 4 ms
// host sync this removes, and never the bottleneck (the GEMM launch is).
//
// expert_map[ge] < 0 (or ge out of [0, num_experts)) skips that routing
// entry, matching moe_align_block_size_with_map. max_EM is the host's
// constant high-water bound; sorted_ids/expert_ids beyond the real EM are
// written as the padding sentinels (N / -1) so the GEMM grid can be sized
// at max_EM/block_size and early-out the padding blocks/rows (the kernels
// already guard `flat < N` and `expert_ids[b] < 0`).
// ======================================================================
__global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_map,
    int M, int top_k, int block_size, int num_experts,
    int N, int max_EM, int local_n,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids_out,
    int32_t* __restrict__ out_em) {
  extern __shared__ int32_t smem[];
  int32_t* s_ids    = smem;                  // [N]
  int32_t* s_le     = s_ids + N;             // [N]  local expert per flat i
  int32_t* s_count  = s_le + N;              // [local_n]
  int32_t* s_offset = s_count + local_n;     // [local_n]  exclusive prefix
  int32_t* s_rank   = s_offset + local_n;    // [local_n]  running scatter idx

  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;  // >= max(N, local_n) (rounded to a warp)

  // Phase 0 — init (parallel): zero counts/ranks, pad buffers with the
  // padding sentinels (N for sorted_ids, -1 for expert_ids).
  for (int i = tid; i < local_n; i += nthr) { s_count[i] = 0; s_rank[i] = 0; }
  for (int i = tid; i < max_EM; i += nthr) sorted_ids[i] = N;
  const int max_blocks = max_EM / block_size;
  for (int i = tid; i < max_blocks; i += nthr) expert_ids_out[i] = -1;
  __syncthreads();

  // Phase A (parallel) — load topk_ids, compute local expert, histogram.
  int my_le = -1;
  if (tid < N) {
    int ge = topk_ids[tid];
    s_ids[tid] = ge;
    int le;
    if (ge < 0 || ge >= num_experts) {
      le = -1;                                  // bad routing -> skip
    } else if (expert_map != nullptr) {
      le = expert_map[ge];                       // global -> local (-1 = unmapped)
    } else {
      le = ge;                                   // no sharding: 1:1
    }
    if (le < 0 || le >= local_n) le = -1;        // bounds / invalid -> skip
    s_le[tid] = le;
    my_le = le;
  }
  __syncthreads();
  if (my_le >= 0) atomicAdd(&s_count[my_le], 1);
  __syncthreads();

  // Phase B (serial, thread 0) — padded counts + exclusive prefix, EM.
  if (tid == 0) {
    int EM = 0;
    for (int e = 0; e < local_n; ++e) {
      int cnt = s_count[e];
      int padded = (cnt == 0) ? 0
                                : ((cnt + block_size - 1) / block_size) * block_size;
      s_offset[e] = EM;
      EM += padded;
    }
    if (EM == 0) EM = block_size;               // min one block (matches Python)
    if (EM > max_EM) EM = max_EM;               // guard (shouldn't trigger)
    *out_em = EM;
  }
  __syncthreads();

  // Phase C (serial, thread 0) — scatter sorted_ids in ascending
  // (local_expert, flat_index) order, then expert_ids per real block.
  if (tid == 0) {
    for (int i = 0; i < N; ++i) {
      int le = s_le[i];
      if (le < 0) continue;                     // expert_map == -1 / bad -> skip
      int pos = s_offset[le] + s_rank[le];
      if (pos < max_EM) sorted_ids[pos] = i;
      s_rank[le] += 1;
    }
    int blk_idx = 0;
    for (int e = 0; e < local_n; ++e) {
      int cnt = s_count[e];
      if (cnt == 0) continue;
      int padded = ((cnt + block_size - 1) / block_size) * block_size;
      int nblocks = padded / block_size;
      for (int b = 0; b < nblocks && blk_idx < max_blocks; ++b) {
        // Real block iff it contains >= 1 real token (matches CPU:
        // expert_ids[b] = e iff b*block_size < cnt); padding blocks
        // within an expert are already -1 from Phase 0.
        if (b * block_size < cnt) expert_ids_out[blk_idx] = e;
        blk_idx += 1;
      }
    }
  }
}

void moe_align_block_size_hip(
    const int32_t* topk_ids,
    const int32_t* expert_map,
    int M, int top_k, int block_size, int num_experts, int local_n,
    int max_EM,
    int32_t* sorted_ids,
    int32_t* expert_ids,
    int32_t* out_em,
    void* stream) {
  const int N = M * top_k;
  // Single block; threads cover N and local_n (both <= 1024). Round up to
  // a multiple of 64 (wavefront) and never below 64 so the warp/coop loop
  // has at least one wave to run.
  int need = (N > local_n) ? N : local_n;
  need = ((need + 63) / 64) * 64;
  if (need < 64) need = 64;
  if (need > 1024) need = 1024;  // caller guards N <= 1024
  const int nthr = need;
  // Shared: s_ids[N] + s_le[N] + s_count[local_n] + s_offset[local_n]
  //       + s_rank[local_n]   (<= (2*1024 + 3*1024) ints = 20 KiB).
  const int shared_ints = 2 * N + 3 * local_n;
  const size_t shared_bytes = (size_t)shared_ints * sizeof(int32_t);
  hipStream_t s = static_cast<hipStream_t>(stream);
  moe_align_block_size_kernel<<<1, nthr, shared_bytes, s>>>(
      topk_ids, expert_map, M, top_k, block_size, num_experts,
      N, max_EM, local_n, sorted_ids, expert_ids, out_em);
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
