// vkernels/kernels/gemm_bf16.cpp -- CPU reference (oracle) implementation.
//
// Implements the bf16 GEMM C = alpha * A @ B + beta * C (issue #29) as a
// straight-line CPU reference that matches the HIP MFMA kernel on the
// Kimi-K3 projection shapes.  Always compiled; independent of GPU toolkit.
//
// bf16 is stored as uint16_t IEEE 754 bit patterns.  The reference converts
// inputs to fp32, accumulates in fp32, and stores with the same
// round-to-nearest-even (RNE) as f32bits_to_bf16 in moe_device.hip so that
// the host oracle and the device kernel agree to the last bit.  The RNE
// helper is replicated from moe_fused.cpp to keep this translation unit
// self-contained on a host-only build.
#include "vkernels/kernels/gemm_bf16.hpp"

#include <cstring>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

namespace {

// bf16 = the top 16 bits of fp32; RNE via round-half-to-even.
uint16_t f32bits_to_bf16_local(uint32_t bits) {
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

uint32_t bf16_to_f32bits_local(uint16_t b) {
  // bf16 occupies the top 16 bits of fp32; the low 17 mantissa bits are 0.
  return static_cast<uint32_t>(b) << 16;
}

float bf16_to_float_local(uint16_t b) {
  uint32_t bits = bf16_to_f32bits_local(b);
  float f;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}

uint16_t float_to_bf16_local(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(float));
  return f32bits_to_bf16_local(bits);
}

}  // namespace

void gemm_bf16_cpu(std::size_t M, std::size_t N, std::size_t K, float alpha,
                   const uint16_t* A, const uint16_t* B, float beta,
                   uint16_t* C) {
  VK_EXPECTS(M == 0 || N == 0 || K == 0 || A != nullptr, "A must not be null");
  VK_EXPECTS(M == 0 || N == 0 || K == 0 || B != nullptr, "B must not be null");
  VK_EXPECTS(M == 0 || N == 0 || C != nullptr, "C must not be null");

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k)
        acc += bf16_to_float_local(A[i * K + k]) * bf16_to_float_local(B[k * N + j]);
      const float prev = (beta != 0.0f) ? bf16_to_float_local(C[i * N + j]) : 0.0f;
      C[i * N + j] = float_to_bf16_local(alpha * acc + beta * prev);
    }
  }
}

void gemm_bf16_config_for(std::size_t M, std::size_t N, std::size_t K,
                          int* bm, int* bn, int* bk, int* threads) {
  // The selector is per-arch: the tile that saturates MI300A's 228 CUs is
  // the wrong one for GB10's 48 SMs, so the two branches below are chosen
  // independently against each chip's on-device autotuner.  BK is fixed at
  // 64 for every architecture -- every K3 K is a multiple of 64.
  *bk = 64;

#if VKERNELS_HAS_CUDA
  (void)K;  // GB10's tile choice is independent of K (measured on all K3 shapes)
  // --- NVIDIA (GB10 / sm_121, 48 SMs, LPDDR) ------------------------------
  // These tiles were regenerated against the cp.async double-buffered kernel
  // (gemm_bf16_kernel_db) that the CUDA public path now uses -- the pipeline
  // shifts the optimum from the synchronous kernel's (32,64) to (16,64) for
  // serving and back to (64,64) for warmup (see the two autotuner matrices in
  // docs/performance/gemm-bf16/gb10.md).
  //
  // Serving M <= 64: (16,64) is within ~15% of the per-shape optimum on
  // almost every K3 shape and wins most of them; the small BM keeps the block
  // count high and BN=64 gives the async copies a full row to work on. The
  // one outlier is 896x7168 (N <= 1024), where (16,16) is ~40% faster at 64
  // (90us vs 126us), so it gets its own branch.
  //
  //   (16,16): (16/16)*(16/16)*32 = 32 threads
  //   (16,64): (16/16)*(64/16)*32 = 128 threads
  //   (64,64): (64/16)*(64/16)*32 = 256 threads
  if (M <= 64) {
    if (N <= 1024) {
      *bm = 16;
      *bn = 16;
      *threads = 32;
    } else {
      *bm = 16;
      *bn = 64;
      *threads = 128;
    }
  } else {
    // Warmup / prefill. The selector reports the effective (64,64) output
    // footprint; on CUDA the kernel realises it as a 4-way M-grouped
    // (16,64) cross-tile-B-reuse kernel (see gemm_bf16.cu). The M=8192
    // autotuner measured that form faster than the flat (64,64) cp.async
    // tile on every K3 shape -- 2-6% on the large-K shapes and 26-52% on
    // the small-K ones (K = 128..1536, where B reuse dominates).
    *bm = 64;
    *bn = 64;
    *threads = 256;
  }
#else
  // --- AMD (MI300A / gfx942) and host-only builds -------------------------
  // N is bounds-checked per tile inside the kernel; the K3 shapes are all
  // multiples of 16 (N) and 64 (K), so no config needs to pad K or pick a
  // BN that divides N.
  (void)N;
  (void)K;
  if (M <= 64) {
    // Serving / decode: tiny M, memory-bound, and the block count comes
    // almost entirely from the N-tiles.  A small BN (=> more blocks) is what
    // saturates the 228 CUs -- the on-device autotuner in bench_gemm_bf16
    // measured (16,16) to beat (16,64) by 1.4-2.3x on every K3 serving shape.
    // (64,64) is ~11-15% better on three small-N/small-K shapes
    // (896x7168, 7168x768, 2304x1536); that marginal gain is left to the
    // autotuner in production rather than to a brittle host heuristic.
    *bm = 16;
    *bn = 16;
    *threads = 64;  // 1 wavefront (one per 16-row fragment)
  } else {
    // Warmup / prefill: large M, compute-bound once cross-tile B reuse
    // lands (issue #77). The selector reports the EFFECTIVE output footprint
    // (rm*bm, bn); hip::gemm_bf16 realises it as the reuse + LDS
    // double-buffer kernel: (128, 64) = (32, 64) M-tiles x RM=4 per block
    // (512 threads = RM*(bm/16)*64 wavefront lanes), halving the global B
    // re-reads vs the old flat (64, 64) tile and pipelining the tile kt+1
    // loads against tile kt's MFMAs. threads matches the reuse kernel's
    // block size (bm/16)*64 with bm = the effective 128-row footprint.
    *bm = 128;
    *bn = 64;
    *threads = 512;  // (128/16)*64 = 4 wavefronts
  }
#endif
}

}  // namespace vkernels::kernels
