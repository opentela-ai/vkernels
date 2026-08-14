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
  (void)N;  // N is bounds-checked per tile inside the kernel; the K3 shapes
  (void)K;  // are all multiples of 16 (N) and 64 (K), so no config needs to
            // pad K or pick a BN that divides N.
  *bk = 64;
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
    // Warmup / prefill: large M, compute-bound; (64,64) keeps the block
    // count sane (~13k at M=8192) and reaches ~45% of HBM bandwidth.
    *bm = 64;
    *bn = 64;
    *threads = 256;  // 4 wavefronts (one per 16-row fragment)
  }
}

}  // namespace vkernels::kernels
