// vkernels/kernels/glm_moe.hpp
//
// GLM-5.3-Flash block-FP8 expert primitives (issue #64).
//
// GLM MoE weights are E4M3FN with FP32 scales per 128x128 block:
//
//   w      : [N, K]        uint8, E4M3FN codes, row-major (row = output)
//   scales : [N/128, K/128] float32, scale[n/128][k/128] multiplies block
//   x      : [M, K]        bfloat16 activations (decode: M = 1 or 2)
//   out    : [M, N]        bfloat16
//
//   out[m][n] = bf16( sum_k fp32(x[m][k]) * scale[n/128][k/128] * e4m3(w[n][k]) )
//
// E4M3FN (OCP FP8, finite-only): 1 sign, 4 exponent bits (bias 7), 3
// mantissa bits; no infinities; the encoding e=15,m=7 (0x7F/0xFF) is the
// only NaN, everything else is finite with max magnitude 448. Decode:
//   e>0 : (-1)^s * 2^(e-7) * (1 + m/8)      (normals)
//   e=0 : (-1)^s * 2^(-6) * (m/8)           (subnormals, m=0 -> +0)
//
// The decode-fused GEMV streams the FP8 weights directly (no BF16/FP32
// materialization of the expert buffer): at M=1 the op is memory-bound on
// the weight bytes (AI ~ 2 FLOP/byte), so fusing dequant into the dot
// removes both the materialized buffer traffic and its allocation.
//
// Two-implementation model:
//   glm_moe.cpp — CPU reference (oracle), always compiled
//   glm_moe.hip — HIP kernels (gfx942), compiled with VKERNELS_HAS_HIP
#pragma once

#include <cstdint>

namespace vkernels::kernels {

// ---------------------------------------------------------------------------
// glm_fp8_block_gemv_cpu — dequant-fused block-FP8 GEMV (CPU oracle)
// ---------------------------------------------------------------------------
// Contract: M in {1,2}, N % 128 == 0, K % 128 == 0, K <= 4096.
// Accumulation in float32; one round-to-nearest-even to bfloat16 at the
// end. The NaN encoding (0x7F/0xFF) decodes to NaN on both CPU and GPU.
// Divergence note: the CPU oracle decodes the reserved NaN encodings
// (0x7F/0xFF) to NaN; the GPU's branchless decode maps them to +-480.
// Weight tensors never carry them (they are the only non-finite codes).
void glm_fp8_block_gemv_cpu(const uint16_t* x, const uint8_t* w,
                            const float* scales, uint16_t* out,
                            int M, int N, int K);

// E4M3FN decode shared by the tests (exact CPU/GPU agreement).
float glm_e4m3_to_f32_cpu(uint8_t v);

}  // namespace vkernels::kernels

// HIP declarations only (no hip/hip_runtime.h here — glm_moe.cpp is a
// CPU TU that includes this header; the .hip TU includes the runtime
// first, matching kda.hpp's pattern).
#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// Device counterpart of glm_fp8_block_gemv_cpu, split-K (see glm_moe.hip):
// partial kernel grid (N/32, sk) + fixed-order reduce. sk in {1,2,4,8}
// dividing K/128; glm_fp8_gemv_pick_sk gives the occupancy heuristic the
// plain wrapper uses (>= ~2 blocks per CU on 228-CU MI300A).
int glm_fp8_gemv_pick_sk(int N, int K);

void glm_fp8_block_gemv(const uint16_t* x, const uint8_t* w,
                        const float* scales, uint16_t* out,
                        int M, int N, int K);

// Caller-owned fp32 partials scratch, M*N*sk floats (bench/autotune hook;
// contents clobbered each call).
void glm_fp8_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w,
                                     const float* scales, uint16_t* out,
                                     float* part, int M, int N, int K,
                                     int sk);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
