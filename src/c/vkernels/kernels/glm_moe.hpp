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

// Split-K pick for the HIP block-FP8 GEMV launch (host arithmetic, always
// compiled — same seam as mla_fwd_split_for). Fills MI300A's 228 CUs with
// ~4 blocks each where the segment budget allows, within {1,2,4,8} and
// divisibility of K/128. glm_fp8_block_gemv consults this on every launch;
// the tuning store (bench_glm_fp8_gemv --persist) overrides it per
// (N, K) when a measured winner exists for the device arch.
int gemv_pick_sk(int N, int K);
int glm_fp8_gemv_pick_sk(int N, int K);

// e4m3fnuz decode (issue #148): the hardware v_cvt_pk_f32_fp8 table —
// bias 8, subnormals m*2^-10, NaN only at 0x80. CPU oracle counterpart of
// the fnuz-storage decode GEMV below.
float glm_fnuz_e4m3_to_f32_cpu(uint8_t v);

// glm_fp8_fnuz_block_gemv_cpu — FNUZ-storage oracle (issue #148): w holds
// e4m3fnuz bytes and scales are the DOUBLED fp32 scales (the
// e4m3fn_to_fnuz_inplace convention), value = fnuz(code) * s2. Same shape
// contract as glm_fp8_block_gemv_cpu.
void glm_fp8_fnuz_block_gemv_cpu(
    const uint16_t* x, const uint8_t* w, const float* scales, uint16_t* out, int M, int N, int K);

// fnuz fp8 encode (issue #148): round-to-nearest-even to the e4m3fnuz
// lattice, bit-exact vs hardware v_cvt_pk_fp8_f32 (verified empirically
// by diag_mfma_issue148.hip: ties go to the even mantissa — 2.375 -> 2.5
// (0x4a), 2.125 -> 2.0 (0x48) — and the two-call packing streams the
// values low-byte first).
uint8_t glm_fp8_fnuz_quant_cpu(float v, float inv);

// glm_fp8_fnuz_mfma_oracle_cpu — tensor-core fp8 GEMV oracle (issue
// #148): models the fused fp8 payload quantize of
// glm_fp8_fnuz_mfma_gemv_kernel (per token, per 128-k group: amax ->
// s2a = amax/224 floor 1e-4; payload = RNE(x/s2a) fnuz codes), then
// out[m][n] = bf16(payload x fnuz(w) x scales) in fp32. Without the
// payload model this kernel's fp8 rounding noise (|w| spans 0.01..240)
// dominates the 2e-2 gate — the plain fnuz oracle cannot validate it.
void glm_fp8_fnuz_mfma_oracle_cpu(
    const uint16_t* x, const uint8_t* w, const float* scales, uint16_t* out, int M, int N, int K);

}  // namespace vkernels::kernels

// HIP declarations only (no hip/hip_runtime.h here — glm_moe.cpp is a
// CPU TU that includes this header; the .hip TU includes the runtime
// first, matching kda.hpp's pattern).
#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// Every entry takes the launch stream as a trailing ``void* stream``
// (a hipStream_t; nullptr = the null stream). The overloads without it are
// the pre-stream signatures and launch on the null stream. Contract
// violations (shape/split outside the contract below) throw
// std::invalid_argument via VK_EXPECTS; launch failures throw
// std::runtime_error. Internal partials scratch is stream-ordered
// (Scratch<T>), so the call returns without a device synchronization.

// Device counterpart of glm_fp8_block_gemv_cpu, split-K (see glm_moe.hip):
// partial kernel grid (N/32, sk) + fixed-order reduce. sk in {1,2,4,8}
// dividing K/128; the split pick is vkernels::kernels::gemv_pick_sk (host
// arithmetic above, tuning-store aware).
void glm_fp8_block_gemv(const uint16_t* x, const uint8_t* w, const float* scales, uint16_t* out,
                        int M, int N, int K, void* stream);
void glm_fp8_block_gemv(const uint16_t* x, const uint8_t* w, const float* scales, uint16_t* out,
                        int M, int N, int K);

// Caller-owned fp32 partials scratch, M*N*sk floats (bench/autotune hook;
// contents clobbered each call).
void glm_fp8_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w, const float* scales,
                                     uint16_t* out, float* part, int M, int N, int K, int sk,
                                     void* stream);
void glm_fp8_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w, const float* scales,
                                     uint16_t* out, float* part, int M, int N, int K, int sk);

// FNUZ-storage decode GEMV (issue #148): w holds e4m3fnuz bytes (the #71
// in-place conversion's payload-halved encoding) and scales are the
// DOUBLED fp32 scales, so value = fnuz(code) * s2 reproduces the original
// e4m3fn weight exactly. Decode is the hardware fnuz converter
// (v_cvt_pk_f32_fp8) instead of the manual bit-assembly; same split-K
// structure, sk heuristic and shape contract as glm_fp8_block_gemv.
void glm_fp8_fnuz_block_gemv(const uint16_t* x, const uint8_t* w, const float* scales,
                             uint16_t* out, int M, int N, int K, void* stream);
void glm_fp8_fnuz_block_gemv(const uint16_t* x, const uint8_t* w, const float* scales,
                             uint16_t* out, int M, int N, int K);

void glm_fp8_fnuz_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w,
                                          const float* scales, uint16_t* out, float* part,
                                          int M, int N, int K, int sk, void* stream);
void glm_fp8_fnuz_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w,
                                          const float* scales, uint16_t* out, float* part,
                                          int M, int N, int K, int sk);

// MFMA tensor-core decode GEMV (issue #148): same FNUZ storage contract
// and fused activation quantize, but the fp8 dot runs on the CDNA3
// tensor core (v_mfma_f32_16x16x32_fp8_fp8, wave64 fragments) instead of
// per-code cvt+FMA — no fp8 decode arithmetic at all. Same shape
// contract; requires N % 64 == 0 (implied by N % 128 == 0).
void glm_fp8_fnuz_mfma_block_gemv(const uint16_t* x, const uint8_t* w, const float* scales,
                                  uint16_t* out, int M, int N, int K, void* stream);
void glm_fp8_fnuz_mfma_block_gemv(const uint16_t* x, const uint8_t* w, const float* scales,
                                  uint16_t* out, int M, int N, int K);

void glm_fp8_fnuz_mfma_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w,
                                               const float* scales, uint16_t* out,
                                               float* part, int M, int N, int K, int sk,
                                               void* stream);
void glm_fp8_fnuz_mfma_block_gemv_with_scratch(const uint16_t* x, const uint8_t* w,
                                               const float* scales, uint16_t* out,
                                               float* part, int M, int N, int K, int sk);

// Debug hook (probe_mfma_fp8): also dumps block (0,0)'s quantized x
// payload words and per-group activation scales. Callers may pass nullptr.
void glm_fp8_fnuz_mfma_block_gemv_debug(const uint16_t* x, const uint8_t* w,
                                        const float* scales, uint16_t* out, float* part,
                                        int M, int N, int K, int sk, uint32_t* dbg_sxa,
                                        float* dbg_s2a, void* stream);
void glm_fp8_fnuz_mfma_block_gemv_debug(const uint16_t* x, const uint8_t* w,
                                        const float* scales, uint16_t* out, float* part,
                                        int M, int N, int K, int sk, uint32_t* dbg_sxa,
                                        float* dbg_s2a);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
