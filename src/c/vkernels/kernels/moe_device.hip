// vkernels/kernels/moe_device.hip
//
// Device-side declarations shared by moe.hip and moe_fused.hip.
// Included only when VKERNELS_HAS_HIP is set.
#pragma once

#if VKERNELS_HAS_HIP
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

namespace vkernels::kernels::hip {

// ======================================================================
//  E2M1 nibble decode → float bit pattern (uint32_t)
// ======================================================================
__host__ __device__ __forceinline__ uint32_t fp4nib_to_f32bits(int nibble) {
  int s = (nibble >> 3) & 1;
  int e = (nibble >> 1) & 3;
  int m = nibble & 1;
  if (e == 0) {
    if (m == 0) return s ? 0x80000000u : 0x00000000u;
    else        return (s ? 0x80000000u : 0u) | 0x3E800000u;
  } else if (e == 3) {
    if (m == 0) return (s ? 0xFF800000u : 0x7F800000u);
    else        return 0x7FC00000u;
  } else {
    uint32_t eb = (127u + static_cast<uint32_t>(e) - 1u) << 23;
    uint32_t mb = m ? 0x400000u : 0u;
    return (s ? 0x80000000u : 0u) | eb | mb;
  }
}

// Round float (as uint32_t bits) → bf16 (uint16_t, RNE).
__host__ __device__ __forceinline__ uint16_t f32bits_to_bf16(uint32_t bits) {
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// bf16 (uint16, IEEE-754 top-16 bits) → float (zero-extend the low 17
// mantissa bits). The inverse of f32bits_to_bf16's round; every bf16 device
// kernel (gemm_bf16, moe_aux, moe_fused, dsa, mhc) loads weights/activations
// through this single definition.
__host__ __device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  float f;
  uint32_t u = static_cast<uint32_t>(b) << 16;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}

// Round a float value → bf16 (uint16_t, RNE). Convenience wrapper around
// f32bits_to_bf16 for the common "I have a float, not raw uint32 bits" call
// site (the device kernels' store epilogue).
__host__ __device__ __forceinline__ uint16_t f2bf(float v) {
  uint32_t b;
  __builtin_memcpy(&b, &v, sizeof(float));
  return f32bits_to_bf16(b);
}

// OCP FP8 e4m3fnuz -> fp32 (GLM-5.3-Flash on MI300A stores its KV cache and
// the DSA-indexer query as torch.float8_e4m3fnuz). e4m3fnuz layout: 1 sign |
// 4 exponent (BIAS 8) | 3 mantissa, NO infinity, NaN at 0x7F (exp=15,
// mant=7), max finite 0x7E = 2^(15-8)*(1+6/8) = 224.0, and NO negative zero
// (0x00 AND 0x80 both decode to +0.0). Subnormals (exp==0, mant!=0) scale
// as m * 2^(1-bias) = m * 2^-7. Mirrors fp4nib_to_f32bits / bf16_to_f32:
// pure IEEE bit manipulation, no FP8 header. __host__ as well as __device__
// so the GPU cross-check harness (meta/benchmarks/test_dsa_topk_correct.hip)
// can dequant the fp8 source on the host -- the value BOTH the CPU oracle
// (dsa_topk_logits_cpu, fed the recovered fp32) and the device kernel
// (dsa.hip, dequanting on load) see. This MUST match
// torch.tensor(b, dtype=torch.uint8).view(torch.float8_e4m3fnuz).to(torch.float32)
// exactly; the harness cross-checks the two dequant paths.
__host__ __device__ __forceinline__ float fp8e4m3fnuz_to_f32(uint8_t b) {
  const uint32_t s = static_cast<uint32_t>(b >> 7) & 1u;     // sign
  const uint32_t e = static_cast<uint32_t>(b >> 3) & 0xFu;   // exponent (bias 8)
  const uint32_t m = static_cast<uint32_t>(b) & 0x7u;        // mantissa (3)
  if ((b & 0x7Fu) == 0u) return 0.0f;                        // +0 (0x00 AND 0x80)
  float f;
  if (e == 15u && m == 7u) {                                 // 0x7F = NaN -> qNaN
    const uint32_t qnan = 0x7fc00000u;
    __builtin_memcpy(&f, &qnan, sizeof(f));
  } else if (e == 0u) {                                      // subnormal: m*2^-7
    const float v = static_cast<float>(m) * 0x1p-7f;         // m*2^(1-8), exact
    f = s ? -v : v;
  } else {                                                   // normal: 2^(e-8)*(1+m/8)
    const uint32_t bits = (s << 31) | ((e + 119u) << 23) | (m << 20);
    __builtin_memcpy(&f, &bits, sizeof(f));
  }
  return f;
}

// ======================================================================
//  K16 bf16 MFMA (device-only, inline asm)
//  NOTE: Uses generic "v" constraints (no hardcoded VGPRs).
//  The compiler handles register allocation.  The MFMA instruction
//  syntax uses register *indices*, so we extract them and format
//  the range syntax (v[lo:hi]) for each operand group.
// ======================================================================
__device__ __forceinline__ void mfma_f32_16x16x16bf16(
    float c[4], const uint32_t a[2], const uint32_t b[2],
    int cbsz = 0, int abid = 0, int blgp = 0) {
  (void)cbsz; (void)abid; (void)blgp;
  register float   r0 asm("v0"), r1 asm("v1"), r2 asm("v2"), r3 asm("v3");
  register uint32_t s0 asm("v4"), s1 asm("v5");
  register uint32_t t0 asm("v6"), t1 asm("v7");
  // Simple direct assignment — compiler emits v_mov_b32
  r0 = c[0]; r1 = c[1]; r2 = c[2]; r3 = c[3];
  s0 = a[0]; s1 = a[1]; t0 = b[0]; t1 = b[1];
  __asm__ __volatile__(
      "v_mfma_f32_16x16x16bf16_1k v[0:3], v[4:5], v[6:7], v[0:3]"
      : "+v"(r0), "+v"(r1), "+v"(r2), "+v"(r3)
      : "v"(s0), "v"(s1), "v"(t0), "v"(t1));
  c[0] = r0; c[1] = r1; c[2] = r2; c[3] = r3;
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
