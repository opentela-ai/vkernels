// vkernels/kernels/device_numeric.cuh
//
// The bf16 / fp8 e4m3fnuz numeric conversions shared by EVERY backend device
// TU: moe_device.hip (HIP moe family), dsa.hip + dsa.cu (the HIP and CUDA
// dsa_topk indexers), gemm_bf16.hip + gemm_bf16.cu, and the GPU
// cross-check harnesses. The bodies are pure IEEE bit manipulation -- no
// vendor intrinsics -- so ONE definition compiles under hipcc, nvcc and the
// __host__ passes of both. Include only from device-compiled sources
// (after the vendor runtime header); plain host TUs use their own
// .cpp-oracle copies instead.
//
// This header exists because the "device output matches the CPU oracle"
// invariant of these kernels rests on every dequant/round call site agreeing
// bit-for-bit (fp8e4m3fnuz_to_f32 additionally has to match torch's
// uint8->float8_e4m3fnuz cast exactly). Until the CUDA port each backend TU
// carried its own copy of that promise; now the promise is made once.
#pragma once

#include <cstdint>
#include <cstring>

namespace vkernels::kernels {

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
// so the GPU cross-check harnesses (test_dsa_topk_correct.{hip,cu}) can
// dequant the fp8 source on the host -- the value BOTH the CPU oracle
// (dsa_topk_logits_cpu, fed the recovered fp32) and the device kernels (the
// dsa_topk TUs, dequanting on load) see. This MUST match
// torch.tensor(b, dtype=torch.uint8).view(torch.float8_e4m3fnuz).to(torch.float32)
// exactly; the harnesses cross-check the two dequant paths.
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

}  // namespace vkernels::kernels
