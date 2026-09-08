// vkernels/kernels/moe_device.hip
//
// Device-side declarations shared by moe.hip and moe_fused.hip.
// Included only when VKERNELS_HAS_HIP is set.
#pragma once

#if VKERNELS_HAS_HIP
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

#include "vkernels/kernels/device_numeric.cuh"  // bf16<->f32 + fp8 e4m3fnuz conversions

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

// The bf16<->f32 and fp8 e4m3fnuz conversions formerly defined here now live
// in device_numeric.cuh (included above): the CUDA TUs need them too, and the
// definitions must not fork per backend. They keep resolving unqualified from
// this namespace (vkernels::kernels::hip) via the enclosing vkernels::kernels.

// ======================================================================
//  K16 bf16 MFMA (device-only, inline asm)
//  NOTE: Uses generic "v" constraints (no hardcoded VGPRs).
//  The compiler handles register allocation.  The MFMA instruction
//  syntax uses register *indices*, so we extract them and format
//  the range syntax (v[lo:hi]) for each operand group.
//
//  Portability: the register-asm body only compiles on AMD device
//  passes, so the SIGNATURE stays visible in every pass (HIP host
//  passes type-check __global__ callers of this helper) while the
//  body is guarded. AMD device passes for unsupported targets #error
//  instead of silently mis-executing; CUDA (nvcc) passes -- which
//  never call this -- get an empty body.
// ======================================================================
__device__ __forceinline__ void mfma_f32_16x16x16bf16(
    float c[4], const uint32_t a[2], const uint32_t b[2],
    int cbsz = 0, int abid = 0, int blgp = 0) {
#if defined(__gfx942__) || defined(__gfx90a__) || defined(__gfx908__)
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
#elif defined(__HIP_DEVICE_COMPILE__)
#error "mfma_f32_16x16x16bf16: unsupported AMD device target"
#else
  // HIP host pass or nvcc (host or device) pass: this helper is
  // gfx942-family-only and nothing outside AMD gemm kernels calls it;
  // keep the body empty so the TU still type-checks.
  (void)c; (void)a; (void)b; (void)cbsz; (void)abid; (void)blgp;
#endif
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
