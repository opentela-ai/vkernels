// vkernels/kernels/moe.hip — HIP (AMD CDNA3 gfx942) implementations.
//
// These are the GPU-accelerated paths compiled only when VKERNELS_HAS_HIP.
// Each kernel targets the CDNA3 ISA and uses gfx942-native instructions to
// replace the CDNA4-only paths in AITER's flydsl MXFP4 MoE pipeline.
//
// Following the vkernels two-implementation model (see elementwise.cu,
// gemm.cu), the HIP launchers and device functions live in
// vkernels::kernels::hip, while the CPU references stay in
// vkernels::kernels (moe.cpp).
#include "vkernels/kernels/moe.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>

#  include <cstdlib>
#  include <cstring>

#  include "vkernels/util/error.hpp"

namespace vkernels::kernels::hip {

// ======================================================================
// #12 — Software direct-to-LDS fill (GFX942_SW_LDS_FILL)
// ======================================================================
//
// Device-side implementation. Replaces the CDNA4-only
// `rocdl.raw_ptr_buffer_load_lds` with vectorised global loads into VGPRs
// followed by LDS stores at lane-major addresses.
//
// Each thread loads `elements_per_thread` bf16 values (typically 8, for a
// 128-bit / 16-byte load) from `global_src` and writes them to `lds_dst`
// at the same lane-major offset the hardware path would have produced:
//
//   lds_dst[lane_id * elements_per_thread + k] =
//       global_src[lane_id * elements_per_thread + k]   for k in [0, ept-1]

__device__ void direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                                     int elements_per_thread) {
  // Each thread loads 4 dwords (128 bits = 8 bf16 values) from global and
  // writes them to LDS. The address in global is lane-major: lane N gets
  // elements [N*ept .. N*ept+ept-1].
  int tid = threadIdx.x;
  const uint32_t* src =
      reinterpret_cast<const uint32_t*>(global_src) + tid * (elements_per_thread / 2);
  uint32_t* dst =
      reinterpret_cast<uint32_t*>(lds_dst) + tid * (elements_per_thread / 2);

  // 128-bit load via uint4 — compiler emits global_load_dwordx4 for
  // aligned 128-bit pointer dereferences on AMD GPUs.
  uint4 v = *reinterpret_cast<const uint4*>(src);
  dst[0] = v.x;
  dst[1] = v.y;
  dst[2] = v.z;
  dst[3] = v.w;
}

// ======================================================================
// #13 — Software fp4→bf16 dequant (GFX942_SW_CVT)
// ======================================================================

// Helper: convert a single fp4 nibble (E2M1) to a float bit pattern (uint32_t).
// Mirrors the CPU reference in moe.cpp.
__host__ __device__ uint32_t fp4nib_to_f32bits(int nibble) {
  int s = (nibble >> 3) & 1;
  int e = (nibble >> 1) & 3;
  int m = nibble & 1;

  if (e == 0) {
    if (m == 0) {
      return s ? 0x80000000u : 0x00000000u;  // ±0
    } else {
      return (s ? 0x80000000u : 0u) | 0x3E800000u;  // ±0.25 = 2^(-2)
    }
  } else if (e == 3) {
    if (m == 0) {
      return (s ? 0xFF800000u : 0x7F800000u);  // ±inf
    } else {
      return 0x7FC00000u;  // NaN
    }
  } else {
    // Normal: (-1)^s × 2^(e-1) × (1 + m×0.5)
    uint32_t exp_bits = (127u + static_cast<uint32_t>(e) - 1u) << 23;
    uint32_t mant_bits = m ? 0x400000u : 0u;
    return (s ? 0x80000000u : 0u) | exp_bits | mant_bits;
  }
}

// Round a float (as uint32_t bits) to bf16 (uint16_t, RNE).
__host__ __device__ uint16_t f32bits_to_bf16(uint32_t bits) {
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// Device kernel: dequantise packed fp4 → bf16 with an optional scale.
__global__ void fp4_to_bf16_kernel(const uint8_t* __restrict__ packed,
                                   uint16_t* __restrict__ out,
                                   std::size_t n_packed, float scale) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_packed) return;

  uint8_t byte = packed[i];
  // Low nibble first, then high nibble.
  uint32_t flo = fp4nib_to_f32bits(byte & 0x0F);
  uint32_t fhi = fp4nib_to_f32bits((byte >> 4) & 0x0F);

  // Apply scale: reinterpret as float, multiply, round back.
  float vlo, vhi;
  std::memcpy(&vlo, &flo, sizeof(float));
  std::memcpy(&vhi, &fhi, sizeof(float));
  vlo *= scale;
  vhi *= scale;

  uint32_t bits_lo, bits_hi;
  std::memcpy(&bits_lo, &vlo, sizeof(float));
  std::memcpy(&bits_hi, &vhi, sizeof(float));

  out[i * 2] = f32bits_to_bf16(bits_lo);
  out[i * 2 + 1] = f32bits_to_bf16(bits_hi);
}

void fp4_to_bf16_dequant(const uint8_t* packed, uint16_t* out,
                         std::size_t n, float scale) {
  VK_EXPECTS(packed != nullptr, "packed must not be null");
  VK_EXPECTS(out != nullptr, "out must not be null");
  VK_EXPECTS(n > 0, "n must be positive");

  constexpr int kBlock = 256;
  int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  fp4_to_bf16_kernel<<<blocks, kBlock>>>(packed, out, n, scale);
  hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess, "hip fp4_to_bf16 launch failed");
}

// ======================================================================
// #14 — Platform async-copy gate (GFX942_ASYNC_OFF)
// ======================================================================
//
// On CDNA3 (gfx942) the async-copy path in the flydsl fused-MoE pipeline
// misbehaves and should default to OFF. On CDNA4 (gfx950) it works
// correctly and should default to ON.

bool use_async_copy_default() {
  // Environment variable override: K3_NO_ASYNC=1 forces off, =0 forces on.
  const char* env = std::getenv("K3_NO_ASYNC");
  if (env) {
    return env[0] != '1';
  }

  // Query the device architecture to detect gfx942.
  int device_id = 0;
  hipError_t err = hipGetDevice(&device_id);
  if (err != hipSuccess) {
    return false;  // conservative default
  }

  hipDeviceProp_t props;
  err = hipGetDeviceProperties(&props, device_id);
  if (err != hipSuccess) {
    return false;
  }

  // gcnArchName is e.g. "gfx90a:sramecc+:xnack-" or "gfx942:sramecc+:xnack-".
  // gfx942 (CDNA3, e.g. MI300X/MI300A): async copy OFF.
  if (std::strstr(props.gcnArchName, "gfx942") != nullptr) {
    return false;
  }

  // All other architectures (gfx950, gfx90a, ...): async copy ON.
  return true;
}

// ======================================================================
// #15 — K32 bf16 MFMA → two K16 split (GFX942_K16_SPLIT)
// ======================================================================
//
// On gfx942 the CDNA4-only `v_mfma_f32_16x16x32_bf16` (K32 bf16 MFMA) does
// not select. This function issues a single K16 bf16 MFMA
// (`v_mfma_f32_16x16x16bf16`) on gfx942. By calling it twice with the
// low and high K halves of the A and B fragments, the caller emulates a
// K32 MFMA:
//
//   hip::mfma_f32_16x16x16bf16(c, a_lo, b_lo, ...);  // K=0..15
//   hip::mfma_f32_16x16x16bf16(c, a_hi, b_hi, ...);  // K=16..31
//
// Register layout for the 16×16×16bf16 MFMA (gfx942):
//   C[0..3]: 4 VGPRs (float32 accumulators, read+write)
//   A[0..1]: 2 VGPRs (each holds 2 packed bf16 values = 4 bf16 total)
//   B[0..1]: 2 VGPRs (each holds 2 packed bf16 values = 4 bf16 total)
//
// Instruction (CDNA2-level, available on CDNA3 gfx942):
//   v_mfma_f32_16x16x16bf16 v[cdst0:cdst3], v[a0:a1], v[b0:b1], v[cdst0:cdst3]
//                            cbsz:N abid:M blgp:P

__device__ void mfma_f32_16x16x16bf16(float c[4], const uint32_t a[2],
                                      const uint32_t b[2],
                                      int cbsz, int abid, int blgp) {
  // The v_mfma_f32_16x16x16bf16_1k instruction requires consecutive VGPR
  // ranges for accumulator (4 regs) and operands (2+2 regs).  Use explicit
  // register variables to guarantee contiguity on gfx90a+.
  register float r0 asm("v0"), r1 asm("v1"), r2 asm("v2"), r3 asm("v3");
  register uint32_t s0 asm("v4"), s1 asm("v5");
  register uint32_t t0 asm("v6"), t1 asm("v7");
  __builtin_memcpy(&r0, &c[0], 4); __builtin_memcpy(&r1, &c[1], 4);
  __builtin_memcpy(&r2, &c[2], 4); __builtin_memcpy(&r3, &c[3], 4);
  s0 = a[0]; s1 = a[1]; t0 = b[0]; t1 = b[1];

  __asm__ __volatile__(
      "v_mfma_f32_16x16x16bf16_1k v[0:3], v[4:5], v[6:7], v[0:3]"
      : "+v"(r0), "+v"(r1), "+v"(r2), "+v"(r3)
      : "v"(s0), "v"(s1), "v"(t0), "v"(t1));

  __builtin_memcpy(&c[0], &r0, 4); __builtin_memcpy(&c[1], &r1, 4);
  __builtin_memcpy(&c[2], &r2, 4); __builtin_memcpy(&c[3], &r3, 4);
  (void)cbsz; (void)abid; (void)blgp;
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
