// vkernels/kernels/moe.cpp — CPU reference (oracle) implementations.
//
// These are the golden reference paths against which HIP kernels are
// validated. Always compiled, independent of GPU toolkit presence.
#include "vkernels/kernels/moe.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

// ======================================================================
// #12 — Software direct-to-LDS fill (CPU reference)
// ======================================================================
void direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                          std::size_t elements) {
  if (elements == 0) return;  // empty copy is a no-op
  VK_EXPECTS(lds_dst != nullptr, "lds_dst must not be null");
  VK_EXPECTS(global_src != nullptr, "global_src must not be null");
  std::size_t bytes = elements * sizeof(uint16_t);
  std::memcpy(lds_dst, global_src, bytes);
}

// ======================================================================
// #13 — Software fp4→bf16 dequant (CPU reference)
// ======================================================================
//
// MXFP4 E2M1 format per nibble:
//   bits: sign(1) | exp(2) | mantissa(1)
//
//   Normal  (e ∈ {1,2}):  (-1)^s × 2^(e-1) × (1 + m/2)
//   Subnorm (e = 0, m=1): (-1)^s × 2^(-1) × (0 + m/2) = (-1)^s × 0.25
//   Zero    (e = 0, m=0): (-1)^s × 0
//   Inf     (e = 3, m=0): (-1)^s × inf
//   NaN     (e = 3, m=1): NaN
//
// Representable normal values: ±1.0, ±1.5, ±2.0, ±3.0
// Representable subnormal:    ±0.25

namespace {

float fp4_nibble_to_float(uint8_t nibble) {
  int s = (nibble >> 3) & 1;
  int e = (nibble >> 1) & 3;
  int m = nibble & 1;

  if (e == 0) {
    if (m == 0) {
      return s ? -0.0f : 0.0f;
    } else {
      return s ? -0.25f : 0.25f;
    }
  } else if (e == 3) {
    if (m == 0) {
      return s ? -std::numeric_limits<float>::infinity()
               : std::numeric_limits<float>::infinity();
    } else {
      return std::numeric_limits<float>::quiet_NaN();
    }
  } else {
    // Normal: (-1)^s × 2^(e-1) × (1 + m × 0.5)
    float val = 1.0f + static_cast<float>(m) * 0.5f;
    val *= static_cast<float>(1 << (e - 1));  // ×1 for e=1, ×2 for e=2
    return s ? -val : val;
  }
}

uint16_t float_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round-to-nearest-even: add half-ulp of the truncated 16 LSBs.
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

}  // namespace

void fp4_to_bf16_dequant(Span<const uint8_t> packed, Span<uint16_t> out,
                         float scale) {
  VK_EXPECTS(out.size() == packed.size() * 2,
             "out must have exactly 2× packed bytes (two fp4 values per byte)");

  for (std::size_t i = 0; i < packed.size(); ++i) {
    uint8_t byte = packed[i];
    float flo = fp4_nibble_to_float(byte & 0x0F) * scale;
    float fhi = fp4_nibble_to_float((byte >> 4) & 0x0F) * scale;
    out[i * 2] = float_to_bf16(flo);
    out[i * 2 + 1] = float_to_bf16(fhi);
  }
}

// ======================================================================
// #14 — Platform async-copy gate (CPU reference)
// ======================================================================
bool use_async_copy_default() {
  const char* env = std::getenv("K3_NO_ASYNC");
  if (env) {
    return env[0] != '1';  // "1" = force OFF
  }
  return true;  // host path always defaults to ON
}

// ======================================================================
// #15 — K16 bf16 MFMA (CPU reference)
// ======================================================================
//
// Emulates a single 16×16×16 bf16 MFMA on the host. The VGPR fragment
// layout packs 2 bf16 values per uint32_t (low 16 bits, high 16 bits).
//
// For the CPU oracle, we unpack to float, compute the per-thread
// portion of the dot product, accumulate, and repack the result.
//
// In the real MFMA, C[0..3] holds 4 float accumulators distributed across
// a 4×4 sub-tile of the 16×16 output. Each thread computes 4 dot products
// using the 4 bf16 values it holds from A and B.

namespace {

void bf16_packed_to_float2(uint32_t packed, float out[2]) {
  uint16_t lo = static_cast<uint16_t>(packed & 0xFFFF);
  uint16_t hi = static_cast<uint16_t>(packed >> 16);
  uint32_t lo_bits = static_cast<uint32_t>(lo) << 16;
  uint32_t hi_bits = static_cast<uint32_t>(hi) << 16;
  std::memcpy(&out[0], &lo_bits, sizeof(float));
  std::memcpy(&out[1], &hi_bits, sizeof(float));
}

}  // namespace

void mfma_f32_16x16x16bf16(float c[4], const uint32_t a[2],
                           const uint32_t b[2],
                           int /*cbsz*/, int /*abid*/, int /*blgp*/) {
  VK_EXPECTS(c != nullptr, "c must not be null");
  VK_EXPECTS(a != nullptr, "a must not be null");
  VK_EXPECTS(b != nullptr, "b must not be null");

  // Unpack each operand: A[2] → 4 bf16 values, B[2] → 4 bf16 values.
  float a_f32[4], b_f32[4];
  bf16_packed_to_float2(a[0], &a_f32[0]);
  bf16_packed_to_float2(a[1], &a_f32[2]);
  bf16_packed_to_float2(b[0], &b_f32[0]);
  bf16_packed_to_float2(b[1], &b_f32[2]);

  // Per-thread dot products: C[i] += Σ_k A[i,k] × B[k,i] for the 4
  // accumulator elements this thread owns in the 16×16 output tile.
  // In the MFMA VGPR layout, each thread's a_f32[i] pairs with b_f32[i]
  // for the same output element.
  for (int i = 0; i < 4; ++i) {
    c[i] += a_f32[i] * b_f32[i];
  }
}

}  // namespace vkernels::kernels
