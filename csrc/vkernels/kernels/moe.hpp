// vkernels/kernels/moe.hpp
//
// AMD gfx942 (CDNA3 / MI300A) MXFP4 W4A16 MoE low-level kernel primitives.
//
// These fill the gaps where CDNA4-only (gfx950) instructions are used in
// the AITER flydsl MXFP4 fused-MoE path and do not lower on gfx942:
//   #12 — GFX942_SW_LDS_FILL  (rocdl.raw_ptr_buffer_load_lds fallback)
//   #13 — GFX942_SW_CVT       (fp4→bf16 software dequant)
//   #14 — GFX942_ASYNC_OFF    (platform-default async-copy gate)
//   #15 — GFX942_K16_SPLIT    (K32 bf16 MFMA → two K16)
//
// Every operation follows the vkernels two-implementation model:
//   moe.cpp   — CPU reference (oracle), always compiled, in vkernels::kernels
//   moe.hip   — HIP implementation, compiled with VKERNELS_HAS_HIP, in
//               vkernels::kernels::hip (matching the cuda:: namespace convention)
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/util/span.hpp"

namespace vkernels::kernels {

// ---------------------------------------------------------------------------
// #12 — Software direct-to-LDS fill (CPU reference)
// ---------------------------------------------------------------------------
// Copies a tile from global memory to LDS using a plain memcpy. On GPU the
// HIP implementation (hip::direct_lds_fill_bf16) uses vectorised global
// loads + LDS stores to replace the CDNA4-only raw_ptr_buffer_load_lds.
void direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                          std::size_t elements);

// ---------------------------------------------------------------------------
// #13 — Software fp4→bf16 dequant
// ---------------------------------------------------------------------------
// Converts packed fp4 (E2M1 microscaling: sign|exp2|mant1, two values per
// byte, low nibble first) to bf16 (uint16_t bit patterns).
//
// out.size() must be exactly 2 × packed.size().
// scale defaults to 1.0 (no scaling); pass a per-block scale otherwise.
//
// fp4 representable values: 0, ±0.25, ±1.0, ±1.5, ±2.0, ±3.0, ±inf, NaN
void fp4_to_bf16_dequant(Span<const uint8_t> packed, Span<uint16_t> out,
                         float scale = 1.0f);

// ---------------------------------------------------------------------------
// #14 — Platform async-copy gate
// ---------------------------------------------------------------------------
// Returns whether async copy (hardware copy-engine overlap) should be used.
//
// On gfx942 (CDNA3) it misbehaves and defaults to OFF.
// On other architectures it defaults to ON.
// The K3_NO_ASYNC env var overrides: "0"=ON, "1"=OFF.
bool use_async_copy_default();

// ---------------------------------------------------------------------------
// #15 — K16 bf16 MFMA (CPU reference)
// ---------------------------------------------------------------------------
// For gfx942, the K32 bf16 MFMA (CDNA4-only) is emulated by calling this
// K16 function twice (once for K=0..15, once for K=16..31).
//
//   C[0..3] += A[0..1] × B[0..1]   (16×16×16 bf16, accumulator fp32)
//
// `c`: 4 × float accumulators (updated in-place)
// `a`: 2 × uint32_t — packed bf16 A fragment
// `b`: 2 × uint32_t — packed bf16 B fragment
// `cbsz`, `abid`, `blgp`: MFMA control flags (typically 0)
void mfma_f32_16x16x16bf16(float c[4], const uint32_t a[2],
                           const uint32_t b[2],
                           int cbsz = 0, int abid = 0, int blgp = 0);

}  // namespace vkernels::kernels
