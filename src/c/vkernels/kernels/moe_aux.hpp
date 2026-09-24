// vkernels/kernels/moe_aux.hpp
//
// MXFP4 MoE orchestration ops — the per-block quantization, token→expert
// gather, and routed scatter-reduce that bracket a grouped MXFP4 GEMM.
//
// On gfx950 these are provided by AITER's `module_moe_mxfp4_aux` (JIT
// kernels whose 82 KB LDS requirement exceeds the 64 KB limit of gfx942 /
// MI300A). vkernels re-implements them as portable host references + HIP
// kernels so a self-contained W4A4 MoE serving path can be assembled on
// gfx942:
//
//   align     (moe_align_block_size — see moe_fused.hpp)
//     → sort      (mxfp4_moe_sort)        gather A    [M, hidden] → [EM, hidden]
//     → quant     (mxfp4_moe_quant)       E2M1+ue8m0 per token/block
//     → sort_scales (mxfp4_moe_sort_scales) gather scales to sorted order
//     → grouped GEMM (fused_moe_mxfp4 — see moe_fused.hpp)
//     → scatter_reduce[_q] (below)        routed combine → [M, hidden]
//
// The fp4 (E2M1) and scale (ue8m0, `s << 23`) layouts are identical to the
// weight decode in moe.cpp, so a W4A4 GEMM that uses the same format for A
// and W is numerically symmetric.
//
// Two-implementation model (same as moe / moe_fused):
//   moe_aux.cpp — CPU reference (oracle), always compiled, in vkernels::kernels
//   moe_aux.hip — HIP kernels, compiled with VKERNELS_HAS_HIP,
//                 in vkernels::kernels::hip
#pragma once

#include <cstdint>

namespace vkernels::kernels {

// Largest finite E2M1 value (matches `fp4_nibble_to_float` in moe.cpp:
// e=2, m=1 → 2^1 × 1.5 = 3.0). Activation groups are scaled by a ue8m0
// factor of 2^e with e = ceil(log2(amax / FP4_MAX)) so that every element
// of the group lands inside [-FP4_MAX, +FP4_MAX] without saturation.
inline constexpr float kMxFp4Max = 3.0f;

// ---------------------------------------------------------------------------
// #16 — mxfp4_moe_quant: per-token, per-group MXFP4 activation quantization
// ---------------------------------------------------------------------------
//
//   A       [M, hidden]              uint16  bf16 bit patterns
//   packed  [M, hidden / 2]          uint8   two E2M1 nibbles per byte
//                                        (low nibble = even K index)
//   scales  [M, hidden / group_size] uint8   ue8m0 (2^(s-127); 0xFF = zero)
//
// For each (token m, group g of `group_size` consecutive hidden elements):
//   amax = max_i |bf16_to_float(A[m, g*group_size + i])|
//   if amax == 0 (or non-finite): scales[m,g] = 0xFF, every nibble in the
//       group is 0, and the group dequantizes to exactly zero.
//   else: e = ceil(log2(amax / FP4_MAX)); sb = clamp(e + 127, 0, 254);
//         scale = 2^(sb - 127). Each element x = bf16_to_float(A[...]) / scale
//         is rounded to the nearest representable E2M1 value (ties round to
//         the larger magnitude) and packed into the low/high nibble.
//
// Constraints: hidden % 2 == 0, hidden % group_size == 0, group_size > 0.
void mxfp4_moe_quant(const uint16_t* A, uint8_t* packed, uint8_t* scales,
                     int M, int hidden, int group_size);

// ---------------------------------------------------------------------------
// #17 — mxfp4_moe_sort: gather activations into sorted (block-aligned) order
// ---------------------------------------------------------------------------
//
//   A          [M, hidden] uint16  bf16 activations (token order)
//   sorted_ids [EM]        int32   flat topk indices (token*top_k + sel)
//                                  from moe_align_block_size; an entry
//                                  >= M*top_k marks a padding row.
//   A_sorted   [EM, hidden] uint16 row r = A[sorted_ids[r] / top_k] when
//                                  sorted_ids[r] < M*top_k, else zero.
//
// The gather is exact for bf16 (a 16-bit copy). Padding rows are zeroed so
// that a subsequent `mxfp4_moe_quant` produces a 0xFF scale and zero nibbles
// for them, exactly like a real zero activation.
void mxfp4_moe_sort(const uint16_t* A, const int32_t* sorted_ids,
                    uint16_t* A_sorted, int M, int hidden, int top_k, int EM);

// ---------------------------------------------------------------------------
// #18 — mxfp4_moe_sort_scales: gather per-token scales into sorted order
// ---------------------------------------------------------------------------
//
//   scales         [M, n_groups] uint8  ue8m0 scales (token order)
//   sorted_ids     [EM]          int32  flat topk indices (as above)
//   scales_sorted  [EM, n_groups] uint8 row r = scales[sorted_ids[r] / top_k]
//                                       when real, else zero.
//
// Used after `mxfp4_moe_quant` (which produces scales in token order) when
// the grouped GEMM consumes activations and scales in sorted row order.
// Structurally identical to `mxfp4_moe_sort` on a uint8 tensor.
void mxfp4_moe_sort_scales(const uint8_t* scales, const int32_t* sorted_ids,
                           uint8_t* scales_sorted, int M, int n_groups,
                           int top_k, int EM);

// ---------------------------------------------------------------------------
// Fusion — mxfp4_moe_sorted_quant: gather + MXFP4 quantize in one pass
// ---------------------------------------------------------------------------
//
//   A          [M, hidden]      uint16 bf16 activations (token order)
//   sorted_ids [EM]             int32  flat topk indices (as mxfp4_moe_sort;
//                                    an entry outside [0, M*top_k) marks a
//                                    padding row)
//   packed     [EM, hidden / 2] uint8  two E2M1 nibbles per byte, SORTED row
//                                    order (low nibble = even K index)
//   scales     [EM, n_groups]   uint8  ue8m0 per-group scales, SORTED row order
//
// Per sorted row r this computes exactly
//   row_r = A[sorted_ids[r] / top_k]  (real row)  or the all-zero row (padding)
//   (packed[r, :], scales[r, :]) = mxfp4_moe_quant(row_r)
// in one pass, without materializing the [EM, hidden] bf16 A_sorted
// intermediate. The composition contracts it must satisfy (and which the
// tests check bit-exactly) are:
//
//   mxfp4_moe_sorted_quant(A, ids, pk, sc)
//     == mxfp4_moe_quant(mxfp4_moe_sort(A, ids))          (packed AND scales)
//   for REAL rows r (0 <= sorted_ids[r] < M*top_k) also:
//     scales[r, :] == mxfp4_moe_sort_scales(mxfp4_moe_quant(A).scales, ids)[r]
//
// Padding rows quantize exactly like a real all-zero activation (scale
// 0xFF + all-zero nibbles), matching the sort→quant composition. The
// standalone quant→sort_scales route instead writes literal 0 scale bytes
// for padding rows — a pre-existing asymmetry between the two standalone
// routes (the grouped GEMM never reads padding rows); this op follows
// sort→quant, the route it replaces. The composition contracts above are
// checked bit-exactly by tests/kernels/moe/test_moe_aux_fused.cpp. This
// removes one write + one read of EM*hidden*2 bytes (the A_sorted
// round-trip) and makes the separate sort_scales launch unnecessary.
// The three standalone ops stay fully intact; this op is additive.
void mxfp4_moe_sorted_quant(const uint16_t* A, const int32_t* sorted_ids,
                            uint8_t* packed, uint8_t* scales,
                            int M, int hidden, int group_size, int top_k,
                            int EM);

// ---------------------------------------------------------------------------
// #19 — mxfp4_moe_scatter_reduce: routed output combine (float32 partials)
// ---------------------------------------------------------------------------
//
//   partial    [EM, width] fp32  per-expert GEMM output, indexed by SORTED
//                               ROW (the row order of sorted_ids).
//   topk_w     [EM]        fp32  routing weights, sorted to match sorted_ids.
//   sorted_ids [EM]        int32 flat topk indices (>= M*top_k = padding).
//   out        [M, width]  fp32  accumulates in place — caller must zero it.
//
// For each real sorted row r (sorted_ids[r] < M*top_k):
//   token = sorted_ids[r] / top_k
//   out[token, :] += partial[r, :] * topk_w[r]
//
// This is the bias-free form of `moe_combine_cpu` (see moe_fused.hpp); bias
// is added separately in the W4A4 path. Matches AITER `mxfp4_moe_scatter_reduce`.
//
// HIP note (#145): the device path in moe_aux.hip implements this as an
// atomic-free token-major gather via an inverse permutation of sorted_ids.
// Given the contract that sorted_ids is a bijection on [0, M*top_k), every
// output element receives exactly top_k contributions, so the HIP kernel
// fully OVERWRITES `out` — a caller-side zero-init memset is unnecessary
// there (the CPU reference above keeps the accumulate-into-zeroed form).
void mxfp4_moe_scatter_reduce(const float* partial, const float* topk_w,
                              const int32_t* sorted_ids, float* out,
                              int M, int width, int top_k, int EM);

// ---------------------------------------------------------------------------
// #19q — mxfp4_moe_scatter_reduce_q: routed combine of a quantized partial
// ---------------------------------------------------------------------------
//
//   partial_q  [EM, width / 2]            uint8  packed E2M1 (as mxfp4_moe_quant)
//   partial_s  [EM, width / group_size]   uint8  ue8m0 per-group scales
//   topk_w     [EM]                       float  routing weights (sorted)
//   sorted_ids [EM]                       int32  flat topk indices (as above)
//   out        [M, width]                 float  accumulates in place (zero-init)
//
// Identical to `mxfp4_moe_scatter_reduce` except the per-expert partial is
// kept in the MXFP4 (E2M1 + ue8m0) layout and dequantized inline — the
// "Q" combine AITER ships on gfx950 to cut the scatter bandwidth. `width`
// must be divisible by `group_size` and by 2.
void mxfp4_moe_scatter_reduce_q(const uint8_t* partial_q,
                                const uint8_t* partial_s,
                                const float* topk_w,
                                const int32_t* sorted_ids, float* out,
                                int M, int width, int top_k, int EM,
                                int group_size);

}  // namespace vkernels::kernels

// HIP kernels (gfx942) live in moe_aux.hip under namespace
// vkernels::kernels::hip with identical signatures; they are self-declared
// there (not re-declared here) so the host reference names stay unique.
