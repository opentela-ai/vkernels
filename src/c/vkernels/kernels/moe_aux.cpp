// vkernels/kernels/moe_aux.cpp — CPU reference (oracle) for the MXFP4 MoE
// orchestration ops declared in moe_aux.hpp.
//
// Always compiled, independent of GPU toolkit presence. The fp4 (E2M1) and
// ue8m0 (`s << 23`) helpers are byte-identical to the weight decode in
// moe.cpp so that `mxfp4_moe_quant` is the exact inverse of the dequant and
// a W4A4 GEMM using the same layout for A and W is numerically symmetric.
#include "vkernels/kernels/moe_aux.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

namespace {

// Reinterpret a bf16 (uint16) bit pattern as float32 (zero-extend; bf16 and
// float32 share the high 16 bits and bias, so this is exact).
float bf16_to_float(uint16_t bits) {
  float f;
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// E2M1 nibble → float32 (identical to fp4_nibble_to_float in moe.cpp).
float fp4_nibble_to_float(uint8_t nibble) {
  int s = (nibble >> 3) & 1;
  int e = (nibble >> 1) & 3;
  int m = nibble & 1;
  if (e == 0) {
    if (m == 0) return s ? -0.0f : 0.0f;
    return s ? -0.25f : 0.25f;
  }
  if (e == 3) {
    if (m == 0) return s ? -std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::infinity();
    return std::numeric_limits<float>::quiet_NaN();
  }
  float val = (1.0f + static_cast<float>(m) * 0.5f) *
              static_cast<float>(1 << (e - 1));  // ×1 (e=1), ×2 (e=2)
  return s ? -val : val;
}

// Inverse of fp4_nibble_to_float for magnitudes ≤ FP4_MAX (= 3.0).  Given
// `x` (already divided by the per-group scale), round to the nearest
// representable E2M1 magnitude; ties round to the LARGER magnitude.  Returns
// the 4-bit nibble code (sign bit = 0x8).
//
//   magnitude |v|     code     value
//   < 0.125       0    0.0
//   < 0.625       1    0.25
//   < 1.25        2    1.0
//   < 1.75        3    1.5
//   < 2.5         4    2.0
//   else (≤ 3.0)  5    3.0
uint8_t float_to_fp4_nib(float x) {
  float v = std::fabs(x);
  uint8_t mag;
  if (v < 0.125f) mag = 0;        // 0.0
  else if (v < 0.625f) mag = 1;   // 0.25
  else if (v < 1.25f) mag = 2;    // 1.0
  else if (v < 1.75f) mag = 3;    // 1.5
  else if (v < 2.5f) mag = 4;     // 2.0
  else mag = 5;                   // 3.0 (|v| ≤ FP4_MAX guaranteed)
  if (x < 0.0f) mag |= 0x8;
  return mag;
}

// ue8m0 scale byte → float32 (0xFF → 0.0, else 2^(s - 127)).
float ue8m0_to_float(uint8_t s) {
  if (s == 0xFF) return 0.0f;
  float f;
  uint32_t bits = static_cast<uint32_t>(s) << 23;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// ceil(log2(v)) for v > 0, as an int.  std::ceil(std::log2(v)) is reliable
// for the bf16 dynamic range; the caller clamps the result.
int ceil_log2(float v) {
  return static_cast<int>(std::ceil(std::log2(v)));
}

}  // namespace

// ---------------------------------------------------------------------------
// #16 — mxfp4_moe_quant
// ---------------------------------------------------------------------------
void mxfp4_moe_quant(const uint16_t* A, uint8_t* packed, uint8_t* scales,
                     int M, int hidden, int group_size) {
  VK_EXPECTS(A != nullptr, "A must not be null");
  VK_EXPECTS(packed != nullptr, "packed must not be null");
  VK_EXPECTS(scales != nullptr, "scales must not be null");
  VK_EXPECTS(M >= 0 && hidden >= 0, "M and hidden must be non-negative");
  VK_EXPECTS(group_size > 0, "group_size must be positive");
  VK_EXPECTS(hidden % group_size == 0,
             "hidden must be a multiple of group_size");
  VK_EXPECTS(hidden % 2 == 0, "hidden must be even (two nibbles per byte)");

  const int n_groups = hidden / group_size;
  for (int m = 0; m < M; ++m) {
    const uint16_t* row = A + static_cast<std::size_t>(m) * hidden;
    uint8_t* pk = packed + static_cast<std::size_t>(m) * (hidden / 2);
    uint8_t* sc = scales + static_cast<std::size_t>(m) * n_groups;
    for (int g = 0; g < n_groups; ++g) {
      float amax = 0.0f;
      const int base = g * group_size;
      for (int i = 0; i < group_size; ++i) {
        float a = bf16_to_float(row[base + i]);
        float aa = std::fabs(a);
        if (aa > amax) amax = aa;
      }
      uint8_t sb;
      float scale;
      // Exactly mirrors the CPU reference: a zero or non-finite amax produces
      // the 0xFF (zero) scale and zero nibbles for the whole group. For a
      // real amax the scale byte is clamped to [1, 254] (never 0) so the
      // emitted scale is a normal power of two that decodes identically via
      // the simple `s << 23` ue8m0 path on host and device.
      if (!(amax > 0.0f) || !std::isfinite(amax)) {
        sb = 0xFF;
        scale = 0.0f;
      } else {
        int e = ceil_log2(amax / kMxFp4Max);
        int s = e + 127;
        if (s < 1) s = 1;
        if (s > 254) s = 254;
        sb = static_cast<uint8_t>(s);
        scale = ue8m0_to_float(sb);
      }
      sc[g] = sb;
      if (sb == 0xFF) {
        for (int kp = 0; kp < group_size / 2; ++kp) pk[base / 2 + kp] = 0;
        continue;
      }
      for (int i = 0; i < group_size; i += 2) {
        uint8_t lo = float_to_fp4_nib(bf16_to_float(row[base + i]) / scale);
        uint8_t hi = float_to_fp4_nib(bf16_to_float(row[base + i + 1]) / scale);
        pk[base / 2 + i / 2] = (hi << 4) | (lo & 0x0F);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// #17 — mxfp4_moe_sort  (gather A into sorted row order)
// ---------------------------------------------------------------------------
void mxfp4_moe_sort(const uint16_t* A, const int32_t* sorted_ids,
                    uint16_t* A_sorted, int M, int hidden, int top_k, int EM) {
  VK_EXPECTS(A != nullptr, "A must not be null");
  VK_EXPECTS(sorted_ids != nullptr, "sorted_ids must not be null");
  VK_EXPECTS(A_sorted != nullptr, "A_sorted must not be null");
  VK_EXPECTS(M >= 0 && hidden >= 0 && top_k > 0 && EM >= 0,
             "M, hidden, top_k, EM must be valid");
  const int N = M * top_k;
  for (int r = 0; r < EM; ++r) {
    uint16_t* dst = A_sorted + static_cast<std::size_t>(r) * hidden;
    int flat = sorted_ids[r];
    if (flat >= 0 && flat < N) {
      int token = flat / top_k;
      const uint16_t* src = A + static_cast<std::size_t>(token) * hidden;
      for (int j = 0; j < hidden; ++j) dst[j] = src[j];
    } else {
      for (int j = 0; j < hidden; ++j) dst[j] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// #18 — mxfp4_moe_sort_scales  (gather per-token scales into sorted order)
// ---------------------------------------------------------------------------
void mxfp4_moe_sort_scales(const uint8_t* scales, const int32_t* sorted_ids,
                           uint8_t* scales_sorted, int M, int n_groups,
                           int top_k, int EM) {
  VK_EXPECTS(scales != nullptr, "scales must not be null");
  VK_EXPECTS(sorted_ids != nullptr, "sorted_ids must not be null");
  VK_EXPECTS(scales_sorted != nullptr, "scales_sorted must not be null");
  VK_EXPECTS(M >= 0 && n_groups >= 0 && top_k > 0 && EM >= 0,
             "M, n_groups, top_k, EM must be valid");
  const int N = M * top_k;
  for (int r = 0; r < EM; ++r) {
    uint8_t* dst = scales_sorted + static_cast<std::size_t>(r) * n_groups;
    int flat = sorted_ids[r];
    if (flat >= 0 && flat < N) {
      int token = flat / top_k;
      const uint8_t* src = scales + static_cast<std::size_t>(token) * n_groups;
      for (int j = 0; j < n_groups; ++j) dst[j] = src[j];
    } else {
      for (int j = 0; j < n_groups; ++j) dst[j] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// #19 — mxfp4_moe_scatter_reduce  (routed combine, float32 partials)
// ---------------------------------------------------------------------------
void mxfp4_moe_scatter_reduce(const float* partial, const float* topk_w,
                              const int32_t* sorted_ids, float* out,
                              int M, int width, int top_k, int EM) {
  VK_EXPECTS(partial != nullptr, "partial must not be null");
  VK_EXPECTS(topk_w != nullptr, "topk_w must not be null");
  VK_EXPECTS(sorted_ids != nullptr, "sorted_ids must not be null");
  VK_EXPECTS(out != nullptr, "out must not be null");
  VK_EXPECTS(M >= 0 && width >= 0 && top_k > 0 && EM >= 0,
             "M, width, top_k, EM must be valid");
  const int N = M * top_k;
  for (int r = 0; r < EM; ++r) {
    int flat = sorted_ids[r];
    if (flat < 0 || flat >= N) continue;  // padding row
    int token = flat / top_k;
    float w = topk_w[r];
    const float* src = partial + static_cast<std::size_t>(r) * width;
    float* dst = out + static_cast<std::size_t>(token) * width;
    for (int j = 0; j < width; ++j) dst[j] += src[j] * w;
  }
}

// ---------------------------------------------------------------------------
// #19q — mxfp4_moe_scatter_reduce_q  (routed combine of a quantized partial)
// ---------------------------------------------------------------------------
void mxfp4_moe_scatter_reduce_q(const uint8_t* partial_q,
                                const uint8_t* partial_s,
                                const float* topk_w,
                                const int32_t* sorted_ids, float* out,
                                int M, int width, int top_k, int EM,
                                int group_size) {
  VK_EXPECTS(partial_q != nullptr, "partial_q must not be null");
  VK_EXPECTS(partial_s != nullptr, "partial_s must not be null");
  VK_EXPECTS(topk_w != nullptr, "topk_w must not be null");
  VK_EXPECTS(sorted_ids != nullptr, "sorted_ids must not be null");
  VK_EXPECTS(out != nullptr, "out must not be null");
  VK_EXPECTS(M >= 0 && width >= 0 && top_k > 0 && EM >= 0,
             "M, width, top_k, EM must be valid");
  VK_EXPECTS(group_size > 0, "group_size must be positive");
  VK_EXPECTS(width % 2 == 0, "width must be even (two nibbles per byte)");
  VK_EXPECTS(width % group_size == 0,
             "width must be a multiple of group_size");

  const int N = M * top_k;
  const int n_groups = width / group_size;
  for (int r = 0; r < EM; ++r) {
    int flat = sorted_ids[r];
    if (flat < 0 || flat >= N) continue;  // padding row
    int token = flat / top_k;
    float w = topk_w[r];
    const uint8_t* pq = partial_q + static_cast<std::size_t>(r) * (width / 2);
    const uint8_t* ps = partial_s + static_cast<std::size_t>(r) * n_groups;
    float* dst = out + static_cast<std::size_t>(token) * width;
    for (int g = 0; g < n_groups; ++g) {
      float scale = ue8m0_to_float(ps[g]);
      const int base = g * group_size;
      for (int i = 0; i < group_size; i += 2) {
        float lo = fp4_nibble_to_float(pq[base / 2 + i / 2] & 0x0F) * scale;
        float hi = fp4_nibble_to_float((pq[base / 2 + i / 2] >> 4) & 0x0F) * scale;
        dst[base + i] += lo * w;
        dst[base + i + 1] += hi * w;
      }
    }
  }
}

}  // namespace vkernels::kernels
