// vkernels/kernels/moe_aux.hip — HIP kernels for the MXFP4 MoE orchestration
// ops (gfx942 portable replacement for AITER's gfx950-only module_moe_mxfp4_aux).
//
// These are gather / per-block quantization / scatter-reduce kernels. They
// are memory-bound, so unlike the fused MFMA GEMM they need no LDS tiling
// and launch with one block per (token, group) or per sorted row. The
// scatter-reduce kernels use atomicAdd into `out` (several sorted rows map
// to the same token), matching `down_combine_kernel` in moe_fused.hip.
//
// Numerically identical to the CPU reference in moe_aux.cpp: the fp4 (E2M1)
// and ue8m0 (`s << 23`) helpers mirror moe.cpp / moe_fused.hip.
#include "vkernels/kernels/moe_aux.hpp"

#include <cstdint>
#include <cmath>

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#  include "vkernels/kernels/moe_device.hip"
#  include "vkernels/util/error.hpp"

namespace vkernels::kernels::hip {

namespace {

// bf16 (uint16) → float32 (exact; shared bias/high bits).
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  float f;
  uint32_t u = static_cast<uint32_t>(b) << 16;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}

// E2M1 nibble → float32, via the shared bit-pattern decode in moe_device.hip
// (identical to fp4_nibble_to_float in moe.cpp: 0/±0.25/±1.0/±1.5/±2.0/±3.0/±inf/NaN).
__device__ __forceinline__ float fp4_nib_to_f32(uint8_t n) {
  uint32_t bits = fp4nib_to_f32bits(n);
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

// Inverse of fp4_nib_to_f32 for |v| ≤ FP4_MAX (= 3.0).  Ties round to the
// larger magnitude (matches the CPU reference).
__device__ __forceinline__ uint8_t f32_to_fp4_nib(float x) {
  float v = __builtin_fabsf(x);
  uint8_t mag;
  if (v < 0.125f) mag = 0;        // 0.0
  else if (v < 0.625f) mag = 1;   // 0.25
  else if (v < 1.25f) mag = 2;    // 1.0
  else if (v < 1.75f) mag = 3;    // 1.5
  else if (v < 2.5f) mag = 4;     // 2.0
  else mag = 5;                   // 3.0
  if (x < 0.0f) mag |= 0x8;
  return mag;
}

// ue8m0 scale byte → float32 (0xFF → 0.0, else 2^(s-127)).
__device__ __forceinline__ float ue8m0_decode(uint8_t s) {
  if (s == 0xFF) return 0.0f;
  float f;
  uint32_t bits = static_cast<uint32_t>(s) << 23;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

// ceil(log2(v)) for v > 0.
__device__ __forceinline__ int ceil_log2_f(float v) {
  return static_cast<int>(__builtin_ceilf(__builtin_log2f(v)));
}

}  // namespace

// ======================================================================
// #16 — mxfp4_moe_quant
//   grid  = M * n_groups   (one block per (token, group))
//   block = group_size     (one thread per element in the group)
// ======================================================================
__global__ void mxfp4_moe_quant_kernel(
    const uint16_t* __restrict__ A, uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scales, int hidden, int n_groups, int group_size) {
  int mg = blockIdx.x;
  int m = mg / n_groups;
  int g = mg % n_groups;
  int t = threadIdx.x;  // 0 .. group_size-1
  int base = g * group_size;

  __shared__ float s_amax;
  __shared__ uint8_t s_nib[256];  // group_size ≤ 256
  if (t == 0) s_amax = 0.0f;
  __syncthreads();

  float a = (t < group_size) ? __builtin_fabsf(bf16_to_f32(A[(size_t)m * hidden + base + t])) : 0.0f;
  // Block-wide amax reduction (group_size ≤ 256 → single __syncthreads tree).
  __shared__ float s_red[256];
  s_red[t] = a;
  __syncthreads();
  for (int s = group_size / 2; s > 0; s >>= 1) {
    if (t < s) s_red[t] = (s_red[t] > s_red[t + s]) ? s_red[t] : s_red[t + s];
    __syncthreads();
  }
  if (t == 0) s_amax = s_red[0];
  __syncthreads();

  // Exactly mirrors the CPU reference: a zero or non-finite amax produces
  // the 0xFF (zero) scale and zero nibbles for the whole group. For a real
  // amax the scale byte is clamped to [1, 254] (never 0) so the emitted
  // scale is a normal power of two that decodes identically via `s << 23`.
  // Every thread computes the same (sb, scale) from the shared amax.
  uint8_t sb;
  float scale;
  if (!(s_amax > 0.0f) || !__builtin_isfinite(s_amax)) {
    sb = 0xFF;
    scale = 0.0f;
  } else {
    int e = ceil_log2_f(s_amax / 3.0f);
    int s = e + 127;
    if (s < 1) s = 1;
    if (s > 254) s = 254;
    sb = static_cast<uint8_t>(s);
    scale = ue8m0_decode(sb);
  }

  if (t == 0) scales[(size_t)m * n_groups + g] = sb;
  if (sb == 0xFF) {
    if (t < group_size / 2)
      packed[(size_t)m * (hidden / 2) + base / 2 + t] = 0;
    return;
  }

  // Each thread quantizes its element into shared, then pairs pack a byte.
  if (t < group_size)
    s_nib[t] = f32_to_fp4_nib(bf16_to_f32(A[(size_t)m * hidden + base + t]) / scale);
  __syncthreads();
  if (t < group_size / 2) {
    uint8_t lo = s_nib[2 * t] & 0x0F;
    uint8_t hi = s_nib[2 * t + 1] & 0x0F;
    packed[(size_t)m * (hidden / 2) + base / 2 + t] = (hi << 4) | lo;
  }
}

// ======================================================================
// #17/#18 — mxfp4_moe_sort / mxfp4_moe_sort_scales
//   grid  = EM   (one block per sorted row)
//   block = min(width, 1024) threads, strided over the row width
//   Reads flat = sorted_ids[r]; if real, copies src[token] else zeroes dst.
// ======================================================================
__global__ void moe_sort_kernel(
    const void* __restrict__ src_v, const int32_t* __restrict__ sorted_ids,
    void* __restrict__ dst_v, int width, int M, int top_k, int elem) {
  int r = blockIdx.x;
  int flat = sorted_ids[r];
  const bool real = (flat >= 0 && flat < M * top_k);
  int token = real ? (flat / top_k) : 0;
  // Element dispatch by `elem` (sizeof of one element: 1 or 2).
  const uint8_t* src8 = static_cast<const uint8_t*>(src_v);
  uint8_t* dst8 = static_cast<uint8_t*>(dst_v);
  int row_bytes = width * elem;
  const uint8_t* srow = real ? (src8 + (size_t)token * row_bytes) : nullptr;
  uint8_t* drow = dst8 + (size_t)r * row_bytes;
  for (int j = threadIdx.x; j < row_bytes; j += blockDim.x) {
    drow[j] = real ? srow[j] : 0;
  }
}

// ======================================================================
// #19 — mxfp4_moe_scatter_reduce (float32 partials)
//   grid  = EM   (one block per sorted row)
//   block = min(width, 1024) threads, strided over `width`
//   For each real row r: out[token] += partial[r] * topk_w[r]  (atomic)
// ======================================================================
__global__ void moe_scatter_reduce_kernel(
    const float* __restrict__ partial, const float* __restrict__ topk_w,
    const int32_t* __restrict__ sorted_ids, float* __restrict__ out,
    int width, int M, int top_k) {
  int r = blockIdx.x;
  int flat = sorted_ids[r];
  if (flat < 0 || flat >= M * top_k) return;
  int token = flat / top_k;
  float w = topk_w[r];
  const float* src = partial + (size_t)r * width;
  float* dst = out + (size_t)token * width;
  for (int j = threadIdx.x; j < width; j += blockDim.x)
    atomicAdd(&dst[j], src[j] * w);
}

// ======================================================================
// #19q — mxfp4_moe_scatter_reduce_q (quantized partials, inline dequant)
// ======================================================================
__global__ void moe_scatter_reduce_q_kernel(
    const uint8_t* __restrict__ partial_q, const uint8_t* __restrict__ partial_s,
    const float* __restrict__ topk_w, const int32_t* __restrict__ sorted_ids,
    float* __restrict__ out, int width, int M, int top_k,
    int n_groups, int group_size) {
  int r = blockIdx.x;
  int flat = sorted_ids[r];
  if (flat < 0 || flat >= M * top_k) return;
  int token = flat / top_k;
  float w = topk_w[r];
  const uint8_t* pq = partial_q + (size_t)r * (width / 2);
  const uint8_t* ps = partial_s + (size_t)r * n_groups;
  float* dst = out + (size_t)token * width;
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    uint8_t byte = pq[j / 2];
    uint8_t nib = (j & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    float scale = ue8m0_decode(ps[j / group_size]);
    float v = fp4_nib_to_f32(nib) * scale;
    atomicAdd(&dst[j], v * w);
  }
}

}  // namespace vkernels::kernels::hip

// ======================================================================
//  Host launchers — mirror the CPU reference signatures.
// ======================================================================
namespace vkernels::kernels::hip {

namespace {
int block_of(int n) { return n < 1024 ? (n > 0 ? n : 1) : 1024; }
}  // namespace

void mxfp4_moe_quant(const uint16_t* A, uint8_t* packed, uint8_t* scales,
                     int M, int hidden, int group_size) {
  int n_groups = hidden / group_size;
  int gs = group_size < 256 ? group_size : 256;
  mxfp4_moe_quant_kernel<<<dim3(M * n_groups), dim3(gs)>>>(
      A, packed, scales, hidden, n_groups, group_size);
  hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess, "mxfp4_moe_quant launch failed");
}

void mxfp4_moe_sort(const uint16_t* A, const int32_t* sorted_ids,
                    uint16_t* A_sorted, int M, int hidden, int top_k, int EM) {
  moe_sort_kernel<<<dim3(EM), dim3(block_of(hidden))>>>(
      A, sorted_ids, A_sorted, hidden, M, top_k, 2);
  hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess, "mxfp4_moe_sort launch failed");
}

void mxfp4_moe_sort_scales(const uint8_t* scales, const int32_t* sorted_ids,
                           uint8_t* scales_sorted, int M, int n_groups,
                           int top_k, int EM) {
  moe_sort_kernel<<<dim3(EM), dim3(block_of(n_groups))>>>(
      scales, sorted_ids, scales_sorted, n_groups, M, top_k, 1);
  hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess, "mxfp4_moe_sort_scales launch failed");
}

void mxfp4_moe_scatter_reduce(const float* partial, const float* topk_w,
                              const int32_t* sorted_ids, float* out,
                              int M, int width, int top_k, int EM) {
  moe_scatter_reduce_kernel<<<dim3(EM), dim3(block_of(width))>>>(
      partial, topk_w, sorted_ids, out, width, M, top_k);
  hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess, "mxfp4_moe_scatter_reduce launch failed");
}

void mxfp4_moe_scatter_reduce_q(const uint8_t* partial_q,
                                const uint8_t* partial_s,
                                const float* topk_w,
                                const int32_t* sorted_ids, float* out,
                                int M, int width, int top_k, int EM,
                                int group_size) {
  int n_groups = width / group_size;
  moe_scatter_reduce_q_kernel<<<dim3(EM), dim3(block_of(width))>>>(
      partial_q, partial_s, topk_w, sorted_ids, out, width, M, top_k,
      n_groups, group_size);
  hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess, "mxfp4_moe_scatter_reduce_q launch failed");
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
