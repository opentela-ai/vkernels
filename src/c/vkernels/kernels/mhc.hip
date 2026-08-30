// vkernels/kernels/mhc.hip
//
// MHC multi-head hybrid-attention pre-norm (gfx942 / MI300A), issue #51
// part 2. Two of the tilelang MHC kernels from
// ``sglang/kernels/ops/layernorm/mhc.py`` fault/abort on gfx942:
//
//   * ``mhc_pre_gemm_sqrsum_splitk_kernel`` stage-0 aborts JIT init() with
//     "Requested dynamic shared memory 98304 exceeds device limit 65536":
//     tilelang's HIP codegen (jit/adapter/wrapper.py
//     PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY_HIP) only QUERIES
//     hipDeviceAttributeMaxSharedMemoryPerBlock and aborts if the request
//     exceeds it -- it never calls hipFuncSetAttribute to opt in to the
//     larger (~228 KB) dynamic-shared cap. At num_stages=2 a 256-wide
//     hidden block's per-stage shared (x_smem 16 KB bf16 + fn_smem 32 KB
//     fp32 = 49152) doubles to 98304 > 65536. The sglang-side workaround is
//     hidden_block 256->128; the proper fix is a tilelang HIP dynamic-smem
//     opt-in.
//   * ``mhc_post_tilelang`` for the same reason.
//
// This file re-implements both as gfx942 HIP kernels that run WITHOUT the
// workaround: they use *static* shared memory sized to fit MI300A's 64 KB
// non-optin cap (no pipelined double-buffer, no dynamic-shared request, no
// hipFuncSetAttribute opt-in). They are correctness-first baselines; the
// tiled, register-pressure-managed GEMM (mirroring the tilelang
// token_block x hidden_block + swizzled-shared structure) is a documented
// throughput follow-on.
//
// Storage follows the established gfx942 bf16 convention (gemm_bf16.hip,
// moe_device.hip, dsa.hip): bf16 is carried as raw uint16 IEEE-754 bit
// patterns and converted to/from fp32 with bf16_to_f32 (zero-extend of the
// top 16 bits) and f32bits_to_bf16 (round-to-nearest-even, the SAME helper
// the CPU oracle uses). The kernel is therefore bf16-tolerant against the
// oracle exactly as gemm_bf16 / dsa are.
//
// Two-implementation model (mirrors mla.{hpp,cpp,hip} and dsa.{hpp,cpp,hip}):
//   mhc.cpp  -- CPU reference (oracle), always compiled, fp32 throughout.
//   mhc.hip  -- this file, compiled with VKERNELS_HAS_HIP, fp32 accumulation
//               + bf16 storage. Cross-checked against the oracle by
//               test_mhc_correct.hip.
#include <cstdint>

#include "vkernels/kernels/mhc.hpp"

#if VKERNELS_HAS_HIP
#  include "vkernels/kernels/moe_device.hip"  // shared bf16<->f32 + fp4 helpers
#  include <hip/hip_runtime.h>

namespace vkernels::kernels::hip {

// Largest supported hidden width (hc_mult * hidden_size) on the device path.
// 28672 * 2 bytes (bf16) = 56 KB static shared, within MI300A's 64 KB
// non-optin cap. Covers both sglang shapes (16384 = GLM-5.3 hc_mult=4
// hidden=4096; 28672 = hidden=7168).
static constexpr int kMaxHcHidden = 28672;


// ---------------------------------------------------------------------------
//  mhc_pre_gemm_sqrsum
//
//  One block per token, 32 threads (one warp). The token row x[n,:] is
//  staged once in static shared memory as bf16 (<= 56 KB). Lane o (the
//  warp lane) owns output column o (hc_mult3 <= 32) and streams its fn
//  column from global memory. The per-token squared-sum is a strided
//  lane partial + warp-shuffle reduction.
//
//  This is a correctness-first baseline: each lane reads its own contiguous
//  fn row (stride hc_hidden_size across the warp at a fixed h), so the fn
//  reads are not coalesced and the kernel is memory-bound -- exactly the
//  binding resource the tiled, swizzled-shared tilelang version targets.
//  No dynamic-shared-memory workaround is needed: there is no dynamic
//  shared memory at all (a single static staging buffer, no pipelining).
// ---------------------------------------------------------------------------
__global__ void mhc_pre_gemm_sqrsum_kernel(
    const uint16_t* __restrict__ x,   // [num_tokens, hc_hidden_size] bf16
    const float* __restrict__ fn,     // [hc_mult3, hc_hidden_size] fp32
    float* __restrict__ out,          // [num_tokens, hc_mult3] fp32
    float* __restrict__ sqrsum,       // [num_tokens] fp32
    int num_tokens, int hc_mult3, int hc_hidden_size) {
  const int n = blockIdx.x;
  if (n >= num_tokens) return;
  const int o = threadIdx.x;  // warp lane == output column (hc_mult3 <= 32)

  __shared__ uint16_t sX[kMaxHcHidden];  // staged token row (bf16)
  for (int h = o; h < hc_hidden_size; h += 32)
    sX[h] = x[(size_t)n * hc_hidden_size + h];
  __syncthreads();

  // Per-token squared-sum: strided lane partial + warp-shuffle reduce.
  float sq = 0.0f;
  for (int h = o; h < hc_hidden_size; h += 32) {
    float xv = bf16_to_f32(sX[h]);
    sq += xv * xv;
  }
  for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off);
  if (o == 0) sqrsum[n] = sq;

  // GEMM: out[n, o] = sum_h x[n, h] * fn[o, h], for o in [0, hc_mult3).
  if (o < hc_mult3) {
    const float* fn_row = fn + (size_t)o * hc_hidden_size;
    float acc = 0.0f;
    for (int h = 0; h < hc_hidden_size; ++h)
      acc += bf16_to_f32(sX[h]) * fn_row[h];
    out[(size_t)n * hc_mult3 + o] = acc;
  }
}

// ---------------------------------------------------------------------------
//  mhc_post
//
//  One block per (token, output-head j), 256 threads. The a column
//  a[n, :, j] (hc fp32 values) and c[n, j] (scalar) are staged once in a
//  small static shared buffer. Each thread owns a strided slice of the
//  hidden dim and computes
//
//    out[n, j, h] = c[n, j] * d[n, h] + sum_k a[n, k, j] * b[n, k, h]
//
//  accumulating in fp32 and writing bf16. b and d are streamed from global
//  (bf16 -> fp32 on load). No dynamic-shared-memory workaround.
// ---------------------------------------------------------------------------
__global__ void mhc_post_kernel(
    const float* __restrict__ a,      // [num_tokens, hc, hc] fp32
    const uint16_t* __restrict__ b,   // [num_tokens, hc, hidden] bf16
    const float* __restrict__ c,      // [num_tokens, hc] fp32
    const uint16_t* __restrict__ d,   // [num_tokens, hidden] bf16
    uint16_t* __restrict__ out,       // [num_tokens, hc, hidden] bf16
    int num_tokens, int hc, int hidden) {
  const int n = blockIdx.x;
  const int j = blockIdx.y;
  if (n >= num_tokens || j >= hc) return;
  const int tid = threadIdx.x;

  __shared__ float sCol[64];  // a[n, :, j] (<= 63) + c[n, j] at index [hc]
  const float* an = a + (size_t)n * hc * hc;
  for (int k = tid; k < hc; k += blockDim.x) sCol[k] = an[(size_t)k * hc + j];
  if (tid == 0) sCol[hc] = c[(size_t)n * hc + j];
  __syncthreads();
  const float cj = sCol[hc];
  const float* sA = sCol;

  const uint16_t* bn = b + (size_t)n * hc * hidden;     // [hc, hidden]
  const uint16_t* dn = d + (size_t)n * hidden;          // [hidden]
  uint16_t* on = out + (((size_t)n * hc + j) * hidden); // [hidden]

  for (int h = tid; h < hidden; h += blockDim.x) {
    float acc = cj * bf16_to_f32(dn[h]);
    for (int k = 0; k < hc; ++k)
      acc += sA[k] * bf16_to_f32(bn[(size_t)k * hidden + h]);
    on[h] = f2bf(acc);
  }
}

// ---------------------------------------------------------------------------
//  Launchers (no-throw, mirroring dsa.hip: the device C ABI calls through
//  with no try/catch, so validation is the host C ABI's job via VK_CAPI_TRY
//  and the Python layer's. These launchers only guard the one case that
//  would corrupt the static shared staging buffer.)
// ---------------------------------------------------------------------------
void mhc_pre_gemm_sqrsum(int num_tokens, int hc_mult3, int hc_hidden_size,
                         const void* x, const void* fn,
                         void* out, void* sqrsum) {
  if (num_tokens <= 0 || hc_mult3 <= 0 || hc_hidden_size <= 0) return;
  if (hc_mult3 > 32 || hc_hidden_size > kMaxHcHidden) return;
  dim3 grid(num_tokens);
  dim3 block(32);  // one warp per token; lane == output column
  mhc_pre_gemm_sqrsum_kernel<<<grid, block, 0, 0>>>(
      reinterpret_cast<const uint16_t*>(x), reinterpret_cast<const float*>(fn),
      reinterpret_cast<float*>(out), reinterpret_cast<float*>(sqrsum),
      num_tokens, hc_mult3, hc_hidden_size);
}

void mhc_post(int num_tokens, int hc, int hidden,
              const void* a, const void* b, const void* c, const void* d,
              void* out) {
  if (num_tokens <= 0 || hc <= 0 || hidden <= 0) return;
  if (hc > 63) return;  // sCol[64] backstop
  dim3 grid(num_tokens, hc);
  dim3 block(256);
  mhc_post_kernel<<<grid, block, 0, 0>>>(
      reinterpret_cast<const float*>(a), reinterpret_cast<const uint16_t*>(b),
      reinterpret_cast<const float*>(c), reinterpret_cast<const uint16_t*>(d),
      reinterpret_cast<uint16_t*>(out), num_tokens, hc, hidden);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
