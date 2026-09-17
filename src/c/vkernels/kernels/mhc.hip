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
//  Grid (num_tokens, hc_mult3): one block per (token, output column o),
//  256 threads (issue #79). The token row x[n,:] is staged once per block
//  in static shared memory as bf16 (<= 56 KB); the block then streams ITS
//  OWN fn row fn[o,:] contiguously -- threads read consecutive fp32
//  elements, so the fn walk is fully coalesced (the old one-warp-per-token
//  layout had each lane striding a different fn row: un-coalesced, and at
//  decode n=1 the whole GEMM ran on ONE block = 0.1% of MI300A HBM).
//
//  Static shared budget (no dynamic-shared request, no hipFuncSetAttribute
//  opt-in): sX 28672*2 = 57344 B + sRed 256*4 = 1024 B = 58368 B < 65536 B,
//  so the wider hc_hidden=28672 branch still fits MI300A's 64 KB non-optin
//  cap.
//
//  Per-output accumulation is a strided-lane partial + shared tree reduce,
//  fp32 end to end -- the same associativity class as the CPU oracle's
//  left-to-right sum (max_rel <= 2e-5, fp32 storage, gate unchanged).
//  Block o == 0 additionally computes sqrsum[n] off the staged row; no
//  atomics, no second kernel/pass. No dynamic-shared-memory workaround is
//  needed: there is no dynamic shared memory at all.
// ---------------------------------------------------------------------------
__global__ void mhc_pre_gemm_sqrsum_kernel(
    const uint16_t* __restrict__ x,   // [num_tokens, hc_hidden_size] bf16
    const float* __restrict__ fn,     // [hc_mult3, hc_hidden_size] fp32
    float* __restrict__ out,          // [num_tokens, hc_mult3] fp32
    float* __restrict__ sqrsum,       // [num_tokens] fp32
    int num_tokens, int hc_mult3, int hc_hidden_size) {
  const int n = blockIdx.x;
  const int o = blockIdx.y;              // output column (issue #79 split)
  if (n >= num_tokens || o >= hc_mult3) return;
  const int tid = threadIdx.x;

  __shared__ uint16_t sX[kMaxHcHidden];  // staged token row (bf16)
  __shared__ float sRed[256];            // tree-reduction scratch
  for (int h = tid; h < hc_hidden_size; h += 256)
    sX[h] = x[(size_t)n * hc_hidden_size + h];
  __syncthreads();

  // GEMM: out[n, o] = sum_h x[n, h] * fn[o, h]. Each block owns exactly one
  // output element; the fn row walk is coalesced (256 consecutive fp32 =
  // 1 KB per iteration).
  const float* fn_row = fn + (size_t)o * hc_hidden_size;
  float acc = 0.0f;
  for (int h = tid; h < hc_hidden_size; h += 256)
    acc += bf16_to_f32(sX[h]) * fn_row[h];
  sRed[tid] = acc;
  __syncthreads();
#pragma unroll
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sRed[tid] += sRed[tid + off];
    __syncthreads();
  }
  if (tid == 0) out[(size_t)n * hc_mult3 + o] = sRed[0];

  // Per-token squared-sum, computed by the o == 0 block off the staged row.
  if (o == 0) {
    float sq = 0.0f;
    for (int h = tid; h < hc_hidden_size; h += 256) {
      float xv = bf16_to_f32(sX[h]);
      sq += xv * xv;
    }
    __syncthreads();  // all threads past the out reduction before reusing sRed
    sRed[tid] = sq;
    __syncthreads();
#pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
      if (tid < off) sRed[tid] += sRed[tid + off];
      __syncthreads();
    }
    if (tid == 0) sqrsum[n] = sRed[0];
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
                         void* out, void* sqrsum, void* stream) {
  if (num_tokens <= 0 || hc_mult3 <= 0 || hc_hidden_size <= 0) return;
  if (hc_mult3 > 32 || hc_hidden_size > kMaxHcHidden) return;
  dim3 grid(num_tokens, hc_mult3);  // one block per (token, column) (#79)
  dim3 block(256);                  // coalesced fn-row walk + tree reduce
  mhc_pre_gemm_sqrsum_kernel<<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
      reinterpret_cast<const uint16_t*>(x), reinterpret_cast<const float*>(fn),
      reinterpret_cast<float*>(out), reinterpret_cast<float*>(sqrsum),
      num_tokens, hc_mult3, hc_hidden_size);
}

void mhc_post(int num_tokens, int hc, int hidden,
              const void* a, const void* b, const void* c, const void* d,
              void* out, void* stream) {
  if (num_tokens <= 0 || hc <= 0 || hidden <= 0) return;
  if (hc > 63) return;  // sCol[64] backstop
  dim3 grid(num_tokens, hc);
  dim3 block(256);
  mhc_post_kernel<<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
      reinterpret_cast<const float*>(a), reinterpret_cast<const uint16_t*>(b),
      reinterpret_cast<const float*>(c), reinterpret_cast<const uint16_t*>(d),
      reinterpret_cast<uint16_t*>(out), num_tokens, hc, hidden);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
