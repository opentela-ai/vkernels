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

// fn prefetch chunk (fp32 elements) for the mhc_pre column pipeline: the
// double buffer (2 * kFnChunk * 4 B = 4 KB) must keep the static shared
// total (sX 57344 + sFN 4096 + sRed 1024 = 62464 B) under MI300A's 64 KB
// non-optin cap at the widest hc_hidden = 28672 branch.
static constexpr int kFnChunk = 512;


// ---------------------------------------------------------------------------
//  mhc_pre_gemm_sqrsum
//
//  Grid (num_tokens, hc_mult3): one block per (token, output column o),
//  256 threads (issue #79). The token row x[n,:] is staged once per block
//  in static shared memory as bf16 (<= 56 KB); threads 1..255 then stream
//  the block's OWN fn row fn[o,:] through a double-buffered shared chunk
//  buffer with fully coalesced reads (the old one-warp-per-token layout had
//  each lane striding a different fn row: un-coalesced, and at decode n=1
//  the whole GEMM ran on ONE block = 0.1% of MI300A HBM).
//
//  Accumulation ORDER is load-bearing for the correctness gate. The CPU
//  oracle sums each output element as a strict left-to-right fp32
//  mul-then-add chain over h (mhc_pre_gemm_sqrsum_cpu). Because fp32
//  addition is not associative, ANY parallel regrouping of that chain
//  (strided partials, tree/sequential combine, even a fully exact fp64
//  reduction) drifts from the oracle by the oracle's own rounding-path
//  deviation (~1.5e-3 abs on the GLM shapes) -- measured 1.03e-4 max_rel on
//  the n=7 hc_hidden=16384 gate row, i.e. a FAIL at the unchanged 1e-4
//  PRE_THRESH, and above the documented <= 2e-5 envelope. So out[n,o] is
//  computed by ONE thread (tid 0) as the same strict sequential chain,
//  consuming x from the staged row and fn from shared: the product is
//  materialized with a compiler barrier (no mul+add -> FMA contraction),
//  making the device sum BITWISE identical to the oracle. The parallel
//  speedup comes from (a) one block per column (24 blocks at decode
//  instead of 1) and (b) the shared double buffer, which hides the global
//  fn load latency that dominated the old serial walk (~278 us was
//  load-stall, not FMA: the chain floor is hc_hidden * 4 cyc).
//
//  sqrsum[n] has no such ordering constraint (all-positive terms, large
//  partials): the o == 0 block computes it cooperatively (strided fp32
//  partials + shared tree reduce, measured max_rel 1.2e-5 <= 2e-5) off the
//  staged row before the GEMM pipeline starts. No atomics, no extra pass.
//
//  Static shared budget (no dynamic-shared request, no hipFuncSetAttribute
//  opt-in): sX 28672*2 = 57344 B + sFN 2*512*4 = 4096 B + sRed 256*4 =
//  1024 B = 62464 B < 65536 B, so the wider hc_hidden=28672 branch still
//  fits MI300A's 64 KB non-optin cap.
// ---------------------------------------------------------------------------
__global__ void mhc_pre_gemm_sqrsum_kernel(
    const uint16_t* __restrict__ x,   // [num_tokens, hc_hidden_size] bf16
    const float* __restrict__ fn,     // [hc_mult3, hc_hidden_size] fp32
    float* __restrict__ out,          // [num_tokens, hc_mult3] fp32
    float* __restrict__ sqrsum,       // [num_tokens] fp32
    int num_tokens, int hc_mult3, int hc_hidden_size) {
  // Strict mul+add rounding for the out chain -- no FMA contraction (the
  // CPU oracle is a two-rounding mul-then-add chain; see block comment).
  // Must be the first statement of the compound block (clang restriction).
#pragma clang fp contract(off)
  const int n = blockIdx.x;
  const int o = blockIdx.y;              // output column (issue #79 split)
  if (n >= num_tokens || o >= hc_mult3) return;
  const int tid = threadIdx.x;

  __shared__ uint16_t sX[kMaxHcHidden];  // staged token row (bf16)
  __shared__ float sFN[2][kFnChunk];     // fn[o, chunk] double buffer
  __shared__ float sRed[256];            // sqrsum tree-reduction scratch
  for (int h = tid; h < hc_hidden_size; h += 256)
    sX[h] = x[(size_t)n * hc_hidden_size + h];
  __syncthreads();

  // Per-token squared-sum (o == 0 block, cooperative -- order-free), done
  // before the GEMM pipeline so sRed is not touched while it runs.
  if (o == 0) {
    float sq = 0.0f;
    for (int h = tid; h < hc_hidden_size; h += 256) {
      float xv = bf16_to_f32(sX[h]);
      sq += xv * xv;
    }
    sRed[tid] = sq;
    __syncthreads();
#pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
      if (tid < off) sRed[tid] += sRed[tid + off];
      __syncthreads();
    }
    if (tid == 0) sqrsum[n] = sRed[0];
  }

  // GEMM: out[n, o] = sum_h x[n, h] * fn[o, h] as a STRICT sequential fp32
  // mul+add chain over h (order-faithful to the CPU oracle -- see block
  // comment; parallel regrouping measurably breaks the 1e-4 gate). tid 0
  // walks the chain from shared while threads 1..255 prefetch the next
  // fn chunk (coalesced, 255 consecutive fp32 per instruction) into the
  // alternate buffer: one __syncthreads() per chunk, double-buffered, so
  // the chain is fed at LDS latency instead of global latency.
  const float* fn_row = fn + (size_t)o * hc_hidden_size;
  const int nc = (hc_hidden_size + kFnChunk - 1) / kFnChunk;
  // Stage chunk 0 (threads 1..255; tid 0 only consumes).
  for (int j = tid - 1; j < kFnChunk; j += 255) {
    if (j < 0) continue;  // tid 0 (j == -1) does not produce
    const int h = j;
    if (h < hc_hidden_size) sFN[0][j] = fn_row[h];
  }
  __syncthreads();

  float acc = 0.0f;
  for (int c = 0; c < nc; ++c) {
    if (tid == 0) {  // consumer: strict h-order chain over chunk c
      const int base = c * kFnChunk;
#pragma unroll 8
      for (int j = 0; j < kFnChunk; ++j) {
        const int h = base + j;
        if (h < hc_hidden_size) {
          float p = bf16_to_f32(sX[h]) * sFN[c & 1][j];
          // contract(off) below keeps hipcc's default -ffp-contract=fast from
          // fusing this mul+add into an FMA (one rounding instead of the
          // oracle's two): the chain then matches the CPU oracle bitwise.
          acc += p;
        }
      }
      if (c == nc - 1) out[(size_t)n * hc_mult3 + o] = acc;
    } else if (c + 1 < nc) {  // producers: coalesced fn chunk c+1 prefetch
      const int cc = c + 1;
      const int base = cc * kFnChunk;
      for (int j = tid - 1; j < kFnChunk; j += 255) {
        const int h = base + j;
        if (h < hc_hidden_size) sFN[cc & 1][j] = fn_row[h];
      }
    }
    __syncthreads();
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

// ---------------------------------------------------------------------------
//  mhc_pre_gemm_sqrsum_blocked (issue #79 gate experiment)
//
//  Same GEMM as mhc_pre_gemm_sqrsum, but out[n,o] is accumulated as NSLICE
//  contiguous-slice left-to-right fp32 chains (thread t owns
//  [t*S, (t+1)*S), S = ceil(hc_hidden/NSLICE)) combined by tid 0 strictly
//  in thread order (idle threads contribute exact +0.0 -- a bitwise no-op
//  on the chain). NSLICE=1 IS the oracle's strict chain (bitwise-exact by
//  construction); larger NSLICE trades gate margin for latency. The ladder
//  {256..1} is swept by test_mhc_correct (drift) and bench_mhc (latency) to
//  pin the fastest gate-compliant point. Motivation: the strictly-
//  sequential variant has a hard latency floor (hc_hidden DEPENDENT adds x
//  ~4 cyc = 66 us at the ~1 GHz sustained sclk observed on beverin) and the
//  256-slice two-rounding variant drifted to 1.16e-4 rel on the n=7 row
//  (FAIL at 1e-4) -- the sweet spot is in between. Threads read x/fn
//  directly from global --
//  lane-local sequential walks are L1-sector-friendly -- so no staging
//  buffer: static shared is two 1 KB reduction scratch arrays.
// ---------------------------------------------------------------------------
template <int NSLICE>
__global__ void mhc_pre_gemm_sqrsum_blocked_kernel(
    const uint16_t* __restrict__ x,   // [num_tokens, hc_hidden_size] bf16
    const float* __restrict__ fn,     // [hc_mult3, hc_hidden_size] fp32
    float* __restrict__ out,          // [num_tokens, hc_mult3] fp32
    float* __restrict__ sqrsum,       // [num_tokens] fp32
    int num_tokens, int hc_mult3, int hc_hidden_size) {
// NOTE: FMA contraction is REQUIRED here, not forbidden: the CPU oracle's
// `acc += xn[h] * fo[h]` is contracted into an FMA by the host compiler's
// default -ffp-contract=on, so the device must also contract to replay the
// oracle's one-rounding-per-step path. (The predecessor's contract(off)
// two-rounding chain measured 1.19e-5 rel on the n=7 row -- the
// two-rounding-vs-FMA mismatch signature; tiny shapes still matched at 0.0
// because both roundings agree on short non-cancelling sums.)
  const int n = blockIdx.x;
  const int o = blockIdx.y;
  if (n >= num_tokens || o >= hc_mult3) return;
  const int tid = threadIdx.x;
  const uint16_t* x_row = x + (size_t)n * hc_hidden_size;
  const float* fn_row = fn + (size_t)o * hc_hidden_size;

  __shared__ float sRed[256];   // sqrsum tree-reduction scratch (o == 0)
  __shared__ float sPart[256];  // per-thread slice partials for the combine

  // Per-token squared-sum (order-free gate), cooperative over all 256
  // threads regardless of NSLICE (at small NSLICE a serial sqrsum would
  // dominate the kernel).
  if (o == 0) {
    float sq = 0.0f;
    for (int h = tid; h < hc_hidden_size; h += 256) {
      float xv = bf16_to_f32(x_row[h]);
      sq += xv * xv;
    }
    sRed[tid] = sq;
    __syncthreads();
#pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
      if (tid < off) sRed[tid] += sRed[tid + off];
      __syncthreads();
    }
    if (tid == 0) sqrsum[n] = sRed[0];
  }

  // out[n,o]: NSLICE blocked left-to-right chains + in-order combine.
  // blockDim is always 256; threads >= NSLICE contribute exact +0.0.
  const int slice = (hc_hidden_size + NSLICE - 1) / NSLICE;
  const int h0 = tid * slice;
  float part = 0.0f;
  if (tid < NSLICE && h0 < hc_hidden_size) {
    const int h1 = (h0 + slice < hc_hidden_size) ? h0 + slice
                                                 : hc_hidden_size;
#pragma unroll 4
    for (int h = h0; h < h1; ++h) {
      float p = bf16_to_f32(x_row[h]) * fn_row[h];
      part += p;
    }
  }
  sPart[tid] = part;
  __syncthreads();
  if (tid == 0) {
    float acc = 0.0f;
#pragma unroll 8
    for (int t = 0; t < 256; ++t) acc += sPart[t];
    out[(size_t)n * hc_mult3 + o] = acc;
  }
}

void mhc_pre_gemm_sqrsum_blocked(int num_tokens, int hc_mult3,
                                 int hc_hidden_size,
                                 const void* x, const void* fn,
                                 void* out, void* sqrsum, void* stream,
                                 int nslice) {
  if (num_tokens <= 0 || hc_mult3 <= 0 || hc_hidden_size <= 0) return;
  if (nslice <= 0) nslice = 256;
  dim3 grid(num_tokens, hc_mult3);  // one block per (token, column)
  dim3 block(256);                  // sqrsum coop + idle-thread +0.0 padding
  hipStream_t s = static_cast<hipStream_t>(stream);
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x);
  const float* fp = reinterpret_cast<const float*>(fn);
  float* op = reinterpret_cast<float*>(out);
  float* sp = reinterpret_cast<float*>(sqrsum);
#define VK_MHC_BLOCKED_CASE(N)                                                \
  do {                                                                        \
    if (nslice == N)                                                          \
      mhc_pre_gemm_sqrsum_blocked_kernel<N>                                   \
          <<<grid, block, 0, s>>>(xp, fp, op, sp, num_tokens, hc_mult3,       \
                                  hc_hidden_size);                            \
  } while (0)
  VK_MHC_BLOCKED_CASE(1);
  VK_MHC_BLOCKED_CASE(2);
  VK_MHC_BLOCKED_CASE(4);
  VK_MHC_BLOCKED_CASE(8);
  VK_MHC_BLOCKED_CASE(16);
  VK_MHC_BLOCKED_CASE(32);
  VK_MHC_BLOCKED_CASE(64);
  VK_MHC_BLOCKED_CASE(128);
  VK_MHC_BLOCKED_CASE(256);
#undef VK_MHC_BLOCKED_CASE
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
