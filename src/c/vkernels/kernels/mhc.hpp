// vkernels/kernels/mhc.hpp
//
// MHC — Multi-head hybrid-attention **pre-norm** for GLM-5.3-Flash /
// DeepSeek-V3 (issue #51, part 2). Two of the tilelang MHC kernels from
// ``sglang/kernels/ops/layernorm/mhc.py`` GPU-fault / JIT-abort on gfx942
// (MI300A): the split-k ``mhc_pre_gemm_sqrsum`` stage-0 kernel aborts JIT
// init() with "Requested dynamic shared memory 98304 exceeds device limit
// 65536" because tilelang's HIP codegen never calls ``hipFuncSetAttribute``
// to opt in to the larger dynamic-shared cap (a per-stage double-buffer
// pushes a 256-wide hidden block past MI300A's 64 KB *non-optin* shared
// cap). The sglang-side workaround is to patch the kernel's
// ``hidden_block 256 -> 128``.
//
// vkernels re-implements these two kernels as portable host references + HIP
// kernels that run on gfx942 **without** that workaround: the HIP kernels
// use *static* shared memory sized to fit MI300A's non-optin cap (no
// pipelined double-buffer, no dynamic-shared request, no
// ``hipFuncSetAttribute`` opt-in). This is the "vkernels HIP MHC path" that
// acceptance #3 references.
//
// Two-implementation model (mirrors mla.{hpp,cpp,hip} and dsa.{hpp,cpp,hip}):
//
//   mhc_pre_gemm_sqrsum : the pre-norm GEMM + squared-sum reduction.
//       x [num_tokens, hc_hidden_size] bf16   (hc_hidden_size = hc_mult*hidden)
//       fn [hc_mult3,    hc_hidden_size] fp32 (hc_mult3 = hc_mult*(2+hc_mult))
//       out [num_tokens, hc_mult3]        fp32  =  x @ fn^T
//       sqrsum [num_tokens]               fp32  =  sum_h x[n,h]^2
//       (hc_mult3 <= 32; hc_hidden_size <= 28672 on the HIP path)
//
//   mhc_post : the post-attention combine.
//       a [num_tokens, hc, hc] fp32   = comb_res_mix
//       b [num_tokens, hc, h ] bf16   = residual
//       c [num_tokens, hc]    fp32   = post_layer_mix (squeeze(-1))
//       d [num_tokens, h ]    bf16   = x  (the per-token input)
//       out[n,j,h] = c[n,j] * d[n,h]  +  sum_k a[n,k,j] * b[n,k,h]   (bf16)
//
// The complex fused kernels (``mhc_pre_big_fuse`` — RMSNorm + mix extraction
// + Sinkhorn normalisation, and ``mhc_fused_post_pre_fma``) remain on the
// tilelang path; issue #51 accepts either this vkernels HIP MHC path OR an
// upstream tilelang ``hipFuncSetAttribute`` opt-in for them.
#include <cstddef>
#include <cstdint>

namespace vkernels::kernels {

// CPU reference (oracle) for mhc_pre_gemm_sqrsum, fp32 throughout.
//
//   hc_hidden_size = hc_mult * hidden_size
//   hc_mult3       = hc_mult * (2 + hc_mult)   (must be <= 32)
//
// For each token n:
//   out[n, o] = sum_{h=0}^{hc_hidden_size-1} x[n, h] * fn[o, h]   (o in [0,hc_mult3))
//   sqrsum[n] = sum_{h=0}^{hc_hidden_size-1} x[n, h] * x[n, h]
//
// `out` must not alias any input. `x` is laid out row-major as
// [num_tokens, hc_hidden_size]; `fn` as [hc_mult3, hc_hidden_size].
void mhc_pre_gemm_sqrsum_cpu(int num_tokens, int hc_mult, int hidden_size,
                             const float* x, const float* fn,
                             float* out, float* sqrsum);

// CPU reference (oracle) for mhc_post, fp32 throughout (the bf16 round-trip
// the device kernel does is the only divergence).
//
//   out[n, j, h] = c[n, j] * d[n, h] + sum_{k=0}^{hc-1} a[n, k, j] * b[n, k, h]
//
// `a` [num_tokens, hc, hc], `b` [num_tokens, hc, hidden], `c` [num_tokens, hc],
// `d` [num_tokens, hidden], `out` [num_tokens, hc, hidden]. `out` must not
// alias any input.
void mhc_post_cpu(int num_tokens, int hc, int hidden,
                  const float* a, const float* b, const float* c,
                  const float* d, float* out);

}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// HIP pre-norm GEMM + squared-sum (gfx942). `x` is a bf16 device pointer
// (`const uint16_t*` IEEE-754 bit pattern); `fn`, `out`, `sqrsum` are fp32
// device pointers. Online accumulation in fp32, bf16 storage for `x`.
// `hc_mult3 <= 32`; `hc_hidden_size <= 28672` (static shared memory budget).
// No dynamic-shared-memory workaround: per-block shared is a single static
// `hc_hidden_size`-wide bf16 staging buffer (<= 56 KB), well within MI300A's
// 64 KB non-optin cap.
void mhc_pre_gemm_sqrsum(int num_tokens, int hc_mult3, int hc_hidden_size,
                         const void* x, const void* fn,
                         void* out, void* sqrsum);

// HIP post-attention combine (gfx942). `a` (`comb_res_mix`) and `c`
// (`post_layer_mix`) are fp32 device pointers; `b` (`residual`) and `d`
// (`x`) are bf16 device pointers; `out` is bf16. fp32 accumulation, bf16
// output. One block per (token, output-head j).
void mhc_post(int num_tokens, int hc, int hidden,
              const void* a, const void* b, const void* c, const void* d,
              void* out);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
