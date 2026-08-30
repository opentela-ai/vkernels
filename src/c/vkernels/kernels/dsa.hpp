// vkernels/kernels/dsa.hpp
//
// DeepseekSparseAttn (DSA) sparse-MLA forward — the GLM-5.3-Flash / DeepSeek-V3
// sparse-attention path (issue #51). An *indexer* (out of scope here) has
// already selected, per query token, the top-`topk` most relevant KV tiles
// (`indices`). This kernel scores each query against those selected keys and
// produces the combined attention output.
//
// The query/key are `dim + tail_dim` wide and the value is `d_v = dim - tail_dim`
// wide (the absorbed-MLA form: the nope key latent IS the value, width `d_v`,
// and the decoupled RoPE key contributes the extra `tail_dim`). Both shapes
// MUST work:
//
//   * GLM-5.3-Flash : qk_nope_head_dim = v_head_dim = 256, qk_rope_head_dim = 0
//                     -> dim = 256, tail_dim = 0, d_v = 256. There is NO rope
//                        tail, so the tail QK dot is skipped entirely. This is
//                        the shape tilelang cannot compile (a zero-size
//                        `Q_tail_buf`/`K_tail_shared` GEMM); the HIP kernel
//                        below guards `tail_dim == 0` and skips that dot.
//   * DeepSeek-V3   : qk_nope_head_dim = v_head_dim = 512, qk_rope_head_dim = 64
//                     -> dim = 576, tail_dim = 64, d_v = 512.
//
// Tensors (kv_group == 1; all heads share one key/value head):
//
//   q       : [1, S_q,  H,  dim + tail_dim]  bf16 (device) / fp32 (oracle)
//             q[i,h] = [ q_main (dim)  |  q_tail (tail_dim) ]   (tail may be 0)
//   kv      : [1, S_kv, kv_group, dim + tail_dim]  bf16 / fp32
//             kv[j,0] = [ v (d_v)  |  k_nope_extra (tail_dim)  |  k_rope (tail_dim) ]
//             so v[j]    = kv[j][0 : d_v]              (the value)
//                k_main[j]= kv[j][0 : dim]             (nope key, width dim)
//                k_tail[j]= kv[j][dim : dim + tail_dim] (rope key, width tail_dim)
//   indices : [1, S_q, kv_group, topk]   int32  (padded to a multiple of 64;
//             entries < 0 or >= S_kv are masked kpool tail tokens)
//   out     : [1, S_q, H, d_v]           bf16 / fp32   (combined)
//   lse     : [1, S_q, H]                fp32, nullable  (base-2 log-sum-exp)
//
// Computation (per query token i, head h; the selected keys are gathered via
// `indices[i,0,k]`):
//
//   raw[i,h,k]   = q_main[i,h] . k_main[idx] + (tail_dim>0) q_tail[i,h] . k_tail[idx]
//   score[i,h,k] = sm_scale * raw[i,h,k]            (sm_scale includes log2(e))
//   w[i,h,k]     = 2^(score[i,h,k] - lse[i,h])       (masked keys -> weight 0)
//   out[i,h]     = Σ_k w[i,h,k] * v[idx]              (d_v wide)
//   lse[i,h]     = log2( Σ_k 2^score[i,h,k] )          (base-2, when return_lse)
//
// `sm_scale = (1 / sqrt(dim + tail_dim)) * log2(e)`, so the weights are the
// standard natural-exp softmax `exp(raw/sqrt(dim+tail_dim)) / Σ` expressed in
// base-2 (FlashAttention's log2 trick). Masked keys contribute weight 0; if
// every selected key is masked the output row is 0 and the LSE is -inf.
//
// Two-implementation model (mirrors mla.{hpp,cpp,hip} from issue #21):
//   dsa.cpp  -- CPU reference (oracle), always compiled, in vkernels::kernels.
//               Two-pass numerically-stable base-2 softmax, fp32 throughout.
//   dsa.hip  -- HIP kernel (gfx942), compiled with VKERNELS_HAS_HIP. Online
//               softmax, fp32 accumulation, bf16 storage. Each block owns one
//               (query, head) pair and streams its `topk` selected keys,
//               accumulating the combined output directly (the per-group
//               partial_o / partial_lse from the tilelang kernel are an
//               internal parallelism detail; the combined result is
//               grouping-independent). Matches the oracle within tolerance.
#include <cstddef>
#include <cstdint>

namespace vkernels::kernels {

// CPU reference (oracle). Computes the sparse-MLA forward in fp32 with a
// numerically-stable two-pass base-2 softmax over the `topk` selected keys.
//
//   d_v = dim - tail_dim   (must be > 0; tail_dim may be 0)
//   topk % (block_I * inner_iter) must be 0 (the kernel's group tiling; the
//       oracle result is grouping-independent so this is validated, not used)
//   kv_group must be 1 (a single head_kv group is shared by all H heads)
//
// `lse` may be null when `return_lse` is false; otherwise it must be large
// enough for `S_q * H` fp32 values. `out` must not alias any input.
void dsa_sparse_fwd_cpu(int S_q, int S_kv, int H, int dim, int tail_dim,
                        int topk, int kv_group, int block_I, int inner_iter,
                        float sm_scale, bool return_lse,
                        const float* q, const float* kv,
                        const int32_t* indices, float* out, float* lse);

// Per-shape (decode vs prefill) launch tile selector. Writes the tile the
// HIP kernel should use:
//   decode  (S_q <= 8) : one query row per block, one wavefront (64 threads)
//   prefill (S_q >  8) : BQ query rows per block, 256 threads (4 wavefronts)
// `block_I` is the per-pass key tile (default 64); `inner_iter` repeats it so
// that `block_I * inner_iter` divides `topk` (default 1; raise it to trade
// register pressure for fewer passes on large topk).
void dsa_config_for(int S_q, int H, int dim, int topk, int* bq, int* threads,
                    int* block_I, int* inner_iter);

}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// HIP sparse-MLA forward (gfx942). Online softmax in fp32, bf16 storage,
// bf16-tolerant against the CPU oracle. `q`/`kv` are bf16 device pointers
// (`const __hip_bfloat16*`); `indices` is int32 device; `out` is bf16 device;
// `lse` is fp32 device (nullable when `return_lse` is false). Device pointers
// must reside in device or host-pinned memory. `tail_dim == 0` is handled by
// skipping the rope-tail dot at runtime (no zero-size GEMM).
void dsa_sparse_fwd(int S_q, int S_kv, int H, int dim, int tail_dim,
                    int topk, int kv_group, int block_I, int inner_iter,
                    float sm_scale, bool return_lse,
                    const void* q, const void* kv, const void* indices,
                    void* out, void* lse);

// Explicit-tile entry point (offline autotuner hook). Dispatches the concrete
// (bq, block_I, inner_iter) tile; threads is derived as max(bq,1)*64 capped
// at 256. `bn_kv` is reserved for a future tiled-key prefetch.
void dsa_sparse_fwd_with_tile(int S_q, int S_kv, int H, int dim, int tail_dim,
                              int topk, int kv_group, float sm_scale,
                              bool return_lse,
                              const void* q, const void* kv,
                              const void* indices, void* out, void* lse,
                              int bq, int block_I, int inner_iter, int bn_kv);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
