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
//   out must not alias q, kv, indices, or lse (the two-pass softmax writes a
//       row into `out` then reads it back; a shared buffer corrupts the run)
//
// `lse` may be null when `return_lse` is false; otherwise it must be large
// enough for `S_q * H` fp32 values. The HIP kernel (dsa.hip) enforces the
// same preconditions at launch, converting a misconfigured caller into a
// named error instead of a silent hang (issue #57).
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

// ---------------------------------------------------------------------------
//  DSA paged-MQA gated top-k logits (issue #51, the kpool>1 indexer path).
//
//  The INDEXER computes, per query token and paged KV tile, a gated logit
//  that the subsequent top-k selects over (GLM-5.3-Flash `index_n_heads=32`,
//  `index_head_dim=128`, block_size=64 hardcoded in deep_gemm). The tilelang
//  kernel JIT-compiles on gfx942 but the launched kernel NEVER returns for
//  num_heads in {32,64}; deep_gemm.fp8_paged_mqa_logits is CUDA-only. This is
//  the portable replacement — the FIRST stage (kpool logits) that feeds the
//  PR #52 sparse-MLA forward.
//
//  Computation (per batch b, KV token t = i*block + j; i in
//  [0, ceildiv(seq_len[b], block)), j in [0, block)):
//
//    out[b, t] = k_scale[ page[b,i], j ]
//              * Sum_h ( max(0, Sum_d Q[b,h,d] * K[ page[b,i], j,d ] )
//                        * gate[b, h] )
//
//  Tokens t >= seq_len[b] are LEFT UNWRITTEN — exactly as the tilelang path
//  leaves them with `clean_logits=False` (the wrapper allocates with
//  `new_empty`, i.e. UNINITIALISED; sglang's `topk_from_pooled_history_logits`
//  masks invalid positions via `group_lengths`/`topk_offsets`/`seq_lens` BEFORE
//  the top-k, so unwritten/zero/garbage are all excluded). The HIP caller and
//  the unit test ZERO the output first (strictly safer than the original's
//  `new_empty`, harmless under the same masking).
//
//  Tensors (kv_group == 1; `page_size = block = 64`, `head_dim = 128`):
//
//    q        : [bs, H, D]         fp8 e4m3fnuz (device) / fp32 (oracle)
//    kv       : [num_blocks, block, D]                 fp8 e4m3fnuz / fp32
//    k_scale  : [num_blocks, block]                    fp32 (per-token
//               scale, packed in the trailing `block*4` bytes of each KV
//               block on the device path; passed separately on the CPU path)
//    gate     : [bs, H]                               fp32 (the indexer's
//               `_get_logits_head_gate` weight, after `.squeeze(2)`)
//    seq_lens : [bs]                                  int32 (the POOLED valid
//               KV count per batch — `pool_seqlens` in sglang)
//    page_table: [bs, max_table_len]                  int32 (the POOLED page
//               table — `pool_block_tables` in sglang; page_table[b,i] indexes
//               the `num_blocks` dim of kv/k_scale)
//    out      : [bs, max_seq_len]                     fp32
//               (max_seq_len = max_table_len * block; ZERO the output first)
//
//  `split_kv` is a PERFORMANCE knob only — the grouped logit is
//  grouping-independent (mirrors the forward's block_I/inner_iter), so any
//  positive split_kv yields the same top-k. The tilelang reference uses
//  `split_kv = max(1, min(max_seq_len // block, NUM_CU // batch_size))`
//  with `NUM_CU = 256` (MI300A); the HIP caller replicates this.
void dsa_topk_logits_cpu(int batch_size, int num_heads, int head_dim, int block,
                         int max_table_len, int num_blocks,
                         const float* q, const float* kv,
                         const float* k_scale, const float* gate,
                         const int32_t* seq_lens, const int32_t* page_table,
                         float* out);

// Whether an indexer shape fits gfx942's 64 KB NON-OPTIN dynamic-LDS cap
// under the fp32-Q kernel (dsa_topk_logits_kernel) -- the test the HIP
// launcher (dsa.hip::dsa_topk_logits) runs to pick the fast path. The
// kernel stages Q (H*D fp32), the per-head gate (H fp32), one K tile
// (B*D fp32) and its per-token scales (B fp32), so the request is
//
//   shmem = (H*D + H + B*D + B) * 4   bytes
//
// The GLM-5.3-Flash config (index_n_heads=32, index_head_dim=128,
// block=64) stages 49,536 B < 65,536 and is VERIFIED (see
// meta/benchmarks/test_dsa_topk_correct.hip). gfx942's opt-in ceiling
// EQUALS the non-optin cap (hipFuncSetAttribute(MaxDynamicSharedMemorySize,
// N>65536) returns hipSuccess but the launch silently never runs; verified
// on a CSCS beverin node -- see the KB note mi300a-dynamic-lds-no-optin), so
// doubling index_n_heads to 64 (66,048 B) CANNOT raise the cap. Instead the
// launcher dispatches dsa_topk_logits_kernel_fp8q (Q staged raw, dequantised
// on the fly -- bit-identical output), selected by dsa_topk_logits_fits_lds_fp8q
// below. Shapes fitting NEITHER variant are refused (a stderr diagnostic +
// no-op, leaving the caller's zeroed output to surface the (wrong, all-zero)
// top-k downstream) rather than silently launching a block the driver drops.
//
// Pure arithmetic on a documented device constant -- safe to call from host
// code, so the sglang backend can query it before launching and the host
// unit tests (tests/kernels/attn/test_dsa.cpp::DsaTopk::FitsLdsCap) assert
// on it.
bool dsa_topk_logits_fits_lds(int num_heads, int head_dim, int block);

// Whether an indexer shape fits gfx942's 64 KB non-optin dynamic-LDS cap
// under the fp8-Q kernel (dsa_topk_logits_kernel_fp8q) -- the launcher's
// FALLBACK for shapes dsa_topk_logits_fits_lds refuses. The fp8-Q kernel
// stages Q as RAW fp8 e4m3fnuz (1 byte/element, dequantised on the fly in
// the dot loop with the SAME fp8e4m3fnuz_to_f32 helper), and the gate (H),
// one K tile (B*D) and its per-token scales (B) as fp32, so the request is
//
//   shmem = (H + B*D + B) * 4 + H*D   bytes
//
// -- the fp32-Q request minus the 3*H*D bytes raw-fp8 Q saves. At the
// GLM-5.3 2x indexer (H=64, D=128, B=64) that is 41,472 B (vs the fp32-Q
// kernel's 66,048 B). The dequanted Q values, the fp32 accumulator and the
// written output are BIT-IDENTICAL to the fp32-Q kernel, so the H=32
// cross-check (max_rel < 1e-3) carries over. Pure arithmetic on the same
// device cap as dsa_topk_logits_fits_lds; host unit tests
// (tests/kernels/attn/test_dsa.cpp::DsaTopk::FitsLdsFp8q) assert on it.
bool dsa_topk_logits_fits_lds_fp8q(int num_heads, int head_dim, int block);

// Whether the indexer shape fits gfx942's 64 KB NON-OPTIN dynamic-LDS cap
// under the MFMA kernel (dsa_topk_logits_kernel_mfma) -- the launcher's
// FAST path, picked FIRST by dsa_topk_logits at every width it fits (the
// smallest-footprint variant; e.g. H=32 which the fp32-Q kernel ALSO fits).
// The kernel stages Q transposed as bf16 sQt[D][H] (loaded ONCE per block,
// reused across every page in the split -- fp8->bf16 is lossless, verified
// by meta/benchmarks/check_fp8_bf16_exact before this kernel existed), one
// bf16 K-tile sK[B][BK=64] (reloaded per K-tile per page), the per-head
// gate sGate[H] and the per-token scales sKscale[B] as fp32, so the request
// is
//
//   shmem = (D*H + B*BK)*2 + (H + B)*4   bytes   (BK = 64, fixed)
//
// -- the SMALLEST of the three variants at every GLM-5.3 indexer width:
// 16,768 B at H=32, 25,088 B at H=64, 41,728 B at H=128 (vs the fp8-Q
// kernel's 37,248 / 41,472 / 49,920 B and the fp32-Q kernel's
// 49,536 / 66,048 / 98,432 B). Shape constraints (the verified
// 16x16x16bf16_1k fragment needs exact multiples): num_heads % 16 == 0
// (kNF = H/16 column fragments), head_dim % 64 == 0 (BK=64 K-tiles) and
// block % 16 == 0 (BM = B, one wavefront per 16-row fragment). H not a
// multiple of 16 (e.g. 246) cannot use this kernel and is refused by the
// launcher exactly as before. Pure arithmetic on the same device cap as
// dsa_topk_logits_fits_lds; host unit tests
// (tests/kernels/attn/test_dsa.cpp::DsaTopk::FitsLdsMfma) assert on it.
bool dsa_topk_logits_fits_lds_mfma(int num_heads, int head_dim, int block);

// The optimal split_kv for the HIP dsa_topk_logits indexer (issue #51) --
// the single source of truth for the formula the hip_capi.hpp ABI
// docstring used to restate (with the wrong NUM_CU). The kernel's grid is
// (batch_size, split_kv); each split block loops over a ceildiv(max_seq_len,
// block)/split_kv slice of the page range, so split_kv is PERFECT ONLY
// (grouping-independent -- any positive value yields the same logits; see
// test_dsa_topk_correct.hip case split=2). The binding constraint at decode
// is occupancy (one wavefront per block on MI300A's 228 CUs), so more splits
// is strictly better until EITHER the page range is exhausted (one page per
// split) OR the grid saturates the CUs:
//
//   split_kv = max(1, min( ceildiv(max_seq_len, block), NUM_CU / batch_size ))
//
// with NUM_CU = 228 (MI300A / gfx942; hipDeviceProp_t::multiProcessorCount,
// verified on a CSCS beverin node). At bs=1, max_seq_len=4096, block=64 this
// is 64 (one page per split, 64 of 228 CUs used) -- measured 21.5 ms -> 0.36 ms
// (59.9x; meta/benchmarks/bench_dsa_topk.hip). At bs=64 the same shape is
// min(64, 228/64) = 3 (192 of 228 CUs).
//
// Pure arithmetic on a documented device constant -- safe to call from host
// code, so the sglang backend can call this before launching instead of
// hand-computing the split. Host unit tests: tests/kernels/attn/test_dsa.cpp
// ::DsaTopk::SplitFor. Raise NUM_CU here alongside the fp8-Q variant's cap
// (dsa_topk_logits_fits_lds_fp8q) + tiled-key/MFMA for devices with a
// different CU count.
int dsa_topk_logits_split_for(int batch_size, int max_seq_len, int block);

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

// DSA paged-MQA gated top-k logits (gfx942), issue #51. Same computation
// as dsa_topk_logits_cpu (see above for the formula and the left-unwritten
// contract), but the caller passes the FP8 e4m3fnuz Q/K raw and the device
// dequants on load (mirrors dsa_sparse_fwd's bf16->fp32). `head_dim` is
// fixed at 128 (no rope tail).
//
// `q_fp8`/`kvcache_u8` are fp8 e4m3fnuz device pointers; `kvcache_u8` is the
// `(num_blocks, block*(head_dim+4))` uint8 view (B*D fp8 keys, then B fp32
// per-token scales). `weight`/`seq_lens`/`page_table` are fp32/int32 device
// pointers; `out` is fp32 device (ZERO the output first).
//
// Dispatches on the staged-bytes cap (gfx942's 64 KB NON-OPTIN dynamic-LDS
// limit; NO hipFuncSetAttribute opt-in past it -- see dsa_topk_logits_fits_lds
// and the KB note mi300a-dynamic-lds-no-optin): shapes that fit the MFMA
// kernel (bf16 sQt[D][H] staged once, bf16 K-tile reloaded -- the SMALLEST
// variant; see dsa_topk_logits_fits_lds_mfma) take the fast path; shapes
// the MFMA kernel refuses but the fp32-Q kernel fits take that (the verified
// GLM-5.3 indexer H=32, D=128, B=64 -> 49,536 B); shapes fitting NEITHER but
// the fp8-Q kernel (Q staged raw, dequantised on the fly -- bit-identical
// output; see dsa_topk_logits_fits_lds_fp8q) take the fallback; shapes
// fitting NONE are refused (a stderr diagnostic + no-op, leaving `out` as
// the caller provided).
//
// `dsa_topk_logits_with_variant` (below) is the explicit-variant hook the
// autotuner/correctness harness use to FORCE a specific kernel (0=auto,
// 1=fp32q, 2=fp8q, 3=mfma) at a given shape, so every path stays exercised
// even when the auto dispatcher would pick a different one.
void dsa_topk_logits(int batch_size, int num_heads, int head_dim, int block,
                     int max_table_len, int max_seq_len, int split_kv,
                     const void* q_fp8, const void* kvcache_u8,
                     const void* weight, const void* seq_lens,
                     const void* page_table, void* out);

// Explicit-variant entry point (offline-autotuner / correctness hook).
// Runs a SPECIFIC kernel at the given shape, bypassing the auto dispatcher
// in dsa_topk_logits above:
//   variant 0 -> auto (mfma-if-fits, else fp32q-if-fits, else fp8q-if-fits,
//                      else refuse); identical to dsa_topk_logits
//   variant 1 -> dsa_topk_logits_kernel        (fp32-Q)
//   variant 2 -> dsa_topk_logits_kernel_fp8q    (fp8-Q)
//   variant 3 -> dsa_topk_logits_kernel_mfma    (bf16 MFMA)
// An explicit variant (1/2/3) that does NOT fit the shape's LDS cap is
// refused with a stderr diagnostic + no-op (the caller zeroed `out`, which
// is what stays) rather than launch a block gfx942's driver would drop --
// mirrors dsa_topk_logits's contract. Same pointers/contract as
// dsa_topk_logits.
void dsa_topk_logits_with_variant(int batch_size, int num_heads, int head_dim,
                                  int block, int max_table_len,
                                  int max_seq_len, int split_kv,
                                  const void* q_fp8, const void* kvcache_u8,
                                  const void* weight, const void* seq_lens,
                                  const void* page_table, void* out,
                                  int variant);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
