// vkernels/kernels/dsa_kpool.hpp
//
// DeepseekSparseAttn (DSA) kpool-cache compress/write kernels — the
// GLM-5.3-Flash / DeepSeek-V3 persistent kpool index cache path (issue #60).
// When ``index_kpool > 1``, a ``kpool-1`` live tail of recent tokens is
// appended to the fixed ``index_topk`` columns and held in a persistent
// cache. The cache COMPRESS/WRITE path is a *separate* set of kernels from
// the attention FORWARD that issue #51 / PR #52 targeted: the prefill
// (short-context) kernel ``kpool_assemble_softmax_rotate_write_cache`` and
// the decode kernel ``kpool_decode_update_and_maybe_write_cache`` in sglang's
// ``kpool_fp8_index.py``.
//
// The problem. Both Triton kernels declare ``tl.float8e4nv`` in their
// signatures (the cache is ``torch.float8_e4m3fn``). SM80 (A100) Triton
// cannot declare ``*fp8e4nv`` -- there is no native FP8 on A100 -- so the
// kernels JIT-fail on the first forward (job 82822), before MoE is ever
// reached. There is no config-only escape: ``_compress_write`` runs on every
// forward.
//
// The fix, mirroring PR #52's ``dsa.{hpp,cpp,hip}`` two-implementation model
// and the ``gemm_bf16`` bf16-storage convention: a native CPU oracle (stable
// fp32) + a gfx942 HIP kernel (bf16 storage + fp32 accumulation, NO
// ``fp8e4nv`` anywhere). bf16 has 8 exponent bits (range ~3.4e38) and 7
// mantissa bits -- ample dynamic range for the Hadamard-transformed
// softmax-weighted mean WITHOUT a per-vector scale, so the fp8 K region +
// fp32 scale region of the original ``[num_pages, slots_per_page,
// (head_dim+4)]`` uint8 cache collapses to a flat ``[num_pages,
// slots_per_page, head_dim]`` bf16 cache. The per-vector scale is simply
// dropped (the whole point of leaving fp8).
//
// Computation (GLM-5.3: head_dim == index_head_dim == 128, the 128-point
// Hadamard). Both kernels reduce a pool of ``pool_size`` slots to ONE
// cache vector per pool via an online (FlashAttention-style) softmax, then
// apply the normalized 128-point Hadamard transform to that weighted mean:
//
//   mean   = Sum_s softmax(score_s + ape[s]) . K_s            (fp32, online)
//   cache  = Hadamard128(mean)                                 (bf16 storage)
//
//   Prefill  (kpool_assemble_softmax_rotate_write_cache):
//     the ``pool_size`` slots for pool ``r`` are the first ``n_from_tail[r]``
//     tail slots (ring-buffered at ``tail_logical_base[r]`` in
//     ``tail_k/tail_score[req_pool_idx[r]]``) followed by the remaining
//     ``pool_size - n_from_tail[r]`` chunk slots (at ``chunk_src_start[r]
//     + (slot - n_from_tail[r])`` in ``chunk_k/chunk_score``). The result is
//     written to ``out[ loc[r]//slots_per_page, loc[r]%slots_per_page ]``;
//     rows with ``write_mask[r] == false`` (when ``write_mask`` is non-null)
//     are skipped, exactly as the Triton kernel returns early.
//
//   Decode (kpool_decode_update_and_maybe_write_cache):
//     when the pool is COMPLETE (``pos % pool_size == pool_size - 1`` and the
//     row is ``pos_valid``), the SAME weighted-mean + Hadamard is computed
//     over the pool's ``pool_size`` slots -- EXCEPT the current token's own
//     key/score (``key[r]``, ``slot_score[r]``) is substituted in for slot
//     ``pos % pool_size`` (a two-pass softmax: max, then weighted sum, so the
//     current slot is read only once). The result is written to
//     ``out[ packed_page, pool_id % slots_per_page ]`` where ``pool_id =
//     pos // pool_size`` and ``packed_page = block_tables[r,
//     clamp(pool_id//slots_per_page*pool_size, 0, block_table_cols-1)]``.
//     UNCONDITIONALLY (every forward, even when the pool is not complete),
//     the current key/score is written into the live tail at
//     ``tail_k/tail_score[req, pos % tail_size]`` (the ``kpool-1`` live-tail
//     maintenance) -- masked by ``pos_valid`` so an invalid row is a no-op.
//
// Two-implementation model (mirrors mla.{hpp,cpp,hip}, dsa.{hpp,cpp,hip}):
//   dsa_kpool.cpp  -- CPU reference (oracle), always compiled, stable fp32.
//                    Cross-checked by tests/kernels/attn/test_dsa_kpool.cpp.
//   dsa_kpool.hip  -- gfx942 HIP kernel, compiled with VKERNELS_HAS_HIP.
//                    bf16 storage + fp32 accumulation, NO fp8e4nv.
//                    Cross-checked by meta/benchmarks/test_dsa_kpool_correct.hip.
//
// Storage. Following the established gfx942 bf16 convention (gemm_bf16.hip,
// moe_device.hip, dsa.hip), bf16 is carried as raw uint16 IEEE-754 bit
// patterns and converted to/from fp32 with bf16_to_f32 (zero-extend of the
// top 16 bits) and f32bits_to_bf16 (round-to-nearest-even). The CPU oracle
// keeps full fp32 (no bf16 rounding) so it is a STRICT upper bound on the
// device kernel's precision; the device is bf16-tolerant against it exactly
// as dsa_sparse_fwd / gemm_bf16 are.
//
// Per-vector scale: DROPPED (bf16's range is ample; the original fp8 cache
// needed a per-vector scale to fit fp8's +/-448 range). The native cache is
// therefore ``[num_pages, slots_per_page, head_dim]`` bf16 (no scale region),
// half the storage-dtype byte count of the original fp8 K + fp32 scale layout
// at head_dim=128 (2 bytes/elem vs (1+4)/128 ~= 0.04 -- the original was
// dense uint8 with a 4-byte scale; the native is dense bf16 with none).

#include <cstdint>

namespace vkernels::kernels {

// Whether ``token_topk / pool_size`` (``group_topk``) is one of the radix
// specialisations the kpool top-k transform supports. Mirrors sglang's
// ``kpool_topk_group_topk_supported``; never throws.
bool dsa_kpool_group_topk_supported(int32_t group_topk);

// ``ceildiv(num_draft_tokens, pool_size)`` -- the number of CLOSED pools the
// indexer writes per forward (``kpool_max_closed_pools`` in
// kpool_fp8_index.py). Pure host arithmetic; never throws.
int32_t dsa_kpool_max_closed_pools(int32_t num_draft_tokens, int32_t pool_size);

// DSA kpool-cache compress/write -- PREFILL / short-context path
// (kpool_assemble_softmax_rotate_write_cache in sglang's kpool_fp8_index.py).
// Stable fp32 oracle; see the file header for the computation. ``write_mask``
// is null (no masking) or ``[n_pools]`` int32 with a per-row gate (a non-zero
// value writes, a zero skips -- mirrors the Triton ``write_mask`` bool). The
// output cache is ``[num_pages, slots_per_page, head_dim]`` fp32 (the caller
// zeroes it first; only written rows are touched).
//
// Tensors (all fp32; contiguous, dims passed explicitly):
//   chunk_k        : [num_chunks, head_dim]
//   chunk_score    : [num_chunks, head_dim]
//   tail_k         : [n_reqs, tail_size, head_dim]
//   tail_score     : [n_reqs, tail_size, head_dim]
//   ape            : [pool_size, head_dim]   (the indexer's per-pool-slope bias)
//   req_pool_idx   : [n_pools] int32         (in [0, n_reqs))
//   n_from_tail    : [n_pools] int32         (in [0, pool_size])
//   chunk_src_start: [n_pools] int32
//   tail_logical_base: [n_pools] int32
//   loc            : [n_pools] int32         (in [0, num_pages*slots_per_page))
//   write_mask     : [n_pools] int32 or null (non-zero writes)
//   out            : [num_pages, slots_per_page, head_dim] fp32 (ZERO first)
//
// Throws std::invalid_argument on null pointers (when n_pools>0), bad dims,
// or an out-of-range ``req_pool_idx``/``chunk_src_start``/``loc``. Empty
// (n_pools==0) is a no-op.
void dsa_kpool_assemble_cpu(int n_pools, int pool_size, int head_dim,
                            int tail_size, int slots_per_page, int num_pages,
                            int num_chunks, int n_reqs,
                            const float* chunk_k, const float* chunk_score,
                            const float* tail_k, const float* tail_score,
                            const float* ape,
                            const int32_t* req_pool_idx,
                            const int32_t* n_from_tail,
                            const int32_t* chunk_src_start,
                            const int32_t* tail_logical_base,
                            const int32_t* loc, const int32_t* write_mask,
                            float* out);

// DSA kpool-cache compress/write + live-tail update -- DECODE path
// (kpool_decode_update_and_maybe_write_cache in sglang's kpool_fp8_index.py).
// Stable fp32 oracle; see the file header for the computation. The compressed
// K is written ONLY for rows whose pool is complete (``pos % pool_size ==
// pool_size-1``) AND ``pos_valid`` (req in range, cache_loc != 0, pos in
// [0, seq_len)). The current key/score is written into the live tail for
// EVERY pos_valid row (masked to a no-op otherwise). ``tail_k``/``tail_score``
// are IN-PLACE: the read uses the pre-update values, the write overwrites
// ``[req, pos % tail_size]``.
//
// Tensors (fp32 unless noted; contiguous, dims passed explicitly):
//   key            : [batch, head_dim] fp32 (the current token's K)
//   slot_score     : [batch, head_dim] fp32 (the current token's score)
//   tail_k         : [n_reqs, tail_size, head_dim] fp32 (IN-PLACE updated)
//   tail_score     : [n_reqs, tail_size, head_dim] fp32 (IN-PLACE updated)
//   ape            : [pool_size, head_dim] fp32
//   block_tables   : [batch, block_table_cols] int32
//   req_pool_indices: [batch] int32           (<0 or >= n_reqs -> row invalid)
//   positions      : [batch] int32
//   seq_lens       : [batch] int32
//   out_cache_loc  : [batch] int32            (== 0 -> no compressed-K write)
//   out            : [num_pages, slots_per_page, head_dim] fp32 (ZERO first)
//
// Throws std::invalid_argument on null pointers (when batch>0), bad dims, or
// an out-of-range ``block_tables`` entry / ``out`` write index. Empty
// (batch==0) is a no-op.
void dsa_kpool_decode_update_cpu(int batch, int pool_size, int head_dim,
                                 int tail_size, int slots_per_page,
                                 int block_table_cols, int n_reqs,
                                 int num_pages,
                                 const float* key, const float* slot_score,
                                 float* tail_k, float* tail_score,
                                 const float* ape,
                                 const int32_t* block_tables,
                                 const int32_t* req_pool_indices,
                                 const int32_t* positions,
                                 const int32_t* seq_lens,
                                 const int32_t* out_cache_loc, float* out);

}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// DSA kpool-cache compress/write -- PREFILL path (gfx942). bf16 storage +
// fp32 accumulation, NO fp8e4nv. Same computation as dsa_kpool_assemble_cpu
// (see above for the formula and the write-mask contract); the per-pool-row
// weighted mean is rounded to bf16 on store via f2bf (moe_device.hip), the
// ONLY lossy step (no per-vector scale -- bf16's range is ample). The output
// cache is ``[num_pages, slots_per_page, head_dim]`` bf16 device. Pointers
// are device memory; ``out`` must be a bf16 device pointer (zeroed first).
void dsa_kpool_assemble(int n_pools, int pool_size, int head_dim,
                        int tail_size, int slots_per_page, int num_pages,
                        int num_chunks, int n_reqs,
                        const void* chunk_k, const void* chunk_score,
                        const void* tail_k, const void* tail_score,
                        const void* ape,
                        const void* req_pool_idx, const void* n_from_tail,
                        const void* chunk_src_start,
                        const void* tail_logical_base,
                        const void* loc, const void* write_mask, void* out);

// DSA kpool-cache compress/write + live-tail update -- DECODE path (gfx942).
// bf16 storage + fp32 accumulation, NO fp8e4nv. Same computation as
// dsa_kpool_decode_update_cpu (see above for the complete-pool gate, the
// current-token substitution, and the in-place live-tail write). ``tail_k``
// and ``tail_score`` are IN-PLACE bf16 device buffers; ``key``/``slot_score``
// are bf16 device; ``out`` is a bf16 device pointer (zeroed first). ``ape``,
// ``block_tables``, ``req_pool_indices``, ``positions``, ``seq_lens``,
// ``out_cache_loc`` are fp32/int32 device (ape is fp32 -- the gate weight
// stays fp32 end-to-end, mirroring dsa_topk_logits).
void dsa_kpool_decode_update(int batch, int pool_size, int head_dim,
                             int tail_size, int slots_per_page,
                             int block_table_cols, int n_reqs, int num_pages,
                             const void* key, const void* slot_score,
                             void* tail_k, void* tail_score,
                             const void* ape,
                             const void* block_tables,
                             const void* req_pool_indices,
                             const void* positions, const void* seq_lens,
                             const void* out_cache_loc, void* out);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
