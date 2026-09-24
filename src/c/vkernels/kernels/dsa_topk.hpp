// vkernels/kernels/dsa_topk.hpp
//
// Pool-level top-k transform used between the DSA indexer logits and sparse
// attention forward.  The transform selects K pool groups, expands every
// selected group to `pool_size` token indices, and optionally maps the raw
// token indices through a page table or a per-row ragged offset.
#pragma once

#include <cstdint>

namespace vkernels::kernels {

// The group-top-k specialisations validated by sglang's kpool transform.
bool dsa_topk_transform_group_topk_supported(int32_t group_topk);

// Host correctness oracle. `dst_token_indices` is contiguous [batch_size,
// out_cols]. Scores are row-major with `score_stride` elements between rows;
// `row_starts`, when present, offsets the valid score range within each row.
//
// `page_table_row_index`, when present, selects the page-table row for each
// score row (otherwise the score-row index is used); page_table has
// `batch_size` rows, so every explicit row index must be in [0, batch_size).
// `page_table` and
// `topk_indices_offset` are mutually exclusive. When `seq_lens` is present,
// out_cols must be token_topk + pool_size - 1 and the final partial pool is
// appended after the selected history tokens.
void dsa_topk_transform_cpu(int32_t batch_size,
                            const float* score,
                            const int32_t* lengths,
                            int32_t* dst_token_indices,
                            int64_t score_stride,
                            int32_t pool_size,
                            int32_t token_topk,
                            int32_t out_cols,
                            const int32_t* page_table,
                            int64_t page_table_stride,
                            const int32_t* page_table_row_index,
                            const int32_t* topk_indices_offset,
                            const int32_t* row_starts,
                            const int32_t* seq_lens);

} // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// gfx942 HIP implementation. One 1024-thread workgroup owns one score row and
// runs the same two-stage radix selection as sglang's reference kernel. All
// pointers are device pointers and the launch uses the default stream.
void dsa_topk_transform(int32_t batch_size,
                        const float* score,
                        const int32_t* lengths,
                        int32_t* dst_token_indices,
                        int64_t score_stride,
                        int32_t pool_size,
                        int32_t token_topk,
                        int32_t out_cols,
                        const int32_t* page_table,
                        int64_t page_table_stride,
                        const int32_t* page_table_row_index,
                        const int32_t* topk_indices_offset,
                        const int32_t* row_starts,
                        const int32_t* seq_lens);

// FUSED top-k chain (fusion lane dsa-topk-tailfold): the scalar
// dsa_topk_logits GEMV (the AUTO dispatcher's scalar arms -- fp32-Q, else
// fp8-Q; the MFMA fast paths live in dsa.hip) AND dsa_topk_transform on the
// complete logits rows, in ONE launch when VK_DSA_TOPK_FUSED=1: a
// cooperative grid of (batch, split_kv) 1024-thread blocks, phase 1 =
// logits (lane = KV token), grid sync, phase 2 (split block) = the verbatim
// transform row. With the gate OFF (the default) this entry IS the proven
// two-launch chain, so callers may wire it unconditionally.
//
// Fallback contract: any condition the fused path cannot honour (gate off,
// invalid transform parameters, unsupported group_topk, no scalar logits
// variant under the device cap, no cooperative-launch support, occupancy 0,
// grid beyond the co-residency capacity -- split_kv is CLAMPED to it, it is
// perf-only/grouping-independent) silently runs the proven chain instead.
// `logits` doubles as the transform's `score` buffer: pass score_stride ==
// max_seq_len and zero/canary-fill up front -- cells with t >= seq_len[b]
// are never written and the transform only reads [row_start, row_start +
// lengths[row]) per row, exactly the caller contract of the two-launch
// chain. All pointers are device pointers; default stream.
//
// `q_variant`: 0 = auto (fp32-Q, else fp8-Q), 1 = fp32-Q, 2 = fp8-Q; both
// scalar variants produce BIT-IDENTICAL logits (gfx942 caps: H=32 -> fp32-Q,
// H=64 -> fp8-Q; GB10: both fp32-Q). Explicit variant lets a caller pin the
// proven one for bit-exact A/B against the fused path.
void dsa_topk_logits_transform_fused(int batch_size, int num_heads, int head_dim,
                                     int block, int max_table_len, int max_seq_len,
                                     int split_kv, const void* q_fp8,
                                     const void* kvcache_u8, const void* weight,
                                     const void* seq_lens, const void* page_table,
                                     void* logits, int q_variant,
                                     const int32_t* t_lengths,
                                     int32_t* dst_token_indices,
                                     int64_t score_stride, int32_t pool_size,
                                     int32_t token_topk, int32_t out_cols,
                                     const int32_t* t_page_table,
                                     int64_t t_page_table_stride,
                                     const int32_t* page_table_row_index,
                                     const int32_t* topk_indices_offset,
                                     const int32_t* row_starts,
                                     const int32_t* t_seq_lens);

} // namespace vkernels::kernels::hip
#endif // VKERNELS_HAS_HIP
