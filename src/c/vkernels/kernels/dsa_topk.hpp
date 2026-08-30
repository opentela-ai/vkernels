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

} // namespace vkernels::kernels::hip
#endif // VKERNELS_HAS_HIP
