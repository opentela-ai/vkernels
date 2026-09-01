// vkernels/kernels/dsa_topk.cpp -- host oracle for the DSA kpool transform.
#include "vkernels/kernels/dsa_topk.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {
namespace {

int32_t transform_token(int32_t raw_token,
                        const int32_t* page_table_row,
                        const int32_t* topk_indices_offset,
                        int32_t offset) {
  if (page_table_row != nullptr) return page_table_row[raw_token];
  if (topk_indices_offset != nullptr) return raw_token + offset;
  return raw_token;
}

} // namespace

bool dsa_topk_transform_group_topk_supported(int32_t group_topk) {
  switch (group_topk) {
  case 128:
  case 160:
  case 192:
  case 224:
  case 256:
  case 512:
    return true;
  default:
    return false;
  }
}

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
                            const int32_t* seq_lens) {
  VK_EXPECTS(batch_size >= 0, "batch_size must be non-negative");
  VK_EXPECTS(score_stride >= 0, "score_stride must be non-negative");
  VK_EXPECTS(pool_size > 1, "pool_size must be greater than one");
  VK_EXPECTS(token_topk > 0 && token_topk % pool_size == 0,
             "token_topk must be a positive multiple of pool_size");
  const int32_t group_topk = token_topk / pool_size;
  VK_EXPECTS(dsa_topk_transform_group_topk_supported(group_topk), "unsupported pool-level top-k");
  const int32_t tail_cols = seq_lens == nullptr ? 0 : pool_size - 1;
  VK_EXPECTS(out_cols == token_topk + tail_cols, "out_cols does not match the transform layout");
  VK_EXPECTS(page_table == nullptr || topk_indices_offset == nullptr,
             "page_table and topk_indices_offset are mutually exclusive");
  VK_EXPECTS(page_table == nullptr || page_table_stride > 0,
             "page_table_stride must be positive when a page table is used");
  VK_EXPECTS(batch_size == 0 || score != nullptr, "score must not be null");
  VK_EXPECTS(batch_size == 0 || lengths != nullptr, "lengths must not be null");
  VK_EXPECTS(batch_size == 0 || dst_token_indices != nullptr, "dst_token_indices must not be null");
  if (batch_size == 0) return;

  std::vector<int32_t> selected;
  for (int32_t row = 0; row < batch_size; ++row) {
    const int32_t length = lengths[row];
    const int32_t row_start = row_starts == nullptr ? 0 : row_starts[row];
    VK_EXPECTS(length >= 0, "lengths entries must be non-negative");
    VK_EXPECTS(row_start >= 0, "row_starts entries must be non-negative");
    VK_EXPECTS(static_cast<int64_t>(row_start) + length <= score_stride,
               "valid score range exceeds score_stride");
    VK_EXPECTS(length <= std::numeric_limits<int32_t>::max() / pool_size,
               "expanded token index overflows int32");

    int32_t* dst = dst_token_indices + static_cast<size_t>(row) * out_cols;
    std::fill(dst, dst + out_cols, -1);
    const int32_t valid_groups = std::min(length, group_topk);
    if (valid_groups > 0) {
      selected.resize(length);
      std::iota(selected.begin(), selected.end(), 0);
      if (length > group_topk) {
        const float* score_row = score + static_cast<size_t>(row) * score_stride + row_start;
        std::nth_element(selected.begin(),
                         selected.begin() + group_topk,
                         selected.end(),
                         [score_row](int32_t a, int32_t b) {
                           if (score_row[a] == score_row[b]) return a < b;
                           return score_row[a] > score_row[b];
                         });
        selected.resize(group_topk);
      }
    }

    const int32_t page_row = page_table_row_index == nullptr ? row : page_table_row_index[row];
    VK_EXPECTS(page_table == nullptr || (page_row >= 0 && page_row < batch_size),
               "page_table_row_index entries must address a page-table row");
    const int32_t* page_table_entry =
        page_table == nullptr ? nullptr
                              : page_table + static_cast<size_t>(page_row) * page_table_stride;
    const int32_t offset = topk_indices_offset == nullptr ? 0 : topk_indices_offset[row];

    const int32_t history_len = valid_groups * pool_size;
    for (int32_t col = 0; col < history_len; ++col) {
      const int32_t group_rank = col / pool_size;
      const int32_t slot = col % pool_size;
      const int32_t raw_token = selected[group_rank] * pool_size + slot;
      VK_EXPECTS(page_table_entry == nullptr || raw_token < page_table_stride,
                 "raw token exceeds page_table_stride");
      dst[col] = transform_token(raw_token, page_table_entry, topk_indices_offset, offset);
    }

    if (seq_lens != nullptr) {
      VK_EXPECTS(seq_lens[row] >= 0, "seq_lens entries must be non-negative");
      const int32_t tail_count = seq_lens[row] % pool_size;
      for (int32_t tail = 0; tail < tail_count; ++tail) {
        const int32_t raw_token = length * pool_size + tail;
        VK_EXPECTS(page_table_entry == nullptr || raw_token < page_table_stride,
                   "tail token exceeds page_table_stride");
        dst[history_len + tail] =
            transform_token(raw_token, page_table_entry, topk_indices_offset, offset);
      }
    }
  }
}

} // namespace vkernels::kernels
