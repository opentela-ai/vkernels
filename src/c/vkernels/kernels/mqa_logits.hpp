// vkernels/kernels/mqa_logits.hpp — weighted ReLU MQA logits (lightning
// indexer scoring), borrowed from DeepGEMM's kernel family
// (fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits, MIT (c) 2025 DeepSeek).
//
// For every query row m with per-row KV span [cu_seq_len_k_start[m],
// cu_seq_len_k_end[m]) over the shared single-head KV cache:
//
//     out[m, n] = sum_h weights[m, h] * relu( sum_d q[m, h, d] * kv[n, d] )
//
// and -inf outside the row's span. (DeepGEMM additionally compresses the
// output to [M, max_seqlen_k] packing each row's span at column zero; this
// tree keeps the dense [M, N] layout — the compression is a serving-side
// layout concern owned by the caller.)
//
// Layouts (row-major): q [M, H, D], kv [N, D], weights [M, H], cu_* [M]
// (int), out [M, N].
#pragma once

#include <cstddef>

#include "vkernels/util/span.hpp"

namespace vkernels::kernels {

void mqa_logits(std::size_t M, std::size_t N, std::size_t H, std::size_t D,
                Span<const float> q, Span<const float> kv,
                Span<const float> weights,
                Span<const int> cu_seq_len_k_start,
                Span<const int> cu_seq_len_k_end, Span<float> out);

namespace cuda {
// Device-pointer variant; same contract as the CPU oracle above.
void mqa_logits(std::size_t M, std::size_t N, std::size_t H, std::size_t D,
                Span<const float> q, Span<const float> kv,
                Span<const float> weights,
                Span<const int> cu_seq_len_k_start,
                Span<const int> cu_seq_len_k_end, Span<float> out);
}  // namespace cuda
}  // namespace vkernels::kernels
