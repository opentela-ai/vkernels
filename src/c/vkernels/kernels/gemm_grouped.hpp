// vkernels/kernels/gemm_grouped.hpp — masked m-grouped GEMM for
// CUDA-graph MoE decode, borrowed from DeepGEMM's
// m_grouped_fp8_gemm_nt_masked contract (MIT (c) 2025 DeepSeek).
//
// Groups share the M/N/K problem shape; a device-resident ``masked_m``
// gives the valid row count per group, so a graph captured without knowing
// per-expert token counts still computes exactly the live rows:
//
//     C[g, m, n] = sum_k A[g, m, k] * B[g, n, k]        (NT: B used transposed)
//
// for m < masked_m[g]; rows at or beyond masked_m[g] are LEFT UNTOUCHED —
// callers pre-poison them (NaN) to detect leakage. Layouts (row-major):
// A [G, M, K], B [G, N, K], masked_m [G] (int), C [G, M, N].
#pragma once

#include <cstddef>

#include "vkernels/util/span.hpp"

namespace vkernels::kernels {

void m_grouped_gemm_nt_masked(std::size_t G, std::size_t M, std::size_t N,
                              std::size_t K, Span<const float> A,
                              Span<const float> B, Span<const int> masked_m,
                              Span<float> C);

namespace cuda {
// Device-pointer variant; same contract as the CPU oracle above.
void m_grouped_gemm_nt_masked(std::size_t G, std::size_t M, std::size_t N,
                              std::size_t K, Span<const float> A,
                              Span<const float> B, Span<const int> masked_m,
                              Span<float> C);
}  // namespace cuda
}  // namespace vkernels::kernels
