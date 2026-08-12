// vkernels/kernels/gemm.hpp
//
// SGEMM: C = alpha * A @ B + beta * C, row-major, all MxN, MxK, KxN.
#pragma once

#include "vkernels/util/span.hpp"

namespace vkernels::kernels {

void gemm(std::size_t M, std::size_t N, std::size_t K, float alpha,
          Span<const float> A, Span<const float> B, float beta, Span<float> C);

}  // namespace vkernels::kernels
