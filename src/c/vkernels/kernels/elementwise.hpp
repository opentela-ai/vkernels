// vkernels/kernels/elementwise.hpp
//
// Element-wise kernels. The CPU reference implementations below are always
// compiled and treated as the correctness oracle; the CUDA launchers live in
// elementwise.cu and are compiled only when VKERNELS_HAS_CUDA.
#pragma once

#include "vkernels/util/span.hpp"

namespace vkernels::kernels {

// out = a + b
void add(Span<const float> a, Span<const float> b, Span<float> out);

// out = alpha * x
void scale(Span<const float> x, float alpha, Span<float> out);

// out = max(x, 0)
void relu(Span<const float> x, Span<float> out);

}  // namespace vkernels::kernels
