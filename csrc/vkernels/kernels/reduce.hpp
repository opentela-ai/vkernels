// vkernels/kernels/reduce.hpp
#pragma once

#include "vkernels/util/span.hpp"

namespace vkernels::kernels {

// Sum of all elements. Returns the reduced value in `out`.
void sum(Span<const float> x, float& out);

// Maximum of all elements.
void max(Span<const float> x, float& out);

}  // namespace vkernels::kernels
