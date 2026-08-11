// vkernels/kernels/elementwise.cpp — CPU reference (oracle) implementation.
#include "vkernels/kernels/elementwise.hpp"

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

namespace {
void check_same(Span<const float> a, Span<const float> b, Span<float> out) {
  VK_EXPECTS(a.size() == b.size(), "a and b must have equal length");
  VK_EXPECTS(a.size() == out.size(), "out must have the same length as inputs");
}
}  // namespace

void add(Span<const float> a, Span<const float> b, Span<float> out) {
  check_same(a, b, out);
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
}

void scale(Span<const float> x, float alpha, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = alpha * x[i];
}

void relu(Span<const float> x, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

}  // namespace vkernels::kernels
