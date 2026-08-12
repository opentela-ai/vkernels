// vkernels/kernels/reduce.cpp — CPU reference (oracle) implementation.
#include "vkernels/kernels/reduce.hpp"

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

void sum(Span<const float> x, float& out) {
  VK_EXPECTS(x.size() > 0, "cannot reduce an empty span");
  float acc = 0.0f;
  for (std::size_t i = 0; i < x.size(); ++i) acc += x[i];
  out = acc;
}

void max(Span<const float> x, float& out) {
  VK_EXPECTS(x.size() > 0, "cannot reduce an empty span");
  float m = x[0];
  for (std::size_t i = 1; i < x.size(); ++i)
    if (x[i] > m) m = x[i];
  out = m;
}

}  // namespace vkernels::kernels
