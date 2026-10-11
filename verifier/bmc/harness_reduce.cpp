// verifier/bmc/harness_reduce.cpp
//
// Bounded model check of the reduction CPU oracle
// (src/c/vkernels/kernels/reduce.cpp).  Proves that `sum` reproduces the
// sequential left-to-right accumulation the oracle specifies, and that `max`
// returns an element of the input that dominates every element.  Run via
// cbmc/run.sh; see verifier/README.md.
#include <cstddef>

#include "verify_support.h"
#include "vkernels/kernels/reduce.hpp"

using vkernels::Span;
namespace vk = vkernels::kernels;

namespace {
constexpr std::size_t kN = 4;
}

int main() {
  float x[kN];
  for (std::size_t i = 0; i < kN; ++i) x[i] = VK_NONDET_FLOAT();

  const Span<const float> X(x, kN);

  // sum == the oracle's sequential accumulation order.
  float s = 0.0f;
  vk::sum(X, s);
  float expected = 0.0f;
  for (std::size_t i = 0; i < kN; ++i) expected += x[i];
  VK_ASSERT(vk_feq(s, expected), "sum: equals sequential accumulation");

  // max is (a) one of the elements and (b) >= every element.
  float m = 0.0f;
  vk::max(X, m);
  bool found = false;
  for (std::size_t i = 0; i < kN; ++i)
    if (vk_feq(x[i], m)) found = true;
  VK_ASSERT(found, "max: result equals some input element");
  for (std::size_t i = 0; i < kN; ++i)
    VK_ASSERT(!(x[i] > m), "max: result dominates every element");
  return 0;
}
