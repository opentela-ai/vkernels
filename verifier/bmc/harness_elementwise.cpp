// verifier/bmc/harness_elementwise.cpp
//
// Bounded model check of the element-wise CPU oracle
// (src/c/vkernels/kernels/elementwise.cpp).  Proves, for every nondeterministic
// input of the fixed (small) bound below, that each oracle matches its
// mathematical specification element-wise and performs no out-of-bounds
// access.  Run via cbmc/run.sh; see verifier/README.md.
#include <cstddef>

#include "verify_support.h"
#include "vkernels/kernels/elementwise.hpp"

using vkernels::Span;
namespace vk = vkernels::kernels;

namespace {
constexpr std::size_t kN = 4;
}

int main() {
  float a[kN], b[kN], x[kN], out[kN];
  for (std::size_t i = 0; i < kN; ++i) {
    a[i] = VK_NONDET_FLOAT();
    b[i] = VK_NONDET_FLOAT();
    x[i] = VK_NONDET_FLOAT();
    out[i] = VK_NONDET_FLOAT();
  }

  const Span<const float> A(a, kN);
  const Span<const float> B(b, kN);
  const Span<const float> X(x, kN);
  const Span<float> O(out, kN);

  // out = a + b, element-wise.
  vk::add(A, B, O);
  for (std::size_t i = 0; i < kN; ++i)
    VK_ASSERT(vk_feq(O[i], a[i] + b[i]), "add: out[i] == a[i] + b[i]");

  // out = alpha * x, element-wise.
  const float alpha = VK_NONDET_FLOAT();
  vk::scale(X, alpha, O);
  for (std::size_t i = 0; i < kN; ++i)
    VK_ASSERT(vk_feq(O[i], alpha * x[i]), "scale: out[i] == alpha * x[i]");

  // out = max(x, 0), element-wise.
  vk::relu(X, O);
  for (std::size_t i = 0; i < kN; ++i)
    VK_ASSERT(vk_feq(O[i], x[i] > 0.0f ? x[i] : 0.0f),
              "relu: out[i] == max(x[i], 0)");
  return 0;
}
