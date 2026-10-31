// vkernels/kernels/common/dispatch.hpp — runtime int -> compile-time constant.
//
// Launchers pick a kernel template instantiation from a small runtime value
// (segment count, M, head dim, ...). Instead of a hand-written switch or a
// #define/#undef case table per launcher:
//
//   common::switch_over<2, 4, 8>(segs, [&](auto c) {
//     launch(kernel<decltype(c)::value>, ...);
//   });
//
// calls the functor once with std::integral_constant<int, V> for the first
// listed V equal to ``value`` and returns whether one matched. Exactly the
// listed values are instantiated, in the listed order, so a table converted
// to this helper generates the same set of kernel instantiations.
// Host-only (plain C++17); usable from .cpp, .cu and .hip TUs.
#pragma once

#include <type_traits>
#include <utility>

namespace vkernels::kernels::common {

template <int... Vs, class F>
bool switch_over(int value, F&& f) {
  return ((value == Vs ? (f(std::integral_constant<int, Vs>{}), true) : false) || ...);
}

// As switch_over, but calls f with ``Fallback`` when no listed value matches.
template <int Fallback, int... Vs, class F>
void switch_over_or(int value, F&& f) {
  if (!switch_over<Vs...>(value, f)) f(std::integral_constant<int, Fallback>{});
}

}  // namespace vkernels::kernels::common
