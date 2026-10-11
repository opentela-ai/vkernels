// vkernels/kernels/common/epilogue.hpp — composable GEMM epilogue operators.
//
// The store-side math of ``D = C + A @ B`` expressed as small structs that
// both the CPU oracle (gemm.cpp) and the CUDA kernels (gemm.cu and friends)
// instantiate, so a fused epilogue is one type, not a copy of the kernel
// with the last line edited. Layout borrowed from DeepGEMM's
// ``include/deep_gemm/epilogue/operators.cuh`` layering, scaled down to
// this repo's two-implementation model.
//
// Every epilogue is ``float apply(float acc, float c_prev) const``:
// ``acc`` is the K-loop accumulator, ``c_prev`` the existing output
// element (valid only when the epilogue reads it — see Linear's beta rule).
// Host- and device-usable: no CUDA headers, no exceptions.
#pragma once

namespace vkernels::kernels::epilogue {

// C = alpha * acc + beta * C_prev, with BLAS beta semantics: beta == 0
// must not read C_prev (0 * NaN would poison the store; this is the rule
// the CUDA gemm always honored and the CPU oracle now shares).
struct Linear {
  float alpha;
  float beta;

  constexpr float apply(float acc, float c_prev) const {
    const float prev = (beta != 0.0f) ? c_prev : 0.0f;
    return alpha * acc + beta * prev;
  }
};

// relu(base(...)). Matches the standalone relu kernel's fmaxf semantics:
// a non-positive value (including NaN from the base) maps to 0.
template <typename Base>
struct Relu {
  Base base;

  constexpr float apply(float acc, float c_prev) const {
    const float v = base.apply(acc, c_prev);
    return (v > 0.0f) ? v : 0.0f;
  }
};

using LinearRelu = Relu<Linear>;

}  // namespace vkernels::kernels::epilogue
