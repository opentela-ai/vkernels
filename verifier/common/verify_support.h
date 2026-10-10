// verifier/common/verify_support.h
//
// Helpers shared by the bounded-model-checking harnesses.  These harnesses
// are never compiled into libvkernels or the production test binaries (see
// verifier/README.md).  The macros have two modes:
//
//   * under a prover they map to its assume/assert/nondeterministic
//     intrinsics, so a harness encodes a proof obligation.  The runner
//     selects the prover with -DVK_PROVER_ESBMC / -DVK_PROVER_CBMC.
//   * under a normal host compiler they map to no-ops (assume), assertions
//     (assert) and a fixed value (nondet), so the harness still type-checks
//     and runs as a smoke test.  This keeps verifier/ lint-clean.
//
// Postconditions must use VK_ASSERT (not the standard assert) so the proof
// obligation survives into the prover.
#pragma once

#include <cstddef>

#if defined(VK_PROVER_ESBMC)
// ESBMC (Clang frontend) recognizes nondet_* without a declaration.
#  define VK_VERIFY_MODEL 1
#  define VK_ASSUME(cond) __ESBMC_assume(cond)
#  define VK_ASSERT(cond, msg) __ESBMC_assert(cond, msg)
#  define VK_NONDET_FLOAT() nondet_float()
#elif defined(VK_PROVER_CBMC) || defined(__CPROVER__) || defined(__CPROVER)
#  define VK_VERIFY_MODEL 1
#  define VK_ASSUME(cond) __CPROVER_assume(cond)
#  define VK_ASSERT(cond, msg) __CPROVER_assert(cond, msg)
extern "C" float nondet_float(void);
#  define VK_NONDET_FLOAT() nondet_float()
#else
#  include <cassert>
#  define VK_VERIFY_MODEL 0
#  define VK_ASSUME(cond) ((void)0)
#  define VK_ASSERT(cond, msg) assert(cond)
#  define VK_NONDET_FLOAT() (0.0f)
#endif

// NaN-tolerant IEEE equality: true when the two are equal or both NaN.
// IEEE `==` is false for NaN == NaN, which would make a proof over
// unconstrained nondeterministic floats fail spuriously.  The kernels state
// no NaN semantics, so "both NaN" is treated as agreement; a real mismatch
// (one NaN, one finite) still fails.
inline bool vk_feq(float a, float b) {
  return a == b || (a != a && b != b);
}
