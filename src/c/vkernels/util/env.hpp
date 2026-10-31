// vkernels/util/env.hpp — environment knobs, one parsing idiom.
//
// Kernel launchers expose A/B and diagnostic switches as environment
// variables. Two lifetimes exist and both are deliberate:
//
//   * Process-cached: read once at first use, then a plain load. Use on
//     launch paths whose knob is fixed for the life of the process:
//
//       static const int mode = vkernels::util::env_int("VK_DSA_STAGE", 1);
//
//     (the function-local static is the cache; initialisation is
//     thread-safe and runs exactly once).
//
//   * Re-read per call: only where a harness flips the variable in-process
//     between calls (tests/kernels/attn/test_kda_chunked_fused_gpu.cpp,
//     tests/kernels/attn/test_dsa_topk_fused_gpu.cpp, tests/kernels/moe/
//     test_moe.cpp toggle VK_KDA_CHUNKED_FUSED / VK_DSA_TOPK_FUSED /
//     K3_NO_ASYNC). Call the helper directly without a static.
//
// Host-only; never call from device code.
#pragma once

#include <cstdlib>
#include <cstring>

namespace vkernels::util {

// The raw value, or nullptr when unset.
inline const char* env_str(const char* name) { return std::getenv(name); }

// std::atoi of the value; ``fallback`` when unset.
inline int env_int(const char* name, int fallback) {
  const char* e = std::getenv(name);
  return e ? std::atoi(e) : fallback;
}

// True when the value is exactly ``value`` (unset never matches).
inline bool env_equals(const char* name, const char* value) {
  const char* e = std::getenv(name);
  return e != nullptr && std::strcmp(e, value) == 0;
}

// Boolean switch: set, non-empty and not exactly "0".
inline bool env_flag(const char* name) {
  const char* e = std::getenv(name);
  return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
}

}  // namespace vkernels::util
