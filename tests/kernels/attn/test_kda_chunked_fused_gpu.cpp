// tests/kernels/attn/test_kda_chunked_fused_gpu.cpp
//
// GPU test for the VK_KDA_CHUNKED_FUSED=1 launch-chain fusion of the chunked
// WY KDA forward (kda.hip #L8): the per-key log-cumsum launch is folded into
// the gram-kernel prologue, so the chain runs 3 launches instead of 4.
//
// Two gates, mirroring the fusion-lane contract (NOTES-fusion-dsa-kda-launches):
//   1. BIT-IDENTITY: fused-chain out + final state == proven-chain out +
//      final state, element for element (the fused prologue computes L with
//      the SAME statements in the SAME order, so every downstream float must
//      be identical — a single ULP difference is a failure).
//   2. ORACLE PARITY: both chains match kda_delta_rule_fwd_state_cpu (the
//      chunked-scan state API, the CPU mirror of the WY form) to the device
//      test tolerance used by meta/benchmarks/test_kda_chunked.hip (2e-2).
//
// The device code only exists under VKERNELS_HAS_HIP (real HIP build on
// gfx942, or the HIP-on-NVIDIA shim in the CUDA build). Without it this TU
// compiles to a self-reported skip so the host-only CI stays green.
#include "minitest.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/kda.hpp"

#if VKERNELS_HAS_HIP

#include <cstdlib>

using namespace vkernels::kernels;

namespace {

// Deterministic inputs in the style of meta/benchmarks/test_kda_chunked.hip:
// contractive per-key forget gate in (0.05, 1.0], L2-normalised keys, beta in
// (0, 1] — the WY contract regime.
float rnd(unsigned seed, int i) {
  unsigned x = seed * 2654435761u + (unsigned)i * 40503u;
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)(x % 200000u) - 100000) / 100000.0f;
}

float rgate(unsigned seed, int i) {
  unsigned x = seed * 2654435761u + (unsigned)i * 40503u;
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return 0.05f + 0.95f * (float)(x % 100000u) / 100000.0f;
}

void normalize_keys(std::vector<float>& k, int tokens, int D) {
  for (int x = 0; x < tokens; ++x) {
    float* kk = k.data() + (size_t)x * D;
    float ss = 0.0f;
    for (int d = 0; d < D; ++d) ss += kk[d] * kk[d];
    const float inv = 1.0f / std::sqrt(ss + 1e-12f);
    for (int d = 0; d < D; ++d) kk[d] *= inv;
  }
}

struct Stats {
  double max_abs, max_rel;
};

Stats cmp(const std::vector<float>& got, const std::vector<float>& ref) {
  Stats s{0.0, 0.0};
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = std::fabs((double)got[i] - ref[i]);
    if (e > s.max_abs) s.max_abs = e;
    const double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    if (e / d > s.max_rel) s.max_rel = e / d;
  }
  return s;
}

}  // namespace

TEST(KdaChunkedFused, BitIdentityAndOracleParity) {
  struct Cfg {
    int B, H, S, D;
  };
  const Cfg cfgs[] = {
      {1, 1, 64, 16},    // single chunk, smallest D template
      {1, 1, 64, 128},   // single chunk, largest D template
      {1, 2, 128, 64},   // multi (b,h), 2 chunks
      {1, 1, 512, 128},  // the full K3 bench shape (8 chunks)
      {2, 1, 128, 32},   // multi batch, D=32 template
  };
#ifdef VKERNELS_KDA_CUDA_SHIM
  // HIP-on-NVIDIA shim build: D=128 is not instantiated (NVIDIA 48 KB static
  // shared limit vs the 64 KB gfx942 gram staging); covered on gfx942.
  constexpr bool kShim = true;
#else
  constexpr bool kShim = false;
#endif
  constexpr int kCs = 64;
  constexpr float kThresh = 2e-2f;  // device test tolerance (test_kda_chunked.hip)

  for (const auto& c : cfgs) {
    if (kShim && c.D > 64) {
      std::printf("  skip B=%d H=%d S=%d D=%d (shim build serves D <= 64)\n",
                  c.B, c.H, c.S, c.D);
      continue;
    }
    const int B = c.B, H = c.H, S = c.S, D = c.D;
    const size_t m = (size_t)B * H * S * D;
    const size_t n = (size_t)B * H * S;
    std::vector<float> q(m), k(m), v(m), g(m), beta(n);
    for (size_t i = 0; i < m; ++i) {
      q[i] = rnd(1, (int)i);
      k[i] = rnd(2, (int)i);
      v[i] = rnd(3, (int)i);
      g[i] = rgate(4, (int)i);
    }
    for (size_t i = 0; i < n; ++i) beta[i] = 0.1f + 0.9f * std::fabs(rnd(5, (int)i));
    normalize_keys(k, B * H * S, D);

    const size_t sf = hip::kda_chunked_scratch_floats(B, H, S, D);
    std::vector<float> scratch(sf);
    const size_t ns = (size_t)B * H * D * D;

    // --- proven chain (gate off; the default) ---
    std::vector<float> state0(ns, 0.0f), out0(m, 0.0f);
    hip::kda_delta_rule_fwd_chunked_with_scratch(
        q.data(), k.data(), v.data(), g.data(), beta.data(), state0.data(),
        out0.data(), scratch.data(), B, H, S, D, kCs, nullptr);

    // --- fused chain (gate on; per-call env read, so setenv works) ---
    setenv("VK_KDA_CHUNKED_FUSED", "1", 1);
    std::vector<float> state1(ns, 0.0f), out1(m, 0.0f);
    hip::kda_delta_rule_fwd_chunked_with_scratch(
        q.data(), k.data(), v.data(), g.data(), beta.data(), state1.data(),
        out1.data(), scratch.data(), B, H, S, D, kCs, nullptr);
    setenv("VK_KDA_CHUNKED_FUSED", "0", 1);

    // Gate 1: bit identity (out AND final state).
    const bool out_bit =
        std::memcmp(out0.data(), out1.data(), m * sizeof(float)) == 0;
    const bool state_bit =
        std::memcmp(state0.data(), state1.data(), ns * sizeof(float)) == 0;
    EXPECT_TRUE(out_bit);
    EXPECT_TRUE(state_bit);

    // Gate 2: oracle parity for both chains (chunked-scan state CPU API,
    // zero initial state).
    std::vector<float> state_ref(ns, 0.0f), out_ref(m, 0.0f);
    kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                 beta.data(), state_ref.data(),
                                 state_ref.data(), out_ref.data(), B, H, S, D,
                                 kCs);
    const Stats s0 = cmp(out0, out_ref);
    const Stats s1 = cmp(out1, out_ref);
    const bool o0 = s0.max_rel < (double)kThresh;
    const bool o1 = s1.max_rel < (double)kThresh;
    EXPECT_TRUE(o0);
    EXPECT_TRUE(o1);
    std::printf("  B=%d H=%d S=%d D=%d: bit(out=%d state=%d); oracle max_rel %.3e / %.3e\n",
                B, H, S, D, (int)out_bit, (int)state_bit, s0.max_rel, s1.max_rel);
  }
}

#else  // !VKERNELS_HAS_HIP

// Host-only builds (no HIP toolchain, no CUDA shim): the device path is not
// compiled. Report the skip explicitly so the host CI run is honest.
TEST(KdaChunkedFused, SkippedWithoutDevicePath) {
  std::printf(
      "  skip: VKERNELS_HAS_HIP not defined (no device kda.hip in this build)\n");
}

#endif  // VKERNELS_HAS_HIP
