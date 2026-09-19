// tests/kernels/attn/test_kda_cp.cpp
//
// Host tests for the KDA chunked-scan state API (#CP in kda.hpp/kda.cpp) —
// the seam context parallelism needs: CP ranks split the sequence into
// shards and hand the linear-attention recurrent state along the rank ring.
// The state layout is the canonical [B, H, D, D] float of the HIP state
// scratch (kda_delta_rule_fwd[_chunked]_with_scratch).
//
// NOTE on slicing: q/k/v/g are [B, H, S, D] and beta [B, H, S] — a sequence
// shard [t0, t0+len) is a STRIDED slice per (b,h), not a flat offset. The
// helpers below gather shards into per-(b,h)-contiguous [B, H, len, D]
// buffers and scatter shard outputs back.
//
// Checks:
//   1. the state-carrying per-token oracle (kda_naive_delta_rule_fwd_state_cpu)
//      hand-checked, and self-composable (segmented == monolithic);
//   2. the chunked WY state forward (kda_delta_rule_fwd_state_cpu) matches
//      the state-carrying oracle at random + seeded states, incl. K3 head
//      shapes;
//   3. RING HANDOFF: composing shard-by-shard (final state of shard i
//      seeded into shard i+1) equals the monolithic run in outputs AND
//      final state — the property that makes CP correctness modular;
//   4. in-place state aliasing (one ring buffer per (b,h)) and the
//      null/chunk contract.
#include "minitest.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/kda.hpp"

using namespace vkernels::kernels;

namespace {

struct Cfg { int B, H, S, D, cs; };

// Random K3-style inputs: q/k/v in [-1,1], per-key-dim gates in [0.3,1.0]
// (no near-zero decay — the recurrence stays numerically stable, same regime
// as test_kda_k3_chunked.cpp), beta in [0.3,1.0], k L2-NORMALISED (the WY
// contract: the explicit Ainv needs the gate-weighted Gram bounded).
void fill_inputs(const Cfg& c, std::mt19937& rng, std::vector<float>& q,
                 std::vector<float>& k, std::vector<float>& v,
                 std::vector<float>& g, std::vector<float>& beta) {
  const size_t n = (size_t)c.B * c.H * c.S;
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  q.resize(n * c.D); k.resize(n * c.D); v.resize(n * c.D);
  g.resize(n * c.D); beta.resize(n);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  for (auto& x : g) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  for (size_t i = 0; i < n; ++i) {  // L2-normalise k (WY contract)
    float* ki = k.data() + i * c.D;
    float ss = 0.0f;
    for (int d = 0; d < c.D; ++d) ss += ki[d] * ki[d];
    const float inv = 1.0f / std::sqrt(ss);
    for (int d = 0; d < c.D; ++d) ki[d] *= inv;
  }
}

// A sequence shard gathered into per-(b,h)-contiguous buffers.
struct Shard {
  std::vector<float> q, k, v, g, beta;  // [B,H,len,D] / [B,H,len]
};

// Gather tokens [t0, t0+len) of the full [B,H,S,D] inputs (strided per
// (b,h)!) into contiguous shard buffers.
Shard slice_shard(const Cfg& c, const std::vector<float>& q,
                  const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<float>& g, const std::vector<float>& beta,
                  int t0, int len) {
  Shard s;
  const size_t elems = (size_t)c.B * c.H * len * c.D;
  s.q.resize(elems); s.k.resize(elems); s.v.resize(elems); s.g.resize(elems);
  s.beta.resize((size_t)c.B * c.H * len);
  for (int b = 0; b < c.B; ++b)
    for (int h = 0; h < c.H; ++h) {
      const size_t src_bh = ((size_t)(b * c.H + h) * c.S + t0) * c.D;
      const size_t dst_bh = ((size_t)(b * c.H + h) * len) * c.D;
      const size_t srcb = (size_t)(b * c.H + h) * c.S + t0;
      const size_t dstb = (size_t)(b * c.H + h) * len;
      for (int t = 0; t < len; ++t) {
        std::memcpy(s.q.data() + dst_bh + (size_t)t * c.D,
                    q.data() + src_bh + (size_t)t * c.D,
                    sizeof(float) * c.D);
        std::memcpy(s.k.data() + dst_bh + (size_t)t * c.D,
                    k.data() + src_bh + (size_t)t * c.D,
                    sizeof(float) * c.D);
        std::memcpy(s.v.data() + dst_bh + (size_t)t * c.D,
                    v.data() + src_bh + (size_t)t * c.D,
                    sizeof(float) * c.D);
        std::memcpy(s.g.data() + dst_bh + (size_t)t * c.D,
                    g.data() + src_bh + (size_t)t * c.D,
                    sizeof(float) * c.D);
        s.beta[dstb + t] = beta[srcb + t];
      }
    }
  return s;
}

}  // namespace

// --- state-carrying per-token oracle: hand-checked --------------------------
// S_in = I, one token k=[0,1] v=[1,0] g=[1,1] beta=.5 q=[1,0]:
//   S' = I; a = S'·k = [0,1]; S = I + .5 (v-a)⊗k = [[1,.5],[0,.5]]
//   o = S q = [1,0]; state_out = [[1,.5],[0,.5]]
TEST(KdaDeltaRuleState, NaiveHandChecked) {
  constexpr int B = 1, H = 1, S = 1, D = 2;
  std::vector<float> q = {1.f, 0.f}, k = {0.f, 1.f}, v = {1.f, 0.f};
  std::vector<float> g = {1.f, 1.f}, beta = {0.5f};
  std::vector<float> sin = {1.f, 0.f, 0.f, 1.f};        // identity
  std::vector<float> sout(4, -1.f), out(S * D, -1.f);
  kda_naive_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                     beta.data(), sin.data(), sout.data(),
                                     out.data(), B, H, S, D);
  EXPECT_NEAR(out[0], 1.0f, 1e-6f);
  EXPECT_NEAR(out[1], 0.0f, 1e-6f);
  EXPECT_NEAR(sout[0], 1.0f, 1e-6f);
  EXPECT_NEAR(sout[1], 0.5f, 1e-6f);
  EXPECT_NEAR(sout[2], 0.0f, 1e-6f);
  EXPECT_NEAR(sout[3], 0.5f, 1e-6f);
}

// --- state-carrying oracle composes: running [0,half) then [half,S) with
// the exported mid state equals the monolithic run (outputs + final state).
// Shards are strided per (b,h) — gathered/scattered via slice_shard. ---
TEST(KdaDeltaRuleState, NaiveCompositionMatchesMonolithic) {
  constexpr int B = 1, H = 2, S = 16, D = 4, half = S / 2;
  Cfg c{B, H, S, D, D /*cs unused for naive*/};
  std::mt19937 rng(5);
  std::vector<float> q, k, v, g, beta;
  fill_inputs(c, rng, q, k, v, g, beta);
  const size_t n = (size_t)B * H * S, nstate = (size_t)B * H * D * D;
  std::vector<float> state(nstate, 0.0f), state_end(nstate, 0.0f);
  std::vector<float> out_mono(n * D, 0.0f);
  kda_naive_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                     beta.data(), state.data(),
                                     state_end.data(), out_mono.data(),
                                     B, H, S, D);
  // two halves: state after half 1 seeds half 2
  const Shard sh1 = slice_shard(c, q, k, v, g, beta, 0, half);
  const Shard sh2 = slice_shard(c, q, k, v, g, beta, half, half);
  std::vector<float> s0(nstate, 0.0f), s1(nstate, -1.f);
  std::vector<float> o1(sh1.q.size(), 0.f), o2(sh2.q.size(), 0.f);
  kda_naive_delta_rule_fwd_state_cpu(sh1.q.data(), sh1.k.data(), sh1.v.data(),
                                     sh1.g.data(), sh1.beta.data(),
                                     s0.data(), s1.data(), o1.data(),
                                     B, H, half, D);
  kda_naive_delta_rule_fwd_state_cpu(sh2.q.data(), sh2.k.data(), sh2.v.data(),
                                     sh2.g.data(), sh2.beta.data(),
                                     s1.data(), s0.data(), o2.data(),
                                     B, H, half, D);
  // scatter both shard outputs back onto the full-sequence layout
  std::vector<float> composed(n * D, 0.0f);
  const float* outs[2] = {o1.data(), o2.data()};
  for (int r = 0; r < 2; ++r)
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int t = 0; t < half; ++t)
          for (int d = 0; d < D; ++d)
            composed[(((size_t)(b * H + h) * S) + r * half + t) * D + d] =
                outs[r][(((size_t)(b * H + h) * half) + t) * D + d];
  for (size_t i = 0; i < n * D; ++i)
    EXPECT_NEAR(composed[i], out_mono[i], 1e-4f);
  for (size_t i = 0; i < nstate; ++i) EXPECT_NEAR(s0[i], state_end[i], 1e-5f);
}

// --- chunked WY state forward vs the state-carrying oracle (zero state),
// incl. the K3 head shapes and the HIP chunk shape (cs=64) ---
TEST(KdaDeltaRuleState, ChunkedWYMatchesNaiveOracleZeroState) {
  const Cfg cfgs[] = {
      {1, 1, 8, 2, 4},      {1, 2, 8, 4, 4},      {2, 1, 16, 8, 8},
      {1, 1, 32, 16, 16},   {1, 1, 64, 32, 32},   // multi-chunk, small
      {1, 1, 128, 64, 64},  {1, 2, 128, 128, 64}, // K3 head dim, HIP chunk
  };
  std::mt19937 rng(4242);
  for (const auto& c : cfgs) {
    std::vector<float> q, k, v, g, beta;
    fill_inputs(c, rng, q, k, v, g, beta);
    const size_t n = (size_t)c.B * c.H * c.S;
    const size_t nstate = (size_t)c.B * c.H * c.D * c.D;
    std::vector<float> s0(nstate, 0.0f), sw(nstate, -1.f), so(nstate, -1.f);
    std::vector<float> naive(n * c.D, 0.0f), wy(n * c.D, 0.0f);
    kda_naive_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                       beta.data(), s0.data(), so.data(),
                                       naive.data(), c.B, c.H, c.S, c.D);
    kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                 beta.data(), s0.data(), sw.data(),
                                 wy.data(), c.B, c.H, c.S, c.D, c.cs);
    float max_out = 0.0f, max_st = 0.0f;
    for (size_t i = 0; i < n * c.D; ++i) {
      const float e = std::fabs(wy[i] - naive[i]);
      if (e > max_out) max_out = e;
      EXPECT_NEAR(wy[i], naive[i], 1e-3f);
    }
    for (size_t i = 0; i < nstate; ++i) {
      const float e = std::fabs(sw[i] - so[i]);
      if (e > max_st) max_st = e;
      EXPECT_NEAR(sw[i], so[i], 1e-3f);
    }
    std::printf("  chunked-vs-oracle B=%d H=%d S=%d D=%d cs=%d: "
                "out max_abs=%.6f state max_abs=%.6f\n",
                c.B, c.H, c.S, c.D, c.cs, max_out, max_st);
  }
}

// --- chunked WY state forward with a NONZERO seeded state (the multi-turn /
// CP mid-stream entry contract): the seeded state is what a previous rank or
// request turn exported; the whole point of the API. Single-chunk decode
// shape (S == cs) and multi-chunk prefill shape. ---
TEST(KdaDeltaRuleState, ChunkedWYSeededStateMatchesOracle) {
  const Cfg cfgs[] = {
      {1, 2, 64, 64, 64},    // one chunk, seeded (decode turn)
      {1, 1, 128, 32, 64},   // two chunks, seeded
      {1, 2, 256, 128, 64},  // K3 head dim, 4 chunks, seeded
  };
  std::mt19937 rng(909090);
  for (const auto& c : cfgs) {
    std::vector<float> q, k, v, g, beta;
    fill_inputs(c, rng, q, k, v, g, beta);
    const size_t n = (size_t)c.B * c.H * c.S;
    const size_t nstate = (size_t)c.B * c.H * c.D * c.D;
    std::vector<float> s0(nstate), sw(nstate, -1.f), so(nstate, -1.f);
    for (auto& x : s0) x = 0.5f * ((rng() % 2000) / 1000.0f - 1.0f);
    std::vector<float> naive(n * c.D, 0.0f), wy(n * c.D, 0.0f);
    kda_naive_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                       beta.data(), s0.data(), so.data(),
                                       naive.data(), c.B, c.H, c.S, c.D);
    kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                 beta.data(), s0.data(), sw.data(),
                                 wy.data(), c.B, c.H, c.S, c.D, c.cs);
    float max_out = 0.0f, max_st = 0.0f;
    for (size_t i = 0; i < n * c.D; ++i) {
      const float e = std::fabs(wy[i] - naive[i]);
      if (e > max_out) max_out = e;
      EXPECT_NEAR(wy[i], naive[i], 1e-3f);
    }
    for (size_t i = 0; i < nstate; ++i) {
      const float e = std::fabs(sw[i] - so[i]);
      if (e > max_st) max_st = e;
      EXPECT_NEAR(sw[i], so[i], 1e-3f);
    }
    std::printf("  seeded B=%d H=%d S=%d D=%d cs=%d: out max_abs=%.6f "
                "state max_abs=%.6f\n",
                c.B, c.H, c.S, c.D, c.cs, max_out, max_st);
  }
}

// --- RING HANDOFF (the CP correctness property): composing shard-by-shard
// with the exported state equals the monolithic run in outputs AND final
// state, at a K3-decode-ish head count and the K3 head dim. Each shard is a
// whole number of internal chunks (shard % cs == 0, the HIP contract). ---
TEST(KdaDeltaRuleState, RingHandoffCompositionMatchesMonolithic) {
  std::mt19937 rng(31337);
  const Cfg cfgs[] = {
      {1, 2, 256, 32, 64},   // 4 shards of one internal chunk each
      {1, 16, 256, 64, 64},  // K3-decode-ish: 16 heads, 4 shards
      {1, 2, 512, 128, 64},  // K3 head dim, 8 chunks over 4 shards
  };
  for (const auto& c : cfgs) {
    std::vector<float> q, k, v, g, beta;
    fill_inputs(c, rng, q, k, v, g, beta);
    const size_t n = (size_t)c.B * c.H * c.S;
    const size_t nstate = (size_t)c.B * c.H * c.D * c.D;
    constexpr int kShards = 4;
    const int shard = c.S / kShards;
    // monolithic reference
    std::vector<float> s_mono_in(nstate, 0.0f), s_mono_out(nstate, 0.0f);
    std::vector<float> out_mono(n * c.D, 0.0f);
    kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                                 beta.data(), s_mono_in.data(),
                                 s_mono_out.data(), out_mono.data(),
                                 c.B, c.H, c.S, c.D, c.cs);
    // ring composition: shard r runs with S_in = S_out(r-1)
    std::vector<float> state_ring(nstate, 0.0f), state_next(nstate, 0.0f);
    std::vector<float> out_ring(n * c.D, 0.0f);
    for (int r = 0; r < kShards; ++r) {
      const Shard sh = slice_shard(c, q, k, v, g, beta, r * shard, shard);
      std::vector<float> o(sh.q.size(), 0.f);
      kda_delta_rule_fwd_state_cpu(sh.q.data(), sh.k.data(), sh.v.data(),
                                   sh.g.data(), sh.beta.data(),
                                   state_ring.data(), state_next.data(),
                                   o.data(), c.B, c.H, shard, c.D, c.cs);
      state_ring = state_next;
      // scatter this shard's outputs back onto the full layout
      for (int b = 0; b < c.B; ++b)
        for (int h = 0; h < c.H; ++h)
          for (int t = 0; t < shard; ++t)
            for (int d = 0; d < c.D; ++d)
              out_ring[(((size_t)(b * c.H + h) * c.S) + r * shard + t) * c.D + d] =
                  o[(((size_t)(b * c.H + h) * shard) + t) * c.D + d];
    }
    float max_out = 0.0f, max_st = 0.0f;
    for (size_t i = 0; i < n * c.D; ++i) {
      const float e = std::fabs(out_ring[i] - out_mono[i]);
      if (e > max_out) max_out = e;
      EXPECT_NEAR(out_ring[i], out_mono[i], 1e-3f);
    }
    for (size_t i = 0; i < nstate; ++i) {
      const float e = std::fabs(state_ring[i] - s_mono_out[i]);
      if (e > max_st) max_st = e;
      EXPECT_NEAR(state_ring[i], s_mono_out[i], 1e-3f);
    }
    std::printf("  ring handoff B=%d H=%d S=%d D=%d cs=%d shards=%d: "
                "out max_abs=%.6f state max_abs=%.6f\n",
                c.B, c.H, c.S, c.D, c.cs, kShards, max_out, max_st);
  }
}

// --- in-place state aliasing: the ring handoff reuses ONE state buffer per
// (b,h) (seed from it, overwrite it). state_in == state_out must be safe. ---
TEST(KdaDeltaRuleState, InPlaceStateAlias) {
  constexpr int B = 1, H = 2, S = 128, D = 32, cs = 64;
  Cfg c{B, H, S, D, cs};
  std::mt19937 rng(777);
  std::vector<float> q, k, v, g, beta;
  fill_inputs(c, rng, q, k, v, g, beta);
  const size_t n = (size_t)B * H * S, nstate = (size_t)B * H * D * D;
  std::vector<float> s0(nstate, 0.0f);
  // separate-buffer reference
  std::vector<float> ref_in = s0, ref_out(nstate, -1.f), out_ref(n * D, 0.f);
  kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                               beta.data(), ref_in.data(), ref_out.data(),
                               out_ref.data(), B, H, S, D, cs);
  // aliased run: same buffer seeded and overwritten
  std::vector<float> st = s0;
  std::vector<float> out_alias(n * D, 0.f);
  kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(), g.data(),
                               beta.data(), st.data(), st.data(),
                               out_alias.data(), B, H, S, D, cs);
  for (size_t i = 0; i < n * D; ++i) EXPECT_NEAR(out_alias[i], out_ref[i], 0.0f);
  for (size_t i = 0; i < nstate; ++i) EXPECT_NEAR(st[i], ref_out[i], 0.0f);
}

// --- contracts: null args + chunk_size must divide S ------------------------
TEST(KdaDeltaRuleState, Contracts) {
  constexpr int B = 1, H = 1, S = 8, D = 4;
  std::vector<float> q(S * D, 1.f), k(S * D, 1.f), v(S * D, 1.f);
  std::vector<float> g(S * D, 1.f), beta(S, 1.f);
  std::vector<float> s(4, 0.f), o(S * D, 0.f);
  EXPECT_THROW(kda_delta_rule_fwd_state_cpu(nullptr, k.data(), v.data(),
                                            g.data(), beta.data(), s.data(),
                                            s.data(), o.data(), B, H, S, D, 4),
               std::invalid_argument);
  EXPECT_THROW(kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(),
                                            g.data(), beta.data(), nullptr,
                                            s.data(), o.data(), B, H, S, D, 4),
               std::invalid_argument);
  EXPECT_THROW(kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(),
                                            g.data(), beta.data(), s.data(),
                                            nullptr, o.data(), B, H, S, D, 4),
               std::invalid_argument);
  EXPECT_THROW(kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(),
                                            g.data(), beta.data(), s.data(),
                                            s.data(), nullptr, B, H, S, D, 4),
               std::invalid_argument);
  EXPECT_THROW(kda_delta_rule_fwd_state_cpu(q.data(), k.data(), v.data(),
                                            g.data(), beta.data(), s.data(),
                                            s.data(), o.data(), B, H, S, D, 3),
               std::invalid_argument);  // 8 % 3 != 0
  EXPECT_THROW(kda_naive_delta_rule_fwd_state_cpu(
                   q.data(), k.data(), v.data(), g.data(), beta.data(),
                   nullptr, s.data(), o.data(), B, H, S, D),
               std::invalid_argument);
  EXPECT_THROW(kda_naive_delta_rule_fwd_state_cpu(
                   q.data(), k.data(), v.data(), g.data(), beta.data(),
                   s.data(), nullptr, o.data(), B, H, S, D),
               std::invalid_argument);
}
