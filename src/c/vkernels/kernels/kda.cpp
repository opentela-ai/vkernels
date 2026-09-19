// vkernels/kernels/kda.cpp -- CPU reference (oracle) for the Kimi Delta
// Attention layer (issue #21). Always compiled (no GPU toolkit).
//
// kda_naive_delta_rule_fwd_cpu is the per-token oracle: it implements the
// recurrence in the file header exactly (O(S·D²) per head) and is the
// correctness reference both for the host tests and the on-device harness.
// The chunked pieces (gate cumsum / intra solve / inter propagation /
// output combine) parallelise that recurrence and are cross-checked against
// the naive oracle in test_kda.cpp at K3 head shapes.
//
// See kda.hpp for the full per-operation contracts and the gate-product
// conventions (G_{a,b} = exp(L_b - L_{a-1}), L_{-1} = 0).
#include "vkernels/kernels/kda.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

namespace {

constexpr float kNegLogFloor = -1.0e9f;  // log(0) clamp (fully forgotten)

// log(g) with a floor so a fully-forgotten gate (g==0) yields ~0 weight
// instead of -inf.
float log_gate(float g) {
  if (g <= 0.0f) return kNegLogFloor;
  return std::log(g);
}

// Within-chunk gate product G_{a,b} = ∏_{l=a}^{b} g_l = exp(L_b - L_{a-1}),
// with L_{-1} = 0. Returns 1.0 when a > b (empty product).
float gate_prod_intra(const float* L, int a, int b) {
  if (a > b) return 1.0f;
  return std::exp(L[b] - (a > 0 ? L[a - 1] : 0.0f));
}

// small helper: dot product of two D-vectors
float dot(const float* x, const float* y, int D) {
  float s = 0.0f;
  for (int d = 0; d < D; ++d) s += x[d] * y[d];
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// #L1 layer_norm_gated
// ---------------------------------------------------------------------------
void kda_layer_norm_gated_cpu(const float* x, const float* weight,
                              const float* gate, float* out,
                              int N, int D, float eps) {
  VK_EXPECTS(N == 0 || D > 0, "D must be positive");
  VK_EXPECTS(N == 0 || x != nullptr, "x must not be null");
  VK_EXPECTS(N == 0 || weight != nullptr, "weight must not be null");
  VK_EXPECTS(N == 0 || gate != nullptr, "gate must not be null");
  VK_EXPECTS(N == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(eps >= 0.0f, "eps must be non-negative");
  for (int n = 0; n < N; ++n) {
    const float* xn = x + (size_t)n * D;
    float* on = out + (size_t)n * D;
    float sq = 0.0f;
    for (int d = 0; d < D; ++d) sq += xn[d] * xn[d];
    const float inv = 1.0f / std::sqrt(sq / static_cast<float>(D) + eps);
    for (int d = 0; d < D; ++d) {
      const float g = gate[(size_t)n * D + d];
      const float silu = g / (1.0f + std::exp(-g));  // g · sigmoid(g)
      on[d] = xn[d] * inv * weight[d] * silu;
    }
  }
}

// ---------------------------------------------------------------------------
// #L2 gate_chunk_cumsum (intra within-chunk + inter cross-chunk)
// ---------------------------------------------------------------------------
void kda_gate_chunk_cumsum_cpu(const float* g, float* intra_log,
                               float* inter_log, int B, int H,
                               int n_chunks, int chunk_size) {
  VK_EXPECTS(B == 0 || H == 0 || n_chunks > 0, "n_chunks must be positive");
  VK_EXPECTS(B == 0 || H == 0 || chunk_size > 0, "chunk_size must be positive");
  VK_EXPECTS(B == 0 || H == 0 || g != nullptr, "g must not be null");
  VK_EXPECTS(B == 0 || H == 0 || intra_log != nullptr, "intra_log must not be null");
  VK_EXPECTS(B == 0 || H == 0 || inter_log != nullptr, "inter_log must not be null");
  if (B == 0 || H == 0 || n_chunks == 0) return;
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t base = ((size_t)b * H + h) * n_chunks;
      float inter_acc = 0.0f;  // exclusive cross-chunk log-cumsum
      for (int c = 0; c < n_chunks; ++c) {
        float* Lc = intra_log + (base + c) * chunk_size;
        const float* gc = g + (base + c) * chunk_size;
        float acc = 0.0f;  // within-chunk inclusive log-cumsum
        for (int t = 0; t < chunk_size; ++t) {
          acc += log_gate(gc[t]);
          Lc[t] = acc;
        }
        inter_log[base + c] = inter_acc;       // I_c = Σ_{c'<c} chunk_log[c']
        inter_acc += acc;                       // chunk_log[c] = L_{C-1} = acc
      }
    }
}

// ---------------------------------------------------------------------------
// #L3 naive per-token oracle
// ---------------------------------------------------------------------------
void kda_naive_delta_rule_fwd_cpu(const float* q, const float* k,
                                  const float* v, const float* g,
                                  const float* beta, float* out,
                                  int B, int H, int S, int D) {
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || q != nullptr, "q must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || k != nullptr, "k must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || v != nullptr, "v must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || g != nullptr, "g must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || beta != nullptr, "beta must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(D > 0, "D must be positive");
  if (B == 0 || H == 0 || S == 0) return;

  std::vector<float> state((size_t)D * D, 0.0f);   // state per (b,h)
  std::vector<float> a(D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      std::fill(state.begin(), state.end(), 0.0f);
      const size_t bh = (size_t)(b * H + h) * S;
      for (int t = 0; t < S; ++t) {
        const float* kt = k + (bh + t) * D;
        const float* vt = v + (bh + t) * D;
        const float* qt = q + (bh + t) * D;
        const float* gt = g + (bh + t) * D;   // per-key-dim forget gate [D]
        const float bt = beta[bh + t];         // scalar delta gate
        // (1) gate: S'[v,k] *= g_t[k]   (per-key-dim, normal space)
        for (int v = 0; v < D; ++v) {
          float* Srow = state.data() + (size_t)v * D;
          for (int k = 0; k < D; ++k) Srow[k] *= gt[k];
        }
        // (2) predict: a[v] = S'[v,:] . k_t   (from GATED state)
        for (int v = 0; v < D; ++v)
          a[v] = dot(state.data() + (size_t)v * D, kt, D);
        // (3) update: S_t[v,k] += beta_t * (v_t[v] - a[v]) * k_t[k]
        for (int v = 0; v < D; ++v) {
          const float ud = vt[v] - a[v];
          float* Srow = state.data() + (size_t)v * D;
          for (int k = 0; k < D; ++k) Srow[k] += bt * ud * kt[k];
        }
        // (4) output: o_t[v] = S_t[v,:] . q_t
        for (int v = 0; v < D; ++v)
          out[(bh + t) * D + v] = dot(state.data() + (size_t)v * D, qt, D);
      }
    }
}

// ---------------------------------------------------------------------------
// #L4 intra-chunk solve (delta-corrected values u_t) — one chunk
// ---------------------------------------------------------------------------
void kda_delta_rule_intra_cpu(const float* q, const float* k, const float* v,
                              const float* g, const float* beta,
                              const float* intra_log,
                              const float* inter_state, float* u,
                              int B, int H, int S, int D, int chunk_size,
                              int chunk_idx) {
  (void)q; (void)g;  // gate enters via intra_log; q unused in the solve
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || k != nullptr, "k must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || v != nullptr, "v must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || beta != nullptr, "beta must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || intra_log != nullptr, "intra_log must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || inter_state != nullptr, "inter_state must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || u != nullptr, "u must not be null");
  VK_EXPECTS(chunk_size > 0 && S % chunk_size == 0, "chunk_size must divide S");
  VK_EXPECTS(chunk_idx >= 0 && chunk_idx * chunk_size < S,
             "chunk_idx out of range");
  if (B == 0 || H == 0 || S == 0) return;

  const int n_chunks = S / chunk_size;
  std::vector<float> pred(D), corr(D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t bh = (size_t)(b * H + h) * S;
      const float* Lc = intra_log + ((size_t)(b * H + h) * n_chunks
                                     + chunk_idx) * chunk_size;
      const float* Cin = inter_state + ((size_t)(b * H + h) * (n_chunks + 1)
                                        + chunk_idx) * D * D;
      for (int t = 0; t < chunk_size; ++t) {
        const int tau = chunk_idx * chunk_size + t;
        const float* kt = k + (bh + tau) * D;
        const float* vt = v + (bh + tau) * D;
        // inter prediction: G_{0,t-1} (C_{c-1} k_t)
        const float G0 = gate_prod_intra(Lc, 0, t - 1);
        for (int d = 0; d < D; ++d)
          pred[d] = G0 * dot(Cin + (size_t)d * D, kt, D);
        // within-chunk delta coupling: Σ_{j<t} G_{j+1,t-1} β_j (k_j·k_t) u_j
        for (int d = 0; d < D; ++d) corr[d] = 0.0f;
        for (int j = 0; j < t; ++j) {
          const int tauj = chunk_idx * chunk_size + j;
          const float* kj = k + (bh + tauj) * D;
          const float* uj = u + (bh + tauj) * D;
          const float Gj = gate_prod_intra(Lc, j + 1, t - 1);
          const float cjk = Gj * beta[bh + tauj] * dot(kj, kt, D);
          for (int d = 0; d < D; ++d) corr[d] += cjk * uj[d];
        }
        float* ut = u + (bh + tau) * D;
        for (int d = 0; d < D; ++d) ut[d] = vt[d] - pred[d] - corr[d];
      }
    }
}

// ---------------------------------------------------------------------------
// #L5 inter-chunk state propagation — one chunk
// ---------------------------------------------------------------------------
void kda_delta_rule_inter_cpu(const float* k, const float* v, const float* g,
                              const float* beta, const float* intra_log,
                              const float* u, float* inter_state,
                              int B, int H, int S, int D, int chunk_size,
                              int chunk_idx) {
  (void)v; (void)g;  // value/gate enter via u and intra_log
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || k != nullptr, "k must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || beta != nullptr, "beta must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || intra_log != nullptr, "intra_log must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || u != nullptr, "u must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || inter_state != nullptr, "inter_state must not be null");
  VK_EXPECTS(chunk_size > 0 && S % chunk_size == 0, "chunk_size must divide S");
  VK_EXPECTS(chunk_idx >= 0 && chunk_idx * chunk_size < S,
             "chunk_idx out of range");
  if (B == 0 || H == 0 || S == 0) return;

  const int n_chunks = S / chunk_size;
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t bh = (size_t)(b * H + h) * S;
      const float* Lc = intra_log + ((size_t)(b * H + h) * n_chunks
                                     + chunk_idx) * chunk_size;
      const float* Cin = inter_state + ((size_t)(b * H + h) * (n_chunks + 1)
                                        + chunk_idx) * D * D;
      float* Cout = inter_state + ((size_t)(b * H + h) * (n_chunks + 1)
                                   + (chunk_idx + 1)) * D * D;
      const float Gfull = gate_prod_intra(Lc, 0, chunk_size - 1);  // G_{0,C-1}
      for (int d = 0; d < D; ++d) {
        const float* Cinrow = Cin + (size_t)d * D;
        float* Crow = Cout + (size_t)d * D;
        for (int e = 0; e < D; ++e) Crow[e] = Gfull * Cinrow[e];
      }
      for (int t = 0; t < chunk_size; ++t) {
        const int tau = chunk_idx * chunk_size + t;
        const float* kt = k + (bh + tau) * D;
        const float* ut = u + (bh + tau) * D;
        const float Gt = gate_prod_intra(Lc, t + 1, chunk_size - 1);
        const float w = Gt * beta[bh + tau];
        for (int d = 0; d < D; ++d) {
          float* Crow = Cout + (size_t)d * D;
          const float wd = w * ut[d];
          for (int e = 0; e < D; ++e) Crow[e] += wd * kt[e];
        }
      }
    }
}

// ---------------------------------------------------------------------------
// #L6 output (intra + inter) combine
// ---------------------------------------------------------------------------
void kda_gla_fwd_o_cpu(const float* q, const float* k, const float* g,
                       const float* beta, const float* intra_log,
                       const float* inter_state, const float* u,
                       float* out, int B, int H, int S, int D, int chunk_size) {
  (void)g;  // gate enters via intra_log
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || q != nullptr, "q must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || k != nullptr, "k must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || beta != nullptr, "beta must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || intra_log != nullptr, "intra_log must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || inter_state != nullptr, "inter_state must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || u != nullptr, "u must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(chunk_size > 0 && S % chunk_size == 0, "chunk_size must divide S");
  if (B == 0 || H == 0 || S == 0) return;

  const int n_chunks = S / chunk_size;
  std::vector<float> inter_out(D), intra_out(D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t bh = (size_t)(b * H + h) * S;
      for (int c = 0; c < n_chunks; ++c) {
        const float* Lc = intra_log + ((size_t)(b * H + h) * n_chunks + c) * chunk_size;
        const float* Cin = inter_state + ((size_t)(b * H + h) * (n_chunks + 1) + c) * D * D;
        for (int t = 0; t < chunk_size; ++t) {
          const int tau = c * chunk_size + t;
          const float* qt = q + (bh + tau) * D;
          // inter term: G_{0,t} (C_{c-1} q_t)
          const float G0t = gate_prod_intra(Lc, 0, t);
          for (int d = 0; d < D; ++d)
            inter_out[d] = G0t * dot(Cin + (size_t)d * D, qt, D);
          // intra term: Σ_{j<=t} G_{j+1,t} β_j (k_j·q_t) u_j
          for (int d = 0; d < D; ++d) intra_out[d] = 0.0f;
          for (int j = 0; j <= t; ++j) {
            const int tauj = c * chunk_size + j;
            const float* kj = k + (bh + tauj) * D;
            const float* uj = u + (bh + tauj) * D;
            const float Gj = gate_prod_intra(Lc, j + 1, t);
            const float cjq = Gj * beta[bh + tauj] * dot(kj, qt, D);
            for (int d = 0; d < D; ++d) intra_out[d] += cjq * uj[d];
          }
          float* ot = out + (bh + tau) * D;
          for (int d = 0; d < D; ++d) ot[d] = inter_out[d] + intra_out[d];
        }
      }
    }
}

// ---------------------------------------------------------------------------
// #L7 chunked forward (orchestrates L2..L6)
// ---------------------------------------------------------------------------
void kda_delta_rule_fwd_cpu(const float* q, const float* k, const float* v,
                            const float* g, const float* beta, float* out,
                            int B, int H, int S, int D, int chunk_size) {
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || q != nullptr, "q must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || k != nullptr, "k must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || v != nullptr, "v must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || g != nullptr, "g must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || beta != nullptr, "beta must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(D > 0, "D must be positive");
  VK_EXPECTS(chunk_size > 0 && S % chunk_size == 0, "chunk_size must divide S");
  if (B == 0 || H == 0 || S == 0) return;

  const int n_chunks = S / chunk_size;
  std::vector<float> intra_log((size_t)B * H * n_chunks * chunk_size);
  std::vector<float> inter_log((size_t)B * H * n_chunks);
  std::vector<float> u((size_t)B * H * S * D);
  // inter_state: [B, H, n_chunks+1, D, D], row 0 = 0 (C_{-1}). The state is
  // carried serially across chunks on the CPU reference — intra(c) needs
  // C_{c-1} from inter(c-1) — so L4 and L5 are interleaved chunk by chunk.
  // (The GPU kernel later recovers chunk-level parallelism by decoupling
  // the pure-intra solve from C_{c-1}.)
  std::vector<float> inter_state((size_t)B * H * (n_chunks + 1) * D * D, 0.0f);

  kda_gate_chunk_cumsum_cpu(g, intra_log.data(), inter_log.data(),
                            B, H, n_chunks, chunk_size);
  for (int c = 0; c < n_chunks; ++c) {
    kda_delta_rule_intra_cpu(q, k, v, g, beta, intra_log.data(),
                             inter_state.data(), u.data(),
                             B, H, S, D, chunk_size, c);
    kda_delta_rule_inter_cpu(k, v, g, beta, intra_log.data(), u.data(),
                             inter_state.data(), B, H, S, D, chunk_size, c);
  }
  kda_gla_fwd_o_cpu(q, k, g, beta, intra_log.data(), inter_state.data(),
                    u.data(), out, B, H, S, D, chunk_size);
}

// ---------------------------------------------------------------------------
// #CP state-carrying per-token oracle (explicit S_0, exports S_S)
// ---------------------------------------------------------------------------
void kda_naive_delta_rule_fwd_state_cpu(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, const float* state_in, float* state_out, float* out,
    int B, int H, int S, int D) {
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || q != nullptr, "q must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || k != nullptr, "k must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || v != nullptr, "v must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || g != nullptr, "g must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || beta != nullptr, "beta must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || state_in != nullptr,
             "state_in must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || state_out != nullptr,
             "state_out must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(D > 0, "D must be positive");
  if (B == 0 || H == 0 || S == 0) return;

  const size_t state_bh = (size_t)D * D;  // per-(b,h) state elements
  std::vector<float> state(state_bh);
  std::vector<float> a(D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t sh = ((size_t)(b * H + h)) * state_bh;
      std::copy(state_in + sh, state_in + sh + state_bh, state.begin());
      const size_t bh = (size_t)(b * H + h) * S;
      for (int t = 0; t < S; ++t) {
        const float* kt = k + (bh + t) * D;
        const float* vt = v + (bh + t) * D;
        const float* qt = q + (bh + t) * D;
        const float* gt = g + (bh + t) * D;   // per-key-dim forget gate [D]
        const float bt = beta[bh + t];         // scalar delta gate
        // (1) gate: S'[v,k] *= g_t[k]   (per-key-dim, normal space)
        for (int vv = 0; vv < D; ++vv) {
          float* Srow = state.data() + (size_t)vv * D;
          for (int kk = 0; kk < D; ++kk) Srow[kk] *= gt[kk];
        }
        // (2) predict: a[v] = S'[v,:] . k_t   (from GATED state)
        for (int vv = 0; vv < D; ++vv)
          a[vv] = dot(state.data() + (size_t)vv * D, kt, D);
        // (3) update: S_t[v,k] += beta_t * (v_t[v] - a[v]) * k_t[k]
        for (int vv = 0; vv < D; ++vv) {
          const float ud = vt[vv] - a[vv];
          float* Srow = state.data() + (size_t)vv * D;
          for (int kk = 0; kk < D; ++kk) Srow[kk] += bt * ud * kt[kk];
        }
        // (4) output: o_t[v] = S_t[v,:] . q_t
        for (int vv = 0; vv < D; ++vv)
          out[(bh + t) * D + vv] = dot(state.data() + (size_t)vv * D, qt, D);
      }
      // export the final state (after the full read of state_in for this
      // (b,h): state_out may alias state_in — the ring handoff buffer)
      std::copy(state.begin(), state.end(), state_out + sh);
    }
}

// ---------------------------------------------------------------------------
// #CP chunked WY forward with explicit state (affine UT-transform form of
// the K3 per-key-dim recurrence — the math hip::kda_delta_rule_fwd_chunked
// implements, seeded with S_in and exporting S_S so CP ranks can hand the
// state along the rank ring over sequence shards).
//
// Per chunk (local t = 0..C-1, L_t[k] = per-column within-chunk log-cumsum,
// end = C-1), identical to the derivation in test_kda_k3_chunked.cpp:
//   M[t][j] = b_j * Σ_k G_{j+1,t}[k] k_j[k] k_t[k]   (j<t, strict tril)
//   N[t][j] = b_j * Σ_k G_{j+1,t}[k] k_j[k] q_t[k]   (j<=t, incl diag)
//   Ainv = (I + tril(M,-1))^{-1}
//   Kgw[t][k] = exp(L_t[k]) k_t[k]        (G_{0,t} includes g_t: POST-gate)
//   Qgw[t][k] = exp(L_t[k]) q_t[k]
//   Kgb[t][k] = b_t exp(L_end[k]-L_t[k]) k_t[k];  diagG[k] = exp(L_end[k])
//   U_v = Ainv v;  W = Ainv Kgw;  T = N U_v;  P = N W;  Opar = Qgw - P
//   ---- serial state pass (only row v of C meets column v of u/o) ----
//   u = U_v - W C^T;  o = T + Opar C^T;  C = diagG (.) C + Kgb^T u
// with C_{-1} = state_in (the cross-rank delta vs the zero-state form).
void kda_delta_rule_fwd_state_cpu(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, const float* state_in, float* state_out, float* out,
    int B, int H, int S, int D, int chunk_size) {
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || q != nullptr, "q must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || k != nullptr, "k must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || v != nullptr, "v must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || g != nullptr, "g must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || beta != nullptr, "beta must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || state_in != nullptr,
             "state_in must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || state_out != nullptr,
             "state_out must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(D > 0, "D must be positive");
  VK_EXPECTS(chunk_size > 0 && S % chunk_size == 0,
             "chunk_size must divide S");
  if (B == 0 || H == 0 || S == 0) return;

  const int nc = S / chunk_size;
  const int cs = chunk_size;
  const size_t state_bh = (size_t)D * D;
  std::vector<float> L((size_t)cs * D);
  std::vector<float> M((size_t)cs * cs, 0.0f), N((size_t)cs * cs, 0.0f),
      Ainv((size_t)cs * cs, 0.0f);
  std::vector<float> Kgw((size_t)cs * D), Qgw((size_t)cs * D),
      Kgb((size_t)cs * D), diagG(D);
  std::vector<float> U_v((size_t)cs * D), W((size_t)cs * D),
      T((size_t)cs * D), P((size_t)cs * D), Opar((size_t)cs * D),
      u((size_t)cs * D);
  std::vector<float> C(state_bh);  // the carried inter state (per (b,h))
  for (int b = 0; b < B; ++b)
  for (int h = 0; h < H; ++h) {
    const size_t bh = (size_t)(b * H + h) * S;   // token base of this (b,h)
    const size_t sh = ((size_t)(b * H + h)) * state_bh;
    // C_{-1} = state_in (explicit initial state; read fully before the
    // final write, so state_out may alias state_in — the ring buffer)
    std::copy(state_in + sh, state_in + sh + state_bh, C.begin());
    for (int c = 0; c < nc; ++c) {
      const int t0 = c * cs;
      // per-column within-chunk inclusive log-cumsum
      for (int kk = 0; kk < D; ++kk) {
        float acc = 0.0f;
        for (int t = 0; t < cs; ++t) {
          acc += log_gate(g[(bh + t0 + t) * D + kk]);
          L[(size_t)t * D + kk] = acc;
        }
      }
      // grams M (strict tril) / N (incl diag), beta folded in
      for (int t = 0; t < cs; ++t)
      for (int j = 0; j <= t; ++j) {
        const float* kj = k + (bh + t0 + j) * D;
        const float* kt = k + (bh + t0 + t) * D;
        const float* qt = q + (bh + t0 + t) * D;
        const float* Lj = L.data() + (size_t)j * D;
        const float* Lt = L.data() + (size_t)t * D;
        float mk = 0.0f, nk = 0.0f;
        for (int kk = 0; kk < D; ++kk) {
          const float gw = std::exp(Lt[kk] - Lj[kk]);   // G_{j+1,t}[k]
          mk += gw * kj[kk] * kt[kk];
          nk += gw * kj[kk] * qt[kk];
        }
        const float bj = beta[bh + t0 + j];
        N[(size_t)t * cs + j] = bj * nk;
        M[(size_t)t * cs + j] = (j < t) ? bj * mk : 0.0f;
      }
      // Ainv = (I + tril(M,-1))^{-1}, forward substitution per column
      for (int j = 0; j < cs; ++j)
      for (int t = j; t < cs; ++t) {
        float x = (t == j) ? 1.0f : 0.0f;
        if (t > j) {
          for (int l = j; l < t; ++l)
            x -= M[(size_t)t * cs + l] * Ainv[(size_t)l * cs + j];
        }
        Ainv[(size_t)t * cs + j] = x;
      }
      // chunk-local gated operands
      const int lend = cs - 1;
      for (int kk = 0; kk < D; ++kk) diagG[kk] = std::exp(L[(size_t)lend * D + kk]);
      for (int t = 0; t < cs; ++t) {
        const float* kt = k + (bh + t0 + t) * D;
        const float* qt = q + (bh + t0 + t) * D;
        const float* Lt = L.data() + (size_t)t * D;
        const float bt = beta[bh + t0 + t];
        for (int kk = 0; kk < D; ++kk) {
          const float e0 = std::exp(Lt[kk]);                       // G_{0,t}
          Kgw[(size_t)t * D + kk] = e0 * kt[kk];
          Qgw[(size_t)t * D + kk] = e0 * qt[kk];
          Kgb[(size_t)t * D + kk] =
              bt * std::exp(L[(size_t)lend * D + kk] - Lt[kk]) * kt[kk];
        }
      }
      // chunk-local GEMMs
      for (int t = 0; t < cs; ++t)
      for (int d = 0; d < D; ++d) {
        float sv = 0.0f, sw = 0.0f;
        for (int j = 0; j < cs; ++j) {
          sv += Ainv[(size_t)t * cs + j] * v[(bh + t0 + j) * D + d];
          sw += Ainv[(size_t)t * cs + j] * Kgw[(size_t)j * D + d];
        }
        U_v[(size_t)t * D + d] = sv;
        W[(size_t)t * D + d] = sw;
      }
      for (int t = 0; t < cs; ++t)
      for (int d = 0; d < D; ++d) {
        float st = 0.0f, sp = 0.0f;
        for (int j = 0; j <= t; ++j) {
          st += N[(size_t)t * cs + j] * U_v[(size_t)j * D + d];
          sp += N[(size_t)t * cs + j] * W[(size_t)j * D + d];
        }
        T[(size_t)t * D + d] = st;
        P[(size_t)t * D + d] = sp;
        Opar[(size_t)t * D + d] = Qgw[(size_t)t * D + d] - sp;
      }
      // serial state pass (rowblock-splittable: only row v of C appears)
      for (int t = 0; t < cs; ++t)
      for (int d = 0; d < D; ++d) {
        float s = U_v[(size_t)t * D + d];
        for (int kk = 0; kk < D; ++kk)
          s -= W[(size_t)t * D + kk] * C[(size_t)d * D + kk];
        u[(size_t)t * D + d] = s;
      }
      for (int t = 0; t < cs; ++t)
      for (int d = 0; d < D; ++d) {
        float s = T[(size_t)t * D + d];
        for (int kk = 0; kk < D; ++kk)
          s += Opar[(size_t)t * D + kk] * C[(size_t)d * D + kk];
        out[(bh + t0 + t) * D + d] = s;
      }
      for (int d = 0; d < D; ++d)
      for (int kk = 0; kk < D; ++kk) {
        float s = diagG[kk] * C[(size_t)d * D + kk];
        for (int t = 0; t < cs; ++t)
          s += Kgb[(size_t)t * D + kk] * u[(size_t)t * D + d];
        C[(size_t)d * D + kk] = s;
      }
    }
    // export the final state S_S for the next rank in the ring
    std::copy(C.begin(), C.end(), state_out + sh);
  }
}

// ---------------------------------------------------------------------------
// #P pack_bitmatrix (MSB-first)
// ---------------------------------------------------------------------------
void kda_pack_bitmatrix_cpu(const uint8_t* bits, uint8_t* packed,
                            std::size_t n_bits) {
  VK_EXPECTS(n_bits == 0 || bits != nullptr, "bits must not be null");
  VK_EXPECTS(n_bits == 0 || packed != nullptr, "packed must not be null");
  const std::size_t bytes = (n_bits + 7) / 8;
  for (std::size_t i = 0; i < bytes; ++i) packed[i] = 0;
  for (std::size_t k = 0; k < n_bits; ++k) {
    const std::size_t byte = k / 8;
    const std::size_t bit = 7 - (k % 8);  // MSB first
    if (bits[k] != 0) packed[byte] |= static_cast<uint8_t>(1u << bit);
  }
}

}  // namespace vkernels::kernels
