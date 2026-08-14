// vkernels/kernels/kda.hpp
//
// Kimi Delta Attention (KDA) — the delta-rule hybrid layer of Kimi-K3
// (issue #21). KDA is a gated delta-rule linear-attention layer: a state
// matrix S_t (head_dim × head_dim, per head) is updated each token by a
// *delta correction* (β_t (v_t − S_{t−1} k_t) k_tᵀ) and decayed by a forget
// gate g_t, and the output is o_t = S_t q_t. On gfx942 today the AITER /
// Triton chunked kernels GPU-fault (job 586165), so K3 serving disables
// KDA (K3_DISABLE_KDA=1). vkernels re-implements the seven faulting kernels
// as portable host references + HIP kernels so the KDA path is correct on
// MI300A without that flag.
//
//   per-token oracle (the math the chunked kernels parallelise):
//     S_0 = 0
//     for t = 0 .. S-1:
//       a_t = S_{t-1} · k_t                       # prediction from prior state
//       S_t = g_t · S_{t-1} + β_t (v_t - a_t) ⊗ k_t   # forget + delta update
//       o_t = S_t · q_t
//
// (k is L2-normalised by the caller, as in delta-net; the gate g and the
//  delta gate β are scalar per token, broadcast across the head dim.)
//
// Chunking. The naive recurrence is O(S · D²) per head — correct but too
// slow for production. The chunked algorithm (Yang et al., "Parallelizing
// Linear Transformers with the Delta Rule over Sequence Length") splits the
// sequence into chunks of size C and recovers three parallel pieces:
//
//   1. gate cumsum (kda_gate_chunk_cumsum)
//        within-chunk inclusive  L_{c,t} = Σ_{l<=t} log g_{c,l}
//        cross-chunk exclusive   I_c      = Σ_{c'<c} Σ_l log g_{c',l}
//      so a gate product G_{a,b} = exp(Σ_{l=a}^{b} log g_l) is recovered as
//      exp(L_b − L_{a−1}) (L_{−1} = 0).
//
//   2. intra-chunk solve (kda_delta_rule_intra)
//        For each chunk, solve the lower-triangular system for the
//        delta-corrected values u_t (local t = 0 .. C-1):
//          u_t = v_t − G_{0,t-1} (C_{c-1} k_t)
//                    − Σ_{j<t} G_{j+1,t-1} β_j (k_j · k_t) u_j
//        The first term subtracts the inter-chunk prediction; the sum is the
//        within-chunk delta coupling. C_{c-1} is the state entering the
//        chunk (0 for chunk 0).
//
//   3. inter-chunk state + output (kda_delta_rule_inter / kda_gla_fwd_o)
//        C_c = G_{0,C-1} C_{c-1} + Σ_t G_{t+1,C-1} β_t u_t k_tᵀ
//        o_t = G_{0,t} (C_{c-1} q_t)
//                + Σ_{j<=t} G_{j+1,t} β_j (k_j · q_t) u_j
//      The inter call propagates C_c across chunks and produces the
//      inter output term; kda_gla_fwd_o adds the intra output term. Both
//      equal the per-token oracle (kda_naive_delta_rule_fwd_cpu) to within
//      fp32 round-off, verified by the host tests.
//
// Two-implementation model:
//   kda.cpp  -- CPU reference (oracle), always compiled, in vkernels::kernels.
//               kda_naive_delta_rule_fwd_cpu is the per-token oracle; the
//               chunked pieces (gate cumsum / intra / inter / output) are
//               exposed separately and cross-checked against it.
//   kda.hip  -- HIP kernels (gfx942), compiled with VKERNELS_HAS_HIP, in
//               vkernels::kernels::hip. Each mirrors its CPU counterpart.
#include <cstddef>
#include <cstdint>

namespace vkernels::kernels {

// ---------------------------------------------------------------------------
// #L1 — layer_norm_gated_fwd: gated RMSNorm (pre-attention normaliser)
// ---------------------------------------------------------------------------
//   x      : [N, D]            float
//   weight : [D]               float (per-channel scale γ)
//   gate   : [N, D]            float (pre-activation gate)
//   out    : [N, D]            float
//   eps                       RMS denominator regulariser
//
//   rms_n = sqrt(mean(x_n[:]²) + eps)
//   out[n, d] = (x[n, d] / rms_n) * weight[d] * silu(gate[n, d])
//
// silu(g) = g · sigmoid(g). Zero x rows map to zero output (the gate and
// weight still apply, but x/rms = 0/eps = 0).
void kda_layer_norm_gated_cpu(const float* x, const float* weight,
                              const float* gate, float* out,
                              int N, int D, float eps);

// ---------------------------------------------------------------------------
// #L2 — kda_gate_chunk_cumsum: log-gate cumulative sums (intra + inter)
// ---------------------------------------------------------------------------
//   g         : [B, H, n_chunks, chunk_size]   float, gate in *normal* space
//                                              (g > 0; g==0 is clamped to a
//                                              large-negative log)
//   intra_log : [B, H, n_chunks, chunk_size]   float, within-chunk INCLUSIVE
//                                              log-cumsum L_{c,t}
//   inter_log : [B, H, n_chunks]               float, cross-chunk EXCLUSIVE
//                                              log-cumsum I_c
//
//   intra_log[b,h,c,t] = Σ_{l=0}^{t}  log(g[b,h,c,l])
//   chunk_log[b,h,c]   = Σ_{l}        log(g[b,h,c,l])      (= intra_log[..,C-1])
//   inter_log[b,h,c]   = Σ_{c'=0}^{c-1} chunk_log[b,h,c']   (exclusive)
//
// Downstream gate products are recovered as exp(L_b - L_{a-1}) (L_{-1}=0),
// so a within-chunk product is exp(intra_log[t] - (t>0 ? intra_log[t-1] : 0))
// and a cross-chunk product up to chunk c is exp(inter_log[c]).
void kda_gate_chunk_cumsum_cpu(const float* g, float* intra_log,
                               float* inter_log, int B, int H,
                               int n_chunks, int chunk_size);

// ---------------------------------------------------------------------------
// #L3 — kda_naive_delta_rule_fwd: per-token oracle (O(S·D²), reference)
// ---------------------------------------------------------------------------
//   q, k, v : [B, H, S, D]   float (k L2-normalised by the caller)
//   g, β    : [B, H, S]       float (forget gate g, delta gate β)
//   out     : [B, H, S, D]   float
//
// Implements the per-token recurrence in the file header exactly. This is
// the correctness oracle for the chunked kernels; it is too slow for
// production but unambiguous and verifiable by hand.
void kda_naive_delta_rule_fwd_cpu(const float* q, const float* k,
                                  const float* v, const float* g,
                                  const float* beta, float* out,
                                  int B, int H, int S, int D);

// ---------------------------------------------------------------------------
// #L4 — kda_delta_rule_intra: within-chunk delta-corrected value solve
//   (one chunk at a time; the orchestrator interleaves L4 and L5 because
//    chunk c's intra solve needs C_{c-1} produced by inter(c-1).)
// ---------------------------------------------------------------------------
//   q, k, v : [B, H, S, D]   float (the chunk's keys/values/queries)
//   g, β    : [B, H, S]       float (gate / delta gate, chunk slice)
//   intra_log : [B, H, n_chunks, chunk_size]  float (from gate_chunk_cumsum)
//   inter_state : [B, H, n_chunks+1, D, D] float (row chunk_idx = C_{c-1},
//                 the state entering chunk chunk_idx; row 0 must be 0).
//                 Read-only for this call.
//   u       : [B, H, S, D]   float, the solved delta-corrected values
//   chunk_idx : which chunk to solve (0 .. n_chunks-1).
//
// For chunk `chunk_idx`, solves for u_t (local t) the lower-triangular
// system in the file header. `inter_state[b,h,chunk_idx]` is C_{c-1}; it is
// read but not modified.
void kda_delta_rule_intra_cpu(const float* q, const float* k, const float* v,
                              const float* g, const float* beta,
                              const float* intra_log,
                              const float* inter_state, float* u,
                              int B, int H, int S, int D, int chunk_size,
                              int chunk_idx);

// ---------------------------------------------------------------------------
// #L5 — kda_delta_rule_inter: cross-chunk state propagation (one chunk)
// ---------------------------------------------------------------------------
//   k, v, g, β : as above (chunk slices)
//   intra_log  : from gate_chunk_cumsum
//   inter_state : [B, H, n_chunks+1, D, D] float, IN-OUT. row chunk_idx
//                 (= C_{c-1}) is read; row chunk_idx+1 is filled with C_c.
//   chunk_idx : which chunk to propagate (0 .. n_chunks-1).
//
//   C_c = G_{0,C-1} C_{c-1} + Σ_t G_{t+1,C-1} β_t u_t k_tᵀ
void kda_delta_rule_inter_cpu(const float* k, const float* v, const float* g,
                              const float* beta, const float* intra_log,
                              const float* u, float* inter_state,
                              int B, int H, int S, int D, int chunk_size,
                              int chunk_idx);

// ---------------------------------------------------------------------------
// #L6 — kda_gla_fwd_o: output (intra + inter) combine
// ---------------------------------------------------------------------------
//   q, k, g, β : as above
//   intra_log  : from gate_chunk_cumsum
//   inter_state : [B, H, n_chunks+1, D, D] (C_{c-1} read per chunk)
//   u           : [B, H, S, D] from kda_delta_rule_intra
//   out         : [B, H, S, D]
//
//   o_t = G_{0,t} (C_{c-1} q_t) + Σ_{j<=t} G_{j+1,t} β_j (k_j · q_t) u_j
void kda_gla_fwd_o_cpu(const float* q, const float* k, const float* g,
                       const float* beta, const float* intra_log,
                       const float* inter_state, const float* u,
                       float* out, int B, int H, int S, int D, int chunk_size);

// ---------------------------------------------------------------------------
// #L7 — kda_delta_rule_fwd: the chunked forward (orchestrates L2..L6)
// ---------------------------------------------------------------------------
//   q, k, v : [B, H, S, D]   float (k L2-normalised by the caller)
//   g, β    : [B, H, S]       float
//   out     : [B, H, S, D]   float
//   chunk_size  must divide S.
//
// Runs the gate cumsum → intra solve → inter propagation → output combine
// pipeline and writes `out`. Matches kda_naive_delta_rule_fwd_cpu to within
// fp32 round-off (verified by the host tests at K3 head shapes).
void kda_delta_rule_fwd_cpu(const float* q, const float* k, const float* v,
                            const float* g, const float* beta, float* out,
                            int B, int H, int S, int D, int chunk_size);

// ---------------------------------------------------------------------------
// #P — kda_pack_bitmatrix: pack a binary matrix into bytes (MSB first)
// ---------------------------------------------------------------------------
//   bits   : [n_bits]        uint8  (each 0 or 1)
//   packed : [ceil(n_bits/8)] uint8 bit k → byte k/8, bit (7 - k%8) (MSB first)
//
// Used by the KDA weight path to pack the binary/ternary routing / gate
// matrices that AITER's `pack_bitmatrix` produces on gfx950.
void kda_pack_bitmatrix_cpu(const uint8_t* bits, uint8_t* packed,
                            std::size_t n_bits);

}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// HIP kernels (gfx942), one per CPU counterpart above. The cooperative
// forward (kda_delta_rule_fwd) is the correctness-first baseline matching
// the naive oracle; kda_delta_rule_fwd_with_scratch lets the caller own the
// B*H*D*D zero-initialized state scratch (one allocation, reused across
// timed iterations) for benchmarking.
void kda_layer_norm_gated(const float* x, const float* weight,
                          const float* gate, float* out,
                          int N, int D, float eps);

void kda_gate_chunk_cumsum(const float* g, float* intra, float* inter,
                           int B, int H, int n_chunks, int chunk_size);

void kda_delta_rule_fwd(const float* q, const float* k, const float* v,
                        const float* g, const float* beta, float* out,
                        int B, int H, int S, int D, int chunk_size);

void kda_delta_rule_fwd_with_scratch(const float* q, const float* k,
                                     const float* v, const float* g,
                                     const float* beta, float* state,
                                     float* out, int B, int H, int S, int D);

void kda_pack_bitmatrix(const uint8_t* bits, uint8_t* packed,
                        std::size_t n_bits);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
