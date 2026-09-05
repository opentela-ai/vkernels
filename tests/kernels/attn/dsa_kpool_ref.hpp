// tests/kernels/attn/dsa_kpool_ref.hpp
//
// Shared INDEPENDENT reference for the dsa_kpool host tests (issues #60/#61).
// Included by test_dsa_kpool.cpp (bf16 suite) and test_dsa_kpool_fp8.cpp
// (fp8 suite); header-only inline functions, never compiled into the
// library.
//
// Everything here is deliberately STRUCTURALLY DIFFERENT from the
// implementation (dsa_kpool.cpp / dsa_kpool.hip): a dense 128x128
// Walsh-Hadamard matvec instead of the staged ladder, two-pass naive
// softmax instead of the online/FlashAttention form, and a frexp-based fp8
// encoder with a hand-rolled RNE table instead of the impl's bit
// manipulation -- so a bug shared between the impl and the oracle is
// unlikely by construction. The bf16 and fp8 suites share THIS ONE
// reference core (the pooled-vector math is identical by contract; only
// the store differs), so the two test references cannot drift apart.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace dsa_kpool_ref {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Dense normalized 128-point Walsh-Hadamard (independent of the staged
// decomposition the oracle uses): H[i][j] = prod_bits (-1)^(bit_i & bit_j) /
// sqrt(128). Built once; the test applies it as a plain matvec so the only
// residual against the oracle is FMA-vs-mul-add (<<1e-4).
inline const std::vector<float>& dense_hadamard128() {
  static std::vector<float> H(128 * 128);
  static bool built = false;
  if (built) return H;
  const float scale = 1.0f / std::sqrt(128.0f);
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j) {
      int x = i & j, sign = 1;
      while (x) { if (x & 1) sign = -sign; x >>= 1; }
      H[(size_t)i * 128 + j] = (float)sign * scale;
    }
  built = true;
  return H;
}

inline void apply_dense_hadamard(const std::vector<float>& v,
                                 std::vector<float>& out) {
  const std::vector<float>& H = dense_hadamard128();
  for (int i = 0; i < 128; ++i) {
    float s = 0;
    for (int j = 0; j < 128; ++j) s += H[(size_t)i * 128 + j] * v[j];
    out[i] = s;
  }
}

// ---------------------------------------------------------------------------
// PREFILL row core: gather the pool's ``pool_size`` slots (tail prefix at
// logical ``[tail_base, tail_base + n_tail)`` ring-buffered in
// ``tail_k/tail_score[req]``, then the chunk suffix from
// ``chunk_k/chunk_score[chunk_src + s - n_tail]``), add the per-slot ``ape``
// weights, and two-pass-naive-softmax-accumulate them. Fills
// ``acc``/``denom`` (each head_dim). This is the shared reference for the
// bf16 and fp8 PREFILL references -- the weighted mean they store differs
// only in the epilogue.
// ---------------------------------------------------------------------------
inline void assemble_row_accumulate(
    int pool_size, int head_dim, int tail_size, const std::vector<float>& chunk_k,
    const std::vector<float>& chunk_score, const std::vector<float>& tail_k,
    const std::vector<float>& tail_score, const std::vector<float>& ape,
    int req, int n_tail, int chunk_src, int tail_base,
    std::vector<float>& acc, std::vector<float>& denom) {
  // Two-pass naive: full per-slot score, max, then exp/sum/weighted.
  std::vector<float> sc(pool_size * head_dim, -kInf);
  for (int slot = 0; slot < pool_size; ++slot) {
    const float* sp;
    if (slot < n_tail) {
      const int phys = ((tail_base + slot) % tail_size + tail_size) % tail_size;
      sp = tail_score.data() + (size_t)req * tail_size * head_dim
                                + (size_t)phys * head_dim;
    } else {
      const int ci = chunk_src + (slot - n_tail);
      sp = chunk_score.data() + (size_t)ci * head_dim;
    }
    for (int d = 0; d < head_dim; ++d)
      sc[slot * head_dim + d] = sp[d] + ape[slot * head_dim + d];
  }
  std::vector<float> mx(head_dim, -kInf);
  for (int slot = 0; slot < pool_size; ++slot)
    for (int d = 0; d < head_dim; ++d)
      mx[d] = std::max(mx[d], sc[slot * head_dim + d]);
  acc.assign(head_dim, 0.0f);
  denom.assign(head_dim, 0.0f);
  for (int slot = 0; slot < pool_size; ++slot) {
    const float* kp;
    if (slot < n_tail) {
      const int phys = ((tail_base + slot) % tail_size + tail_size) % tail_size;
      kp = tail_k.data() + (size_t)req * tail_size * head_dim
                          + (size_t)phys * head_dim;
    } else {
      const int ci = chunk_src + (slot - n_tail);
      kp = chunk_k.data() + (size_t)ci * head_dim;
    }
    for (int d = 0; d < head_dim; ++d) {
      const float prob = std::exp(sc[slot * head_dim + d] - mx[d]);
      denom[d] += prob;
      acc[d] += kp[d] * prob;
    }
  }
}

// Full PREFILL reference: per-row accumulate + mean + dense Hadamard + fp32
// write to ``out[loc[r]//ssp, loc[r]%ssp]`` (write_mask skips rows).
// Mirrors dsa_kpool_assemble_cpu's contract exactly.
inline void assemble_ref(int n_pools, int pool_size, int head_dim,
                         int tail_size, int slots_per_page,
                         const std::vector<float>& chunk_k,
                         const std::vector<float>& chunk_score,
                         const std::vector<float>& tail_k,
                         const std::vector<float>& tail_score,
                         const std::vector<float>& ape,
                         const std::vector<int32_t>& req_pool_idx,
                         const std::vector<int32_t>& n_from_tail,
                         const std::vector<int32_t>& chunk_src_start,
                         const std::vector<int32_t>& tail_logical_base,
                         const std::vector<int32_t>& loc,
                         const std::vector<int32_t>* write_mask,
                         std::vector<float>& out) {
  std::vector<float> acc, denom, mean(head_dim), h(128);
  for (int r = 0; r < n_pools; ++r) {
    if (write_mask && (*write_mask)[r] == 0) continue;
    assemble_row_accumulate(pool_size, head_dim, tail_size, chunk_k,
                            chunk_score, tail_k, tail_score, ape,
                            req_pool_idx[r], n_from_tail[r],
                            chunk_src_start[r], tail_logical_base[r], acc,
                            denom);
    for (int d = 0; d < head_dim; ++d) mean[d] = acc[d] / denom[d];
    apply_dense_hadamard(mean, h);
    const int lr = loc[r];
    const int page = lr / slots_per_page;
    const int sip = lr % slots_per_page;
    float* dst = out.data() + (size_t)page * slots_per_page * head_dim
                             + (size_t)sip * head_dim;
    for (int d = 0; d < head_dim; ++d) dst[d] = h[d];
  }
}

// ---------------------------------------------------------------------------
// DECODE row core: two-pass accumulate for the COMPLETE pool of request row
// ``r`` (max, then weighted sum under the fixed max, the current token's
// key/score substituted at slot ``pos % pool_size``). Fills acc/denom and
// returns true iff the pool is complete. Reads the tail PRE-update -- the
// caller owns the in-place tail write (the fp32 reference writes it after;
// the fp8 test compares it against the impl's in-place update on separate
// copies of ``tail_k``/``tail_score``).
// ---------------------------------------------------------------------------
inline bool decode_row_accumulate(
    int pool_size, int head_dim, int tail_size,
    const std::vector<float>& key, const std::vector<float>& slot_score,
    const std::vector<float>& tail_k, const std::vector<float>& tail_score,
    const std::vector<float>& ape, int n_reqs, int r,
    const std::vector<int32_t>& req_pool_indices,
    const std::vector<int32_t>& positions,
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& out_cache_loc,
    std::vector<float>& acc, std::vector<float>& denom) {
  const int req_raw = req_pool_indices[r];
  const bool req_valid = req_raw >= 0 && req_raw < n_reqs;
  const int req = req_valid ? req_raw : std::min(std::max(req_raw, 0), n_reqs - 1);
  const int pos_raw = positions[r];
  const int safe_pos = std::max(pos_raw, 0);
  const int seq_len = seq_lens[r];
  const int cache_loc = out_cache_loc[r];
  const bool pos_valid = req_valid && cache_loc != 0 && pos_raw >= 0 &&
                         pos_raw < seq_len;
  const int slot = safe_pos % pool_size;
  if (!(pos_valid && slot == pool_size - 1)) return false;

  const int pool_logical_start = safe_pos - slot;
  const float* cur_k = key.data() + (size_t)r * head_dim;
  const float* cur_s = slot_score.data() + (size_t)r * head_dim;
  std::vector<float> mx(head_dim, -kInf);
  for (int p = 0; p < pool_size; ++p) {
    const int phys = ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
    const float* sb = tail_score.data() + (size_t)req * tail_size * head_dim
                                        + (size_t)phys * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      const float s = (p == slot ? cur_s[d] : sb[d]) + ape[p * head_dim + d];
      mx[d] = std::max(mx[d], s);
    }
  }
  acc.assign(head_dim, 0.0f);
  denom.assign(head_dim, 0.0f);
  for (int p = 0; p < pool_size; ++p) {
    const int phys = ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
    const float* sb = tail_score.data() + (size_t)req * tail_size * head_dim
                                        + (size_t)phys * head_dim;
    const float* kb = tail_k.data() + (size_t)req * tail_size * head_dim
                                    + (size_t)phys * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      const float s = (p == slot ? cur_s[d] : sb[d]) + ape[p * head_dim + d];
      const float prob = std::exp(s - mx[d]);
      denom[d] += prob;
      acc[d] += (p == slot ? cur_k[d] : kb[d]) * prob;
    }
  }
  return true;
}

// Full DECODE reference: per-row accumulate + mean + dense Hadamard + fp32
// write, plus the unconditional in-place live-tail write. Mirrors
// dsa_kpool_decode_update_cpu exactly.
inline void decode_ref(int batch, int pool_size, int head_dim,
                       int tail_size, int slots_per_page, int block_table_cols,
                       int n_reqs, const std::vector<float>& key,
                       const std::vector<float>& slot_score,
                       std::vector<float>& tail_k,
                       std::vector<float>& tail_score,
                       const std::vector<float>& ape,
                       const std::vector<int32_t>& block_tables,
                       const std::vector<int32_t>& req_pool_indices,
                       const std::vector<int32_t>& positions,
                       const std::vector<int32_t>& seq_lens,
                       const std::vector<int32_t>& out_cache_loc,
                       std::vector<float>& out) {
  std::vector<float> acc, denom, mean(head_dim), h(128);
  for (int r = 0; r < batch; ++r) {
    const bool complete = decode_row_accumulate(
        pool_size, head_dim, tail_size, key, slot_score, tail_k, tail_score,
        ape, n_reqs, r, req_pool_indices, positions, seq_lens, out_cache_loc,
        acc, denom);
    if (complete) {
      for (int d = 0; d < head_dim; ++d) mean[d] = acc[d] / denom[d];
      apply_dense_hadamard(mean, h);
      const int safe_pos = std::max(positions[r], 0);
      const int pool_id = safe_pos / pool_size;
      const int pool_page_group = pool_id / slots_per_page;
      int tpr = std::min(std::max(pool_page_group * pool_size, 0), block_table_cols - 1);
      const int packed_page = block_tables[(size_t)r * block_table_cols + tpr];
      const int loc_slot = pool_id % slots_per_page;
      float* dst = out.data() + (size_t)packed_page * slots_per_page * head_dim
                             + (size_t)loc_slot * head_dim;
      for (int d = 0; d < head_dim; ++d) dst[d] = h[d];
    }
    // Unconditional live-tail write (no-op when !pos_valid).
    const int req_raw = req_pool_indices[r];
    const bool req_valid = req_raw >= 0 && req_raw < n_reqs;
    const int req = req_valid ? req_raw : std::min(std::max(req_raw, 0), n_reqs - 1);
    const int pos_raw = positions[r];
    const int safe_pos = std::max(pos_raw, 0);
    const bool pos_valid = req_valid && out_cache_loc[r] != 0 &&
                           pos_raw >= 0 && pos_raw < seq_lens[r];
    if (pos_valid) {
      const int phys_slot = ((safe_pos % tail_size) + tail_size) % tail_size;
      float* tk = tail_k.data() + (size_t)req * tail_size * head_dim
                                 + (size_t)phys_slot * head_dim;
      float* ts = tail_score.data() + (size_t)req * tail_size * head_dim
                                     + (size_t)phys_slot * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        tk[d] = key[(size_t)r * head_dim + d];
        ts[d] = slot_score[(size_t)r * head_dim + d];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// fp8e4m3fn encoder + store (shared by the fp8 tests). Verbatim from the
// original test_dsa_kpool.cpp oracle -- the hand-checked fp8 cases pin
// THIS construction, so it must not be rewritten.
// ---------------------------------------------------------------------------

// bf16 round (RNE) + back to fp32, independent of the impl (uses the IEEE
// top-16-bits convention directly).
inline float ref_round_bf16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  uint32_t lsb = (u >> 16) & 1u;
  uint32_t rounded = u + 0x7FFFu + lsb;
  uint16_t b = static_cast<uint16_t>(rounded >> 16);
  uint32_t ext = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &ext, sizeof(f));
  return f;
}

// fp32 -> fp8e4m3fn RNE. bias=7, max finite=448 (0x7E), NaN=0x7F/0xFF,
// subnormal = mant * 2^(1-7) = mant * 2^-6 (exp=0). Independent construction
// using frexp + explicit RNE on the (2*frac-1)*8 mantissa; saturates to
// +/-448, NaN -> 0x7F.
inline uint8_t ref_f32_to_fp8e4m3fn(float fv) {
  uint32_t bits;
  std::memcpy(&bits, &fv, sizeof(bits));
  uint32_t sign = (bits >> 31) & 1u;
  float amag = std::fabs(fv);
  if (amag != amag) return 0x7Fu;            // NaN
  if (amag == 0.0f) return static_cast<uint8_t>(sign << 7);
  if (amag >= 448.0f) return static_cast<uint8_t>((sign << 7) | 0x7Eu);
  int e;
  float frac = std::frexp(amag, &e);          // [0.5, 1)
  int eu = e - 1;                             // 2*frac in [1,2)
  if (eu < -9) return static_cast<uint8_t>(sign << 7);  // rounds to 0
  // RNE of a value in [0, 8).
  auto rne = [](float x) -> uint32_t {
    float lo = std::floor(x);
    float rem = x - lo;
    if (rem < 0.5f) return static_cast<uint32_t>(lo);
    if (rem > 0.5f) return static_cast<uint32_t>(lo) + 1u;
    // tie -> nearest even
    uint32_t li = static_cast<uint32_t>(lo);
    return (li & 1u) ? li + 1u : li;
  };
  if (eu < -6) {                              // subnormal (exp stored 0)
    uint32_t mant = rne(amag * 64.0f);        // amag / 2^-6
    if (mant >= 8u) return static_cast<uint8_t>((sign << 7) | 0x08u);
    return static_cast<uint8_t>((sign << 7) | mant);
  }
  uint32_t mant = rne((2.0f * frac - 1.0f) * 8.0f);
  uint32_t stored_exp = static_cast<uint32_t>(eu + 7);
  if (mant >= 8u) { mant = 0u; stored_exp += 1u; }
  if (stored_exp == 15u && mant == 7u)        // NaN slot -> max finite
    return static_cast<uint8_t>((sign << 7) | 0x7Eu);
  if (stored_exp > 15u)
    return static_cast<uint8_t>((sign << 7) | 0x7Eu);
  return static_cast<uint8_t>((sign << 7) | (stored_exp << 3) | mant);
}

// Compute the fp8 K + scale for one pool row from precomputed acc/denom
// (bf16-rounded mean, dense Hadamard, bf16 round again, absmax/448 or
// power-of-two scale). The caller writes k_fp8/scale into the page layout.
inline void ref_store_fp8(const std::vector<float>& acc,
                          const std::vector<float>& denom, int head_dim,
                          bool round_scale, std::vector<uint8_t>& k_fp8,
                          float& scale) {
  std::vector<float> x(head_dim), h(128);
  for (int d = 0; d < head_dim; ++d)
    x[d] = ref_round_bf16(acc[d] / denom[d]);
  apply_dense_hadamard(x, h);
  for (int d = 0; d < head_dim; ++d) x[d] = ref_round_bf16(h[d]);
  float absmax = 0.0f;
  for (int d = 0; d < head_dim; ++d) absmax = std::max(absmax, std::fabs(x[d]));
  if (absmax < 1e-4f) absmax = 1e-4f;
  const float fp8_max_inv = 1.0f / 448.0f;
  scale = round_scale ? std::exp2(std::ceil(std::log2(absmax * fp8_max_inv)))
                      : absmax * fp8_max_inv;
  k_fp8.assign(head_dim, 0);
  if (scale == 0.0f) return;
  for (int d = 0; d < head_dim; ++d) {
    float q = x[d] / scale;
    if (q > 448.0f) q = 448.0f;
    else if (q < -448.0f) q = -448.0f;
    k_fp8[d] = ref_f32_to_fp8e4m3fn(q);
  }
}

}  // namespace dsa_kpool_ref
