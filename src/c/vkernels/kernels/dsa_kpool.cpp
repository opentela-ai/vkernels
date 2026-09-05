// vkernels/kernels/dsa_kpool.cpp -- host oracle for the DSA kpool-cache
// compress/write kernels (issue #60). See dsa_kpool.hpp for the computation
// and the bf16-storage / no-fp8e4nv design.
#include "vkernels/kernels/dsa_kpool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Normalized 128-point Hadamard (1/sqrt(128)), the exact stage decomposition
// sglang's kpool_fp8_index.py::_hadamard128 uses (involutory up to fp32
// rounding -- verified offline). Applied to a single 128-vector (the per-pool
// weighted mean), so this is the 1-D form mirroring the Triton kernel's
// reshape(64,2,1)->...->reshape(128) ladder. ``head_dim`` MUST be 128 (the
// GLM-5.3 / DeepSeek-V3 ``index_head_dim``); the loop is fully unrolled and
// bounded so a larger head_dim is a logic error caught up front.
inline void hadamard128(float* x) {
  struct Stage { int groups; int stride; };
  constexpr Stage kStages[] = {{64, 1}, {32, 2}, {16, 4}, {8, 8},
                               {4, 16}, {2, 32}, {1, 64}};
  for (const Stage st : kStages) {
    const int G = st.groups;
    const int S = st.stride;
    const int two_s = 2 * S;
    float tmp[128];
    for (int g = 0; g < G; ++g) {
      const int base = g * two_s;
      for (int k = 0; k < 2; ++k)
        for (int s = 0; s < S; ++s) {
          const float a = x[base + s];
          const float b = x[base + S + s];
          tmp[base + k * S + s] = (k == 0) ? (a + b) : (a - b);
        }
    }
    for (int i = 0; i < 128; ++i) x[i] = tmp[i];
  }
  constexpr float kScale = 0.08838834764831845f;  // 1/sqrt(128)
  for (int i = 0; i < 128; ++i) x[i] *= kScale;
}

// --- Issue #61: host bf16 / fp8e4m3 helpers (the CPU oracle is standalone; ---
// these mirror moe_device.hip's f32bits_to_bf16 / bf16_to_f32 and the OCP
// e4m3fn spec so device + oracle share one construction).
//
// bf16: round-to-nearest-even (identical to f32bits_to_bf16 in
// moe_device.hip) and zero-extend-back (bf16_to_f32). The kpool fp8 store
// rounds the mean AND the Hadamard output to bf16 first, matching sglang's
// `_hadamard_quantize_fp8` (.to(tl.bfloat16).to(tl.float32)) EXACTLY, so the
// fp32 oracle is a STRICT upper bound on the A100 bf16-intermediate store.
inline uint16_t f32bits_to_bf16_host(uint32_t bits) {
  uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;  // round-to-nearest-even
  return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_f32_host(uint16_t b) {
  float f;
  uint32_t u = static_cast<uint32_t>(b) << 16;  // zero-extend low 17 bits
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// Round a float value -> bf16 (RNE) -> back to float (the kpool fp8 store's
// `.to(bf16).to(fp32)` step).
inline float round_bf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bf16_to_f32_host(f32bits_to_bf16_host(bits));
}

// fp32 -> fp8e4m3fn round-to-nearest-even (no __nv_fp8 header; SM80 safe).
// Layout: 1 sign | 4 exponent (BIAS 7) | 3 mantissa. Max finite = 448
// (exp=15, mant=6); exp=15 & mant=7 (0x7F / 0xFF) = NaN. Subnormals
// (exp=0, mant>0) = mant * 2^(1-7) = mant * 2^-6; zero (exp=0, mant=0).
// Mirrors Triton's `tl.store(float8e4m3fn, fp32)` cast (LLVM RNE) so the
// CPU fp8 oracle is byte-for-byte the legacy cache the A100 graph writes.
inline uint8_t f32_to_fp8e4m3fn_rne(float fv) {
  uint32_t bits;
  std::memcpy(&bits, &fv, sizeof(bits));
  const uint32_t sign = (bits >> 31) & 1u;
  const uint32_t mag_bits = bits & 0x7FFFFFFFu;
  float amag;
  std::memcpy(&amag, &mag_bits, sizeof(amag));
  if (amag != amag) return 0x7Fu;                  // NaN -> +NaN
  if (amag == 0.0f) return static_cast<uint8_t>(sign << 7);
  if (amag > 448.0f) return static_cast<uint8_t>((sign << 7) | 0x7Eu);  // saturate
  int e;
  const float frac = std::frexp(amag, &e);  // frac in [0.5,1), amag = frac*2^e
  const int e_unbiased = e - 1;                  // 2*frac in [1,2), amag=(2*frac)*2^e_unbiased
  if (e_unbiased < -9) return static_cast<uint8_t>(sign << 7);  // rounds to 0
  auto round_mant = [](float mf) -> uint32_t {    // RNE of mf in [0,8)
    uint32_t fl = static_cast<uint32_t>(std::floor(mf));
    const float rem = mf - static_cast<float>(fl);
    if (rem < 0.5f) return fl;
    if (rem > 0.5f) return fl + 1u;
    return (fl & 1u) ? fl + 1u : fl;              // tie -> even
  };
  if (e_unbiased < -6) {                          // subnormal (stored exp=0)
    const uint32_t mant = round_mant(amag * 64.0f);  // amag / 2^-6 = amag*2^6
    if (mant >= 8u) return static_cast<uint8_t>((sign << 7) | 0x08u);  // -> smallest normal
    return static_cast<uint8_t>((sign << 7) | mant);
  }
  // normal: stored_exp = e_unbiased + 7 in [1, 15]; mant = round((2*frac-1)*8)
  uint32_t mant = round_mant((2.0f * frac - 1.0f) * 8.0f);
  uint32_t stored_exp = static_cast<uint32_t>(e_unbiased + 7);
  if (mant >= 8u) { mant = 0u; stored_exp += 1u; }  // carry into exponent
  if (stored_exp == 15u && mant == 7u) return static_cast<uint8_t>((sign << 7) | 0x7Eu);  // NaN slot -> max finite
  if (stored_exp > 15u) return static_cast<uint8_t>((sign << 7) | 0x7Eu);  // overflow -> max finite
  return static_cast<uint8_t>((sign << 7) | (stored_exp << 3) | mant);
}

// fp8e4m3fn -> fp32 (inverse; for tests). bias 7, max 448, NaN at 0x7F/0xFF,
// subnormal mant*2^-6. Mirrors the OCP e4m3fn spec (NOT fnuz: bias 7, max 448).
inline float fp8e4m3fn_to_f32(uint8_t b) {
  const uint32_t s = static_cast<uint32_t>(b >> 7) & 1u;
  const uint32_t e = static_cast<uint32_t>(b >> 3) & 0xFu;
  const uint32_t m = static_cast<uint32_t>(b) & 0x7u;
  float f;
  if (e == 15u && m == 7u) {  // NaN -> qNaN
    const uint32_t q = 0x7fc00000u;
    std::memcpy(&f, &q, sizeof(f));
    return f;
  }
  if (e == 0u) {
    if (m == 0u) return s ? -0.0f : 0.0f;
    const float v = static_cast<float>(m) * 0x1p-6f;  // m * 2^(1-7)
    return s ? -v : v;
  }
  const uint32_t bits = (s << 31) | ((e + 119u) << 23) | (m << 20);  // e + (127-7)
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// fp8+scale store epilogue for ONE pool row (mirrors sglang's
// `_hadamard_quantize_fp8`): mean = round_bf16(acc/denom); x = round_bf16(
// hadamard128(mean)); absmax = max_d |x[d]| floored at 1e-4; scale =
// round_scale ? exp2(ceil(log2(absmax/448))) : absmax/448; k_fp8[d] =
// f32_to_fp8e4m3fn_rne(clamp(x[d]/scale, -448, 448)). An empty pool
// (all non-finite denom, e.g. pool_size==0 which is rejected up-front) writes
// all-zero K and a 0 scale. ``k_dst`` is the head_dim fp8 byte region;
// ``scale_dst`` (may be null) receives the single fp32 scale.
inline void store_kpool_fp8(const float* acc, const float* denom,
                            int head_dim, bool round_scale,
                            uint8_t* k_dst, float* scale_dst) {
  float x[128];
  bool any_finite = false;
  for (int d = 0; d < head_dim; ++d) {
    if (denom[d] > 0.0f && std::isfinite(denom[d]) && std::isfinite(acc[d])) {
      x[d] = round_bf16(acc[d] / denom[d]);   // mean, rounded to bf16
      any_finite = true;
    } else {
      x[d] = 0.0f;
    }
  }
  if (head_dim == 128) hadamard128(x);         // in-place Hadamard
  for (int d = 0; d < head_dim; ++d) x[d] = round_bf16(x[d]);  // out -> bf16
  float absmax = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    const float a = std::fabs(x[d]);
    if (a > absmax) absmax = a;
  }
  if (absmax < 1e-4f) absmax = 1e-4f;          // floored (Triton max(absmax,1e-4))
  const float fp8_max_inv = 1.0f / 448.0f;
  float scale;
  if (!any_finite) {
    scale = 0.0f;                              // empty pool -> 0 scale
  } else if (round_scale) {
    scale = std::exp2(std::ceil(std::log2(absmax * fp8_max_inv)));
  } else {
    scale = absmax * fp8_max_inv;
  }
  if (scale_dst) *scale_dst = scale;
  if (scale == 0.0f) {                         // contract: all-zero K, 0 scale
    for (int d = 0; d < head_dim; ++d) k_dst[d] = 0u;
    return;
  }
  for (int d = 0; d < head_dim; ++d) {
    float q = x[d] / scale;
    if (q > 448.0f) q = 448.0f;
    else if (q < -448.0f) q = -448.0f;
    k_dst[d] = f32_to_fp8e4m3fn_rne(q);
  }
}

}  // namespace

bool dsa_kpool_group_topk_supported(int32_t group_topk) {
  switch (group_topk) {
  case 128:
  case 160:
  case 192:
  case 224:
  case 256:
  case 512:
    return true;
  default:
    return false;
  }
}

int32_t dsa_kpool_max_closed_pools(int32_t num_draft_tokens, int32_t pool_size) {
  if (num_draft_tokens <= 0 || pool_size <= 0) return 0;
  return (num_draft_tokens + pool_size - 1) / pool_size;
}


// ---------------------------------------------------------------------------
// Shared PREFILL/DECODE cores. The bf16 and fp8 oracles MUST compute the
// identical pooled vector -- the gather, the online softmax, and every write
// gate are the same computation; only the STORE differs. So the core is
// written ONCE, parameterized on a store callback receiving the completed
// pool's (acc, denom) and its cache slot, and each public function below is
// a thin wrapper supplying its format's store. A drift between the two
// caches is now a compile/link-time impossibility rather than a review
// discipline.
// ---------------------------------------------------------------------------

template <class Store>
void assemble_impl(int n_pools, int pool_size, int head_dim, int tail_size,
                   int slots_per_page, int num_pages, int num_chunks,
                   int n_reqs, const float* chunk_k, const float* chunk_score,
                   const float* tail_k, const float* tail_score,
                   const float* ape, const int32_t* req_pool_idx,
                   const int32_t* n_from_tail, const int32_t* chunk_src_start,
                   const int32_t* tail_logical_base, const int32_t* loc,
                   const int32_t* write_mask, const char* who, Store store) {
  VK_EXPECTS(pool_size > 0 && head_dim == 128 && tail_size > 0 &&
                 slots_per_page > 0 && num_pages > 0 && num_chunks >= 0 &&
                 n_reqs >= 0,
             std::string(who) + ": positive pool_size/tail_size/"
                                "slots_per_page/num_pages, head_dim==128, "
                                "non-negative counts");
  VK_EXPECTS((num_chunks == 0 || (chunk_k && chunk_score)) &&
                 (n_reqs == 0 || (tail_k && tail_score)) &&
                 ape && req_pool_idx && n_from_tail && chunk_src_start &&
                 tail_logical_base && loc,
             std::string(who) + ": non-null tensors when n_pools>0");
  const int total_slots = num_pages * slots_per_page;
  for (int r = 0; r < n_pools; ++r) {
    if (write_mask && write_mask[r] == 0) continue;  // early return row
    const int req = static_cast<int>(req_pool_idx[r]);
    const int n_tail = static_cast<int>(n_from_tail[r]);
    const int chunk_src = static_cast<int>(chunk_src_start[r]);
    const int tail_base = static_cast<int>(tail_logical_base[r]);
    const int lr = static_cast<int>(loc[r]);
    VK_EXPECTS(req >= 0 && req < n_reqs,
               std::string(who) + ": req_pool_idx out of range");
    VK_EXPECTS(n_tail >= 0 && n_tail <= pool_size,
               std::string(who) + ": n_from_tail out of range [0, pool_size)");
    const int n_chunk = pool_size - n_tail;
    if (n_chunk > 0)
      VK_EXPECTS(chunk_src >= 0 && chunk_src + n_chunk - 1 < num_chunks,
                 std::string(who) + ": chunk_src_start out of range");
    VK_EXPECTS(lr >= 0 && lr < total_slots,
               std::string(who) + ": loc out of range [0, num_pages*ssp)");

    // Gather the pool's scores/keys.
    std::vector<float> scores(pool_size * head_dim);
    std::vector<float> keys(pool_size * head_dim);
    for (int slot = 0; slot < pool_size; ++slot) {
      const float* src_score;
      const float* src_key;
      if (slot < n_tail) {
        const int phys = ((tail_base + slot) % tail_size + tail_size) % tail_size;
        src_score = tail_score + (size_t)req * tail_size * head_dim
                                + (size_t)phys * head_dim;
        src_key = tail_k + (size_t)req * tail_size * head_dim
                          + (size_t)phys * head_dim;
      } else {
        const int ci = chunk_src + (slot - n_tail);
        src_score = chunk_score + (size_t)ci * head_dim;
        src_key = chunk_k + (size_t)ci * head_dim;
      }
      const float* a = ape + slot * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        scores[slot * head_dim + d] = src_score[d] + a[d];
        keys[slot * head_dim + d] = src_key[d];
      }
    }
    // Online (FlashAttention-style) softmax: running max, weight-sum
    // denominator and weighted accumulator across the pool's slots.
    std::vector<float> m(head_dim, kNegInf);
    std::vector<float> acc(head_dim, 0.0f);
    std::vector<float> denom(head_dim, 0.0f);
    for (int slot = 0; slot < pool_size; ++slot) {
      const float* s = scores.data() + slot * head_dim;
      const float* k = keys.data() + slot * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        const float score = s[d];
        const float new_m = std::max(m[d], score);
        const float rescale = std::exp(m[d] - new_m);
        const float prob = std::exp(score - new_m);
        denom[d] = denom[d] * rescale + prob;
        acc[d] = acc[d] * rescale + k[d] * prob;
        m[d] = new_m;
      }
    }
    const int page = lr / slots_per_page;
    const int slot_in_page = lr % slots_per_page;
    store(page, slot_in_page, acc.data(), denom.data());
  }
}

template <class Store>
void decode_impl(int batch, int pool_size, int head_dim, int tail_size,
                 int slots_per_page, int block_table_cols, int n_reqs,
                 int num_pages, const float* key, const float* slot_score,
                 float* tail_k, float* tail_score, const float* ape,
                 const int32_t* block_tables, const int32_t* req_pool_indices,
                 const int32_t* positions, const int32_t* seq_lens,
                 const int32_t* out_cache_loc, const char* who, Store store) {
  VK_EXPECTS(pool_size > 0 && head_dim == 128 && tail_size > 0 &&
                 slots_per_page > 0 && block_table_cols > 0 && n_reqs >= 0 &&
                 num_pages >= 0,
             std::string(who) + ": positive pool_size/tail_size/"
                                "slots_per_page/block_table_cols, head_dim=="
                                "128, non-negative counts");
  VK_EXPECTS(key && slot_score && tail_k && tail_score && ape && block_tables &&
                 req_pool_indices && positions && seq_lens && out_cache_loc,
             std::string(who) + ": non-null tensors when batch>0");
  for (int r = 0; r < batch; ++r) {
    const int req_raw = static_cast<int>(req_pool_indices[r]);
    const bool req_valid = req_raw >= 0 && req_raw < n_reqs;
    const int req = req_valid ? req_raw
                              : std::min(std::max(req_raw, 0), n_reqs - 1);
    const int pos_raw = static_cast<int>(positions[r]);
    const int safe_pos = std::max(pos_raw, 0);
    const int seq_len = static_cast<int>(seq_lens[r]);
    const int cache_loc = static_cast<int>(out_cache_loc[r]);
    const bool pos_valid = req_valid && cache_loc != 0 && pos_raw >= 0 &&
                           pos_raw < seq_len;
    const int slot = safe_pos % pool_size;
    const int phys_slot = ((safe_pos % tail_size) + tail_size) % tail_size;

    const float* cur_k = key + (size_t)r * head_dim;
    const float* cur_s = slot_score + (size_t)r * head_dim;

    // Compress the COMPLETE pool (slot == pool_size-1 and pos_valid). The
    // two-pass form is what the device kernel does (max, then weighted sum
    // under the fixed max, current token substituted).
    if (pos_valid && slot == pool_size - 1) {
      const int pool_logical_start = safe_pos - slot;
      std::vector<float> max_score(head_dim, kNegInf);
      for (int p = 0; p < pool_size; ++p) {
        const bool is_current = (p == slot);
        const int phys =
            ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
        const float* s_buf = tail_score + (size_t)req * tail_size * head_dim
                                          + (size_t)phys * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          const float s = (is_current ? cur_s[d] : s_buf[d]) + ape[p * head_dim + d];
          max_score[d] = std::max(max_score[d], s);
        }
      }
      std::vector<float> acc(head_dim, 0.0f), denom(head_dim, 0.0f);
      for (int p = 0; p < pool_size; ++p) {
        const bool is_current = (p == slot);
        const int phys =
            ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
        const float* s_buf = tail_score + (size_t)req * tail_size * head_dim
                                          + (size_t)phys * head_dim;
        const float* k_buf = tail_k + (size_t)req * tail_size * head_dim
                                      + (size_t)phys * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          const float s = (is_current ? cur_s[d] : s_buf[d]) + ape[p * head_dim + d];
          const float prob = std::exp(s - max_score[d]);
          denom[d] += prob;
          acc[d] += (is_current ? cur_k[d] : k_buf[d]) * prob;
        }
      }
      const int pool_id = safe_pos / pool_size;
      const int pool_page_group = pool_id / slots_per_page;
      int token_page_row = pool_page_group * pool_size;
      token_page_row = std::min(std::max(token_page_row, 0), block_table_cols - 1);
      const int packed_page =
          static_cast<int>(block_tables[(size_t)r * block_table_cols +
                                        token_page_row]);
      const int loc_slot = pool_id % slots_per_page;
      VK_EXPECTS(packed_page >= 0 && packed_page < num_pages,
                 std::string(who) + ": block_tables entry out of range "
                                    "[0, num_pages)");
      store(packed_page, loc_slot, acc.data(), denom.data());
    }

    // Unconditional live-tail write (masked to a no-op when !pos_valid).
    if (pos_valid) {
      float* tk_dst = tail_k + (size_t)req * tail_size * head_dim
                              + (size_t)phys_slot * head_dim;
      float* ts_dst = tail_score + (size_t)req * tail_size * head_dim
                                  + (size_t)phys_slot * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        tk_dst[d] = cur_k[d];
        ts_dst[d] = cur_s[d];
      }
    }
  }
}

void dsa_kpool_assemble_cpu(int n_pools, int pool_size, int head_dim,
                            int tail_size, int slots_per_page, int num_pages,
                            int num_chunks, int n_reqs,
                            const float* chunk_k, const float* chunk_score,
                            const float* tail_k, const float* tail_score,
                            const float* ape,
                            const int32_t* req_pool_idx,
                            const int32_t* n_from_tail,
                            const int32_t* chunk_src_start,
                            const int32_t* tail_logical_base,
                            const int32_t* loc, const int32_t* write_mask,
                            float* out) {
  if (n_pools <= 0) return;
  VK_EXPECTS(out, "dsa_kpool_assemble: non-null out when n_pools>0");
  assemble_impl(n_pools, pool_size, head_dim, tail_size, slots_per_page,
                num_pages, num_chunks, n_reqs, chunk_k, chunk_score, tail_k,
                tail_score, ape, req_pool_idx, n_from_tail, chunk_src_start,
                tail_logical_base, loc, write_mask, "dsa_kpool_assemble",
                [out, head_dim, slots_per_page](int page, int slot_in_page,
                                                const float* acc,
                                                const float* denom) {
                  float out_vec[128];
                  for (int d = 0; d < head_dim; ++d) out_vec[d] = acc[d] / denom[d];
                  if (head_dim == 128) hadamard128(out_vec);
                  float* dst = out + (size_t)page * slots_per_page * head_dim
                                     + (size_t)slot_in_page * head_dim;
                  for (int d = 0; d < head_dim; ++d) dst[d] = out_vec[d];
                });
}

void dsa_kpool_decode_update_cpu(int batch, int pool_size, int head_dim,
                                 int tail_size, int slots_per_page,
                                 int block_table_cols, int n_reqs,
                                 int num_pages,
                                 const float* key, const float* slot_score,
                                 float* tail_k, float* tail_score,
                                 const float* ape,
                                 const int32_t* block_tables,
                                 const int32_t* req_pool_indices,
                                 const int32_t* positions,
                                 const int32_t* seq_lens,
                                 const int32_t* out_cache_loc, float* out) {
  if (batch <= 0) return;
  VK_EXPECTS(out, "dsa_kpool_decode_update: non-null out when batch>0");
  decode_impl(batch, pool_size, head_dim, tail_size, slots_per_page,
              block_table_cols, n_reqs, num_pages, key, slot_score, tail_k,
              tail_score, ape, block_tables, req_pool_indices, positions,
              seq_lens, out_cache_loc, "dsa_kpool_decode_update",
              [out, head_dim, slots_per_page](int packed_page, int loc_slot,
                                              const float* acc,
                                              const float* denom) {
                float out_vec[128];
                for (int d = 0; d < head_dim; ++d) out_vec[d] = acc[d] / denom[d];
                if (head_dim == 128) hadamard128(out_vec);
                float* dst = out + (size_t)packed_page * slots_per_page * head_dim
                                   + (size_t)loc_slot * head_dim;
                for (int d = 0; d < head_dim; ++d) dst[d] = out_vec[d];
              });
}

// ---------------------------------------------------------------------------
// Issue #61: fp8+scale store variants. Identical cores to the bf16 oracles
// above -- ONLY the store differs: the compressed K is written as head_dim
// fp8e4m3 bytes + one fp32 scale into the legacy ``[num_pages, ssp*(128+4)]``
// uint8 cache (see dsa_kpool.hpp). The live-tail write is fp32, unchanged.
// ---------------------------------------------------------------------------
void dsa_kpool_assemble_fp8_cpu(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const float* chunk_k, const float* chunk_score, const float* tail_k,
    const float* tail_score, const float* ape, const int32_t* req_pool_idx,
    const int32_t* n_from_tail, const int32_t* chunk_src_start,
    const int32_t* tail_logical_base, const int32_t* loc,
    const int32_t* write_mask, uint8_t* cache_u8, float* round_scale_or_null) {
  if (n_pools <= 0) return;
  VK_EXPECTS(cache_u8, "dsa_kpool_assemble_fp8: non-null cache_u8");
  const bool round_scale = round_scale_or_null && *round_scale_or_null > 0.0f;
  const int page_bytes = slots_per_page * (head_dim + 4);  // K region + scale region
  const int scale_off_f32 = slots_per_page * head_dim / 4;  // scale region start (fp32 idx)
  assemble_impl(n_pools, pool_size, head_dim, tail_size, slots_per_page,
                num_pages, num_chunks, n_reqs, chunk_k, chunk_score, tail_k,
                tail_score, ape, req_pool_idx, n_from_tail, chunk_src_start,
                tail_logical_base, loc, write_mask, "dsa_kpool_assemble_fp8",
                [cache_u8, head_dim, slots_per_page, page_bytes, scale_off_f32,
                 round_scale](int page, int slot_in_page, const float* acc,
                              const float* denom) {
                  uint8_t* k_dst = cache_u8 + (size_t)page * page_bytes
                                   + (size_t)slot_in_page * head_dim;
                  float* scale_dst = reinterpret_cast<float*>(
                      cache_u8 + (size_t)page * page_bytes) + scale_off_f32
                      + slot_in_page;
                  store_kpool_fp8(acc, denom, head_dim, round_scale, k_dst,
                                  scale_dst);
                });
}

void dsa_kpool_decode_update_fp8_cpu(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const float* key, const float* slot_score, float* tail_k,
    float* tail_score, const float* ape, const int32_t* block_tables,
    const int32_t* req_pool_indices, const int32_t* positions,
    const int32_t* seq_lens, const int32_t* out_cache_loc, uint8_t* cache_u8,
    float* round_scale_or_null) {
  if (batch <= 0) return;
  VK_EXPECTS(cache_u8, "dsa_kpool_decode_update_fp8: non-null cache_u8");
  const bool round_scale = round_scale_or_null && *round_scale_or_null > 0.0f;
  const int page_bytes = slots_per_page * (head_dim + 4);
  const int scale_off_f32 = slots_per_page * head_dim / 4;
  decode_impl(batch, pool_size, head_dim, tail_size, slots_per_page,
              block_table_cols, n_reqs, num_pages, key, slot_score, tail_k,
              tail_score, ape, block_tables, req_pool_indices, positions,
              seq_lens, out_cache_loc, "dsa_kpool_decode_update_fp8",
              [cache_u8, head_dim, slots_per_page, page_bytes, scale_off_f32,
               round_scale](int packed_page, int loc_slot, const float* acc,
                            const float* denom) {
                uint8_t* k_dst = cache_u8 + (size_t)packed_page * page_bytes
                                 + (size_t)loc_slot * head_dim;
                float* scale_dst = reinterpret_cast<float*>(
                    cache_u8 + (size_t)packed_page * page_bytes) + scale_off_f32
                    + loc_slot;
                store_kpool_fp8(acc, denom, head_dim, round_scale, k_dst,
                                scale_dst);
              });
}

}  // namespace vkernels::kernels
