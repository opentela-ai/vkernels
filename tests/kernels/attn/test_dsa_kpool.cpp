// tests/kernels/attn/test_dsa_kpool.cpp
//
// Host tests for the DeepseekSparseAttn (DSA) kpool-cache compress/write
// kernels (issue #60): the prefill ``kpool_assemble_softmax_rotate_write_cache``
// path (``dsa_kpool_assemble_cpu``) and the decode
// ``kpool_decode_update_and_maybe_write_cache`` path
// (``dsa_kpool_decode_update_cpu``). Both are the FP8 e4m3 Triton kernels
// sglang JIT-fails on SM80 (A100); these native CPU oracles store bf16-free
// fp32 end-to-end and are the reference the gfx942 HIP kernel is checked
// against (see meta/benchmarks/test_dsa_kpool_correct.hip).
//
// Coverage:
//   * hand-checked cases (single-slot pool, two-equal-slot pool, decode
//     complete-pool write, decode live-tail update, decode not-complete)
//   * an independent dense Walsh-Hadamard reference for the full GLM-5.3
//     shape (H=128, pool_size in {4,8}, tail_size in {64,128}) across a
//     randomized sweep, isolating the indexing + softmax from the Hadamard
//     (the test builds the dense 128x128 normalized Hadamard once and
//     applies it; the oracle uses the staged decomposition -- they agree
//     to ~1e-7, verified offline)
//   * the write_mask gate, the pos_valid gate, the in-place live-tail write,
//     and the chunk/tail slot mixing (n_from_tail boundaries)
//   * null-arg / empty no-op / out-of-range edges
#include "minitest.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "vkernels/kernels/dsa_kpool.hpp"

using vkernels::kernels::dsa_kpool_assemble_cpu;
using vkernels::kernels::dsa_kpool_assemble_fp8_cpu;
using vkernels::kernels::dsa_kpool_decode_update_cpu;
using vkernels::kernels::dsa_kpool_decode_update_fp8_cpu;
using vkernels::kernels::dsa_kpool_group_topk_supported;
using vkernels::kernels::dsa_kpool_max_closed_pools;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Dense normalized 128-point Walsh-Hadamard (independent of the staged
// decomposition the oracle uses): H[i][j] = prod_bits (-1)^(bit_i & bit_j) /
// sqrt(128). Built once; the test applies it as a plain matvec so the only
// residual against the oracle is FMA-vs-mul-add (<<1e-4).
const std::vector<float>& dense_hadamard128() {
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

void apply_dense_hadamard(const std::vector<float>& v, std::vector<float>& out) {
  const std::vector<float>& H = dense_hadamard128();
  for (int i = 0; i < 128; ++i) {
    float s = 0;
    for (int j = 0; j < 128; ++j) s += H[(size_t)i * 128 + j] * v[j];
    out[i] = s;
  }
}

// Independent two-pass reference for the PREFILL path (different code path:
// build the full per-slot score, naive softmax, weighted sum, then the dense
// Hadamard). Mirrors dsa_kpool_assemble_cpu's contract exactly.
void assemble_ref(int n_pools, int pool_size, int head_dim, int tail_size,
                  int slots_per_page, int num_chunks, int n_reqs,
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
  (void)num_chunks; (void)n_reqs;
  for (int r = 0; r < n_pools; ++r) {
    if (write_mask && (*write_mask)[r] == 0) continue;
    const int req = req_pool_idx[r];
    const int n_tail = n_from_tail[r];
    const int chunk_src = chunk_src_start[r];
    const int tail_base = tail_logical_base[r];
    std::vector<float> acc(head_dim, 0.0f), denom(head_dim, 0.0f);
    // Two-pass naive: max, then exp/sum/weighted.
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
    std::vector<float> mean(head_dim), h(128);
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

// Independent two-pass reference for the DECODE path (max, then weighted sum
// under the fixed max, current token substituted; the dense Hadamard; plus
// the in-place live-tail write). Mirrors dsa_kpool_decode_update_cpu exactly.
void decode_ref(int batch, int pool_size, int head_dim, int tail_size,
                int slots_per_page, int block_table_cols, int n_reqs,
                const std::vector<float>& key,
                const std::vector<float>& slot_score,
                std::vector<float>& tail_k, std::vector<float>& tail_score,
                const std::vector<float>& ape,
                const std::vector<int32_t>& block_tables,
                const std::vector<int32_t>& req_pool_indices,
                const std::vector<int32_t>& positions,
                const std::vector<int32_t>& seq_lens,
                const std::vector<int32_t>& out_cache_loc,
                std::vector<float>& out) {
  for (int r = 0; r < batch; ++r) {
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
    const int phys_slot = ((safe_pos % tail_size) + tail_size) % tail_size;
    const float* cur_k = key.data() + (size_t)r * head_dim;
    const float* cur_s = slot_score.data() + (size_t)r * head_dim;
    if (pos_valid && slot == pool_size - 1) {
      const int pool_logical_start = safe_pos - slot;
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
      std::vector<float> acc(head_dim, 0.0f), denom(head_dim, 0.0f);
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
      std::vector<float> mean(head_dim), h(128);
      for (int d = 0; d < head_dim; ++d) mean[d] = acc[d] / denom[d];
      apply_dense_hadamard(mean, h);
      const int pool_id = safe_pos / pool_size;
      const int pool_page_group = pool_id / slots_per_page;
      int tpr = std::min(std::max(pool_page_group * pool_size, 0), block_table_cols - 1);
      const int packed_page = block_tables[(size_t)r * block_table_cols + tpr];
      const int loc_slot = pool_id % slots_per_page;
      float* dst = out.data() + (size_t)packed_page * slots_per_page * head_dim
                             + (size_t)loc_slot * head_dim;
      for (int d = 0; d < head_dim; ++d) dst[d] = h[d];
    }
    if (pos_valid) {
      float* tk = tail_k.data() + (size_t)req * tail_size * head_dim
                                 + (size_t)phys_slot * head_dim;
      float* ts = tail_score.data() + (size_t)req * tail_size * head_dim
                                     + (size_t)phys_slot * head_dim;
      for (int d = 0; d < head_dim; ++d) { tk[d] = cur_k[d]; ts[d] = cur_s[d]; }
    }
  }
}

}  // namespace

// ----------------------------------------------------------------------------
// Arithmetic helpers
// ----------------------------------------------------------------------------

TEST(DsaKpool, GroupTopkSupported) {
  EXPECT_TRUE(dsa_kpool_group_topk_supported(128));
  EXPECT_TRUE(dsa_kpool_group_topk_supported(160));
  EXPECT_TRUE(dsa_kpool_group_topk_supported(192));
  EXPECT_TRUE(dsa_kpool_group_topk_supported(224));
  EXPECT_TRUE(dsa_kpool_group_topk_supported(256));
  EXPECT_TRUE(dsa_kpool_group_topk_supported(512));
  EXPECT_FALSE(dsa_kpool_group_topk_supported(100));
  EXPECT_FALSE(dsa_kpool_group_topk_supported(0));
}

TEST(DsaKpool, MaxClosedPools) {
  EXPECT_EQ(dsa_kpool_max_closed_pools(0, 4), 0);
  EXPECT_EQ(dsa_kpool_max_closed_pools(1, 4), 1);
  EXPECT_EQ(dsa_kpool_max_closed_pools(4, 4), 1);
  EXPECT_EQ(dsa_kpool_max_closed_pools(5, 4), 2);
  EXPECT_EQ(dsa_kpool_max_closed_pools(64, 8), 8);
  EXPECT_EQ(dsa_kpool_max_closed_pools(-3, 4), 0);  // non-positive -> 0
}

// ----------------------------------------------------------------------------
// Hand-checked: PREFFILL assemble, single-slot pool. With pool_size=1 the
// softmax weight is 1 (one slot), so mean = K[0]. K[0] = e_0 (standard basis)
// and Hadamard128(e_0) is the FIRST column of the normalized Hadamard, which
// is all 1/sqrt(128) (~0.08838835).
// ----------------------------------------------------------------------------
TEST(DsaKpool, AssembleSingleSlotUnitVector) {
  constexpr int H = 128, pool_size = 1, tail_size = 1, ssp = 1,
                num_pages = 1, num_chunks = 0, n_reqs = 1, n_pools = 1;
  std::vector<float> chunk_k, chunk_score;  // unused (n_tail == pool_size)
  std::vector<float> tail_k((size_t)n_reqs * tail_size * H, 0.0f);
  std::vector<float> tail_score((size_t)n_reqs * tail_size * H, 0.0f);
  tail_k[0] = 1.0f;  // e_0
  std::vector<float> ape((size_t)pool_size * H, 0.0f);
  std::vector<int32_t> req_pool_idx = {0}, n_from_tail = {pool_size},
                       chunk_src_start = {0}, tail_logical_base = {0}, loc = {0};
  std::vector<float> out((size_t)num_pages * ssp * H, -1.0f);
  dsa_kpool_assemble_cpu(n_pools, pool_size, H, tail_size, ssp, num_pages,
                         num_chunks, n_reqs, chunk_k.data(), chunk_score.data(),
                         tail_k.data(), tail_score.data(), ape.data(),
                         req_pool_idx.data(), n_from_tail.data(),
                         chunk_src_start.data(), tail_logical_base.data(),
                         loc.data(), /*write_mask=*/nullptr, out.data());
  const float expect = 1.0f / std::sqrt(128.0f);
  for (int d = 0; d < H; ++d) EXPECT_NEAR(out[d], expect, 1e-6);
}

// PREFFILL: two equal slots (both from chunk), equal scores -> weights {1,1}/2,
// mean = (e_0 + e_0)/2 = e_0 -> Hadamard128(e_0) = all 1/sqrt(128).
TEST(DsaKpool, AssembleTwoEqualSlotsChunk) {
  constexpr int H = 128, pool_size = 2, tail_size = 4, ssp = 1,
                num_pages = 2, num_chunks = 2, n_reqs = 1, n_pools = 1;
  std::vector<float> chunk_k((size_t)num_chunks * H, 0.0f);
  std::vector<float> chunk_score((size_t)num_chunks * H, 0.0f);
  chunk_k[0] = 1.0f;  // chunk 0 = e_0
  chunk_k[H + 0] = 1.0f;  // chunk 1 = e_0
  std::vector<float> tail_k((size_t)n_reqs * tail_size * H, 0.0f);
  std::vector<float> tail_score((size_t)n_reqs * tail_size * H, 0.0f);
  std::vector<float> ape((size_t)pool_size * H, 0.0f);
  std::vector<int32_t> req_pool_idx = {0}, n_from_tail = {0},  // all chunk
                       chunk_src_start = {0}, tail_logical_base = {0}, loc = {0};
  std::vector<float> out((size_t)num_pages * ssp * H, -1.0f);
  dsa_kpool_assemble_cpu(n_pools, pool_size, H, tail_size, ssp, num_pages,
                         num_chunks, n_reqs, chunk_k.data(), chunk_score.data(),
                         tail_k.data(), tail_score.data(), ape.data(),
                         req_pool_idx.data(), n_from_tail.data(),
                         chunk_src_start.data(), tail_logical_base.data(),
                         loc.data(), nullptr, out.data());
  const float expect = 1.0f / std::sqrt(128.0f);
  for (int d = 0; d < H; ++d) EXPECT_NEAR(out[d], expect, 1e-6);
}

// PREFFILL: n_from_tail mixes chunk + tail; write_mask skips row 1. Two pools,
// row 0 written (loc=0), row 1 skipped (write_mask[1]=0) -> out at loc 1 untouched.
TEST(DsaKpool, AssembleWriteMaskSkipsRow) {
  constexpr int H = 128, pool_size = 1, tail_size = 1, ssp = 2,
                num_pages = 1, num_chunks = 0, n_reqs = 1, n_pools = 2;
  std::vector<float> tail_k((size_t)n_reqs * tail_size * H, 0.0f);
  std::vector<float> tail_score((size_t)n_reqs * tail_size * H, 0.0f);
  tail_k[0] = 1.0f;  // e_0 for both pools (only pool 0 written)
  std::vector<float> ape((size_t)pool_size * H, 0.0f);
  std::vector<int32_t> req_pool_idx = {0, 0}, n_from_tail = {1, 1},
                       chunk_src_start = {0, 0}, tail_logical_base = {0, 0},
                       loc = {0, 1}, write_mask = {1, 0};
  std::vector<float> out((size_t)num_pages * ssp * H, -7.0f);
  dsa_kpool_assemble_cpu(n_pools, pool_size, H, tail_size, ssp, num_pages,
                         num_chunks, n_reqs, nullptr, nullptr,
                         tail_k.data(), tail_score.data(), ape.data(),
                         req_pool_idx.data(), n_from_tail.data(),
                         chunk_src_start.data(), tail_logical_base.data(),
                         loc.data(), write_mask.data(), out.data());
  const float expect = 1.0f / std::sqrt(128.0f);
  for (int d = 0; d < H; ++d) EXPECT_NEAR(out[d], expect, 1e-6);   // slot 0 written
  for (int d = 0; d < H; ++d) EXPECT_NEAR(out[H + d], -7.0f, 0); // slot 1 untouched
}

// ----------------------------------------------------------------------------
// Hand-checked: DECODE complete single-pool write + live-tail update.
// pool_size=1 -> slot always 0 == pool_size-1 -> pool always complete.
// pos=0: pool_id=0, pool_page_group=0, tpr=0, packed_page=block_tables[0,0]=3,
// loc_slot=0 -> out[3,0,:]=Hadamard128(e_0). tail_k[0,0]=e_0, tail_score[0,0]=s.
// ----------------------------------------------------------------------------
TEST(DsaKpool, DecodeCompleteSinglePoolWrite) {
  constexpr int H = 128, pool_size = 1, tail_size = 4, ssp = 2,
                btc = 4, n_reqs = 1, num_pages = 8, batch = 1;
  std::vector<float> key((size_t)batch * H, 0.0f);
  std::vector<float> slot_score((size_t)batch * H, 0.0f);
  key[0] = 1.0f;  // e_0
  std::vector<float> tail_k((size_t)n_reqs * tail_size * H, -9.0f);
  std::vector<float> tail_score((size_t)n_reqs * tail_size * H, -9.0f);
  std::vector<float> ape((size_t)pool_size * H, 0.0f);
  std::vector<int32_t> block_tables = {3, 4, 5, 6};  // [1, btc]
  std::vector<int32_t> req_pool_indices = {0}, positions = {0},
                       seq_lens = {64}, out_cache_loc = {1};  // nonzero -> write
  std::vector<float> out((size_t)num_pages * ssp * H, -1.0f);
  dsa_kpool_decode_update_cpu(batch, pool_size, H, tail_size, ssp, btc, n_reqs,
                              num_pages, key.data(), slot_score.data(),
                              tail_k.data(), tail_score.data(), ape.data(),
                              block_tables.data(), req_pool_indices.data(),
                              positions.data(), seq_lens.data(),
                              out_cache_loc.data(), out.data());
  const float expect = 1.0f / std::sqrt(128.0f);
  const size_t base = (size_t)3 * ssp * H;  // page 3, slot 0
  for (int d = 0; d < H; ++d) EXPECT_NEAR(out[base + d], expect, 1e-6);
  // Live-tail update: tail_k[0, 0%4=0, :] = e_0, tail_score[0,0,:]=0.
  for (int d = 0; d < H; ++d) {
    EXPECT_NEAR(tail_k[d], (d == 0) ? 1.0f : 0.0f, 1e-6);
    EXPECT_NEAR(tail_score[d], 0.0f, 1e-6);
  }
  // Other tail slots untouched.
  for (int s = 1; s < tail_size; ++s)
    for (int d = 0; d < H; ++d)
      EXPECT_NEAR(tail_k[(size_t)s * H + d], -9.0f, 0);
}

// DECODE: pool NOT complete (pool_size=4, pos=2 -> slot=2 != 3) -> no
// compressed-K write, only the live-tail update at phys_slot = 2%tail_size.
TEST(DsaKpool, DecodeNotCompleteOnlyTailUpdate) {
  constexpr int H = 128, pool_size = 4, tail_size = 4, ssp = 2,
                btc = 4, n_reqs = 1, num_pages = 8, batch = 1;
  std::vector<float> key((size_t)batch * H, 0.0f);
  std::vector<float> slot_score((size_t)batch * H, 0.0f);
  key[0] = 2.0f;  // 2*e_0
  std::vector<float> tail_k((size_t)n_reqs * tail_size * H, -9.0f);
  std::vector<float> tail_score((size_t)n_reqs * tail_size * H, -9.0f);
  std::vector<float> ape((size_t)pool_size * H, 0.0f);
  std::vector<int32_t> block_tables = {3, 4, 5, 6};
  std::vector<int32_t> req_pool_indices = {0}, positions = {2},  // slot=2 != 3
                       seq_lens = {64}, out_cache_loc = {1};
  std::vector<float> out((size_t)num_pages * ssp * H, -1.0f);
  dsa_kpool_decode_update_cpu(batch, pool_size, H, tail_size, ssp, btc, n_reqs,
                              num_pages, key.data(), slot_score.data(),
                              tail_k.data(), tail_score.data(), ape.data(),
                              block_tables.data(), req_pool_indices.data(),
                              positions.data(), seq_lens.data(),
                              out_cache_loc.data(), out.data());
  // No compressed-K write anywhere.
  for (float x : out) EXPECT_NEAR(x, -1.0f, 0);
  // Live-tail update at phys_slot = 2%4 = 2.
  const size_t base = (size_t)2 * H;
  for (int d = 0; d < H; ++d) {
    EXPECT_NEAR(tail_k[base + d], (d == 0) ? 2.0f : 0.0f, 1e-6);
    EXPECT_NEAR(tail_score[base + d], 0.0f, 1e-6);
  }
  for (int s = 0; s < tail_size; ++s)
    if (s != 2)
      for (int d = 0; d < H; ++d)
        EXPECT_NEAR(tail_k[(size_t)s * H + d], -9.0f, 0);
}

// DECODE: invalid row (req=-1, cache_loc=0, pos=-1) -> NO write, NO tail
// update (pos_valid is false on every condition).
TEST(DsaKpool, DecodeInvalidRowIsNoOp) {
  constexpr int H = 128, pool_size = 1, tail_size = 4, ssp = 2,
                btc = 4, n_reqs = 1, num_pages = 8, batch = 1;
  std::vector<float> key((size_t)batch * H, 5.0f);
  std::vector<float> slot_score((size_t)batch * H, 5.0f);
  std::vector<float> tail_k((size_t)n_reqs * tail_size * H, -9.0f);
  std::vector<float> tail_score((size_t)n_reqs * tail_size * H, -9.0f);
  std::vector<float> ape((size_t)pool_size * H, 0.0f);
  std::vector<int32_t> block_tables = {3, 4, 5, 6};
  std::vector<int32_t> req_pool_indices = {-1}, positions = {-1},
                       seq_lens = {64}, out_cache_loc = {0};  // cache_loc==0
  std::vector<float> out((size_t)num_pages * ssp * H, -1.0f);
  dsa_kpool_decode_update_cpu(batch, pool_size, H, tail_size, ssp, btc, n_reqs,
                              num_pages, key.data(), slot_score.data(),
                              tail_k.data(), tail_score.data(), ape.data(),
                              block_tables.data(), req_pool_indices.data(),
                              positions.data(), seq_lens.data(),
                              out_cache_loc.data(), out.data());
  for (float x : out) EXPECT_NEAR(x, -1.0f, 0);
  for (float x : tail_k) EXPECT_NEAR(x, -9.0f, 0);
  for (float x : tail_score) EXPECT_NEAR(x, -9.0f, 0);
}

// ----------------------------------------------------------------------------
// MatchesReference: randomized sweep over the full GLM-5.3 shape and smaller
// sanity shapes, for BOTH paths. Tolerance 1e-4 (FMA-vs-mul-add over the
// 128-dim softmax + Hadamard).
// ----------------------------------------------------------------------------
TEST(DsaKpool, AssembleMatchesReference) {
  struct Cfg { int n_pools, pool_size, tail_size, ssp, num_pages, num_chunks,
                   n_reqs; };
  const Cfg cfgs[] = {
      {1, 1, 64, 8, 16, 32, 1},     // single pool, all tail
      {2, 4, 64, 8, 16, 32, 2},     // mixed tail/chunk
      {4, 8, 64, 8, 16, 64, 3},     // GLM-5.3-ish (pool_size=8)
      {3, 8, 128, 16, 8, 32, 2},    // larger tail
      {1, 2, 4, 1, 4, 8, 1},        // tiny
      {5, 4, 64, 8, 16, 48, 4},     // batch of pools
  };
  constexpr int H = 128;
  std::mt19937 rng(20260902);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  for (const auto& c : cfgs) {
    std::vector<float> chunk_k((size_t)c.num_chunks * H),
        chunk_score((size_t)c.num_chunks * H),
        tail_k((size_t)c.n_reqs * c.tail_size * H),
        tail_score((size_t)c.n_reqs * c.tail_size * H),
        ape((size_t)c.pool_size * H);
    for (auto& x : chunk_k) x = rf();
    for (auto& x : chunk_score) x = rf();
    for (auto& x : tail_k) x = rf();
    for (auto& x : tail_score) x = rf();
    for (auto& x : ape) x = rf();
    std::vector<int32_t> req_pool_idx(c.n_pools), n_from_tail(c.n_pools),
        chunk_src_start(c.n_pools), tail_logical_base(c.n_pools), loc(c.n_pools);
    for (int r = 0; r < c.n_pools; ++r) {
      req_pool_idx[r] = static_cast<int32_t>(rng() % c.n_reqs);
      n_from_tail[r] = static_cast<int32_t>(rng() % (c.pool_size + 1));  // [0, pool_size]
      const int n_chunk = c.pool_size - n_from_tail[r];
      chunk_src_start[r] = n_chunk > 0
          ? static_cast<int32_t>(rng() % (c.num_chunks - n_chunk + 1))
          : 0;
      tail_logical_base[r] = static_cast<int32_t>(rng() % c.tail_size);
      loc[r] = static_cast<int32_t>(rng() % (c.num_pages * c.ssp));
    }
    std::vector<float> out((size_t)c.num_pages * c.ssp * H, 0.0f);
    std::vector<float> rout((size_t)c.num_pages * c.ssp * H, 0.0f);
    dsa_kpool_assemble_cpu(c.n_pools, c.pool_size, H, c.tail_size, c.ssp,
                           c.num_pages, c.num_chunks, c.n_reqs,
                           chunk_k.data(), chunk_score.data(),
                           tail_k.data(), tail_score.data(), ape.data(),
                           req_pool_idx.data(), n_from_tail.data(),
                           chunk_src_start.data(), tail_logical_base.data(),
                           loc.data(), nullptr, out.data());
    assemble_ref(c.n_pools, c.pool_size, H, c.tail_size, c.ssp, c.num_chunks,
                 c.n_reqs, chunk_k, chunk_score, tail_k, tail_score, ape,
                 req_pool_idx, n_from_tail, chunk_src_start, tail_logical_base,
                 loc, nullptr, rout);
    float maxd = 0.0f;
    for (size_t i = 0; i < out.size(); ++i)
      maxd = std::max(maxd, std::fabs(out[i] - rout[i]));
    EXPECT_NEAR(maxd, 0.0f, 1e-4f);
  }
}

TEST(DsaKpool, DecodeMatchesReference) {
  struct Cfg { int batch, pool_size, tail_size, ssp, btc, n_reqs, num_pages; };
  const Cfg cfgs[] = {
      {1, 4, 64, 8, 8, 2, 32},
      {2, 4, 64, 8, 8, 2, 32},     // batch > 1
      {3, 8, 64, 8, 8, 3, 32},     // pool_size=8
      {2, 8, 128, 16, 16, 2, 64},  // larger
      {1, 2, 4, 1, 4, 1, 8},       // tiny
      {4, 4, 64, 8, 8, 2, 32},     // several rows
  };
  constexpr int H = 128;
  std::mt19937 rng(82822);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  for (const auto& c : cfgs) {
    std::vector<float> key((size_t)c.batch * H), slot_score((size_t)c.batch * H),
        tail_k((size_t)c.n_reqs * c.tail_size * H),
        tail_score((size_t)c.n_reqs * c.tail_size * H),
        ape((size_t)c.pool_size * H);
    for (auto& x : key) x = rf();
    for (auto& x : slot_score) x = rf();
    for (auto& x : tail_k) x = rf();
    for (auto& x : tail_score) x = rf();
    for (auto& x : ape) x = rf();
    std::vector<int32_t> block_tables((size_t)c.batch * c.btc),
        req_pool_indices(c.batch), positions(c.batch), seq_lens(c.batch),
        out_cache_loc(c.batch);
    for (int r = 0; r < c.batch; ++r)
      for (int b = 0; b < c.btc; ++b)
        block_tables[(size_t)r * c.btc + b] = static_cast<int32_t>(rng() % c.num_pages);
    for (int r = 0; r < c.batch; ++r) {
      // ~half valid (complete on last slot), ~half invalid to exercise gating.
      const bool valid = (rng() & 1) == 0;
      req_pool_indices[r] = valid ? static_cast<int32_t>(rng() % c.n_reqs) : -1;
      seq_lens[r] = c.pool_size * (1 + static_cast<int32_t>(rng() % 8));
      positions[r] = valid ? static_cast<int32_t>(rng() % seq_lens[r]) : -1;
      out_cache_loc[r] = valid ? static_cast<int32_t>(1 + rng() % 16) : 0;
    }
    std::vector<float> out((size_t)c.num_pages * c.ssp * H, 0.0f);
    std::vector<float> rout((size_t)c.num_pages * c.ssp * H, 0.0f);
    std::vector<float> tail_k_ref = tail_k, tail_score_ref = tail_score;
    dsa_kpool_decode_update_cpu(c.batch, c.pool_size, H, c.tail_size, c.ssp,
                                c.btc, c.n_reqs, c.num_pages, key.data(),
                                slot_score.data(), tail_k.data(),
                                tail_score.data(), ape.data(),
                                block_tables.data(), req_pool_indices.data(),
                                positions.data(), seq_lens.data(),
                                out_cache_loc.data(), out.data());
    decode_ref(c.batch, c.pool_size, H, c.tail_size, c.ssp, c.btc, c.n_reqs,
               key, slot_score, tail_k_ref, tail_score_ref,
               ape, block_tables, req_pool_indices, positions, seq_lens,
               out_cache_loc, rout);
    float maxd = 0.0f;
    for (size_t i = 0; i < out.size(); ++i)
      maxd = std::max(maxd, std::fabs(out[i] - rout[i]));
    EXPECT_NEAR(maxd, 0.0f, 1e-4f);
    // In-place tail update must match too.
    float maxtk = 0.0f, maxts = 0.0f;
    for (size_t i = 0; i < tail_k.size(); ++i)
      maxtk = std::max(maxtk, std::fabs(tail_k[i] - tail_k_ref[i]));
    for (size_t i = 0; i < tail_score.size(); ++i)
      maxts = std::max(maxts, std::fabs(tail_score[i] - tail_score_ref[i]));
    EXPECT_NEAR(maxtk, 0.0f, 1e-4f);
    EXPECT_NEAR(maxts, 0.0f, 1e-4f);
  }
}

// ----------------------------------------------------------------------------
// Edge cases: null args rejected when there is work; empty is a no-op;
// out-of-range indices rejected.
// ----------------------------------------------------------------------------
TEST(DsaKpool, AssembleNullArgsThrow) {
  std::vector<float> ape(128, 0.0f);
  EXPECT_THROW(dsa_kpool_assemble_cpu(1, 1, 128, 1, 1, 1, 0, 1, nullptr,
                                      nullptr, nullptr, nullptr, ape.data(),
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr),
               std::invalid_argument);
}

TEST(DsaKpool, AssembleEmptyIsNoOp) {
  std::vector<float> out(128, -1.0f);
  dsa_kpool_assemble_cpu(0, 1, 128, 1, 1, 1, 0, 1, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, out.data());
  for (float x : out) EXPECT_NEAR(x, -1.0f, 0);  // untouched
}

TEST(DsaKpool, AssembleOutOfRangeThrows) {
  std::vector<float> chunk_k(128), chunk_score(128), tail_k(128), tail_score(128),
      ape(128, 0.0f);
  std::vector<int32_t> req = {1}, nt = {0}, css = {0}, tlb = {0}, loc = {0};
  // req_pool_idx=1 >= n_reqs=1 -> throw.
  EXPECT_THROW(dsa_kpool_assemble_cpu(1, 1, 128, 1, 1, 1, 1, 1, chunk_k.data(),
                                      chunk_score.data(), tail_k.data(),
                                      tail_score.data(), ape.data(), req.data(),
                                      nt.data(), css.data(), tlb.data(),
                                      loc.data(), nullptr, chunk_k.data()),
               std::invalid_argument);
  std::vector<int32_t> loc_oob = {5};  // >= num_pages*ssp=1
  std::vector<int32_t> req0 = {0};
  EXPECT_THROW(dsa_kpool_assemble_cpu(1, 1, 128, 1, 1, 1, 1, 1, chunk_k.data(),
                                      chunk_score.data(), tail_k.data(),
                                      tail_score.data(), ape.data(), req0.data(),
                                      nt.data(), css.data(), tlb.data(),
                                      loc_oob.data(), nullptr, chunk_k.data()),
               std::invalid_argument);
}

TEST(DsaKpool, DecodeNullArgsThrow) {
  std::vector<float> ape(128, 0.0f);
  EXPECT_THROW(dsa_kpool_decode_update_cpu(1, 1, 128, 1, 1, 1, 1, 1, nullptr,
                                           nullptr, nullptr, nullptr, ape.data(),
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr),
               std::invalid_argument);
}

TEST(DsaKpool, DecodeEmptyIsNoOp) {
  std::vector<float> out(128, -1.0f);
  dsa_kpool_decode_update_cpu(0, 1, 128, 1, 1, 1, 1, 1, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, out.data());
  for (float x : out) EXPECT_NEAR(x, -1.0f, 0);
}

TEST(DsaKpool, DecodeBlockTablesOutOfRangeThrows) {
  std::vector<float> key(128, 0), slot_score(128, 0), tail_k(128, -9),
      tail_score(128, -9), ape(128, 0);
  key[0] = 1.0f;
  std::vector<int32_t> bt = {9}, rpi = {0}, pos = {0}, sl = {64}, ocl = {1};
  // bt[0]=9 >= num_pages=8 -> the complete-pool write must throw.
  EXPECT_THROW(dsa_kpool_decode_update_cpu(1, 1, 128, 1, 2, 1, 1, 8, key.data(),
                                           slot_score.data(), tail_k.data(),
                                           tail_score.data(), ape.data(),
                                           bt.data(), rpi.data(), pos.data(),
                                           sl.data(), ocl.data(), key.data()),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Issue #61: fp8+scale store epilogue. Independent reference + parity tests
// for ``dsa_kpool_assemble_fp8_cpu`` / ``dsa_kpool_decode_update_fp8_cpu``.
// The fp8 encoder below is structurally DIFFERENT from the implementation
// (explicit float-bit unpacking via std::frexp + a hand-rolled RNE rounding
// table) so a shared bug in either would show up. The cache layout mirrors
// dsa_kpool.hpp: [num_pages, ssp*(128+4)] uint8, K region [0, ssp*128),
// scale region [ssp*128, ssp*128+ssp*4), one fp32 per slot.
// ---------------------------------------------------------------------------

// bf16 round (RNE) + back to fp32, independent of the impl (uses the IEEE
// top-16-bits convention directly).
float ref_round_bf16(float v) {
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
uint8_t ref_f32_to_fp8e4m3fn(float fv) {
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

// Compute the fp8 K + scale for one pool row from precomputed acc/denom.
void ref_store_fp8(const std::vector<float>& acc,
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

TEST(DsaKpoolFp8, AssembleMatchesReference) {
  struct Cfg { int n_pools, pool_size, tail_size, ssp, num_pages, num_chunks,
                   n_reqs; };
  const Cfg cfgs[] = {
      {1, 1, 64, 8, 16, 32, 1}, {2, 4, 64, 8, 16, 32, 2},
      {4, 8, 64, 8, 16, 64, 3}, {3, 8, 128, 16, 8, 32, 2},
      {1, 2, 4, 1, 4, 8, 1},     {5, 4, 64, 8, 16, 48, 4},
  };
  constexpr int H = 128;
  for (int round_mode = 0; round_mode <= 1; ++round_mode) {
    const bool round_scale = (round_mode == 1);
    std::mt19937 rng(20260904 + round_mode);
    auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
    for (const auto& c : cfgs) {
      std::vector<float> chunk_k((size_t)c.num_chunks * H),
          chunk_score((size_t)c.num_chunks * H),
          tail_k((size_t)c.n_reqs * c.tail_size * H),
          tail_score((size_t)c.n_reqs * c.tail_size * H),
          ape((size_t)c.pool_size * H);
      for (auto& x : chunk_k) x = rf();
      for (auto& x : chunk_score) x = rf();
      for (auto& x : tail_k) x = rf();
      for (auto& x : tail_score) x = rf();
      for (auto& x : ape) x = rf();
      std::vector<int32_t> req_pool_idx(c.n_pools), n_from_tail(c.n_pools),
          chunk_src_start(c.n_pools), tail_logical_base(c.n_pools),
          loc(c.n_pools);
      for (int r = 0; r < c.n_pools; ++r) {
        req_pool_idx[r] = static_cast<int32_t>(rng() % c.n_reqs);
        n_from_tail[r] = static_cast<int32_t>(rng() % (c.pool_size + 1));
        const int n_chunk = c.pool_size - n_from_tail[r];
        chunk_src_start[r] = n_chunk > 0
            ? static_cast<int32_t>(rng() % (c.num_chunks - n_chunk + 1))
            : 0;
        tail_logical_base[r] = static_cast<int32_t>(rng() % c.tail_size);
        loc[r] = static_cast<int32_t>(rng() % (c.num_pages * c.ssp));
      }
      const int page_bytes = c.ssp * (H + 4);
      std::vector<uint8_t> out((size_t)c.num_pages * page_bytes, 0u);
      float rs = round_scale ? 1.0f : 0.0f;
      dsa_kpool_assemble_fp8_cpu(c.n_pools, c.pool_size, H, c.tail_size, c.ssp,
                                 c.num_pages, c.num_chunks, c.n_reqs,
                                 chunk_k.data(), chunk_score.data(),
                                 tail_k.data(), tail_score.data(), ape.data(),
                                 req_pool_idx.data(), n_from_tail.data(),
                                 chunk_src_start.data(), tail_logical_base.data(),
                                 loc.data(), nullptr, out.data(), &rs);
      int mism = 0;
      for (int r = 0; r < c.n_pools; ++r) {
        const int req = req_pool_idx[r];
        const int n_tail = n_from_tail[r];
        const int chunk_src = chunk_src_start[r];
        const int tail_base = tail_logical_base[r];
        const int lr = loc[r];
        std::vector<float> sc(c.pool_size * H);
        for (int slot = 0; slot < c.pool_size; ++slot) {
          const float* sp;
          if (slot < n_tail) {
            const int phys = ((tail_base + slot) % c.tail_size + c.tail_size) % c.tail_size;
            sp = tail_score.data() + (size_t)req * c.tail_size * H + (size_t)phys * H;
          } else {
            const int ci = chunk_src + (slot - n_tail);
            sp = chunk_score.data() + (size_t)ci * H;
          }
          for (int d = 0; d < H; ++d)
            sc[slot * H + d] = sp[d] + ape[slot * H + d];
        }
        std::vector<float> mx(H, -kInf);
        for (int slot = 0; slot < c.pool_size; ++slot)
          for (int d = 0; d < H; ++d) mx[d] = std::max(mx[d], sc[slot * H + d]);
        std::vector<float> acc(H, 0.0f), denom(H, 0.0f);
        for (int slot = 0; slot < c.pool_size; ++slot) {
          const float* kp;
          if (slot < n_tail) {
            const int phys = ((tail_base + slot) % c.tail_size + c.tail_size) % c.tail_size;
            kp = tail_k.data() + (size_t)req * c.tail_size * H + (size_t)phys * H;
          } else {
            const int ci = chunk_src + (slot - n_tail);
            kp = chunk_k.data() + (size_t)ci * H;
          }
          for (int d = 0; d < H; ++d) {
            const float prob = std::exp(sc[slot * H + d] - mx[d]);
            denom[d] += prob;
            acc[d] += kp[d] * prob;
          }
        }
        std::vector<uint8_t> k_ref(H);
        float scale_ref = 0.0f;
        ref_store_fp8(acc, denom, H, round_scale, k_ref, scale_ref);
        const int page = lr / c.ssp;
        const int sip = lr % c.ssp;
        const uint8_t* k_dst = out.data() + (size_t)page * page_bytes + (size_t)sip * H;
        const float* scale_dst = reinterpret_cast<const float*>(
            out.data() + (size_t)page * page_bytes) + (c.ssp * H / 4) + sip;
        for (int d = 0; d < H; ++d)
          if (k_dst[d] != k_ref[d]) ++mism;
        if (std::fabs(*scale_dst - scale_ref) > 0.0f) ++mism;
      }
      EXPECT_EQ(mism, 0);
    }
  }
}

TEST(DsaKpoolFp8, DecodeMatchesReference) {
  struct Cfg { int batch, pool_size, tail_size, ssp, btc, n_reqs, num_pages; };
  const Cfg cfgs[] = {
      {1, 4, 64, 8, 8, 2, 32}, {2, 4, 64, 8, 8, 2, 32},
      {3, 8, 64, 8, 8, 3, 32}, {2, 8, 128, 16, 16, 2, 64},
      {1, 2, 4, 1, 4, 1, 8},   {4, 4, 64, 8, 8, 2, 32},
  };
  constexpr int H = 128;
  for (int round_mode = 0; round_mode <= 1; ++round_mode) {
    const bool round_scale = (round_mode == 1);
    std::mt19937 rng(82822 + round_mode);
    auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
    for (const auto& c : cfgs) {
      std::vector<float> key((size_t)c.batch * H),
          slot_score((size_t)c.batch * H),
          tail_k((size_t)c.n_reqs * c.tail_size * H),
          tail_score((size_t)c.n_reqs * c.tail_size * H),
          ape((size_t)c.pool_size * H);
      for (auto& x : key) x = rf();
      for (auto& x : slot_score) x = rf();
      for (auto& x : tail_k) x = rf();
      for (auto& x : tail_score) x = rf();
      for (auto& x : ape) x = rf();
      std::vector<int32_t> block_tables((size_t)c.batch * c.btc),
          req_pool_indices(c.batch), positions(c.batch), seq_lens(c.batch),
          out_cache_loc(c.batch);
      for (int r = 0; r < c.batch; ++r)
        for (int b = 0; b < c.btc; ++b)
          block_tables[(size_t)r * c.btc + b] =
              static_cast<int32_t>(rng() % c.num_pages);
      for (int r = 0; r < c.batch; ++r) {
        const bool valid = (rng() & 1) == 0;
        req_pool_indices[r] = valid ? static_cast<int32_t>(rng() % c.n_reqs) : -1;
        seq_lens[r] = c.pool_size * (1 + static_cast<int32_t>(rng() % 8));
        positions[r] = valid ? static_cast<int32_t>(rng() % seq_lens[r]) : -1;
        out_cache_loc[r] = valid ? static_cast<int32_t>(1 + rng() % 16) : 0;
      }
      const int page_bytes = c.ssp * (H + 4);
      std::vector<uint8_t> out((size_t)c.num_pages * page_bytes, 0u);
      std::vector<float> tail_k_ref = tail_k, tail_score_ref = tail_score;
      float rs = round_scale ? 1.0f : 0.0f;
      dsa_kpool_decode_update_fp8_cpu(
          c.batch, c.pool_size, H, c.tail_size, c.ssp, c.btc, c.n_reqs,
          c.num_pages, key.data(), slot_score.data(), tail_k.data(),
          tail_score.data(), ape.data(), block_tables.data(),
          req_pool_indices.data(), positions.data(), seq_lens.data(),
          out_cache_loc.data(), out.data(), &rs);
      int mism = 0;
      for (int r = 0; r < c.batch; ++r) {
        const int req_raw = req_pool_indices[r];
        const bool req_valid = req_raw >= 0 && req_raw < c.n_reqs;
        const int req = req_valid ? req_raw
                                  : std::min(std::max(req_raw, 0), c.n_reqs - 1);
        const int pos_raw = positions[r];
        const int safe_pos = std::max(pos_raw, 0);
        const int seq_len = seq_lens[r];
        const int cache_loc = out_cache_loc[r];
        const bool pos_valid = req_valid && cache_loc != 0 && pos_raw >= 0 &&
                               pos_raw < seq_len;
        const int slot = safe_pos % c.pool_size;
        const int phys_slot = ((safe_pos % c.tail_size) + c.tail_size) % c.tail_size;
        const float* cur_k = key.data() + (size_t)r * H;
        const float* cur_s = slot_score.data() + (size_t)r * H;
        if (pos_valid && slot == c.pool_size - 1) {
          const int pool_logical_start = safe_pos - slot;
          std::vector<float> mx(H, -kInf);
          for (int p = 0; p < c.pool_size; ++p) {
            const int phys =
                ((pool_logical_start + p) % c.tail_size + c.tail_size) % c.tail_size;
            const float* sb = tail_score_ref.data() +
                              (size_t)req * c.tail_size * H + (size_t)phys * H;
            for (int d = 0; d < H; ++d) {
              const float s = (p == slot ? cur_s[d] : sb[d]) + ape[p * H + d];
              mx[d] = std::max(mx[d], s);
            }
          }
          std::vector<float> acc(H, 0.0f), denom(H, 0.0f);
          for (int p = 0; p < c.pool_size; ++p) {
            const int phys =
                ((pool_logical_start + p) % c.tail_size + c.tail_size) % c.tail_size;
            const float* sb = tail_score_ref.data() +
                              (size_t)req * c.tail_size * H + (size_t)phys * H;
            const float* kb = tail_k_ref.data() +
                              (size_t)req * c.tail_size * H + (size_t)phys * H;
            for (int d = 0; d < H; ++d) {
              const float s = (p == slot ? cur_s[d] : sb[d]) + ape[p * H + d];
              const float prob = std::exp(s - mx[d]);
              denom[d] += prob;
              acc[d] += (p == slot ? cur_k[d] : kb[d]) * prob;
            }
          }
          std::vector<uint8_t> k_ref(H);
          float scale_ref = 0.0f;
          ref_store_fp8(acc, denom, H, round_scale, k_ref, scale_ref);
          const int pool_id = safe_pos / c.pool_size;
          const int pool_page_group = pool_id / c.ssp;
          int tpr = std::min(std::max(pool_page_group * c.pool_size, 0), c.btc - 1);
          const int packed_page = block_tables[(size_t)r * c.btc + tpr];
          const int loc_slot = pool_id % c.ssp;
          const uint8_t* k_dst =
              out.data() + (size_t)packed_page * page_bytes + (size_t)loc_slot * H;
          const float* scale_dst =
              reinterpret_cast<const float*>(out.data() +
                                             (size_t)packed_page * page_bytes) +
              (c.ssp * H / 4) + loc_slot;
          for (int d = 0; d < H; ++d)
            if (k_dst[d] != k_ref[d]) ++mism;
          if (std::fabs(*scale_dst - scale_ref) > 0.0f) ++mism;
        }
        if (pos_valid) {
          const float* tk_ref = tail_k_ref.data() + (size_t)req * c.tail_size * H +
                                (size_t)phys_slot * H;
          const float* ts_ref = tail_score_ref.data() + (size_t)req * c.tail_size * H +
                                (size_t)phys_slot * H;
          const float* tk = tail_k.data() + (size_t)req * c.tail_size * H +
                            (size_t)phys_slot * H;
          const float* ts = tail_score.data() + (size_t)req * c.tail_size * H +
                            (size_t)phys_slot * H;
          for (int d = 0; d < H; ++d) {
            if (std::fabs(tk[d] - cur_k[d]) > 0.0f) ++mism;
            if (std::fabs(ts[d] - cur_s[d]) > 0.0f) ++mism;
            const_cast<float*>(tk_ref)[d] = cur_k[d];
            const_cast<float*>(ts_ref)[d] = cur_s[d];
          }
        }
      }
      EXPECT_EQ(mism, 0);
    }
  }
}

TEST(DsaKpoolFp8, AssembleNullArgsThrow) {
  std::vector<float> ape(128, 0.0f);
  std::vector<uint8_t> out(256, 0u);
  float rs = 0.0f;
  EXPECT_THROW(dsa_kpool_assemble_fp8_cpu(1, 1, 128, 1, 1, 1, 0, 1, nullptr,
                                          nullptr, nullptr, nullptr, ape.data(),
                                          nullptr, nullptr, nullptr, nullptr,
                                          nullptr, nullptr, out.data(), &rs),
               std::invalid_argument);
}

TEST(DsaKpoolFp8, AssembleEmptyIsNoOp) {
  std::vector<uint8_t> out(256, 0xABu);
  float rs = 0.0f;
  dsa_kpool_assemble_fp8_cpu(0, 1, 128, 1, 1, 1, 0, 1, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, out.data(),
                             &rs);
  for (uint8_t x : out) EXPECT_NEAR(static_cast<float>(x), 0xABu, 0);
}

TEST(DsaKpoolFp8, DecodeNullArgsThrow) {
  std::vector<float> ape(128, 0.0f);
  std::vector<uint8_t> out(256, 0u);
  EXPECT_THROW(dsa_kpool_decode_update_fp8_cpu(1, 1, 128, 1, 1, 1, 1, 1,
                                               nullptr, nullptr, nullptr,
                                               nullptr, ape.data(), nullptr,
                                               nullptr, nullptr, nullptr,
                                               nullptr, out.data(), nullptr),
               std::invalid_argument);
}

TEST(DsaKpoolFp8, DecodeEmptyIsNoOp) {
  std::vector<uint8_t> out(256, 0xABu);
  dsa_kpool_decode_update_fp8_cpu(0, 1, 128, 1, 1, 1, 1, 1, nullptr, nullptr,
                                  nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, nullptr, nullptr, out.data(),
                                  nullptr);
  for (uint8_t x : out) EXPECT_NEAR(static_cast<float>(x), 0xABu, 0);
}

// round_scale=NULL behaves identically to round_scale<=0 (raw scale).
TEST(DsaKpoolFp8, AssembleNullRoundScaleIsRaw) {
  constexpr int H = 128;
  std::mt19937 rng(123);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  // n_reqs=1, req=0 (valid but unused since n_tail=0 -> pure chunk path):
  // isolates round_scale=NULL vs round_scale<=0 (both raw, no power-of-two
  // rounding) without any tail-indexing dependence.
  std::vector<float> chunk_k(H), chunk_score(H), tail_k(H), tail_score(H),
      ape(H);
  for (auto& x : chunk_k) x = rf();
  for (auto& x : chunk_score) x = rf();
  for (auto& x : ape) x = rf();
  std::vector<int32_t> rpi = {0}, nt = {0}, css = {0}, tlb = {0}, loc = {0};
  const int ssp = 1, num_pages = 1;
  const int page_bytes = ssp * (H + 4);
  std::vector<uint8_t> out_a((size_t)num_pages * page_bytes, 0u);
  std::vector<uint8_t> out_b((size_t)num_pages * page_bytes, 0u);
  dsa_kpool_assemble_fp8_cpu(1, 1, H, 1, ssp, num_pages, 1, 1, chunk_k.data(),
                             chunk_score.data(), tail_k.data(), tail_score.data(),
                             ape.data(), rpi.data(), nt.data(), css.data(),
                             tlb.data(), loc.data(), nullptr, out_a.data(),
                             nullptr);
  float rs0 = 0.0f;
  dsa_kpool_assemble_fp8_cpu(1, 1, H, 1, ssp, num_pages, 1, 1, chunk_k.data(),
                             chunk_score.data(), tail_k.data(), tail_score.data(),
                             ape.data(), rpi.data(), nt.data(), css.data(),
                             tlb.data(), loc.data(), nullptr, out_b.data(),
                             &rs0);
  int mism = 0;
  for (size_t i = 0; i < out_a.size(); ++i)
    if (out_a[i] != out_b[i]) ++mism;
  EXPECT_EQ(mism, 0);
}
