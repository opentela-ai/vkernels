// tests/kernels/attn/test_dsa_kpool_fp8.cpp
//
// Issue #61: fp8+scale store epilogue -- host-CPU tests for
// ``dsa_kpool_assemble_fp8_cpu`` / ``dsa_kpool_decode_update_fp8_cpu``.
// Split from test_dsa_kpool.cpp (issue #61 review): the fp8 suite shares
// the ONE independent reference core (dense Hadamard, two-pass softmax,
// frexp-based fp8 encoder) with the bf16 suite via dsa_kpool_ref.hpp --
// the pooled-vector math is identical by contract, so the two test
// references must not be able to drift apart.
#include "minitest.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/dsa_kpool.hpp"

#include "dsa_kpool_ref.hpp"

using vkernels::kernels::dsa_kpool_assemble_fp8_cpu;
using vkernels::kernels::dsa_kpool_decode_update_fp8_cpu;

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
        const int lr = loc[r];
        std::vector<float> acc, denom;
        dsa_kpool_ref::assemble_row_accumulate(
            c.pool_size, H, c.tail_size, chunk_k, chunk_score, tail_k,
            tail_score, ape, req_pool_idx[r], n_from_tail[r],
            chunk_src_start[r], tail_logical_base[r], acc, denom);
        std::vector<uint8_t> k_ref(H);
        float scale_ref = 0.0f;
        dsa_kpool_ref::ref_store_fp8(acc, denom, H, round_scale, k_ref,
                                     scale_ref);
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
        const bool pos_valid = req_valid && out_cache_loc[r] != 0 &&
                               pos_raw >= 0 && pos_raw < seq_lens[r];
        const int slot = safe_pos % c.pool_size;
        const int phys_slot = ((safe_pos % c.tail_size) + c.tail_size) % c.tail_size;
        const float* cur_k = key.data() + (size_t)r * H;
        const float* cur_s = slot_score.data() + (size_t)r * H;
        if (pos_valid && slot == c.pool_size - 1) {
          // Shared reference core reads the tail PRE-update: pass the ref
          // copies (the impl mutates tail_k/tail_score in place).
          std::vector<float> acc, denom;
          const bool complete = dsa_kpool_ref::decode_row_accumulate(
              c.pool_size, H, c.tail_size, key, slot_score, tail_k_ref,
              tail_score_ref, ape, c.n_reqs, r, req_pool_indices, positions,
              seq_lens, out_cache_loc, acc, denom);
          ASSERT_TRUE(complete);
          std::vector<uint8_t> k_ref(H);
          float scale_ref = 0.0f;
          dsa_kpool_ref::ref_store_fp8(acc, denom, H, round_scale, k_ref,
                                       scale_ref);
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
