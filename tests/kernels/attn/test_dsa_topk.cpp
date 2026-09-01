// tests/kernels/attn/test_dsa_topk.cpp
#include "minitest.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vkernels/kernels/dsa_topk.hpp"

using vkernels::kernels::dsa_topk_transform_cpu;
using vkernels::kernels::dsa_topk_transform_group_topk_supported;

namespace {

std::vector<int32_t> sorted_prefix(const std::vector<int32_t>& values, int32_t count) {
  std::vector<int32_t> out(values.begin(), values.begin() + count);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<int32_t>
expanded_group_tokens(int32_t group_begin, int32_t group_end, int32_t pool_size) {
  std::vector<int32_t> out;
  for (int32_t group = group_begin; group < group_end; ++group)
    for (int32_t lane = 0; lane < pool_size; ++lane)
      out.push_back(group * pool_size + lane);
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace

TEST(DsaTopk, SupportedGroupTopkValues) {
  EXPECT_TRUE(dsa_topk_transform_group_topk_supported(128));
  EXPECT_TRUE(dsa_topk_transform_group_topk_supported(512));
  EXPECT_FALSE(dsa_topk_transform_group_topk_supported(64));
}

TEST(DsaTopk, ReturnsContiguousHistoryWhenLengthDoesNotExceedK) {
  std::vector<float> score(12);
  for (int i = 0; i < 12; ++i)
    score[i] = static_cast<float>(i);
  std::vector<int32_t> lengths = {3};
  std::vector<int32_t> out(256, -7);
  dsa_topk_transform_cpu(1,
                         score.data(),
                         lengths.data(),
                         out.data(),
                         12,
                         2,
                         256,
                         256,
                         nullptr,
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr);
  for (int32_t i = 0; i < 6; ++i)
    EXPECT_EQ(out[i], i);
  for (int32_t i = 6; i < 256; ++i)
    EXPECT_EQ(out[i], -1);
}

TEST(DsaTopk, AppliesPageTableRowIndex) {
  std::vector<float> score(8);
  for (int i = 0; i < 8; ++i)
    score[i] = static_cast<float>(i);
  std::vector<int32_t> lengths = {2};
  std::vector<int32_t> page_table(32);
  for (int32_t i = 0; i < 32; ++i)
    page_table[i] = 100 + i;
  std::vector<int32_t> page_row = {0};
  std::vector<int32_t> out(256, -7);
  dsa_topk_transform_cpu(1,
                         score.data(),
                         lengths.data(),
                         out.data(),
                         8,
                         2,
                         256,
                         256,
                         page_table.data(),
                         32,
                         page_row.data(),
                         nullptr,
                         nullptr,
                         nullptr);
  EXPECT_EQ(out[0], 100);
  EXPECT_EQ(out[1], 101);
  EXPECT_EQ(out[2], 102);
  EXPECT_EQ(out[3], 103);
}

TEST(DsaTopk, AppliesRaggedOffsets) {
  std::vector<float> score(16);
  for (int i = 0; i < 16; ++i)
    score[i] = static_cast<float>(i);
  std::vector<int32_t> lengths = {2};
  std::vector<int32_t> offsets = {9};
  std::vector<int32_t> out(512, -7);
  dsa_topk_transform_cpu(1,
                         score.data(),
                         lengths.data(),
                         out.data(),
                         16,
                         4,
                         512,
                         512,
                         nullptr,
                         0,
                         nullptr,
                         offsets.data(),
                         nullptr,
                         nullptr);
  for (int32_t i = 0; i < 8; ++i)
    EXPECT_EQ(out[i], 9 + i);
}

TEST(DsaTopk, UsesRowStartsForWinnerSelection) {
  std::vector<float> score(132);
  score[0] = 1000.0f;
  score[1] = 999.0f;
  for (int32_t i = 0; i < 130; ++i)
    score[2 + i] = static_cast<float>(i);
  std::vector<int32_t> lengths = {130};
  std::vector<int32_t> row_starts = {2};
  std::vector<int32_t> out(256, -7);
  dsa_topk_transform_cpu(1,
                         score.data(),
                         lengths.data(),
                         out.data(),
                         132,
                         2,
                         256,
                         256,
                         nullptr,
                         0,
                         nullptr,
                         nullptr,
                         row_starts.data(),
                         nullptr);
  const auto got = sorted_prefix(out, 256);
  const auto want = expanded_group_tokens(2, 130, 2);
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(got[i], want[i]);
}

TEST(DsaTopk, AppendsSequenceTailAfterHistory) {
  std::vector<float> score(16);
  for (int i = 0; i < 16; ++i)
    score[i] = static_cast<float>(i);
  std::vector<int32_t> lengths = {2};
  std::vector<int32_t> seq_lens = {10};
  std::vector<int32_t> out(515, -7);
  dsa_topk_transform_cpu(1,
                         score.data(),
                         lengths.data(),
                         out.data(),
                         16,
                         4,
                         512,
                         515,
                         nullptr,
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         seq_lens.data());
  for (int32_t i = 0; i < 8; ++i)
    EXPECT_EQ(out[i], i);
  EXPECT_EQ(out[8], 8);
  EXPECT_EQ(out[9], 9);
  for (int32_t i = 10; i < 515; ++i)
    EXPECT_EQ(out[i], -1);
}

TEST(DsaTopk, RejectsConflictingMappings) {
  std::vector<float> score(16);
  std::vector<int32_t> lengths = {2};
  std::vector<int32_t> page_table(32, 0);
  std::vector<int32_t> offsets = {1};
  std::vector<int32_t> out(512, -7);
  EXPECT_THROW(dsa_topk_transform_cpu(1,
                                      score.data(),
                                      lengths.data(),
                                      out.data(),
                                      16,
                                      4,
                                      512,
                                      512,
                                      page_table.data(),
                                      32,
                                      nullptr,
                                      offsets.data(),
                                      nullptr,
                                      nullptr),
               std::invalid_argument);
}

TEST(DsaTopk, RejectsOutOfRangePageTableRow) {
  std::vector<float> score(16, 0.0f);
  std::vector<int32_t> lengths = {2};
  std::vector<int32_t> page_table(32, 0);
  std::vector<int32_t> page_rows = {1};
  std::vector<int32_t> out(512, -7);
  EXPECT_THROW(dsa_topk_transform_cpu(1,
                                      score.data(),
                                      lengths.data(),
                                      out.data(),
                                      16,
                                      4,
                                      512,
                                      512,
                                      page_table.data(),
                                      32,
                                      page_rows.data(),
                                      nullptr,
                                      nullptr,
                                      nullptr),
               std::invalid_argument);
}
