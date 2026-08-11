// tests/comm/test_allreduce.cpp
#include "minitest.hpp"

#include <vector>

#include "vkernels/comm/allreduce.hpp"

using vkernels::comm::ring_allreduce;

static std::vector<float> elementwise_sum(const std::vector<std::vector<float>>& locals) {
  std::vector<float> out = locals[0];
  for (size_t r = 1; r < locals.size(); ++r)
    for (size_t i = 0; i < out.size(); ++i) out[i] += locals[r][i];
  return out;
}

static void expect_all_equal(const std::vector<std::vector<float>>& got,
                             const std::vector<float>& expected) {
  for (const auto& g : got) {
    ASSERT_EQ(g.size(), expected.size());
    for (size_t i = 0; i < g.size(); ++i) EXPECT_NEAR(g[i], expected[i], 1e-4);
  }
}

TEST(Allreduce, TwoRanks) {
  std::vector<std::vector<float>> locals = {{1, 2, 3, 4}, {10, 20, 30, 40}};
  auto got = ring_allreduce(locals);
  expect_all_equal(got, {11, 22, 33, 44});
}

TEST(Allreduce, ThreeRanks) {
  std::vector<std::vector<float>> locals = {{1, 2, 3, 4, 5, 6},  //
                                            {1, 1, 1, 1, 1, 1},  //
                                            {2, 2, 2, 2, 2, 2}};
  auto got = ring_allreduce(locals);
  expect_all_equal(got, {4, 5, 6, 7, 8, 9});
}

TEST(Allreduce, FourRanksMatchesReference) {
  std::vector<std::vector<float>> locals = {{1, 2, 3, 4, 5, 6, 7, 8},     //
                                            {8, 7, 6, 5, 4, 3, 2, 1},     //
                                            {0, 0, 0, 0, 0, 0, 0, 0},    //
                                            {1, 1, 1, 1, 1, 1, 1, 1}};
  auto got = ring_allreduce(locals);
  expect_all_equal(got, elementwise_sum(locals));
}

TEST(Allreduce, SingleRankNoOp) {
  std::vector<std::vector<float>> locals = {{1, 2, 3}};
  auto got = ring_allreduce(locals);
  expect_all_equal(got, {1, 2, 3});
}

TEST(Allreduce, EmptyLocalsThrows) {
  std::vector<std::vector<float>> locals;
  EXPECT_THROW(ring_allreduce(locals), std::invalid_argument);
}

TEST(Allreduce, UnequalLengthsThrows) {
  std::vector<std::vector<float>> locals = {{1, 2}, {3}};
  EXPECT_THROW(ring_allreduce(locals), std::invalid_argument);
}

TEST(Allreduce, NotDivisibleByWorldThrows) {
  std::vector<std::vector<float>> locals = {{1, 2, 3}, {4, 5, 6}};  // 3 not divisible by 2
  EXPECT_THROW(ring_allreduce(locals), std::invalid_argument);
}
