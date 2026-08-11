// tests/comm/test_overlap.cpp
#include "minitest.hpp"

#include <vector>

#include "vkernels/comm/overlap.hpp"

using vkernels::comm::OverlapExecutor;

TEST(Overlap, RunsComputeAndComm) {
  OverlapExecutor ex;
  std::vector<int> comm_inputs;
  std::vector<int> compute_outputs;
  int sum = 0;

  auto result = ex.run(5,
                       [&](std::size_t i) {
                         compute_outputs.push_back(static_cast<int>(i + 1));
                         return static_cast<int>(i + 1);
                       },
                       [&](std::size_t i, int value) {
                         comm_inputs.push_back(value);
                         sum += value;
                         (void)i;
                       });

  EXPECT_EQ(result.compute_count, 5u);
  EXPECT_EQ(result.comm_count, 5u);
  EXPECT_TRUE(ex.uses_two_streams());
  // Each compute value was delivered to comm exactly once.
  ASSERT_EQ(comm_inputs.size(), 5u);
  int expected_sum = 1 + 2 + 3 + 4 + 5;
  EXPECT_EQ(sum, expected_sum);
}

TEST(Overlap, ZeroIterations) {
  OverlapExecutor ex;
  bool comm_called = false;
  auto result = ex.run(0, [&](std::size_t) { return 0; },
                       [&](std::size_t, int) { comm_called = true; });
  EXPECT_EQ(result.compute_count, 0u);
  EXPECT_EQ(result.comm_count, 0u);
  EXPECT_FALSE(comm_called);
}
