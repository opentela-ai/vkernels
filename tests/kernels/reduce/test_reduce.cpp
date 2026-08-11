// tests/kernels/reduce/test_reduce.cpp
#include "minitest.hpp"

#include <vector>

#include "vkernels/kernels/reduce.hpp"

using vkernels::kernels::max;
using vkernels::kernels::sum;

TEST(Sum, Basic) {
  std::vector<float> x = {1, 2, 3, 4, 5};
  float out = -1.0f;
  sum(x, out);
  EXPECT_NEAR(out, 15.0, 1e-6);
}

TEST(Sum, SingleElement) {
  std::vector<float> x = {42.0f};
  float out = 0.0f;
  sum(x, out);
  EXPECT_NEAR(out, 42.0, 1e-6);
}

TEST(Sum, EmptyThrows) {
  std::vector<float> x;
  float out = 0.0f;
  EXPECT_THROW(sum(x, out), std::invalid_argument);
}

TEST(Max, FindsMaximum) {
  std::vector<float> x = {3, 1, 4, 1, 5, 9, 2, 6};
  float out = -1.0f;
  max(x, out);
  EXPECT_NEAR(out, 9.0, 1e-6);
}

TEST(Max, SingleElement) {
  std::vector<float> x = {7.0f};
  float out = 0.0f;
  max(x, out);
  EXPECT_NEAR(out, 7.0, 1e-6);
}

TEST(Max, EmptyThrows) {
  std::vector<float> x;
  float out = 0.0f;
  EXPECT_THROW(max(x, out), std::invalid_argument);
}
