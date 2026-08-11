// tests/kernels/elementwise/test_elementwise.cpp
#include "minitest.hpp"

#include <vector>

#include "vkernels/util/span.hpp"
#include "vkernels/kernels/elementwise.hpp"

using vkernels::Span;
using vkernels::kernels::add;
using vkernels::kernels::relu;
using vkernels::kernels::scale;

TEST(Add, Elementwise) {
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {10, 20, 30, 40};
  std::vector<float> out(4);
  add(a, b, out);
  EXPECT_EQ(out[0], 11);
  EXPECT_EQ(out[1], 22);
  EXPECT_EQ(out[2], 33);
  EXPECT_EQ(out[3], 44);
}

TEST(Add, EmptyIsOk) {
  std::vector<float> a, b, out;
  add(a, b, out);  // length 0 == 0
  EXPECT_TRUE(out.empty());
}

TEST(Add, MismatchedInputsThrows) {
  std::vector<float> a = {1, 2}, b = {1}, out(2);
  EXPECT_THROW(add(a, b, out), std::invalid_argument);
}

TEST(Add, MismatchedOutputThrows) {
  std::vector<float> a = {1, 2}, b = {1, 2}, out(3);
  EXPECT_THROW(add(a, b, out), std::invalid_argument);
}

TEST(Scale, Elementwise) {
  std::vector<float> x = {1, 2, 3};
  std::vector<float> out(3);
  scale(x, 2.5f, out);
  EXPECT_NEAR(out[0], 2.5, 1e-6);
  EXPECT_NEAR(out[1], 5.0, 1e-6);
  EXPECT_NEAR(out[2], 7.5, 1e-6);
}

TEST(Scale, MismatchedOutputThrows) {
  std::vector<float> x = {1, 2, 3}, out(2);
  EXPECT_THROW(scale(x, 1.0f, out), std::invalid_argument);
}

TEST(Relu, PositiveAndNegative) {
  std::vector<float> x = {-3, -1, 0, 1, 3};
  std::vector<float> out(5);
  relu(x, out);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 0);
  EXPECT_EQ(out[3], 1);
  EXPECT_EQ(out[4], 3);
}

TEST(Relu, MismatchedOutputThrows) {
  std::vector<float> x = {1}, out(2);
  EXPECT_THROW(relu(x, out), std::invalid_argument);
}
