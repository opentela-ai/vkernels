// tests/core/test_span.cpp
#include "minitest.hpp"

#include <vector>

#include "vkernels/util/span.hpp"

using vkernels::Span;

TEST(Span, DefaultIsEmpty) {
  Span<const int> s;
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size(), 0u);
  EXPECT_TRUE(s.data() == nullptr);
}

TEST(Span, FromPointerAndSize) {
  int data[] = {1, 2, 3};
  Span<int> s(data, 3);
  EXPECT_EQ(s.size(), 3u);
  EXPECT_EQ(s[0], 1);
  EXPECT_EQ(s[2], 3);
  int count = 0;
  for (int x : s) count += x;
  EXPECT_EQ(count, 6);
}

TEST(Span, FromContainer) {
  std::vector<int> v = {4, 5, 6};
  Span<int> s(v);
  EXPECT_EQ(s.size(), 3u);
  EXPECT_EQ(s[1], 5);
}

TEST(Span, First) {
  int data[] = {7, 8, 9, 10};
  Span<int> s(data, 4);
  Span<int> head = s.first(2);
  EXPECT_EQ(head.size(), 2u);
  EXPECT_EQ(head[0], 7);
  EXPECT_EQ(head[1], 8);
}

TEST(Span, IndexOutOfRangeThrows) {
  int data[] = {1};
  Span<int> s(data, 1);
  EXPECT_THROW(s[1], std::invalid_argument);
}

TEST(Span, FirstOutOfRangeThrows) {
  int data[] = {1, 2};
  Span<int> s(data, 2);
  EXPECT_THROW(s.first(5), std::invalid_argument);
}
