// tests/core/test_allocator.cpp
#include "minitest.hpp"

#include <new>

#include "vkernels/core/allocator.hpp"

using vkernels::Buffer;
using vkernels::allocate;
using vkernels::deallocate;

TEST(Allocator, AllocateAndDeallocate) {
  int* p = allocate<int>(4);
  EXPECT_TRUE(p != nullptr);
  for (int i = 0; i < 4; ++i) p[i] = i;
  EXPECT_EQ(p[3], 3);
  deallocate(p);
}

TEST(Allocator, AllocateZeroReturnsNull) {
  int* p = allocate<int>(0);
  EXPECT_TRUE(p == nullptr);
  deallocate(p);  // safe on nullptr
}

TEST(Allocator, AllocateOverflowThrows) {
  EXPECT_THROW(allocate<int>(SIZE_MAX / sizeof(int) + 1), std::invalid_argument);
}

TEST(Buffer, BasicLifecycle) {
  Buffer<int> b(3);
  EXPECT_EQ(b.size(), 3u);
  b[0] = 10;
  b[1] = 20;
  b[2] = 30;
  EXPECT_EQ(b[0], 10);
  EXPECT_EQ(b[1], 20);
  EXPECT_EQ(b[2], 30);

  Buffer<int> moved = std::move(b);
  EXPECT_EQ(moved.size(), 3u);
  EXPECT_EQ(moved[2], 30);

  Buffer<int> assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.size(), 3u);
  EXPECT_EQ(assigned[0], 10);

  const Buffer<int>& cb = assigned;
  EXPECT_EQ(cb[1], 20);
  EXPECT_EQ(cb[2], 30);
}

TEST(Buffer, IndexOutOfRangeThrows) {
  Buffer<int> b(2);
  EXPECT_THROW(b[2], std::invalid_argument);
  const Buffer<int>& cb = b;
  EXPECT_THROW(cb[5], std::invalid_argument);
}
