// tests/core/test_error.cpp
#include "minitest.hpp"

#include "vkernels/util/error.hpp"

using vkernels::Code;
using vkernels::Status;
using vkernels::code_name;

TEST(Status, DefaultIsOk) {
  Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(static_cast<int>(s.code()), static_cast<int>(Code::Ok));
  EXPECT_TRUE(s.message().empty());
  s.throw_if_error();  // must not throw
}

TEST(Status, ErrorHoldsCodeAndMessage) {
  Status s(Code::InvalidArgument, "bad size");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(static_cast<int>(s.code()), static_cast<int>(Code::InvalidArgument));
  EXPECT_EQ(s.message(), "bad size");
  EXPECT_ANY_THROW(s.throw_if_error());
}

TEST(CodeName, AllKnownCodes) {
  EXPECT_EQ(std::string(code_name(Code::Ok)), "ok");
  EXPECT_EQ(std::string(code_name(Code::InvalidArgument)), "invalid_argument");
  EXPECT_EQ(std::string(code_name(Code::OutOfRange)), "out_of_range");
  EXPECT_EQ(std::string(code_name(Code::Unsupported)), "unsupported");
  EXPECT_EQ(std::string(code_name(Code::Internal)), "internal");
  // Default branch: an out-of-range enum value.
  EXPECT_EQ(std::string(code_name(static_cast<Code>(999))), "unknown");
}
