// tests/third_party/minitest.hpp
//
// A tiny, dependency-free test harness so the project is fully testable on
// any machine with a C++17 compiler and no network. It supports a small,
// GoogleTest-shaped surface (TEST, EXPECT_*, ASSERT_*); richer matchers can
// be had later by setting -DVKERNELS_TEST_FRAMEWORK=gtest (cmake/Testing.cmake).
//
// Usage in a test file:
//   #include "minitest.hpp"
//   TEST(MySuite, Adds) { EXPECT_EQ(1 + 1, 2); }
//
// Provide main() via tests/test_main.cpp, which calls vkernels::minitest::run().
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkernels::minitest {

struct TestCase {
  const char* suite;
  const char* name;
  std::function<void()> fn;
};

// Defined out-of-line in tests/test_main.cpp so there is exactly one
// shared registry/counter across every translation unit. (An inline
// function-local static can be duplicated per TU under some flag/visibility
// combinations, which silently yields zero registered tests.)
std::vector<TestCase>& registry();
int& failure_count();

// Each TEST() instantiates one of these; its constructor runs during static
// initialisation and pushes the case into the shared registry.
struct Registrar {
  Registrar(const char* suite, const char* name, void (*fn)()) {
    registry().push_back({suite, name, fn});
  }
};

inline void report(const char* file, int line, const char* expr, const std::string& detail) {
  ++failure_count();
  std::fprintf(stderr, "  FAIL %s:%d: %s%s%s\n", file, line, expr,
               detail.empty() ? "" : " (", detail.c_str());
}

// Thrown by ASSERT_* to stop the current test.
struct AbortTest : std::exception {
  const char* what() const noexcept override { return "test aborted"; }
};

template <typename T>
std::string to_string_val(const T& v) {
  if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    return v ? "true" : "false";
  } else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
    std::ostringstream os;
    os << v;
    return os.str();
  } else {
    std::ostringstream os;
    os << v;
    return os.str();
  }
}

inline int run() {
  const auto& cases = registry();
  int passed = 0;
  for (const auto& t : cases) {
    int before = failure_count();
    std::printf("[ RUN      ] %s.%s\n", t.suite, t.name);
    try {
      t.fn();
    } catch (const AbortTest&) {
      // Reported by the ASSERT macro already.
    } catch (const std::exception& e) {
      report("<unknown>", 0, t.name, std::string("unhandled exception: ") + e.what());
    }
    if (failure_count() == before) {
      ++passed;
      std::printf("[       OK ] %s.%s\n", t.suite, t.name);
    } else {
      std::printf("[  FAILED  ] %s.%s\n", t.suite, t.name);
    }
  }
  int total = static_cast<int>(cases.size());
  int failed = total - passed;
  std::printf("\n%d/%d test(s) passed, %d failed\n", passed, total, failed);
  return failed == 0 ? 0 : 1;
}

}  // namespace vkernels::minitest

#define VK_CONCAT_(a, b) a##b
#define VK_CONCAT(a, b) VK_CONCAT_(a, b)

#define TEST(suite, name)                                                           \
  static void VK_CONCAT(vk_test_, __LINE__)();                                      \
  static ::vkernels::minitest::Registrar VK_CONCAT(vk_reg_, __LINE__)(              \
      #suite, #name, &VK_CONCAT(vk_test_, __LINE__));                               \
  static void VK_CONCAT(vk_test_, __LINE__)()

#define VK_EXPECT_(file, line, expr, detail) \
  ::vkernels::minitest::report(file, line, expr, detail)

#define EXPECT_TRUE(x)                                                \
  do {                                                                \
    if (!(x)) VK_EXPECT_(__FILE__, __LINE__, #x, "expected true");    \
  } while (0)

#define EXPECT_FALSE(x)                                               \
  do {                                                                \
    if ((x)) VK_EXPECT_(__FILE__, __LINE__, #x, "expected false");    \
  } while (0)

#define VK_COMPARE_(a, b, op, opp)                                                   \
  do {                                                                               \
    auto&& _va = (a);                                                                \
    auto&& _vb = (b);                                                                \
    if (!(_va op _vb))                                                               \
      VK_EXPECT_(__FILE__, __LINE__, #a " " #opp " " #b,                            \
                 "got " + ::vkernels::minitest::to_string_val(_va) +                \
                     ", expected " + ::vkernels::minitest::to_string_val(_vb));      \
  } while (0)

#define EXPECT_EQ(a, b) VK_COMPARE_(a, b, ==, ==)
#define EXPECT_NE(a, b) VK_COMPARE_(a, b, !=, !=)
#define EXPECT_LT(a, b) VK_COMPARE_(a, b, <, <)
#define EXPECT_LE(a, b) VK_COMPARE_(a, b, <=, <=)
#define EXPECT_GT(a, b) VK_COMPARE_(a, b, >, >)
#define EXPECT_GE(a, b) VK_COMPARE_(a, b, >=, >=)

#define EXPECT_NEAR(a, b, eps)                                                       \
  do {                                                                               \
    double _va = static_cast<double>(a);                                             \
    double _vb = static_cast<double>(b);                                             \
    double _e = static_cast<double>(eps);                                            \
    if (std::fabs(_va - _vb) > _e)                                                   \
      VK_EXPECT_(__FILE__, __LINE__, #a " ~ " #b,                                   \
                 "got " + ::vkernels::minitest::to_string_val(_va) +                \
                     ", expected ~" + ::vkernels::minitest::to_string_val(_vb) +    \
                     " (eps " + ::vkernels::minitest::to_string_val(_e) + ")");      \
  } while (0)

#define VK_ASSERT_(a, b, op, opp)                                                    \
  do {                                                                               \
    auto&& _va = (a);                                                                \
    auto&& _vb = (b);                                                                \
    if (!(_va op _vb)) {                                                             \
      VK_EXPECT_(__FILE__, __LINE__, #a " " #opp " " #b,                            \
                 "got " + ::vkernels::minitest::to_string_val(_va) +                \
                     ", expected " + ::vkernels::minitest::to_string_val(_vb));      \
      throw ::vkernels::minitest::AbortTest();                                       \
    }                                                                                \
  } while (0)

#define ASSERT_TRUE(x)                                                               \
  do {                                                                               \
    if (!(x)) {                                                                      \
      VK_EXPECT_(__FILE__, __LINE__, #x, "expected true");                           \
      throw ::vkernels::minitest::AbortTest();                                       \
    }                                                                                \
  } while (0)
#define ASSERT_EQ(a, b) VK_ASSERT_(a, b, ==, ==)
#define ASSERT_NE(a, b) VK_ASSERT_(a, b, !=, !=)

#define EXPECT_THROW(expr, exc_type)                                                \
  do {                                                                               \
    bool _caught = false;                                                            \
    try {                                                                            \
      expr;                                                                          \
    } catch (const exc_type&) {                                                      \
      _caught = true;                                                                \
    } catch (...) {                                                                  \
      _caught = false;                                                               \
    }                                                                                \
    if (!_caught)                                                                    \
      VK_EXPECT_(__FILE__, __LINE__, #expr, "expected " #exc_type " to be thrown"); \
  } while (0)

#define EXPECT_ANY_THROW(expr)                                                       \
  do {                                                                               \
    bool _caught = false;                                                            \
    try {                                                                            \
      expr;                                                                          \
    } catch (...) {                                                                  \
      _caught = true;                                                                \
    }                                                                                \
    if (!_caught)                                                                    \
      VK_EXPECT_(__FILE__, __LINE__, #expr, "expected an exception");               \
  } while (0)

#define EXPECT_NO_THROW(expr)                                                        \
  do {                                                                               \
    try {                                                                            \
      expr;                                                                          \
    } catch (...) {                                                                  \
      VK_EXPECT_(__FILE__, __LINE__, #expr, "expected no exception");               \
    }                                                                                \
  } while (0)
