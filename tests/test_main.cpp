// tests/test_main.cpp — shared entry point and singletons for the minitest harness.
#include "minitest.hpp"

namespace vkernels::minitest {

std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

int& failure_count() {
  static int c = 0;
  return c;
}

}  // namespace vkernels::minitest

int main() { return ::vkernels::minitest::run(); }
