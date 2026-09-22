// tests/test_main.cpp — shared entry point and singletons for the minitest harness.
#include <cstdlib>

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

int main() {
  // Hermetic by default: a machine-local tuning store (written by --persist
  // bench runs) must never override compiled-in formulas inside tests.
  // Store-exercising tests point VKERNELS_TUNING_CACHE at their own dirs.
  if (::getenv("VKERNELS_TUNING_CACHE") == nullptr) {
    ::setenv("VKERNELS_TUNING_CACHE", "off", 1);
  }
  return ::vkernels::minitest::run();
}
