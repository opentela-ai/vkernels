// tests/core/test_logging.cpp
#include "minitest.hpp"

#include <cstdio>
#include <cstring>

#include "vkernels/util/logging.hpp"

using namespace vkernels::logging;

TEST(Logging, LevelNameAll) {
  EXPECT_EQ(std::string(level_name(Level::Silent)), "SILENT");
  EXPECT_EQ(std::string(level_name(Level::Error)), "ERROR");
  EXPECT_EQ(std::string(level_name(Level::Warn)), "WARN");
  EXPECT_EQ(std::string(level_name(Level::Info)), "INFO");
  EXPECT_EQ(std::string(level_name(Level::Debug)), "DEBUG");
  EXPECT_EQ(std::string(level_name(static_cast<Level>(42))), "?");
}

TEST(Logging, SetAndGetLevel) {
  set_level(Level::Warn);
  EXPECT_TRUE(current_level() == Level::Warn);
  set_level(Level::Debug);  // restore for other tests
  EXPECT_TRUE(current_level() == Level::Debug);
}

TEST(Logging, FiltersByLevel) {
  set_level(Level::Error);
  log(Level::Warn, "should-be-suppressed");  // filtered out
  log(Level::Error, "should-print");         // printed
  set_level(Level::Debug);

  error("err");
  warn("warn");
  info("info");
  debug("dbg");
  log(Level::Silent, nullptr);  // exercises the null-guard branch at level above filter
}
