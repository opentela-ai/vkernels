// vkernels/util/logging.hpp
//
// Minimal stderr logger with severity levels. Kept dependency-free and
// host-only; device-side logging is intentionally not provided.
#pragma once

#include <cstdio>
#include <cstring>

namespace vkernels::logging {

enum class Level { Silent = 0, Error = 1, Warn = 2, Info = 3, Debug = 4 };

inline Level& current_level() {
  static Level level =
#ifdef NDEBUG
      Level::Info
#else
      Level::Debug
#endif
      ;
  return level;
}

inline void set_level(Level l) { current_level() = l; }

inline const char* level_name(Level l) {
  switch (l) {
    case Level::Silent: return "SILENT";
    case Level::Error: return "ERROR";
    case Level::Warn: return "WARN";
    case Level::Info: return "INFO";
    case Level::Debug: return "DEBUG";
  }
  return "?";
}

inline void log(Level l, const char* msg) {
  if (static_cast<int>(l) > static_cast<int>(current_level())) return;
  std::fprintf(stderr, "[vkernels %s] %s\n", level_name(l), msg ? msg : "(null)");
}

inline void error(const char* msg) { log(Level::Error, msg); }
inline void warn(const char* msg) { log(Level::Warn, msg); }
inline void info(const char* msg) { log(Level::Info, msg); }
inline void debug(const char* msg) { log(Level::Debug, msg); }

}  // namespace vkernels::logging
