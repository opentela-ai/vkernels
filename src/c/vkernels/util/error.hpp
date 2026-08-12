// vkernels/util/error.hpp
//
// Lightweight status type plus precondition/contract macros. On the host,
// contract violations throw std::invalid_argument so they surface as test
// failures; inside device code (where exceptions are unavailable) they are
// best-effort and must not break compilation.
#pragma once

#include <exception>
#include <stdexcept>
#include <string>

#include "vkernels/util/annotations.hpp"

namespace vkernels {

enum class Code { Ok = 0, InvalidArgument, OutOfRange, Unsupported, Internal };

inline const char* code_name(Code c) {
  switch (c) {
    case Code::Ok: return "ok";
    case Code::InvalidArgument: return "invalid_argument";
    case Code::OutOfRange: return "out_of_range";
    case Code::Unsupported: return "unsupported";
    case Code::Internal: return "internal";
  }
  return "unknown";
}

// A small Status: either Ok with no message, or an error code + message.
class Status {
 public:
  Status() = default;
  Status(Code code, std::string message) : code_(code), message_(std::move(message)) {}

  bool ok() const { return code_ == Code::Ok; }
  Code code() const { return code_; }
  const std::string& message() const { return message_; }

  // Throw on failure. Used by host code; never call from device code.
  void throw_if_error() const {
    if (!ok()) throw std::runtime_error(code_name(code_) + std::string(": ") + message_);
  }

 private:
  Code code_ = Code::Ok;
  std::string message_;
};

}  // namespace vkernels

// ---------------------------------------------------------------------------
// Contracts
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
#  define VK_EXPECTS(cond, msg) \
    do { (void)(cond); (void)(msg); } while (0)
#  define VK_ENSURES(cond, msg) \
    do { (void)(cond); (void)(msg); } while (0)
#else
#  define VK_EXPECTS(cond, msg)                                              \
    do {                                                                     \
      if (!(cond)) throw ::std::invalid_argument(std::string(msg) + ": " + #cond); \
    } while (0)
#  define VK_ENSURES(cond, msg)                                              \
    do {                                                                     \
      if (!(cond)) throw ::std::runtime_error(std::string(msg) + ": " + #cond); \
    } while (0)
#endif
