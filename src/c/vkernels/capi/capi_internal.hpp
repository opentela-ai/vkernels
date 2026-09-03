// vkernels/capi/capi_internal.hpp — shared plumbing for the C ABI
// implementation TUs (capi.cpp, capi_kernels.cpp, capi_attn.cpp,
// capi_moe.cpp). Not installed, not part of any public contract.
//
// The exported entry points live one domain per translation unit; this
// header carries what they all share: the thread-local last-error state
// and the exception-to-status translation macros. Exceptions (the
// VK_EXPECTS / VK_ENSURES contract checks, std::bad_alloc) cannot cross
// the C ABI, so every exported function wraps its body in VK_CAPI_TRY /
// VK_CAPI_CATCH_RETURN_CODE() (or _RETURN_NULL), which translate them into
// a status code plus a thread-local message readable via vk_last_error() /
// vk_last_error_code(). See capi.hpp for the full contract.
#ifndef VKERNELS_CAPI_CAPI_INTERNAL_HPP_
#define VKERNELS_CAPI_CAPI_INTERNAL_HPP_

#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include "vkernels/capi/capi.hpp"

namespace vkernels::capi {

// The thread-local state itself is defined in capi.cpp; these accessors are
// how the domain TUs (and the macros below) reach it.
void set_last_error(int code, const char* message);
const char* last_error();
int last_error_code();

}  // namespace vkernels::capi

// The C++ library's contract checks throw std::invalid_argument (VK_EXPECTS)
// and std::runtime_error (VK_ENSURES); allocations can throw std::bad_alloc.
// Translate each to the status codes declared in capi.hpp. Both macros set
// the thread-local code/message so vk_last_error_code() is always accurate.
#define VK_CAPI_TRY try {
#define VK_CAPI_CATCH_RETURN_CODE()                                    \
  }                                                                    \
  catch (const std::invalid_argument& e) {                             \
    vkernels::capi::set_last_error(VK_ERROR_INVALID_ARGUMENT, e.what()); \
    return VK_ERROR_INVALID_ARGUMENT;                                  \
  }                                                                    \
  catch (const std::out_of_range& e) {                                 \
    vkernels::capi::set_last_error(VK_ERROR_OUT_OF_RANGE, e.what());   \
    return VK_ERROR_OUT_OF_RANGE;                                      \
  }                                                                    \
  catch (const std::length_error& e) {                                 \
    vkernels::capi::set_last_error(VK_ERROR_INVALID_ARGUMENT, e.what()); \
    return VK_ERROR_INVALID_ARGUMENT;                                  \
  }                                                                    \
  catch (const std::runtime_error& e) {                                \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, e.what());       \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (const std::bad_alloc&) {                                      \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, "out of memory"); \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (const std::exception& e) {                                    \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, e.what());       \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (...) {                                                        \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, "unknown C++ exception"); \
    return VK_ERROR_INTERNAL;                                          \
  }

// Catch variant for handle-returning functions: report the error and return
// nullptr instead of a status code.
#define VK_CAPI_CATCH_RETURN_NULL()                                    \
  }                                                                    \
  catch (const std::invalid_argument& e) {                             \
    vkernels::capi::set_last_error(VK_ERROR_INVALID_ARGUMENT, e.what()); \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::out_of_range& e) {                                 \
    vkernels::capi::set_last_error(VK_ERROR_OUT_OF_RANGE, e.what());   \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::length_error& e) {                                 \
    vkernels::capi::set_last_error(VK_ERROR_INVALID_ARGUMENT, e.what()); \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::runtime_error& e) {                                \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, e.what());       \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::bad_alloc&) {                                      \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, "out of memory"); \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::exception& e) {                                    \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, e.what());       \
    return nullptr;                                                    \
  }                                                                    \
  catch (...) {                                                        \
    vkernels::capi::set_last_error(VK_ERROR_INTERNAL, "unknown C++ exception"); \
    return nullptr;                                                    \
  }

#endif  // VKERNELS_CAPI_CAPI_INTERNAL_HPP_
