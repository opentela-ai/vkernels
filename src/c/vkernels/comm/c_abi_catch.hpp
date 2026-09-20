// vkernels/comm/c_abi_catch.hpp — shared exception-to-status translation
// for the extern "C" wrapper TUs in this directory (p2p_gather_c.cu,
// kv_gather_c.cu, rccl_c.cpp, ...).
//
// Every C ABI entry point in the comm layer catches every C++ exception
// (the VK_EXPECTS / VK_ENSURES contract checks, std::bad_alloc, ...) so
// nothing is thrown across the language boundary; only the status enum and
// its constants differ per wrapper. The common mapping policy lives here:
// contract violations (VK_EXPECTS arrives as std::invalid_argument) map to
// the caller's invalid-argument status, and every other std::exception
// (the VK_ENSURES std::runtime_error, std::bad_alloc, ...) maps to the
// caller's most conservative "something was wrong" status.
//
// C++-only: dynamic_cast and exceptions do not exist in device
// compilation, so include this header only from host-only code — the .cu
// wrappers already guard their bodies with #ifndef __CUDA_ARCH__.
#ifndef VKERNELS_COMM_C_ABI_CATCH_HPP_
#define VKERNELS_COMM_C_ABI_CATCH_HPP_

#include <exception>
#include <stdexcept>

namespace vkernels::comm::cabi {

// Map a caught C++ exception to an ABI status code. StatusT is the
// wrapper's status enum; the two constants are its invalid-argument and
// catch-all statuses. Wrappers with an additional live mapping (e.g.
// pipeline_boundary_c.cu's std::runtime_error -> ERR_UNSUPPORTED, or
// rccl_c.cpp's defensively-dead std::out_of_range branch) keep a local
// pre-check and defer the common pair to this helper.
template <typename StatusT, StatusT InvalidArg, StatusT Internal>
inline StatusT translate(const std::exception& e) noexcept {
  if (dynamic_cast<const std::invalid_argument*>(&e) != nullptr) {
    return InvalidArg;
  }
  // Not reachable from the host-compiled wrappers (rccl_c /
  // fabric_import_c tests drive only std::invalid_argument across their
  // ABIs); reachable from the CUDA-only wrappers, which the host coverage
  // gate does not compile.
  return Internal;  // LCOV_EXCL_LINE
}

}  // namespace vkernels::comm::cabi

#endif  // VKERNELS_COMM_C_ABI_CATCH_HPP_
