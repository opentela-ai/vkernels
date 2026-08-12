// vkernels/core/device.hpp
//
// A minimal device abstraction. On the host every "device" is the CPU and
// is a no-op; the CUDA variant (guarded) records the cuda device and offers
// set_current() / sync().
#pragma once

#include "vkernels/util/config.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels {

class Device {
 public:
  // `index == -1` selects the default device (CPU on host, 0 on CUDA).
  explicit Device(int index = -1) : index_(index) {}

  int index() const { return index_; }

#if VKERNELS_HAS_CUDA
  void set_current() const;
  void sync() const;
  bool supports_peer(const Device& other) const;
#else
  void set_current() const {}  // host: always CPU
  void sync() const {}
  bool supports_peer(const Device& other) const {
    (void)other;
    return false;  // single CPU device
  }
#endif

  bool operator==(const Device& other) const { return index_ == other.index_; }
  bool operator!=(const Device& other) const { return !(*this == other); }

 private:
  int index_;
};

inline Device default_device() { return Device{}; }

}  // namespace vkernels
