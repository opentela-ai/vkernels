// tests/core/test_device.cpp
#include "minitest.hpp"

#include "vkernels/core/device.hpp"
#include "vkernels/util/config.hpp"

using vkernels::Device;
using vkernels::default_device;

TEST(Device, Defaults) {
  Device d;
  EXPECT_EQ(d.index(), -1);
  EXPECT_TRUE(d == default_device());
  EXPECT_FALSE(d != default_device());
}

TEST(Device, ExplicitIndex) {
  Device d(2);
  EXPECT_EQ(d.index(), 2);
  EXPECT_FALSE(d == Device(3));
  EXPECT_TRUE(d != Device(3));
}

TEST(Device, HostNoOps) {
  Device d;
  d.set_current();   // no-op on host
  d.sync();          // no-op on host
  EXPECT_FALSE(d.supports_peer(Device(0)));  // single CPU device
}

TEST(Device, CudaFlagReflectsBuild) {
  // Documents the build mode rather than asserting a value.
#if VKERNELS_HAS_CUDA
  EXPECT_TRUE(VKERNELS_HAS_CUDA);
#else
  EXPECT_FALSE(VKERNELS_HAS_CUDA);
#endif
}
