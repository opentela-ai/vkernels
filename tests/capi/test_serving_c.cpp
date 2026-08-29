#include "minitest.hpp"

#include "vkernels/capi/serving_c.h"

TEST(ServingCapi, VersionMatchesHeader) {
  EXPECT_EQ(vkernels_serving_abi_version(), VKERNELS_SERVING_ABI_VERSION);
  EXPECT_EQ(vkernels_serving_abi_version(), 1u);
}
