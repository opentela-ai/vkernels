#include "vkernels/capi/serving_c.h"

extern "C" uint32_t vkernels_serving_abi_version(void) {
  return VKERNELS_SERVING_ABI_VERSION;
}
