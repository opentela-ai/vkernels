// vkernels/capi/serving_c.h
//
// Version probe for the CUDA serving-runtime C ABI exported by
// libvkernels_c.so. Consumers must check this before using ABI structs or
// opaque handles whose layout/contract may evolve independently of the C++
// API and the host-reference `vk_*` C ABI.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VKERNELS_SERVING_ABI_VERSION 1u

uint32_t vkernels_serving_abi_version(void);

#ifdef __cplusplus
}
#endif
