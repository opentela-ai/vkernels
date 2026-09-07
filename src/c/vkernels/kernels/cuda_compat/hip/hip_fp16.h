// ============================================================================
//  Minimal HIP-on-NVIDIA compatibility shim for <hip/hip_fp16.h>.
//
//  Sibling of hip_runtime.h (issue #60 A100/GB10 port). Lets the .hip device
//  sources that include <hip/hip_fp16.h> be compiled by nvcc for NVIDIA GPUs
//  without a ROCm installation. Add ``-I src/c/vkernels/kernels/cuda_compat``
//  to the include path; this header is then found INSTEAD of the real HIP
//  fp16 header.
//
//  HIP's __half / __half2 and the common reinterpret/round intrinsics
//  (__float2half_rn, __half_as_ushort, ...) map 1:1 onto the CUDA
//  <cuda_fp16.h> surface, which nvcc provides natively -- so this shim is
//  little more than the include plus the same ROCm-build guard as
//  hip_runtime.h. Extend it only if a vkernels device source uses an
//  __half* intrinsic that <cuda_fp16.h> lacks (re-express via reinterpret of
//  the __half/.x storage, matching HIP semantics).
//
//  Guarded against accidental use in ROCm builds: there the real HIP fp16
//  header must win (the shim directory is simply not on those builds' include
//  path, and the #error below makes any misconfiguration loud).
// ============================================================================
#ifndef VKERNELS_CUDA_COMPAT_HIP_FP16_H
#define VKERNELS_CUDA_COMPAT_HIP_FP16_H

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || \
    defined(__HIP_DEVICE_COMPILE__)
#error                                                                     \
    "vkernels HIP-on-NVIDIA fp16 shim included in a ROCm/HIP-AMD build -- " \
    "the real <hip/hip_fp16.h> must be used there (cuda_compat must NOT be " \
    "on the include path)."
#endif

#include <cuda_fp16.h>

#endif  // VKERNELS_CUDA_COMPAT_HIP_FP16_H
