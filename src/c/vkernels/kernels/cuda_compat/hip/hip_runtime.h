// ============================================================================
//  Minimal HIP-on-NVIDIA compatibility shim for the vkernels device sources.
//
//  Purpose: lets the .hip device sources (which include <hip/hip_runtime.h>)
//  be compiled by nvcc for NVIDIA GPUs (e.g. A100/bristen) without a ROCm
//  installation. Add ``-I src/c/vkernels/kernels/cuda_compat`` to the include
//  path and ``-DVKERNELS_HAS_HIP``; this header is then found INSTEAD of the
//  real HIP runtime and maps the small HIP API surface the kpool sources use
//  onto the CUDA runtime API.
//
//  NOT a general HIP implementation: only the symbols actually used by the
//  vkernels device sources are mapped (memory, events, streams, properties,
//  error reporting). If a build fails with an unknown ``hip*`` symbol, add
//  the mapping here -- and keep the shim's surface deliberately small.
//
//  Guarded against accidental use in ROCm builds: there the real HIP runtime
//  must win (the shim directory is simply not on those builds' include path,
//  and the #error below makes any misconfiguration loud).
// ============================================================================
#ifndef VKERNELS_CUDA_COMPAT_HIP_RUNTIME_H
#define VKERNELS_CUDA_COMPAT_HIP_RUNTIME_H

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || \
    defined(__HIP_DEVICE_COMPILE__)
#error                                                                     \
    "vkernels HIP-on-NVIDIA shim included in a ROCm/HIP-AMD build -- the "  \
    "real <hip/hip_runtime.h> must be used there (cuda_compat must NOT be " \
    "on the include path)."
#endif

#include <cuda_runtime.h>

// The real HIP runtime header transitively pulls in the C++ std headers its
// users rely on; cuda_runtime.h does not. Provide the ones the vkernels
// device sources use (std::string, std::strstr, printf, exit, ...).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Host-only translation units that transitively pull in device helpers need
// the CUDA keywords neutralized; nvcc TUs define __CUDACC__ and handle them
// natively. CUDA >= 13's crt/host_defines.h defines __host__/__device__ in
// host TUs too (via cuda_runtime.h included above), so guard each define to
// avoid a -Werror redefinition there.
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

// ---- error codes / types --------------------------------------------------
#define hipSuccess cudaSuccess
#define hipError_t cudaError_t
#define hipDeviceProp_t cudaDeviceProp
#define hipEvent_t cudaEvent_t
#define hipStream_t cudaStream_t

// ---- memory / sync --------------------------------------------------------
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
// kda.hip uses the 2-arg 32-lane butterfly __shfl_xor(var, mask); CUDA 13
// removed the unsynced intrinsic, so map to the synced form (full-warp mask
// — every call site in kda.hip has all lanes of a 32-lane warp active).
#define __shfl_xor(var, mask) __shfl_xor_sync(0xFFFFFFFFu, (var), (mask))
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetErrorString cudaGetErrorString
#define hipGetLastError cudaGetLastError

// ---- events ---------------------------------------------------------------
#define hipEventCreate cudaEventCreate
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventDestroy cudaEventDestroy
#define hipEventElapsedTime cudaEventElapsedTime

// ---- device properties / enumeration --------------------------------------
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipGetDeviceCount cudaGetDeviceCount
// cudaDeviceProp has no gcnArchName; map it to name so AMD device-detection
// code (e.g. strstr(p.gcnArchName, "gfx942")) keeps compiling and simply
// falls through to NVIDIA/other device branches.
#define gcnArchName name

#endif  // VKERNELS_CUDA_COMPAT_HIP_RUNTIME_H
