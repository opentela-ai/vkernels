// vkernels/util/config.hpp
//
// Build-time feature flags. VKERNELS_HAS_CUDA and VKERNELS_HAS_HIP are
// injected by CMake (see cmake/CudaSupport.cmake and cmake/HipSupport.cmake)
// on the vkernels target; do not redefine them here.
#pragma once

#ifndef VKERNELS_HAS_CUDA
#  define VKERNELS_HAS_CUDA 0
#endif

#ifndef VKERNELS_HAS_HIP
#  define VKERNELS_HAS_HIP 0
#endif

#if defined(VKERNELS_COVERAGE_BUILD) && !defined(VKERNELS_COVERAGE_BUILD_USED)
#  define VKERNELS_COVERAGE_BUILD_USED 1
#endif
