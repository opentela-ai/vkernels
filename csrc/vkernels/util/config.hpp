// vkernels/util/config.hpp
//
// Build-time feature flags. VKERNELS_HAS_CUDA is injected by CMake (see
// cmake/CudaSupport.cmake) on the vkernels target; do not redefine it here.
#pragma once

#ifndef VKERNELS_HAS_CUDA
#  define VKERNELS_HAS_CUDA 0
#endif

#if defined(VKERNELS_COVERAGE_BUILD) && !defined(VKERNELS_COVERAGE_BUILD_USED)
#  define VKERNELS_COVERAGE_BUILD_USED 1
#endif
