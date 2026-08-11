// vkernels/util/annotations.hpp
//
// Host/device annotation macros so headers that are included from both .cpp
// and .cu translation units can be written once and used on either side.
#pragma once

#if defined(__CUDACC__)
#  define VK_HD __host__ __device__
#  define VK_H  __host__
#  define VK_D  __device__
#else
#  define VK_HD
#  define VK_H
#  define VK_D
#endif
