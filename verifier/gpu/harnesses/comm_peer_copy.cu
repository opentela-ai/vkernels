// verifier/gpu/harnesses/comm_peer_copy.cu
//
// GPUVerify-dialect mirror of the graph-capturable peer copy kernel in
// src/c/vkernels/comm/pipeline_boundary.cu (16-byte uint4 body with a byte
// tail, two independent grid-stride loops).  Body verbatim except
// std::size_t is spelled size_t (the GPUVerify stub provides <stddef.h>).
//
// __requires bounds exclude size_t wraparound of the grid-stride induction
// variables (production payloads are gigabytes, nowhere near 2^64 - stride;
// the bound only rules out astronomically large n4/tail_bytes where
// i + stride could wrap).  Launch dims match the kBlock = 256 launcher.
//
// Invocation:
//   verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=2 \
//       verifier/gpu/harnesses/comm_peer_copy.cu
#include "cuda.h"

__global__ void peer_copy_kernel(uint4* __restrict__ dst,
                                 const uint4* __restrict__ src,
                                 size_t n4, size_t tail_bytes,
                                 unsigned char* dst_tail,
                                 const unsigned char* src_tail) {
  __requires(n4 <= 1024);
  __requires(tail_bytes <= 1024);
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n4; i += stride) dst[i] = src[i];
  // Residual bytes (< 16) via a fresh grid-stride so the index is
  // unambiguous regardless of where each thread left the main loop.
  for (size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       b < tail_bytes; b += stride)
    dst_tail[b] = src_tail[b];
}
