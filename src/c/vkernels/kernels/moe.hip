// vkernels/kernels/moe.hip — HIP (AMD CDNA3 gfx942) implementations.
#include "vkernels/kernels/moe.hpp"
#include <cstdint>
#include <cstddef>
#include "vkernels/kernels/moe_device.hip"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#  include <cstdlib>
#  include <cstring>
#  include "vkernels/util/error.hpp"

namespace vkernels::kernels::hip {

// ======================================================================
// #12 — Software direct-to-LDS fill (GFX942_SW_LDS_FILL)
// ======================================================================
__device__ void direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                                     int elements_per_thread) {
  int tid = threadIdx.x;
  const uint32_t* src = (const uint32_t*)global_src + tid * (elements_per_thread / 2);
  uint32_t* dst = (uint32_t*)lds_dst + tid * (elements_per_thread / 2);
  uint4 v = *(const uint4*)src;
  dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
}

// ======================================================================
// #13 — Software fp4→bf16 dequant (GFX942_SW_CVT)
// ======================================================================
__global__ void fp4_to_bf16_kernel(const uint8_t* __restrict__ packed,
                                   uint16_t* __restrict__ out,
                                   std::size_t n_packed, float scale) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_packed) return;
  uint8_t byte = packed[i];
  uint32_t flo = fp4nib_to_f32bits(byte & 0x0F);
  uint32_t fhi = fp4nib_to_f32bits((byte >> 4) & 0x0F);
  float vlo, vhi;
  __builtin_memcpy(&vlo, &flo, sizeof(float));
  __builtin_memcpy(&vhi, &fhi, sizeof(float));
  vlo *= scale; vhi *= scale;
  uint32_t bits_lo, bits_hi;
  __builtin_memcpy(&bits_lo, &vlo, sizeof(float));
  __builtin_memcpy(&bits_hi, &vhi, sizeof(float));
  out[i * 2] = f32bits_to_bf16(bits_lo);
  out[i * 2 + 1] = f32bits_to_bf16(bits_hi);
}

void fp4_to_bf16_dequant(const uint8_t* packed, uint16_t* out,
                         std::size_t n, float scale) {
  VK_EXPECTS(packed != nullptr, "packed must not be null");
  VK_EXPECTS(out != nullptr, "out must not be null");
  VK_EXPECTS(n > 0, "n must be positive");
  constexpr int kBlock = 256;
  int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  fp4_to_bf16_kernel<<<blocks, kBlock>>>(packed, out, n, scale);
  hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess, "hip fp4_to_bf16 launch failed");
}

// ======================================================================
// #14 — Platform async-copy gate (GFX942_ASYNC_OFF)
// ======================================================================
bool use_async_copy_default() {
  const char* env = std::getenv("K3_NO_ASYNC");
  if (env) return env[0] != '1';
  int device_id = 0;
  hipError_t err = hipGetDevice(&device_id);
  if (err != hipSuccess) return false;
  hipDeviceProp_t props;
  err = hipGetDeviceProperties(&props, device_id);
  if (err != hipSuccess) return false;
  return std::strstr(props.gcnArchName, "gfx942") == nullptr;
}

// ======================================================================
// #15 — K32 bf16 MFMA emulation (GFX942_K16_SPLIT)
// ======================================================================
// K16 is defined in moe_device.hip; K32 emulation calls it twice.

__device__ void mfma_f32_16x16x32bf16(float c[4], const uint32_t a[4],
                                      const uint32_t b[4],
                                      int cbsz, int abid, int blgp) {
  uint32_t a_lo[2] = {a[0], a[1]}, a_hi[2] = {a[2], a[3]};
  uint32_t b_lo[2] = {b[0], b[1]}, b_hi[2] = {b[2], b[3]};
  mfma_f32_16x16x16bf16(c, a_lo, b_lo, cbsz, abid, blgp);
  mfma_f32_16x16x16bf16(c, a_hi, b_hi, cbsz, abid, blgp);
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
