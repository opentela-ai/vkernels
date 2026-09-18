// probe5_f8cfrag.hip -- where do c[0]/c[2] go? Chain two MFMAs with
// different single-row inputs and watch which dev rows accumulate.
//   call1: A row 13 = 1.0 (rest 0), B=1
//   call2: A row 11 = 1.0 (rest 0), B=1
// Under bf16-style C map: row 13 -> dev row 13 (i=1, l>>4=3), row 11 ->
// dev row 11 (i=3, l>>4=2). If c[0]/c[2] accumulate anything, even rows
// will light too.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int   v2i __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v4f mfma_f8(v2i a, v2i b, v4f c) {
  asm volatile("v_mfma_f32_16x16x32_fp8_fp8 %0, %1, %2, %0"
               : "+v"(c) : "v"(a), "v"(b));
  return c;
}

__global__ void probe(const uint8_t* __restrict__ A,   // [16][32]
                      const uint8_t* __restrict__ Bq,  // [16][32]
                      float* __restrict__ C) {
  const int l = threadIdx.x;
  v2i a, b;
  __builtin_memcpy(&a, A + (size_t)l * 8, 8);
  __builtin_memcpy(&b, Bq + (size_t)l * 8, 8);
  v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
  c = mfma_f8(a, b, c);
  c = mfma_f8(a, b, c);   // twice: also doubles every live element
  const int r0 = (l >> 4) * 4;
  for (int i = 0; i < 4; ++i) C[(size_t)(r0 + i) * 16 + (l & 15)] = c[i];
}

static uint8_t hA[512], hB[512]; static float hC[256];
static uint8_t *dA, *dB; static float* dC;

static void run() {
  hipMemcpy(dA, hA, 512, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, 512, hipMemcpyHostToDevice);
  hipMemset(dC, 0, 1024);
  hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, dA, dB, dC);
  hipDeviceSynchronize();
  hipMemcpy(hC, dC, 1024, hipMemcpyDeviceToHost);
}

int main() {
  hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dC, 1024);
  const uint8_t ONE = 0x40, ZRO = 0x00;
  memset(hA, ZRO, 512); memset(hB, ONE, 512);
  for (int k = 0; k < 32; ++k) { hA[13 * 32 + k] = ONE; }   // row 13 only
  run();
  printf("double-accum, A row13=1 (expect dev row13 = 64 if i=1 accumulates):\n");
  for (int r = 0; r < 16; ++r) {
    int nz = 0; float v = 0;
    for (int c = 0; c < 16; ++c) if (hC[r*16+c] != 0) { ++nz; v = hC[r*16+c]; }
    if (nz) printf("  dev row %2d: nz=%d val=%.1f\n", r, nz, v);
  }
  // now rows 11 AND 13 in the SAME operand (i=3 and i=1 both live)
  memset(hA, ZRO, 512);
  for (int k = 0; k < 32; ++k) { hA[11 * 32 + k] = ONE; hA[13 * 32 + k] = ONE; }
  run();
  printf("single-accum, A rows 11+13 = 1:\n");
  for (int r = 0; r < 16; ++r) {
    int nz = 0; float v = 0;
    for (int c = 0; c < 16; ++c) if (hC[r*16+c] != 0) { ++nz; v = hC[r*16+c]; }
    if (nz) printf("  dev row %2d: nz=%d val=%.1f\n", r, nz, v);
  }
  return 0;
}
