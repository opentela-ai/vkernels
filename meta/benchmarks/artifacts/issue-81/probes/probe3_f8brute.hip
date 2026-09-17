// probe3_f8brute.hip -- brute-force the v_mfma_f32_16x16x32_fp8_fp8
// fragment maps (issue #81). For r0 in 0..19: A[r0][*]=1.0 (all else 0),
// B=all-ones -> the dev positions that report 32.0 are exactly the C
// positions of the lanes whose A bytes touch input row r0. Mirror for B
// cols. Includes r0/c0 = 16..19 to detect out-of-bounds row reads.
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

__global__ void probe(const uint8_t* __restrict__ A,   // [32][32] (padded)
                      const uint8_t* __restrict__ Bq,  // [32][32]
                      float* __restrict__ C) {
  const int l  = threadIdx.x;
  const int m  = l & 15;
  const int kq = (l >> 4) & 3;
  v2i a, b;
  __builtin_memcpy(&a, A + (size_t)m * 32 + kq * 8, 8);
  __builtin_memcpy(&b, Bq + (size_t)m * 32 + kq * 8, 8);
  v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
  c = mfma_f8(a, b, c);
  const int r0 = (l >> 4) * 4;
  for (int i = 0; i < 4; ++i) C[(size_t)(r0 + i) * 16 + m] = c[i];
}

static uint8_t hA[1024], hB[1024]; static float hC[256];
static uint8_t *dA, *dB; static float* dC;

static void run() {
  hipMemcpy(dA, hA, 1024, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, 1024, hipMemcpyHostToDevice);
  hipMemset(dC, 0, 1024);
  hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, dA, dB, dC);
  hipDeviceSynchronize();
  hipMemcpy(hC, dC, 1024, hipMemcpyDeviceToHost);
}

int main() {
  hipMalloc(&dA, 1024); hipMalloc(&dB, 1024); hipMalloc(&dC, 1024);
  const uint8_t ONE = 0x40, ZRO = 0x00;

  printf("=== A-row map: A[r0][*]=1, B=1 -> lit output positions (r,c) ===\n");
  for (int r0 = 0; r0 < 20; ++r0) {
    memset(hA, ZRO, 1024); memset(hB, ONE, 1024);
    for (int k = 0; k < 32; ++k) hA[r0 * 32 + k] = ONE;
    run();
    printf("A row %2d -> ", r0);
    int n = 0;
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c)
        if (hC[r * 16 + c] == 32.0f) { printf("(%d,%d) ", r, c); ++n; }
    printf("  [n=%d]\n", n);
  }

  printf("=== B-col map: B[c0][*]=1, A=1 -> lit output positions ===\n");
  for (int c0 = 0; c0 < 20; ++c0) {
    memset(hA, ONE, 1024); memset(hB, ZRO, 1024);
    for (int k = 0; k < 32; ++k) hB[c0 * 32 + k] = ONE;
    run();
    printf("B col %2d -> ", c0);
    int n = 0;
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c)
        if (hC[r * 16 + c] == 32.0f) { printf("(%d,%d) ", r, c); ++n; }
    printf("  [n=%d]\n", n);
  }

  printf("=== K-byte map: A[r0][k0]=1 single element, B all-ones -> which output\n"
         "    positions respond (identifies byte index i within lane fragment) ===\n");
  {
    memset(hA, ZRO, 1024); memset(hB, ONE, 1024);
    // light input row 0, one k0 at a time; combine with A-row map output
    for (int k0 = 0; k0 < 32; ++k0) {
      memset(hA, ZRO, 1024);
      hA[k0] = ONE;   // row 0, col k0
      run();
      printf("A[0][%2d] -> ", k0);
      int n = 0;
      for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
          if (hC[r * 16 + c] == 32.0f) { printf("(%d,%d) ", r, c); ++n; }
      printf("  [n=%d]\n", n);
    }
  }
  return 0;
}
