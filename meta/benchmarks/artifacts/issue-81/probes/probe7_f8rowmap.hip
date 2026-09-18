// probe7_f8rowmap.hip -- pin the A-fragment row map: for r in 0..15 and
// k0 in {0,8,16,24}: A[r][k0]=1 (single element), B=1 -> dump nonzero
// (lane,i) responses. Also mirror for B rows (k-dim) with A=1.
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
                      float* __restrict__ C) {         // [64][4]
  const int l = threadIdx.x;
  v2i a, b;
  __builtin_memcpy(&a, A + (size_t)l * 8, 8);
  __builtin_memcpy(&b, Bq + (size_t)l * 8, 8);
  v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
  c = mfma_f8(a, b, c);
  for (int i = 0; i < 4; ++i) C[(size_t)l * 4 + i] = c[i];
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

  printf("A-map: A[r][k0]=1, B=1 -> nonzero (lane,i)\n");
  for (int r = 0; r < 16; ++r) {
    printf("r=%2d:", r);
    for (int kc = 0; kc < 4; ++kc) {
      int k0 = kc * 8;
      memset(hA, ZRO, 512); memset(hB, ONE, 512);
      hA[r * 32 + k0] = ONE;
      run();
      printf("  [k%d:", k0);
      for (int l = 0; l < 64; ++l)
        for (int i = 0; i < 4; ++i)
          if (hC[l*4+i] != 0) printf(" %d.%d", l, i);
      printf("]");
    }
    printf("\n");
  }

  printf("B-map: Bq[r][k0]=1 (B col r, K k0), A=1 -> nonzero (lane,i)\n");
  for (int r = 0; r < 16; ++r) {
    printf("c=%2d:", r);
    for (int kc = 0; kc < 4; ++kc) {
      int k0 = kc * 8;
      memset(hA, ONE, 512); memset(hB, ZRO, 512);
      hB[r * 32 + k0] = ONE;
      run();
      printf("  [k%d:", k0);
      for (int l = 0; l < 64; ++l)
        for (int i = 0; i < 4; ++i)
          if (hC[l*4+i] != 0) printf(" %d.%d", l, i);
      printf("]");
    }
    printf("\n");
  }
  return 0;
}
