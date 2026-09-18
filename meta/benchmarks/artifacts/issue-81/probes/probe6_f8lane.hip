// probe6_f8lane.hip -- absolute per-lane decode of v_mfma_f32_16x16x32_fp8_fp8
// fragments (issue #81). Each lane stores its OWN c[0..3] to C[l*4+i] -- no
// store-map assumptions. Structured inputs:
//   E1: A row 13 = 1.0 (rest 0), B = 1      -> which (lane,i) hold row-13 sums
//   E2: B col 5  = 1.0 (rest 0), A = 1      -> which (lane,i) hold col-5 sums
//   E3: A[0][k0] = 1.0 for k0 = 0..31 (31 runs), B = 1 -> byte-level A map
//   E4: B[0][k0] = 1.0 for k0 = 0..31 (31 runs), A = 1 -> byte-level B map
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

// print lanes whose any c[i] != 0: "lane l (group g, m m, q q): c0 c1 c2 c3"
static void dump_nz(const char* tag) {
  printf("%s\n", tag);
  for (int l = 0; l < 64; ++l) {
    if (hC[l*4] == 0 && hC[l*4+1] == 0 && hC[l*4+2] == 0 && hC[l*4+3] == 0) continue;
    printf("  lane %2d (g=%d m=%2d q=%d): %6.1f %6.1f %6.1f %6.1f\n",
           l, l >> 4, l & 15, (l >> 2) & 3, hC[l*4], hC[l*4+1], hC[l*4+2], hC[l*4+3]);
  }
}

int main() {
  hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dC, 1024);
  const uint8_t ONE = 0x40, ZRO = 0x00;

  memset(hA, ZRO, 512); memset(hB, ONE, 512);
  for (int k = 0; k < 32; ++k) hA[13 * 32 + k] = ONE;
  run();
  dump_nz("E1: A row13=1, B=1 (full row-13 K-sum = 32.0):");

  memset(hA, ONE, 512); memset(hB, ZRO, 512);
  for (int k = 0; k < 32; ++k) hB[5 * 32 + k] = ONE;
  run();
  dump_nz("E2: B col5=1, A=1 (full col-5 K-sum = 32.0):");

  printf("E3: A[0][k0]=1 single element, B=1 -> nonzero (lane, i): value\n");
  for (int k0 = 0; k0 < 32; ++k0) {
    memset(hA, ZRO, 512); memset(hB, ONE, 512);
    hA[k0] = ONE;   // A[0][k0]
    run();
    printf("  k0=%2d:", k0);
    for (int l = 0; l < 64; ++l)
      for (int i = 0; i < 4; ++i)
        if (hC[l*4+i] != 0) printf(" (l%d,i%d)=%.0f", l, i, hC[l*4+i]);
    printf("\n");
  }

  printf("E4: B[0][k0]=1 single element, A=1 -> nonzero (lane, i): value\n");
  for (int k0 = 0; k0 < 32; ++k0) {
    memset(hA, ONE, 512); memset(hB, ZRO, 512);
    hB[k0] = ONE;
    run();
    printf("  k0=%2d:", k0);
    for (int l = 0; l < 64; ++l)
      for (int i = 0; i < 4; ++i)
        if (hC[l*4+i] != 0) printf(" (l%d,i%d)=%.0f", l, i, hC[l*4+i]);
    printf("\n");
  }
  return 0;
}
