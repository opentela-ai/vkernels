// probe8_f8h2.hip -- H2-packed A fragment (lane l byte j = A[l&15][(l>>4)*8+j]),
// single-element sweep over all rows x K-chunks, per-lane dump.
// Plus i8 MFMA sanity check with same packing to validate methodology.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int   v4i __attribute__((ext_vector_type(4)));
typedef long  v2l __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v4f mfma_f8(long a, long b, v4f c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ v4i mfma_i8(long a, long b, v4i c) {
  return __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, 0, 0, 0);
}

// mode 0: fp8; mode 1: i8 (payload bytes used directly)
__global__ void probe(const uint8_t* __restrict__ A,   // [16][32]
                      const uint8_t* __restrict__ Bq,  // [16][32]
                      float* __restrict__ Cf,          // [64][4]
                      int* __restrict__ Ci,
                      int mode) {
  const int l = threadIdx.x;
  long a, b;
  __builtin_memcpy(&a, A + (size_t)(l & 15) * 32 + (size_t)(l >> 4) * 8, 8);
  __builtin_memcpy(&b, Bq + (size_t)(l & 15) * 32 + (size_t)(l >> 4) * 8, 8);
  if (mode == 0) {
    v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
    c = mfma_f8(a, b, c);
    for (int i = 0; i < 4; ++i) Cf[(size_t)l * 4 + i] = c[i];
  } else {
    long bb;
    __builtin_memcpy(&bb, &b, 8);
    v4i c = {0, 0, 0, 0};
    c = mfma_i8(a, bb, c);
    for (int i = 0; i < 4; ++i) Ci[(size_t)l * 4 + i] = c[i];
  }
}

static uint8_t hA[512], hB[512];
static float hCf[256]; static int hCi[256];
static uint8_t *dA, *dB; static float* dCf; static int* dCi;

static void run(int mode) {
  hipMemcpy(dA, hA, 512, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, 512, hipMemcpyHostToDevice);
  hipMemset(dCf, 0, 1024); hipMemset(dCi, 0, 1024);
  hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, dA, dB, dCf, dCi, mode);
  hipDeviceSynchronize();
  if (mode == 0) hipMemcpy(hCf, dCf, 1024, hipMemcpyDeviceToHost);
  else hipMemcpy(hCi, dCi, 1024, hipMemcpyDeviceToHost);
}

static void dump_nz(int mode, const char* tag) {
  printf("%s\n", tag);
  for (int l = 0; l < 64; ++l) {
    float v[4]; int w[4];
    for (int i = 0; i < 4; ++i) {
      v[i] = hCf[l*4+i]; w[i] = hCi[l*4+i];
    }
    if (mode == 0) { if (!v[0] && !v[1] && !v[2] && !v[3]) continue;
      printf("  lane %2d: %6.1f %6.1f %6.1f %6.1f\n", l, v[0], v[1], v[2], v[3]);
    } else { if (!w[0] && !w[1] && !w[2] && !w[3]) continue;
      printf("  lane %2d: %4d %4d %4d %4d\n", l, w[0], w[1], w[2], w[3]);
    }
  }
}

int main() {
  hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dCf, 1024); hipMalloc(&dCi, 1024);
  const uint8_t ONE = 0x40, ZRO = 0x00;

  // S1: fp8 full row sweep
  printf("S1 fp8: A[r][k0]=1 (H2 pack), B=1 -> nonzero lanes/values\n");
  for (int r = 0; r < 16; r += 5) {
    for (int kc = 0; kc < 4; ++kc) {
      memset(hA, ZRO, 512); memset(hB, ONE, 512);
      hA[r * 32 + kc * 8] = ONE;
      run(0);
      printf("r=%2d k0=%2d:", r, kc * 8);
      for (int l = 0; l < 64; ++l)
        for (int i = 0; i < 4; ++i)
          if (hCf[l*4+i] != 0) printf(" (%d,%d)=%.0f", l, i, hCf[l*4+i]);
      printf("\n");
    }
  }

  // S2: i8 sanity: full row 7
  memset(hA, ZRO, 512); memset(hB, ONE, 512);
  for (int k = 0; k < 32; ++k) hA[7 * 32 + k] = 1;
  run(1);
  dump_nz(1, "S2 i8: A row7=1 (all k), B=1 (expect full sums 32):");

  // S3: i8 single element A[7][3]=1
  memset(hA, ZRO, 512); memset(hB, ONE, 512);
  hA[7 * 32 + 3] = 1;
  run(1);
  printf("S3 i8: A[7][3]=1, B=1 ->");
  for (int l = 0; l < 64; ++l)
    for (int i = 0; i < 4; ++i)
      if (hCi[l*4+i] != 0) printf(" (%d,%d)=%d", l, i, hCi[l*4+i]);
  printf("\n");
  return 0;
}
