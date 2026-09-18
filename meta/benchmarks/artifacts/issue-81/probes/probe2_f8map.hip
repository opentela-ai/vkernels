// probe2_f8map.hip -- deduce the gfx942 v_mfma_f32_16x16x32_fp8_fp8 A/B/C
// fragment layout empirically (issue #81). Structured inputs make each
// experiment's output reveal the mapping:
//   E1: A=B=1.0            -> C[r][c] should be 32.0 everywhere (byte coverage)
//   E4: A[r][k]=v(r), B=1  -> C[r][c]=32*v(r) reveals the ROW map
//   E5: A=1, B[c][k]=w(c)  -> C[r][c]=32*w(c) reveals the COLUMN map
//   E2: A[r][k]=[k==r],B=1 -> C[r][c]=1 tests the K-col group per row
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
                      const uint8_t* __restrict__ Bq,  // [16][32] (N x K)
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

static void run(const uint8_t* hA, const uint8_t* hB, float* hC,
                uint8_t* dA, uint8_t* dB, float* dC) {
  hipMemcpy(dA, hA, 512, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, 512, hipMemcpyHostToDevice);
  hipMemset(dC, 0, 256 * 4);
  hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, dA, dB, dC);
  hipDeviceSynchronize();
  hipMemcpy(hC, dC, 256 * 4, hipMemcpyDeviceToHost);
}

int main() {
  uint8_t hA[512], hB[512]; float hC[256];
  uint8_t *dA, *dB; float* dC;
  hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dC, 1024);
  const uint8_t ONE = 0x40, ZRO = 0x00;   // fnuz 1.0 / +0.0

  // E1: all ones
  memset(hA, ONE, 512); memset(hB, ONE, 512);
  run(hA, hB, hC, dA, dB, dC);
  int bad = 0; for (int i = 0; i < 256; ++i) if (fabsf(hC[i] - 32.0f) > 1e-3) ++bad;
  printf("E1 all-ones: positions != 32.0 : %d/256\n", bad);

  // E4: A row-constant v(r)=r+1 (1..16 exact in fnuz? 16 = 2^4*1.0 exact),
  //     B all ones -> C[r][c] = 32*(r+1) IF row map is identity
  memset(hB, ONE, 512);
  for (int r = 0; r < 16; ++r) {
    uint8_t v;
    switch (r) {  // 1..16, exact e4m3fnuz values
      case 0: v = 0x40; break;  case 1: v = 0x48; break;  // 1, 1.5? use integers:
      default: v = 0; }
    (void)v;
  }
  // integers 1..16: e4m3fnuz value n = 2^(e-8)*(1+m/8); 1=0x40, 2=0x50,
  // 3=0x54, 4=0x60, 5=0x64, 6=0x68, 7=0x6C, 8=0x70, 9=0x74, 10=0x76(1.25*8),
  // ... only powers-of-two-ish are trivially encodable; instead use v(r) = 2^r? too big.
  // Use 1.0 for even rows, 2.0 for odd rows -> C=32 (even row) / 64 (odd row).
  memset(hA, ZRO, 512);
  for (int r = 0; r < 16; ++r) {
    uint8_t v = (r & 1) ? 0x50 : 0x40;  // 2.0 : 1.0
    for (int k = 0; k < 32; ++k) hA[r * 32 + k] = v;
  }
  run(hA, hB, hC, dA, dB, dC);
  printf("E4 row-map (dev C[r][*] value, expect 32 even / 64 odd):\n");
  for (int r = 0; r < 16; ++r) {
    printf("  C[%2d][*]:", r);
    for (int c = 0; c < 16; c += 4) printf(" %5.0f/%5.0f/%5.0f/%5.0f",
      hC[r*16+c], hC[r*16+c+1], hC[r*16+c+2], hC[r*16+c+3]);
    printf("\n");
  }

  // E5: A all ones; B col-constant w(c)=2.0 (odd col) / 1.0 (even col)
  memset(hA, ONE, 512); memset(hB, ZRO, 512);
  for (int c = 0; c < 16; ++c) {
    uint8_t v = (c & 1) ? 0x50 : 0x40;
    for (int k = 0; k < 32; ++k) hB[c * 32 + k] = v;
  }
  run(hA, hB, hC, dA, dB, dC);
  printf("E5 col-map (dev C[*][c], expect 32 even / 64 odd):\n");
  for (int r = 0; r < 4; ++r) {
    printf("  C[%2d][*]:", r);
    for (int c = 0; c < 16; ++c) printf(" %4.0f", hC[r*16+c]);
    printf("\n");
  }

  // E2: A[r][k] = (k == r), B all ones -> C[r][c] = 1.0 if row map identity
  // AND the byte at k=r is live at that row's lanes.
  memset(hA, ZRO, 512); memset(hB, ONE, 512);
  for (int r = 0; r < 16; ++r) hA[r * 32 + r] = ONE;
  run(hA, hB, hC, dA, dB, dC);
  printf("E2 diag (C[r][c] should be 1.0 for all c if row r reads k=r):\n");
  for (int r = 0; r < 16; ++r) {
    printf("  C[%2d][*]:", r);
    for (int c = 0; c < 16; ++c) printf(" %3.0f", hC[r*16+c]);
    printf("\n");
  }
  return 0;
}
