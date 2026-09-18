// probe4_f8decode.hip -- fully decode v_mfma_f32_16x16x32_fp8_fp8 fragment
// maps (issue #81) with information-theoretic probes.
//  P1a-c: lane l's A bytes all = 2^((l>>s) mod 4), s=0,2,4; B=ones. The dev
//         value at each position decodes which lane produced it.
//  P2:    A byte j of EVERY lane = 2^j; B=ones. Dev value = sum of 2^j over
//         the bytes that flow into that output element (binary encoding).
//  P3:    B byte j = 2^j; A=ones. Same for the B operand.
//  P4a-c: B lane-constant 2^((l>>s) mod 4); A=ones.
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
  const int l  = threadIdx.x;
  const int m  = l & 15;
  const int kq = (l >> 4) & 3;
  v2i a, b;
  __builtin_memcpy(&a, A + (size_t)l * 8, 8);   // flat: lane l <- bytes [8l,8l)
  __builtin_memcpy(&b, Bq + (size_t)l * 8, 8);
  v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
  c = mfma_f8(a, b, c);
  const int r0 = (l >> 4) * 4;
  for (int i = 0; i < 4; ++i) C[(size_t)(r0 + i) * 16 + m] = c[i];
}

static uint8_t hA[512], hB[512]; static float hC[256];
static uint8_t *dA, *dB; static float* dC;
static const uint8_t POW[8] = {0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0x78};

static void run() {
  hipMemcpy(dA, hA, 512, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, 512, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, dA, dB, dC);
  hipDeviceSynchronize();
  hipMemcpy(hC, dC, 1024, hipMemcpyDeviceToHost);
}

static void print_mat(const char* tag) {
  printf("%s\n", tag);
  for (int r = 0; r < 16; ++r) {
    printf("  ");
    for (int c = 0; c < 16; ++c) printf("%7.1f", hC[r * 16 + c]);
    printf("\n");
  }
}

int main() {
  hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dC, 1024);
  const uint8_t ONE = 0x40, ZRO = 0x00;

  for (int s = 0; s <= 4; s += 2) {
    memset(hA, ZRO, 512); memset(hB, ONE, 512);
    for (int l = 0; l < 64; ++l) {
      uint8_t v = POW[(l >> s) & 3];
      memset(hA + l * 8, v, 8);
    }
    run();
    char tag[64]; snprintf(tag, 64, "P1%d: A lane-const 2^((l>>%d)&3), B=1", s / 2 + 1, s);
    print_mat(tag);
  }

  memset(hA, ZRO, 512); memset(hB, ONE, 512);
  for (int l = 0; l < 64; ++l)
    for (int j = 0; j < 8; ++j) hA[l * 8 + j] = POW[j];
  run();
  print_mat("P2: A byte j = 2^j (all lanes), B=1  (value = binary code of byte set)");

  memset(hA, ONE, 512); memset(hB, ZRO, 512);
  for (int l = 0; l < 64; ++l)
    for (int j = 0; j < 8; ++j) hB[l * 8 + j] = POW[j];
  run();
  print_mat("P3: B byte j = 2^j (all lanes), A=1");

  for (int s = 0; s <= 4; s += 2) {
    memset(hA, ONE, 512); memset(hB, ZRO, 512);
    for (int l = 0; l < 64; ++l) {
      uint8_t v = POW[(l >> s) & 3];
      memset(hB + l * 8, v, 8);
    }
    run();
    char tag[64]; snprintf(tag, 64, "P4%d: B lane-const 2^((l>>%d)&3), A=1", s / 2 + 1, s);
    print_mat(tag);
  }
  return 0;
}
