// probe_f8mfma.hip -- issue #81: empirically confirm the gfx942 (CDNA3)
// fp8 MFMA (v_mfma_f32_16x16x32_f8) fragment layout + FNUZ numerics before
// building the dsa_topk_logits mfma8 kernel on it.
//
// Method mirrors gemm_bf16.hip's confirmed-fragment approach: ONE wavefront
// runs ONE 16x16x32_f8 MFMA on pseudo-random (non-NaN) e4m3fnuz bytes; the
// host computes the expected fp32 C tile from the SAME dequant helper the
// production kernels use (fp8e4m3fnuz_to_f32). Assumed layout (analogue of
// the confirmed bf16 16x16x16 one, element width 2 -> 1, kMfmaK 16 -> 32):
//   A frag (M=16 rows, K=32): lane l holds a[0..7] = A[l&15][((l>>4)&3)*8+i]
//   B frag (K=32, N=16):      lane l holds b[0..7] = Bq[l&15][((l>>4)&3)*8+i]
//       (Bq is Q's NATIVE [N][K] layout -- the fp8 kernel needs NO transpose)
//   C frag: c[i] = C[((l>>4)*4+i)][l&15]   (same as bf16)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int   v2i __attribute__((ext_vector_type(2)));

__host__ __device__ __forceinline__ float fp8e4m3fnuz_to_f32(uint8_t b) {
  const uint32_t s = static_cast<uint32_t>(b >> 7) & 1u;
  const uint32_t e = static_cast<uint32_t>(b >> 3) & 0xFu;
  const uint32_t m = static_cast<uint32_t>(b) & 0x7u;
  if ((b & 0x7Fu) == 0u) return 0.0f;
  float f;
  if (e == 15u && m == 7u) {
    const uint32_t qnan = 0x7fc00000u;
    __builtin_memcpy(&f, &qnan, sizeof(f));
  } else if (e == 0u) {
    const float v = static_cast<float>(m) * 0x1p-7f;
    f = s ? -v : v;
  } else {
    const uint32_t bits = (s << 31) | ((e + 119u) << 23) | (m << 20);
    __builtin_memcpy(&f, &bits, sizeof(f));
  }
  return f;
}

__device__ __forceinline__ v4f mfma_f8_asm(v2i a, v2i b, v4f c) {
  asm volatile("v_mfma_f32_16x16x32_fp8_fp8 %0, %1, %2, %0"
               : "+v"(c)
               : "v"(a), "v"(b));
  return c;
}

#ifdef F8_BUILTIN
__device__ __forceinline__ v4f mfma_f8_builtin(v2i a, v2i b, v4f c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_f8(a, b, c, 0, 0);
}
#endif

__global__ void probe(const uint8_t* __restrict__ A,   // [16][32] row-major
                      const uint8_t* __restrict__ Bq,  // [16][32] (N x K)
                      float* __restrict__ C) {
  const int l  = threadIdx.x;
  const int m  = l & 15;
  const int kq = (l >> 4) & 3;
  v2i a, b;
  __builtin_memcpy(&a, A + (size_t)m * 32 + kq * 8, 8);
  __builtin_memcpy(&b, Bq + (size_t)m * 32 + kq * 8, 8);
  v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
#ifdef F8_BUILTIN
  c = mfma_f8_builtin(a, b, c);
#else
  c = mfma_f8_asm(a, b, c);
#endif
  const int r0 = (l >> 4) * 4;
  for (int i = 0; i < 4; ++i) C[(size_t)(r0 + i) * 16 + m] = c[i];
}

static uint8_t rnd_fp8_nonnan(unsigned& s) {
  s = s * 1664525u + 1013904223u;
  uint8_t b = (uint8_t)(s >> 24);
  if (b == 0x7F) b = 0x3C;   // scrub the fnuz NaN encoding
  return b;
}

int main() {
  uint8_t hA[16 * 32], hB[16 * 32];
  float hC[16 * 16], hRef[16 * 16];
  unsigned s = 81;
  for (int i = 0; i < 16 * 32; ++i) hA[i] = rnd_fp8_nonnan(s);
  for (int i = 0; i < 16 * 32; ++i) hB[i] = rnd_fp8_nonnan(s);

  // host reference: C[r][c] = sum_k fA[r][k] * fB[c][k], fp32, natural order
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 16; ++c) {
      float acc = 0.0f;
      for (int k = 0; k < 32; ++k)
        acc += fp8e4m3fnuz_to_f32(hA[r * 32 + k]) *
               fp8e4m3fnuz_to_f32(hB[c * 32 + k]);
      hRef[r * 16 + c] = acc;
    }

  uint8_t *dA, *dB; float* dC;
  hipMalloc(&dA, sizeof(hA)); hipMalloc(&dB, sizeof(hB)); hipMalloc(&dC, sizeof(hC));
  hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
  hipMemset(dC, 0, sizeof(hC));
  hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, dA, dB, dC);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) { printf("LAUNCH FAILED: %s\n", hipGetErrorString(err)); return 2; }
  hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost);

  double maxdiff = 0.0; int nbad = 0, nzero = 0;
  for (int i = 0; i < 256; ++i) {
    double d = fabs((double)hC[i] - (double)hRef[i]);
    if (hC[i] == 0.0f && hRef[i] != 0.0f) ++nzero;
    if (d > maxdiff) maxdiff = d;
    if (d > 1e-3 * (fabs(hRef[i]) + 1.0)) ++nbad;
  }
  printf("probe: nbad=%d/256 nzero-when-nonzero=%d maxdiff=%.6g\n", nbad, nzero, maxdiff);
  printf("  sample C[0][0]: dev=%.6f ref=%.6f | C[15][15]: dev=%.6f ref=%.6f\n",
         hC[0], hRef[0], hC[255], hRef[255]);
  printf("  %s\n", nbad == 0 ? "LAYOUT+FNUZ CONFIRMED"
                             : "MISMATCH -- dump follows (dev/ref per row 0..3, col 0..7)");
  if (nbad) {
    for (int r = 0; r < 4; ++r) {
      printf("  row %d: ", r);
      for (int c = 0; c < 8; ++c) printf("(%.3f/%.3f) ", hC[r * 16 + c], hRef[r * 16 + c]);
      printf("\n");
    }
    return 1;
  }
  return 0;
}
