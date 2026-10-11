// meta/benchmarks/probe_mfma_fp8.hip
//
// Empirical fragment-layout probe for v_mfma_f32_16x16x32_fp8_fp8 on
// gfx942 (issue #148). The CDNA wave64 fragment mapping for the fp8 MFMA
// is not spelled out in the rocwmma headers (they only wrap the builtin),
// so this tool DECODES it by construction:
//   * blocks 1:   A byte (lane l, j) = fp8(l%16 + 1), B = all 1.0
//                -> C[lane][slot] = 32*(m+1) reveals the C/A row mapping
//   * blocks 2:   A = all 1.0, B byte = fp8(l%16 + 1)
//                -> C[lane][slot] = 32*(n+1) reveals the C/B column mapping
//   * blocks 3+:  A = ONE-HOT at (lane LP, byte JP); B byte (l, j) =
//                enc(h(tau(l, j))) with h injective over k = 0..31 and
//                tau the candidate k formula. C[m1][n] = h(k1) where k1 is
//                the TRUE k of the A one-hot position -> inverting h gives
//                sigma_A(LP, JP) = (m1, k1) elementwise.
// Host prints the decoded tables. Wave64 is verified via __ballot.
//
// USAGE: probe_mfma_fp8

#include "vkernels/kernels/glm_moe.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#define CK(e, m)                                                                                   \
  do {                                                                                             \
    hipError_t r = (e);                                                                            \
    if (r != hipSuccess) {                                                                         \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));                                   \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

namespace {

typedef float v4f __attribute__((vector_size(16)));
__device__ constexpr v4f kVZero{0.f, 0.f, 0.f, 0.f};

__global__ void probe_kernel(const unsigned long long* a, const unsigned long long* b, float* out) {
  const int blk = blockIdx.x;
  const int lane = threadIdx.x;
  if (blk == 0) { // wave-size check
    // NOTE: the ballot must be executed by ALL lanes of the wave — under
    // divergent execution it would only ever report the participating
    // subset (the lane-0-only versionPrinted 1 on gfx942).
    const unsigned long long m = __ballot(true);
    if (lane == 0) {
      int pop = 0;
      for (int i = 0; i < 64; ++i)
        pop += (int)((m >> i) & 1ull);
      out[0] = (float)pop;
    }
    return;
  }
  const v4f r = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8((long long)a[(size_t)blk * 64 + lane],
                                                           (long long)b[(size_t)blk * 64 + lane],
                                                           kVZero,
                                                           0,
                                                           0,
                                                           0);
  float* o = out + (size_t)blk * 256 + lane * 4;
  o[0] = r[0];
  o[1] = r[1];
  o[2] = r[2];
  o[3] = r[3];
}

// host-side e4m3fnuz encoder for exactly-representable positive values
uint8_t fnuz_enc(float v) {
  if (v == 0.0f) return 0x00u;
  int e;
  std::frexp(v, &e);       // v = f * 2^e, f in [0.5, 1)
  const int E = e - 1 + 8; // bias 8
  if (E < 1 || E > 15) {
    std::printf("enc overflow %g\n", v);
    std::exit(1);
  }
  const float man = v / std::ldexp(1.0f, E - 8) - 1.0f; // in [0, 1)
  const int m = (int)std::lround(man * 8.0f);
  if (m < 0 || m > 7) {
    std::printf("enc mantissa %g\n", v);
    std::exit(1);
  }
  return (uint8_t)((E << 3) | m);
}

float fnuz_dec(uint8_t code) { // for the probe's own sanity check
  const int E = (code >> 3) & 15, m = code & 7;
  return std::ldexp(1.0f + m / 8.0f, E - 8);
}

} // namespace

int run_probe() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  // ---- build the encoded B for the k-sweep: h(k) injective over 0..31
  // h(k) = 2^((k%8)-3) * mult[k/8], mult = {1,3,5,7}  (odd -> unique)
  float hk[32];
  const float mult[4] = {1.f, 3.f, 5.f, 7.f};
  for (int k = 0; k < 32; ++k)
    hk[k] = std::ldexp(1.0f, (k % 8) - 3) * mult[k / 8];
  uint8_t henc[32];
  for (int k = 0; k < 32; ++k)
    henc[k] = fnuz_enc(hk[k]);
  for (int k = 0; k < 32; ++k) // sanity: decoder round-trips
    if (fnuz_dec(henc[k]) != hk[k]) {
      std::printf("roundtrip fail\n");
      return 1;
    }

  // candidate k formula for the B encoding (any injective guess works)
  auto tau = [](int l, int j) { return 8 * (l / 16) + j; };

  const int NBLK = 3 + 5 * 8; // 1 wave-check + 2 maps + 5 lanes x 8 bytes
  const int kLanes[5] = {0, 4, 16, 32, 48};
  std::vector<unsigned long long> fa((size_t)NBLK * 64, 0), fb((size_t)NBLK * 64, 0);
  for (int blk = 1; blk < NBLK; ++blk) {
    for (int l = 0; l < 64; ++l) {
      uint8_t ab[8] = {0, 0, 0, 0, 0, 0, 0, 0}, bb[8];
      if (blk == 1) { // row decode
        for (int j = 0; j < 8; ++j)
          ab[j] = fnuz_enc((float)(l % 16 + 1));
        for (int j = 0; j < 8; ++j)
          bb[j] = 0x40;      // 1.0
      } else if (blk == 2) { // column decode
        for (int j = 0; j < 8; ++j)
          ab[j] = 0x40;
        for (int j = 0; j < 8; ++j)
          bb[j] = fnuz_enc((float)(l % 16 + 1));
      } else { // k sweep
        const int idx = blk - 3;
        const int lp = kLanes[idx / 8], jp = idx % 8;
        if (l == lp) ab[jp] = 0x40; // one-hot 1.0
        for (int j = 0; j < 8; ++j)
          bb[j] = henc[tau(l, j)];
      }
      unsigned long long av = 0, bv = 0;
      for (int j = 0; j < 8; ++j) {
        av |= (unsigned long long)ab[j] << (8 * j);
        bv |= (unsigned long long)bb[j] << (8 * j);
      }
      fa[(size_t)blk * 64 + l] = av;
      fb[(size_t)blk * 64 + l] = bv;
    }
  }

  unsigned long long *da, *db;
  float* dout;
  std::vector<float> out((size_t)NBLK * 256, 0.0f);
  CK(hipMalloc(&da, fa.size() * 8), "ma");
  CK(hipMalloc(&db, fb.size() * 8), "mb");
  CK(hipMalloc(&dout, out.size() * 4), "mo");
  CK(hipMemcpy(da, fa.data(), fa.size() * 8, hipMemcpyHostToDevice), "ca");
  CK(hipMemcpy(db, fb.data(), fb.size() * 8, hipMemcpyHostToDevice), "cb");
  probe_kernel<<<NBLK, 64>>>(da, db, dout);
  CK(hipDeviceSynchronize(), "sync");
  CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost), "co");

  std::printf("wave64 ballot lanes: %d (device warpSize=%d, expect 64)\n", (int)out[0], p.warpSize);

  // C(lane, slot) -> (m, n): decode from blocks 1 and 2
  // C[m][n] with m/n per lane/slot; print the raw tables first
  auto cval = [&](int blk, int lane, int slot) { return out[(size_t)blk * 256 + lane * 4 + slot]; };
  std::printf("\n-- block 1 (A byte = m+1, B = 1): C = 32*(m+1) --\n");
  for (int l = 0; l < 64; l += 16)
    std::printf("  lane %2d: slots %g %g %g %g\n",
                l,
                cval(1, l, 0),
                cval(1, l, 1),
                cval(1, l, 2),
                cval(1, l, 3));
  std::printf("-- block 2 (A = 1, B byte = n+1): C = 32*(n+1) --\n");
  for (int l = 0; l < 64; l += 16)
    std::printf("  lane %2d: slots %g %g %g %g\n",
                l,
                cval(2, l, 0),
                cval(2, l, 1),
                cval(2, l, 2),
                cval(2, l, 3));

  // k-sweep decode: find the single nonzero C and invert
  std::printf("\n-- k sweep: sigma_A(lane, byte) -> (m, k_true) --\n");
  for (int idx = 0; idx < 40; ++idx) {
    const int blk = 3 + idx;
    const int lp = kLanes[idx / 8], jp = idx % 8;
    int m1 = -1, n1 = -1;
    float v = 0.0f;
    for (int l = 0; l < 64 && m1 < 0; ++l)
      for (int s = 0; s < 4; ++s) {
        const float x = cval(blk, l, s);
        if (std::fabs(x) > 1e-3f) {
          m1 = l % 16;
          n1 = l / 16 * 4 + s;
          v = x;
          break;
        }
      }
    if (m1 < 0) {
      std::printf("  A(%2d,%d): NO OUTPUT\n", lp, jp);
      continue;
    }
    // invert: find (l,j) whose encoded h matches v
    int kl = -1;
    for (int l = 0; l < 64 && kl < 0; ++l)
      for (int j = 0; j < 8; ++j)
        if (hk[tau(l, j)] == v) {
          kl = tau(l, j);
          break;
        }
    std::printf("  A(%2d,%d) -> m=%d n*=%d  value=%g  k_true=%s\n",
                lp,
                jp,
                m1,
                n1,
                v,
                kl >= 0 ? std::to_string(kl).c_str() : "??");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// mode 2 (argv[1] == "dbg"): run the real mfma kernel on the test binary's
// exact data recipe (M=2 N=128 K=128, the smallest failing shape) and dump
// its internal quantized activations + scales, compared against CPU
// replicas (amax over 128 vs the kernel's actual group coverage).
float rnd(int seed, int i) { // matches test_glm_fp8_gemv.hip
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f; // [-1, 1)
}
uint16_t f32_to_bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return (uint16_t)(((u + 0x7fffu + ((u >> 16) & 1)) >> 16));
}
float bf16_bits_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}
// host fnuz decode of arbitrary payload (value = (1+m/8)*2^(E-8), bias 8)
float fnuz_val(uint8_t c) {
  if (c == 0x00u || c == 0x80u) return 0.0f;
  const int E = (c >> 3) & 15, m = c & 7;
  float v = std::ldexp(1.0f + m / 8.0f, E - 8);
  return (c & 0x80u) ? -v : v;
}

int run_dbg() {
  const int M = 2, N = 128, K = 128, sk = 1;
  const size_t nx = (size_t)M * K, nw = (size_t)N * K;
  const size_t ns = (size_t)(N / 128) * (K / 128);
  std::vector<uint16_t> x(nx);
  for (size_t i = 0; i < nx; ++i)
    x[i] = f32_to_bf16_bits(rnd(1, (int)i));
  std::vector<uint8_t> w(nw);
  for (size_t i = 0; i < nw; ++i) {
    w[i] = (uint8_t)(rnd(2, (int)i) * 127.0f + 128.0f);
    if (w[i] == 0x80u) w[i] = 0x00u;
  }
  std::vector<float> scales(ns);
  for (size_t i = 0; i < ns; ++i)
    scales[i] = 0.0625f * (float)(1 + (i % 128));

  uint16_t* dx;
  uint8_t* dw;
  float *ds, *dpart, *ds2a;
  uint32_t* dsxa;
  uint16_t* dout;
  CK(hipMalloc(&dx, nx * 2), "ax");
  CK(hipMalloc(&dw, nw), "aw");
  CK(hipMalloc(&ds, ns * 4), "as");
  CK(hipMalloc(&dout, (size_t)M * N * 2), "ao");
  CK(hipMalloc(&dpart, (size_t)M * N * sk * 4), "ap");
  CK(hipMalloc(&dsxa, nx), "adxa");
  CK(hipMalloc(&ds2a, 64 * 4), "as2a");
  CK(hipMemcpy(dx, x.data(), nx * 2, hipMemcpyHostToDevice), "cx");
  CK(hipMemcpy(dw, w.data(), nw, hipMemcpyHostToDevice), "cw");
  CK(hipMemcpy(ds, scales.data(), ns * 4, hipMemcpyHostToDevice), "cs");
  vkernels::kernels::hip::glm_fp8_fnuz_mfma_block_gemv_debug(
      dx, dw, ds, dout, dpart, M, N, K, sk, dsxa, ds2a);
  CK(hipDeviceSynchronize(), "sync");
  std::vector<uint16_t> host((size_t)M * N);
  CK(hipMemcpy(host.data(), dout, host.size() * 2, hipMemcpyDeviceToHost), "co");
  std::vector<uint32_t> sxa(nx);
  CK(hipMemcpy(sxa.data(), dsxa, nx, hipMemcpyDeviceToHost), "cxa");
  std::vector<float> s2a(2);
  CK(hipMemcpy(s2a.data(), ds2a, 8, hipMemcpyDeviceToHost), "cs2a");

  std::printf("kernel s2a: tok0=%.6g tok1=%.6g\n", s2a[0], s2a[1]);
  // CPU replica of the dequantized activations
  for (int m = 0; m < M; ++m) {
    float amax = 0.0f;
    for (int i = 0; i < K; ++i)
      amax = std::fmax(amax, std::fabs(bf16_bits_to_f32(x[(size_t)m * K + i])));
    std::printf("cpu amax tok%d=%.6g  s2a_should=%.6g\n", m, amax, std::fmax(amax, 1e-4f) / 224.0f);
    std::printf("tok%d payload[0..7] =", m);
    for (int i = 0; i < 8; ++i)
      std::printf(" %02x(%g)",
                  (unsigned)((sxa[(size_t)m * K / 4 + i / 4] >> (8 * (i % 4))) & 0xff),
                  fnuz_val((uint8_t)((sxa[(size_t)m * K / 4 + i / 4] >> (8 * (i % 4))) & 0xff)));
    std::printf("\n       x        [0..7] =");
    for (int i = 0; i < 8; ++i)
      std::printf(" %g", bf16_bits_to_f32(x[(size_t)m * K + i]));
    std::printf("\n");
  }
  // oracle values for the first few outputs
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < 4; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        const uint8_t wc = w[(size_t)n * K + k];
        const float wv = fnuz_val(wc);
        const int blk = (n / 128) * (K / 128) + k / 128;
        acc += bf16_bits_to_f32(x[(size_t)m * K + k]) * scales[blk] * wv;
      }
      uint32_t ub = (uint32_t)host[(size_t)m * N + n] << 16;
      float kv;
      std::memcpy(&kv, &ub, 4);
      std::printf("out[m=%d][n=%d] kernel=%g oracle=%g\n", m, n, kv, acc);
    }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "dbg") return run_dbg();
  return run_probe();
}
