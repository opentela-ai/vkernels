// probe9_benchcheck.hip -- is the mfma8 launch actually running in the
// bench contexts that report 0.0 us? Event-timed single launches at
// split=1/8/16/32/64 with output checksums + hipGetLastError checks.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "vkernels/kernels/dsa.hpp"

#define CK(e, m) do { hipError_t r = (e); if (r != hipSuccess) { \
  std::printf("%s: %s\n", m, hipGetErrorString(r)); std::exit(1); } } while (0)

static uint8_t q_u8[64 * 128], kv_u8[64 * 64 * 132], w[64 * 4];
static int32_t sl[1], pt[64];

int main() {
  const int bs = 1, H = 32, D = 128, B = 64, mt = 64, nb = 64;
  const int max_seq_len = mt * B;  // 4096
  for (size_t i = 0; i < sizeof(q_u8); ++i) q_u8[i] = (uint8_t)(i * 7 + 1);
  for (size_t i = 0; i < sizeof(kv_u8); ++i) kv_u8[i] = (uint8_t)(i * 13 + 3);
  for (int i = 0; i < H; ++i) ((float*)w)[i] = 0.5f;
  sl[0] = max_seq_len;
  for (int i = 0; i < mt; ++i) pt[i] = (int32_t)((i * 13) % nb);

  uint8_t *dq, *dkv; float *dw, *dout; int32_t *dsl, *dpt;
  CK(hipMalloc(&dq, sizeof(q_u8)), "q");
  CK(hipMalloc(&dkv, sizeof(kv_u8)), "kv");
  CK(hipMalloc(&dw, sizeof(w)), "w");
  CK(hipMalloc(&dsl, 4), "sl");
  CK(hipMalloc(&dpt, sizeof(pt)), "pt");
  CK(hipMalloc(&dout, max_seq_len * 4), "out");
  CK(hipMemcpy(dq, q_u8, sizeof(q_u8), hipMemcpyHostToDevice), "cq");
  CK(hipMemcpy(dkv, kv_u8, sizeof(kv_u8), hipMemcpyHostToDevice), "ckv");
  CK(hipMemcpy(dw, w, sizeof(w), hipMemcpyHostToDevice), "cw");
  CK(hipMemcpy(dsl, sl, 4, hipMemcpyHostToDevice), "csl");
  CK(hipMemcpy(dpt, pt, sizeof(pt), hipMemcpyHostToDevice), "cpt");

  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0), "e0"); CK(hipEventCreate(&e1), "e1");

  for (int split : {1, 8, 16, 32, 64}) {
    // warmup
    for (int i = 0; i < 30; ++i) {
      CK(hipMemset(dout, 0, max_seq_len * 4), "zero");
      vkernels::kernels::hip::dsa_topk_logits_with_variant(
          bs, H, D, B, mt, max_seq_len, split, dq, dkv, dw, dsl, dpt, dout, 4);
    }
    CK(hipDeviceSynchronize(), "warmup sync");
    hipError_t le = hipGetLastError();
    std::printf("split=%2d warmup lasterr=%s\n", split, hipGetErrorString(le));

    double sum = 0; int n = 0, zeros = 0;
    float checksum = 0;
    for (int i = 0; i < 20; ++i) {
      CK(hipMemset(dout, 0, max_seq_len * 4), "zero");
      CK(hipEventRecord(e0), "rec0");
      vkernels::kernels::hip::dsa_topk_logits_with_variant(
          bs, H, D, B, mt, max_seq_len, split, dq, dkv, dw, dsl, dpt, dout, 4);
      CK(hipEventRecord(e1), "rec1");
      CK(hipEventSynchronize(e1), "sync1");
      float ms = 0;
      CK(hipEventElapsedTime(&ms, e0, e1), "elapsed");
      if (ms <= 0.0f) ++zeros; else { sum += ms * 1000.0f; ++n; }
      if (i == 19) {
        std::vector<float> out(max_seq_len);
        CK(hipMemcpy(out.data(), dout, max_seq_len * 4, hipMemcpyDeviceToHost), "rd");
        double s = 0; int nz = 0;
        for (float v : out) { s += v; if (v != 0) ++nz; }
        checksum = (float)s;
        // count of nonzero slots must equal seq_len (one logit per token)
        std::printf("           nz=%d/%d checksum=%.3f\n", nz, max_seq_len, s);
      }
    }
    std::printf("split=%2d  mean_us=%.2f over n=%d (zero-elapsed samples: %d)\n",
                split, n ? sum / n : -1.0, n, zeros);
    (void)checksum;
  }
  return 0;
}
