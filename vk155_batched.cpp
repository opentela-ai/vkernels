// Issue #155 batched probe: hip::dsa_sparse_fwd decode latency wall.
// Batched timing methodology (see docs/performance/dsa/gfx942.md):
//   - ~15 s of sustained launches FIRST so DVFS reaches boost sclk
//   - 20 warmup launches + hipDeviceSynchronize per batch
//   - 1000 launches inside ONE hipEvent pair; >=4 consecutive batches
//   - output checksum (D2H sum) to prove the launches are real
#include <hip/hip_runtime.h>
#include <vkernels/kernels/dsa.hpp>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <functional>

using namespace vkernels::kernels;

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

static void batched(const char* tag, int batches, int per_batch,
                    const std::function<void()>& launch,
                    const uint16_t* d_out, size_t n_out) {
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  printf("  %s\n", tag);
  for (int k = 0; k < batches; ++k) {
    for (int w = 0; w < 20; ++w) launch();
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < per_batch; ++i) launch();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms = 0; CK(hipEventElapsedTime(&ms, a, b));
    printf("    batch %d: %.2f us/launch", k, ms * 1000.0 / per_batch);
    if (k == batches - 1 && d_out && n_out) {
      std::vector<uint16_t> chk(n_out);
      CK(hipMemcpy(chk.data(), d_out, n_out * 2, hipMemcpyDeviceToHost));
      double s = 0; for (uint16_t v : chk) s += (double)v;  // raw bf16 bits
      printf("   out-bitsum=%.0f", s);
    }
    printf("\n");
  }
}

struct Bufs {
  uint16_t *dq = nullptr, *dkv = nullptr, *dout = nullptr;
  int32_t* didx = nullptr;
  float* dlse = nullptr;
  float *dpart_out = nullptr, *dpart_lse = nullptr;
  size_t cap_q = 0, cap_kv = 0, cap_out = 0, cap_idx = 0;
};

static void ensure(Bufs& B, size_t nq, size_t nkv, size_t nout, size_t nidx) {
  if (nq > B.cap_q) { if (B.dq) hipFree(B.dq); CK(hipMalloc(&B.dq, nq * 2)); B.cap_q = nq; }
  if (nkv > B.cap_kv) { if (B.dkv) hipFree(B.dkv); CK(hipMalloc(&B.dkv, nkv * 2)); B.cap_kv = nkv; }
  if (nout > B.cap_out) { if (B.dout) hipFree(B.dout); CK(hipMalloc(&B.dout, nout * 2)); B.cap_out = nout; }
  if (nidx > B.cap_idx) { if (B.didx) hipFree(B.didx); CK(hipMalloc(&B.didx, nidx * 4)); B.cap_idx = nidx; }
  if (!B.dlse) CK(hipMalloc(&B.dlse, (size_t)8192 * 64 * 4));
}

// One shape: measure plain dsa_sparse_fwd (the public entry) and, for
// reference, dsa_sparse_fwd_split at the split_for recommendation.
static void run_shape(Bufs& B, int S_q, int S_kv, int H, int dim, int tail_dim,
                      int topk, bool with_split_ref) {
  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;
  const float LOG2E = 1.4426950408889634f;
  const float sm_scale = (1.0f / std::sqrt((float)W)) * LOG2E;
  const size_t nq = (size_t)S_q * H * W, nkv = (size_t)S_kv * W;
  const size_t nout = (size_t)S_q * H * d_v, nidx = (size_t)S_q * topk;
  ensure(B, nq, nkv, nout, nidx);

  std::vector<uint16_t> q(nq), kv(nkv);
  std::vector<int32_t> idx(nidx);
  // bf16 bit patterns from fp32 sources (kernel takes raw bf16 bits).
  auto to_bf = [](float v) {
    uint32_t b; std::memcpy(&b, &v, 4);
    uint32_t lsb = (b >> 16) & 1; b += 0x7FFFu + lsb; return (uint16_t)(b >> 16);
  };
  for (size_t i = 0; i < nq; ++i) q[i] = to_bf(rnd(1, (int)(i % 1000003)));
  for (size_t i = 0; i < nkv; ++i) kv[i] = to_bf(rnd(2, (int)(i % 1000003)));
  for (int i = 0; i < S_q; ++i)
    for (int k = 0; k < topk; ++k)
      idx[(size_t)i * topk + k] =
          (k % 10 == 9) ? -1 : (int32_t)(((unsigned)(i * 31 + k * 7)) % (unsigned)S_kv);
  CK(hipMemcpy(B.dq, q.data(), nq * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(B.dkv, kv.data(), nkv * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(B.didx, idx.data(), nidx * 4, hipMemcpyHostToDevice));

  char tag[160];
  snprintf(tag, sizeof tag, "dsa_sparse_fwd S_q=%d S_kv=%d H=%d dim=%d tail=%d topk=%d:",
           S_q, S_kv, H, dim, tail_dim, topk);
  batched(tag, 4, 1000, [&] {
    hip::dsa_sparse_fwd(S_q, S_kv, H, dim, tail_dim, topk, 1, 64, 1, sm_scale,
                        false, B.dq, B.dkv, B.didx, B.dout, B.dlse);
  }, B.dout, nout);

  if (with_split_ref) {
    int split = dsa_sparse_fwd_split_for(S_q, H, topk, 64, 0, dim, tail_dim);
    size_t pout = (size_t)S_q * H * split * d_v, plse = (size_t)S_q * H * split;
    CK(hipMalloc(&B.dpart_out, pout * 4)); CK(hipMalloc(&B.dpart_lse, plse * 4));
    snprintf(tag, sizeof tag, "  (ref) dsa_sparse_fwd_split split=%d:", split);
    batched(tag, 4, 1000, [&] {
      hip::dsa_sparse_fwd_split(S_q, S_kv, H, dim, tail_dim, topk, 1, 64, 1,
                                sm_scale, false, split, B.dq, B.dkv, B.didx,
                                B.dout, B.dlse, B.dpart_out, B.dpart_lse);
    }, B.dout, nout);
    hipFree(B.dpart_out); hipFree(B.dpart_lse);
    B.dpart_out = nullptr; B.dpart_lse = nullptr;
  }
}

int main() {
  CK(hipInit(0));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
  printf("=== vk155 batched dsa probe === GPU: %s (%s)\n", p.name, p.gcnArchName);
  const char* env = std::getenv("VK_DSA_DECODE_SPLIT");
  printf("VK_DSA_DECODE_SPLIT=%s\n", env ? env : "(unset)");

  Bufs B;
  // DVFS warmup: ~15 s of sustained load before the first measurement.
  printf("DVFS warmup (~15 s of launches)...\n");
  {
    const int S_q = 1, S_kv = 2048, H = 64, dim = 256, topk = 2048, W = dim;
    const size_t nq = (size_t)S_q * H * W, nkv = (size_t)S_kv * W;
    const size_t nidx = (size_t)S_q * topk;
    ensure(B, nq, nkv, (size_t)S_q * H * dim, nidx);
    std::vector<uint16_t> q(nq, 0x3c00), kv(nkv, 0x3c00);
    std::vector<int32_t> idx(nidx);
    for (int k = 0; k < topk; ++k) idx[k] = (k % 10 == 9) ? -1 : (int32_t)(k % S_kv);
    CK(hipMemcpy(B.dq, q.data(), nq * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(B.dkv, kv.data(), nkv * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(B.didx, idx.data(), nidx * 4, hipMemcpyHostToDevice));
    const float sm_scale = 1.4426950408889634f / std::sqrt((float)dim);
    for (int i = 0; i < 4000; ++i) {
      hip::dsa_sparse_fwd(S_q, S_kv, H, dim, 0, topk, 1, 64, 1, sm_scale, false,
                          B.dq, B.dkv, B.didx, B.dout, B.dlse);
    }
    CK(hipDeviceSynchronize());
  }

  // Serving decode shapes (the issue targets).
  run_shape(B, 1, 256, 64, 256, 0, 256, true);    // topk=256 serving
  run_shape(B, 1, 256, 64, 256, 0, 128, true);    // topk=128 serving
  run_shape(B, 1, 2048, 64, 256, 0, 2048, true);  // full topk=2048
  // DSv3 decode (tail > 0).
  run_shape(B, 1, 256, 16, 576, 64, 256, true);
  // Short + prefill (regression guards).
  run_shape(B, 64, 256, 64, 256, 0, 128, false);      // short
  run_shape(B, 8192, 4096, 1, 256, 0, 128, false);    // prefill (doc prefill row)

  printf("=== vk155 done ===\n");
  return 0;
}
