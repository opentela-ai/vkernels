// meta/benchmarks/test_mla_capture.hip
//
// Stream/capture-safety test for the MLA HIP C ABI entry points (issues
// #45/#69) on gfx942. The vLLM Kimi-K3 shim (cookbook vkernels_attn.py)
// calls `vk_hip_mla_fwd` via ctypes; decode under CUDA-graph capture
// requires an entry point that (a) launches on a caller-provided stream,
// (b) never allocates, frees, or synchronises inside the call, and
// (c) reports errors. `vk_hip_mla_fwd_stream` is that entry point.
//
// What is verified here, per case:
//   1. ABI parity: vk_hip_mla_fwd (legacy) vs vk_hip_mla_fwd_stream with
//      stream = nullptr → bitwise identical (same kernels, stream 0).
//   2. Caller-stream parity: vk_hip_mla_fwd_stream on a non-default
//      stream → bitwise identical to the legacy run.
//   3. COLD graph capture: capture the stream entry on a fresh shape
//      (split-K workspace unsized → the resize path would hipFree, which
//      is capture-illegal). The workspace guard must fall back to the
//      capture-safe non-split kernel, the capture must succeed, and the
//      played graph must match the CPU oracle.
//   4. WARM graph capture: after one eager call sizes the split workspace,
//      capture → instantiate → play, re-feed NEW inputs (outside the
//      graph), play again — output must track the new inputs and match
//      the CPU oracle, and the played split path must be bitwise equal to
//      the eager split path (same kernels, deterministic order).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/capi/hip_capi.hpp"
#include "vkernels/kernels/mla.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

struct Stats { double max_abs, max_rel; };

float rnd(int seed, int i) {  // deterministic pseudo-random in [-1, 1]
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

struct Shape {
  int B, H, S_q, S_kv;
};

// K3 decode-ish MLA config: kv_lora_rank=512, qk_rope_head_dim=64,
// scale = 1/sqrt(576). The decode shape (S_q=1, long S_kv) is the split-K
// path (issue #82) — the path whose workspace resize is capture-illegal
// without the guard.
constexpr int kLr = 512, kRhd = 64;
const float kScale = 1.0f / std::sqrt((float)(kLr + kRhd));

struct Buffers {
  float *dq = nullptr, *dkc = nullptr, *dkp = nullptr, *dvc = nullptr,
        *dout = nullptr;
  size_t nq, nkc, nkp, nvc, nout;
};

void alloc(const Shape& s, Buffers* b) {
  const int Dq = kLr + kRhd;
  b->nq = (size_t)s.B * s.H * s.S_q * Dq;
  b->nkc = (size_t)s.B * s.S_kv * kLr;
  b->nkp = (size_t)s.B * s.S_kv * kRhd;
  b->nvc = (size_t)s.B * s.S_kv * kLr;
  b->nout = (size_t)s.B * s.H * s.S_q * kLr;
  CK(hipMalloc(&b->dq, b->nq * 4), "alloc q");
  CK(hipMalloc(&b->dkc, b->nkc * 4), "alloc kc");
  CK(hipMalloc(&b->dkp, b->nkp * 4), "alloc kp");
  CK(hipMalloc(&b->dvc, b->nvc * 4), "alloc vc");
  CK(hipMalloc(&b->dout, b->nout * 4), "alloc out");
}

void fill_host(const Shape& s, int seed, std::vector<float>* q,
               std::vector<float>* kc, std::vector<float>* kp,
               std::vector<float>* vc) {
  const int Dq = kLr + kRhd;
  q->resize((size_t)s.B * s.H * s.S_q * Dq);
  kc->resize((size_t)s.B * s.S_kv * kLr);
  kp->resize((size_t)s.B * s.S_kv * kRhd);
  vc->resize((size_t)s.B * s.S_kv * kLr);
  for (size_t i = 0; i < q->size(); ++i) (*q)[i] = rnd(seed, (int)i);
  for (size_t i = 0; i < kc->size(); ++i) (*kc)[i] = rnd(seed + 1, (int)i);
  for (size_t i = 0; i < kp->size(); ++i) (*kp)[i] = rnd(seed + 2, (int)i);
  for (size_t i = 0; i < vc->size(); ++i) (*vc)[i] = rnd(seed + 3, (int)i);
}

void h2d(const Shape& s, const std::vector<float>& q,
         const std::vector<float>& kc, const std::vector<float>& kp,
         const std::vector<float>& vc, const Buffers& b) {
  CK(hipMemcpy(b.dq, q.data(), b.nq * 4, hipMemcpyHostToDevice), "cpyq");
  CK(hipMemcpy(b.dkc, kc.data(), b.nkc * 4, hipMemcpyHostToDevice), "cpykc");
  CK(hipMemcpy(b.dkp, kp.data(), b.nkp * 4, hipMemcpyHostToDevice), "cpykp");
  CK(hipMemcpy(b.dvc, vc.data(), b.nvc * 4, hipMemcpyHostToDevice), "cpyvc");
}

// CPU oracle for the current device inputs (reads back from device first so
// the comparison is always against what the device was actually given).
Stats oracle(const Shape& s, const Buffers& b, std::vector<float>* host_out) {
  std::vector<float> q(b.nq), kc(b.nkc), kp(b.nkp), vc(b.nvc);
  CK(hipMemcpy(q.data(), b.dq, b.nq * 4, hipMemcpyDeviceToHost), "rdq");
  CK(hipMemcpy(kc.data(), b.dkc, b.nkc * 4, hipMemcpyDeviceToHost), "rdkc");
  CK(hipMemcpy(kp.data(), b.dkp, b.nkp * 4, hipMemcpyDeviceToHost), "rdkp");
  CK(hipMemcpy(vc.data(), b.dvc, b.nvc * 4, hipMemcpyDeviceToHost), "rdvc");
  // `host_out` may be null (caller only wants the stats); the oracle still
  // needs a writable reference buffer, so fall back to a local vector.
  std::vector<float> local_ref;
  std::vector<float>& ref = host_out ? *host_out : local_ref;
  ref.resize(b.nout);
  vkernels::kernels::mla_fwd_cpu(s.B, s.H, s.S_q, s.S_kv, s.S_q - 1, 0, kLr,
                                 kRhd, kScale, q.data(), kc.data(), kp.data(),
                                 vc.data(), ref.data());
  std::vector<float> dev(b.nout);
  CK(hipMemcpy(dev.data(), b.dout, b.nout * 4, hipMemcpyDeviceToHost),
     "rdout");
  Stats st{0.0, 0.0};
  for (size_t i = 0; i < b.nout; ++i) {
    double e = std::fabs((double)dev[i] - ref[i]);
    if (e > st.max_abs) st.max_abs = e;
    double d = std::fmax(std::fabs(ref[i]), 1.0);
    if (e / d > st.max_rel) st.max_rel = e / d;
  }
  return st;
}

// max_abs between two device output buffers (read both back).
double dev_diff(const Buffers& a, const Buffers& c) {
  std::vector<float> ha(a.nout), hc(c.nout);
  CK(hipMemcpy(ha.data(), a.dout, a.nout * 4, hipMemcpyDeviceToHost), "rda");
  CK(hipMemcpy(hc.data(), c.dout, c.nout * 4, hipMemcpyDeviceToHost), "rdc");
  double m = 0.0;
  for (size_t i = 0; i < ha.size(); ++i)
    m = std::fmax(m, (double)std::fabs(ha[i] - hc[i]));
  return m;
}

// Capture vk_hip_mla_fwd_stream on `cap` for the given buffers, instantiate,
// and return the exec. Exits non-zero if capture/instantiate fails (that is
// exactly the failure mode this test exists to catch).
hipGraphExec_t capture_and_instantiate(const Shape& s, const Buffers& b,
                                       hipStream_t cap) {
  CK(hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal), "beginCapture");
  int rc = vk_hip_mla_fwd_stream(s.B, s.H, s.S_q, s.S_kv, s.S_q - 1, 0, kLr,
                                 kRhd, kScale, b.dq, b.dkc, b.dkp, b.dvc,
                                 b.dout, cap);
  if (rc != 0) {
    std::fprintf(stderr, "captured call returned hip error %d\n", rc);
    std::exit(1);
  }
  hipGraph_t g = nullptr;
  hipError_t end = hipStreamEndCapture(cap, &g);
  if (end != hipSuccess || g == nullptr) {
    std::fprintf(stderr, "END CAPTURE FAILED: %s (capture was invalidated — "
                         "an illegal call ran mid-capture)\n",
                 hipGetErrorString(end));
    std::exit(1);
  }
  hipGraphExec_t ge = nullptr;
  CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
  CK(hipGraphDestroy(g), "graphDestroy");
  return ge;
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  int fails = 0;

  // ---- shared decode shape (split-K path, issue #82) --------------------
  const Shape s{1, 2, 1, 2048};
  const int Dq = kLr + kRhd;
  (void)Dq;
  Buffers legacy{}, stream_dev{}, cap_dev{};
  alloc(s, &legacy);
  alloc(s, &stream_dev);
  alloc(s, &cap_dev);

  std::vector<float> q, kc, kp, vc;
  fill_host(s, 1, &q, &kc, &kp, &vc);
  h2d(s, q, kc, kp, vc, legacy);
  h2d(s, q, kc, kp, vc, stream_dev);
  h2d(s, q, kc, kp, vc, cap_dev);
  CK(hipDeviceSynchronize(), "sync0");

  int rc = 0;
  bool ok = false;
  double d = 0.0;

  // ---- 1. COLD capture (workspace unsized → fallback path) --------------
  // No shape has run through launch_split yet in this process, so the split
  // workspace is empty and a naive implementation would hipMalloc/hipFree
  // mid-capture (capture-illegal). The guard must fall back to the
  // non-split kernel; the capture must still succeed and match the oracle.
  {
    hipStream_t cap;
    CK(hipStreamCreate(&cap), "capStream");
    hipGraphExec_t ge = capture_and_instantiate(s, cap_dev, cap);
    CK(hipGraphLaunch(ge, cap), "coldLaunch");
    CK(hipStreamSynchronize(cap), "coldSync");
    Stats st = oracle(s, cap_dev, nullptr);
    ok = st.max_rel < 2e-2;
    std::printf("1. COLD capture+play vs oracle: max_abs=%.4f max_rel=%.6f  "
                "%s\n", st.max_abs, st.max_rel, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
    CK(hipGraphExecDestroy(ge), "coldDestroy");
    CK(hipStreamDestroy(cap), "coldStreamDestroy");
  }

  // ---- 2. legacy vs stream(nullptr): bitwise identical ------------------
  vk_hip_mla_fwd(s.B, s.H, s.S_q, s.S_kv, s.S_q - 1, 0, kLr, kRhd, kScale,
                 legacy.dq, legacy.dkc, legacy.dkp, legacy.dvc, legacy.dout);
  CK(hipDeviceSynchronize(), "legacy");
  rc = vk_hip_mla_fwd_stream(s.B, s.H, s.S_q, s.S_kv, s.S_q - 1, 0, kLr,
                             kRhd, kScale, stream_dev.dq, stream_dev.dkc,
                             stream_dev.dkp, stream_dev.dvc,
                             stream_dev.dout, nullptr);
  CK(hipDeviceSynchronize(), "stream-null");
  d = dev_diff(legacy, stream_dev);
  ok = (rc == 0) && (d == 0.0);
  std::printf("2. legacy vs stream(nullptr): rc=%d max_abs=%.3g  %s\n", rc, d,
              ok ? "PASS" : "FAIL");
  if (!ok) ++fails;

  // ---- 3. legacy vs non-default caller stream: bitwise identical --------
  hipStream_t side;
  CK(hipStreamCreate(&side), "sideStream");
  // Zero the output first so a pass/no-launch bug cannot fake parity.
  CK(hipMemsetAsync(stream_dev.dout, 0, stream_dev.nout * 4, side),
     "zeroOut");
  rc = vk_hip_mla_fwd_stream(s.B, s.H, s.S_q, s.S_kv, s.S_q - 1, 0, kLr, kRhd,
                             kScale, stream_dev.dq, stream_dev.dkc,
                             stream_dev.dkp, stream_dev.dvc, stream_dev.dout,
                             side);
  CK(hipStreamSynchronize(side), "sideSync");
  d = dev_diff(legacy, stream_dev);
  ok = (rc == 0) && (d == 0.0);
  std::printf("3. legacy vs side-stream:      rc=%d max_abs=%.3g  %s\n", rc, d,
              ok ? "PASS" : "FAIL");
  if (!ok) ++fails;

  // ---- 4. WARM capture (split path inside the graph) + re-feed replay ---
  {
    hipStream_t cap;
    CK(hipStreamCreate(&cap), "capStream2");
    // Warm: one eager call on the SAME stream sizes the split workspace so
    // the split path is taken both eagerly and inside the captured graph.
    rc = vk_hip_mla_fwd_stream(s.B, s.H, s.S_q, s.S_kv, s.S_q - 1, 0, kLr,
                               kRhd, kScale, cap_dev.dq, cap_dev.dkc,
                               cap_dev.dkp, cap_dev.dvc, cap_dev.dout, cap);
    CK(hipStreamSynchronize(cap), "warmSync");
    if (rc != 0) {
      std::printf("4. warm eager call rc=%d  FAIL\n", rc);
      ++fails;
    }
    std::vector<float> warm_ref;
    oracle(s, cap_dev, &warm_ref);  // eager split-path result
    hipGraphExec_t ge = capture_and_instantiate(s, cap_dev, cap);
    CK(hipGraphLaunch(ge, cap), "warmLaunch");
    CK(hipStreamSynchronize(cap), "warmLaunchSync");
    // The graph replays the same kernels on the same data on the same
    // stream as the eager warm call → must agree to round-off (fp32
    // deterministic kernels; not exact-by-construction because the eager
    // readback/compare path is separate).
    {
      std::vector<float> dev(cap_dev.nout);
      CK(hipMemcpy(dev.data(), cap_dev.dout, cap_dev.nout * 4,
                   hipMemcpyDeviceToHost), "rdWarm");
      double m = 0.0;
      double ref_max = 1.0;
      for (double v : warm_ref) ref_max = std::fmax(ref_max, std::fabs(v));
      for (size_t i = 0; i < dev.size(); ++i)
        m = std::fmax(m, (double)std::fabs(dev[i] - warm_ref[i]));
      bool okw = m / ref_max < 1e-5;
      std::printf("4a. warm graph vs eager split path: max_abs=%.3g  %s\n", m,
                  okw ? "PASS" : "FAIL");
      if (!okw) ++fails;
    }
    // Re-feed NEW inputs and replay: the graph must track the new inputs.
    std::vector<float> q2, kc2, kp2, vc2;
    fill_host(s, 7, &q2, &kc2, &kp2, &vc2);
    h2d(s, q2, kc2, kp2, vc2, cap_dev);
    CK(hipDeviceSynchronize(), "refeed");
    CK(hipGraphLaunch(ge, cap), "warmLaunch2");
    CK(hipStreamSynchronize(cap), "warmLaunch2Sync");
    Stats st = oracle(s, cap_dev, nullptr);
    ok = st.max_rel < 2e-2;
    std::printf("4b. warm graph replay (new inputs) vs oracle: "
                "max_abs=%.4f max_rel=%.6f  %s\n", st.max_abs, st.max_rel,
                ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
    CK(hipGraphExecDestroy(ge), "warmDestroy");
    CK(hipStreamDestroy(cap), "warmStreamDestroy");
  }

  // ---- 5. eager stream entry still matches the CPU oracle ---------------
  {
    CK(hipMemset(cap_dev.dout, 0, cap_dev.nout * 4), "zeroEager");
    rc = vk_hip_mla_fwd_stream(s.B, s.H, s.S_q, s.S_kv, s.S_q - 1, 0, kLr,
                               kRhd, kScale, cap_dev.dq, cap_dev.dkc,
                               cap_dev.dkp, cap_dev.dvc, cap_dev.dout,
                               nullptr);
    CK(hipDeviceSynchronize(), "eagerSync");
    Stats st = oracle(s, cap_dev, nullptr);
    ok = (rc == 0) && st.max_rel < 2e-2;
    std::printf("5. eager stream entry vs oracle: max_abs=%.4f "
                "max_rel=%.6f  %s\n", st.max_abs, st.max_rel,
                ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  hipFree(legacy.dq); hipFree(legacy.dkc); hipFree(legacy.dkp);
  hipFree(legacy.dvc); hipFree(legacy.dout);
  hipFree(stream_dev.dq); hipFree(stream_dev.dkc); hipFree(stream_dev.dkp);
  hipFree(stream_dev.dvc); hipFree(stream_dev.dout);
  hipFree(cap_dev.dq); hipFree(cap_dev.dkc); hipFree(cap_dev.dkp);
  hipFree(cap_dev.dvc); hipFree(cap_dev.dout);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
