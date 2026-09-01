// test_dsa_precond_correct.hip -- issue #57 regression test.
//
// The DSA sparse-fwd launch path (dsa.hip dsa_sparse_fwd / dsa_sparse_fwd_with_tile,
// exposed through the C ABI vk_hip_dsa_sparse_fwd) DOCUMENTED preconditions in
// dsa.hpp but enforced NONE before issue #57. An unmet precondition launched a
// kernel that silently never returned (gfx942 "D" state, no traceback). The
// CPU oracle (test_dsa.cpp / dsa_sparse_fwd_cpu) never exercised the device
// launch path, so the hang had no regression coverage.
//
// This harness is the on-device regression surface for the issue #57 fix:
//   (1) a VALID call still returns VK_OK and matches the CPU oracle to bf16
//       tolerance (the happy path is preserved on gfx942), and
//   (2) every unmet precondition now returns VK_ERROR_INVALID_ARGUMENT with a
//       named message in microseconds instead of hanging -- a wall-time
//       timeout guards the regression.
//
// Built only when VKERNELS_BUILD_BENCHMARKS and VKERNELS_HAS_HIP (see
// meta/benchmarks/CMakeLists.txt). Run on an MI300A (gfx942) node:
//   ./build/<cfg>/meta/benchmarks/test_dsa_precond_correct
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/capi/capi.hpp"        // VK_OK / VK_ERROR_* / vk_last_error
#include "vkernels/capi/hip_capi.hpp"    // vk_hip_dsa_sparse_fwd
#include "vkernels/kernels/dsa.hpp"      // dsa_sparse_fwd_cpu (oracle)

using vkernels::kernels::dsa_sparse_fwd_cpu;

namespace {

int g_fail = 0;

#define CK(expr, message)                                                                          \
  do {                                                                                             \
    const hipError_t status = (expr);                                                              \
    if (status != hipSuccess) {                                                                    \
      std::fprintf(stderr, "%s: %s\n", message, hipGetErrorString(status));                        \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

template <typename T> T* device_copy(const std::vector<T>& host) {
  T* device = nullptr;
  CK(hipMalloc(&device, host.size() * sizeof(T)), "hipMalloc");
  CK(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
     "hipMemcpy H2D");
  return device;
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lsb = (u >> 16) & 1;
  const uint32_t bias = 0x7FFFu + lsb;  // round-to-nearest-even
  return static_cast<uint16_t>((u + bias) >> 16);
}

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void check(bool cond, const char* message) {
  if (cond) {
    std::printf("  PASS: %s\n", message);
  } else {
    std::printf("  FAIL: %s\n", message);
    ++g_fail;
  }
}

// (1) A valid decode-shape call must still succeed and match the CPU oracle.
void run_valid_on_gpu() {
  const int S_q = 1, S_kv = 64, H = 1, dim = 64, tail_dim = 0, topk = 64;
  const int W = dim + tail_dim, d_v = dim - tail_dim;
  // dsa_config_for(1,1,64,64) = (bq=1, th=64, block_I=64, inner_iter=1) ->
  // topk % (block_I*inner_iter) == 0 holds (the documented happy path).
  const float sm_scale =
      (1.0f / std::sqrt(static_cast<float>(W))) * 1.4426950408889634f;

  std::vector<float> q(static_cast<size_t>(S_q) * H * W);
  std::vector<float> kv(static_cast<size_t>(S_kv) * W);
  std::vector<int32_t> idx(static_cast<size_t>(S_q) * topk);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.1f * static_cast<float>((i * 7) % 17) - 0.8f;
  for (size_t i = 0; i < kv.size(); ++i) kv[i] = 0.1f * static_cast<float>((i * 5) % 13) - 0.6f;
  for (int i = 0; i < S_q; ++i)
    for (int k = 0; k < topk; ++k) idx[static_cast<size_t>(i) * topk + k] = k;

  // CPU oracle (fp32 throughout).
  std::vector<float> out_ref(static_cast<size_t>(S_q) * H * d_v, 0.0f);
  dsa_sparse_fwd_cpu(S_q, S_kv, H, dim, tail_dim, topk, 1, 64, 1, sm_scale, false,
                     q.data(), kv.data(), idx.data(), out_ref.data(), nullptr);

  // q/kv -> bf16, upload to device.
  std::vector<uint16_t> qbf(q.size()), kvbf(kv.size());
  for (size_t i = 0; i < q.size(); ++i) qbf[i] = f32_to_bf16(q[i]);
  for (size_t i = 0; i < kv.size(); ++i) kvbf[i] = f32_to_bf16(kv[i]);

  uint16_t* dq = device_copy(qbf);
  uint16_t* dkv = device_copy(kvbf);
  uint16_t* dout = nullptr;
  int32_t* didx = device_copy(idx);
  CK(hipMalloc(&dout, static_cast<size_t>(S_q) * H * d_v * 2), "hipMalloc out");

  const auto t0 = std::chrono::steady_clock::now();
  const int rc = vk_hip_dsa_sparse_fwd(S_q, S_kv, H, dim, tail_dim, topk, 1, 64, 1,
                                       sm_scale, /*return_lse=*/0,
                                       static_cast<const void*>(dq),
                                       static_cast<const void*>(dkv),
                                       static_cast<const void*>(didx),
                                       static_cast<void*>(dout), nullptr);
  CK(hipDeviceSynchronize(), "hipDeviceSynchronize");
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  check(rc == VK_OK, "valid call returned VK_OK");
  if (rc != VK_OK) std::printf("    vk_last_error: %s\n", vk_last_error());
  std::printf("    valid call wall time: %.3f ms\n", ms);

  std::vector<uint16_t> hout(static_cast<size_t>(S_q) * H * d_v);
  CK(hipMemcpy(hout.data(), dout, hout.size() * 2, hipMemcpyDeviceToHost),
     "hipMemcpy D2H");
  float maxabs = 0.0f;
  for (size_t i = 0; i < hout.size(); ++i)
    maxabs = std::max(maxabs, std::fabs(bf16_to_f32(hout[i]) - out_ref[i]));
  char buf[128];
  std::snprintf(buf, sizeof(buf),
                "GPU matches CPU oracle (max abs err %.4f, bf16 tol ~0.05)", maxabs);
  check(maxabs < 0.05f, buf);

  CK(hipFree(dq), "hipFree");
  CK(hipFree(dkv), "hipFree");
  CK(hipFree(dout), "hipFree");
  CK(hipFree(didx), "hipFree");
}

// (2) An unmet precondition must return VK_ERROR_INVALID_ARGUMENT with a
// named message in well under the wall-time timeout (was: silent hang).
void run_invalid(const char* name, int S_q, int H, int topk, int bi, int ii,
                 const void* q, const void* kv, const void* idx, const void* out,
                 const void* lse, int return_lse, const char* expect_msg) {
  const auto t0 = std::chrono::steady_clock::now();
  const int rc = vk_hip_dsa_sparse_fwd(S_q, 1, H, 64, 0, topk, 1, bi, ii, 1.0f,
                                       return_lse, q, kv, idx,
                                       const_cast<void*>(out),
                                       const_cast<void*>(lse));
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("  [%s] rc=%d (%s) in %.3f ms\n", name, rc, vk_last_error(), ms);
  bool ok = (rc == VK_ERROR_INVALID_ARGUMENT) && (ms < 2000.0);
  if (expect_msg && !std::strstr(vk_last_error(), expect_msg)) ok = false;
  check(ok, "invalid input returned named error in <2s (no hang)");
}

}  // namespace

int main() {
  std::printf("=== issue #57 HIP precondition regression (MI300 / gfx942) ===\n");
  int device = 0;
  CK(hipGetDevice(&device), "hipGetDevice");
  hipDeviceProp_t props{};
  CK(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
  std::printf("GPU: %s\n", props.name);

  run_valid_on_gpu();

  // Invalid divisibility: topk=2, block_I=4, inner_iter=1 -> 2 % 4 != 0.
  // Non-null host dummies: the wrapper validates BEFORE launch, so no device
  // pointer is ever dereferenced.
  float dummy[4] = {0, 0, 0, 0};
  int32_t didx[2] = {0, 0};
  std::printf("[divisibility] topk=2, block_I=4, inner_iter=1\n");
  run_invalid("divisibility", 1, 1, 2, 4, 1, dummy, dummy, didx, dummy, nullptr, 0,
              "multiple of");

  // Invalid alias: topk=2, block_I=2, inner_iter=1 (2 % 2 == 0, valid div),
  // but out == q.
  std::printf("[alias] out == q\n");
  run_invalid("alias", 1, 1, 2, 2, 1, dummy, dummy, didx, dummy, nullptr, 0, "alias");

  // Invalid alias with lse: out == lse when return_lse.
  float lse[2] = {0, 0};
  std::printf("[alias-lse] out == lse, return_lse=1\n");
  run_invalid("alias-lse", 1, 1, 2, 2, 1, dummy, dummy, didx, lse, lse, 1, "alias");

  std::printf("=== RESULT: %s (failures=%d) ===\n", g_fail ? "FAILED" : "ALL PASS",
              g_fail);
  return g_fail ? 1 : 0;
}
