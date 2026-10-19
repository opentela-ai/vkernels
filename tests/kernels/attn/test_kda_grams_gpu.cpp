#include "vkernels/kernels/kda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime.h>

static void check(hipError_t e) {
  if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
static float* device(const std::vector<float>& a) {
  float* p = nullptr;
  check(hipMalloc(&p, a.size() * sizeof(float)));
  check(hipMemcpy(p, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice));
  return p;
}
int main() {
  try {
    hipDeviceProp_t prop;
    check(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s\n", prop.name, prop.gcnArchName);
    for (int D : {16, 32, 64, 128}) {
      for (bool forgetting : {false, true}) {
        constexpr int B = 1, H = 2, nc = 2, cs = 64, S = nc * cs;
        std::mt19937 rng(51 + D);
        std::uniform_real_distribution<float> rf(-1.0f, 1.0f), gate(0.1f, 0.99f);
        const size_t m = (size_t)B * H * S * D;
        std::vector<float> q(m), k(m), beta(B * H * S), L(m);
        for (auto& x : q)
          x = rf(rng);
        for (auto& x : k)
          x = rf(rng);
        for (auto& x : beta)
          x = 0.25f + 0.5f * std::abs(rf(rng));
        for (int bh = 0; bh < B * H; ++bh) {
          for (int t = 0; t < S; ++t) {
            float norm = 0;
            for (int d = 0; d < D; ++d)
              norm += k[((size_t)bh * S + t) * D + d] * k[((size_t)bh * S + t) * D + d];
            for (int d = 0; d < D; ++d)
              k[((size_t)bh * S + t) * D + d] /= std::sqrt(norm);
          }
          for (int c = 0; c < nc; ++c)
            for (int d = 0; d < D; ++d) {
              float sum = 0;
              for (int t = 0; t < cs; ++t) {
                if (forgetting) sum += std::log(gate(rng));
                L[(((size_t)bh * nc + c) * cs + t) * D + d] = sum;
              }
            }
        }
        std::vector<float> M(B * H * nc * cs * cs, -999), N(M), refM(M), refN(M);
        vkernels::kernels::kda_k3_grams_blocked_cpu(q.data(),
                                                    k.data(),
                                                    beta.data(),
                                                    L.data(),
                                                    refM.data(),
                                                    refN.data(),
                                                    B,
                                                    H,
                                                    nc,
                                                    S,
                                                    D,
                                                    cs,
                                                    16);
        float* dq = device(q);
        float* dk = device(k);
        float* db = device(beta);
        float* dl = device(L);
        float* dm = device(M);
        float* dn = device(N);
        vkernels::kernels::hip::kda_k3_grams_blocked(
            dq, dk, db, dl, dm, dn, B, H, nc, S, D, 16, nullptr);
        check(hipGetLastError());
        check(hipDeviceSynchronize());
        check(hipMemcpy(M.data(), dm, M.size() * sizeof(float), hipMemcpyDeviceToHost));
        check(hipMemcpy(N.data(), dn, N.size() * sizeof(float), hipMemcpyDeviceToHost));
        float errorM = 0, errorN = 0;
        for (size_t i = 0; i < M.size(); ++i) {
          if (!std::isfinite(M[i]) || !std::isfinite(N[i]))
            throw std::runtime_error("Nonfinite output");
          errorM = std::max(errorM, std::abs(M[i] - refM[i]));
          errorN = std::max(errorN, std::abs(N[i] - refN[i]));
        }
        std::printf("D=%d forgetting=%d M_abs=%g N_abs=%g\n", D, (int)forgetting, errorM, errorN);
        for (float* p : {dq, dk, db, dl, dm, dn})
          check(hipFree(p));
        if (errorM > 2e-4f || errorN > 2e-4f)
          throw std::runtime_error("MFMA grams disagree with blocked CPU oracle");
      }
    }
    std::puts("PASS: 8 MFMA device/oracle comparisons");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
