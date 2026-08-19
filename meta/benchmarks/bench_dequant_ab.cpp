// meta/benchmarks/bench_dequant_ab.cpp
//
// Host-only A/B for the MXFP4 weight-tile dequant (issue #41).
//
// The CPU oracle (fused_moe_mxfp4_cpu) spends its time dequanting the
// routed experts' MXFP4 weights.  This bench isolates that dequant and
// times the two bit-identical implementations exposed in moe_fused.hpp:
//
//   dequant_weight_tile       — optimized: LUT nibble/scale decode +
//                               per-group ue8m0 scale hoisting
//   dequant_weight_tile_ref   — golden reference: original per-byte loop
//
// The two are asserted bit-identical by test_moe_fused's Fp4DequantLUTBitExact
// over every byte x scale; this bench re-asserts that on its own data and
// then times both over the *exact* tile grid the oracle walks for one
// real expert's gate+up weights at the requested shape:
//
//   num_n_blocks = ispp      / BLOCK_N(64)
//   num_k_blocks = hidden    / BLOCK_K(64)
//   tiles/expert = 2 * num_n_blocks * num_k_blocks          (gate + up)
//
// This is the dominant cost in the oracle (the real-expert dequant; the
// padding rows dequant nothing once skipped), so the speedup measured here
// is the speedup the oracle sees on this hardware.  Run anywhere (no GPU);
// run on a beverin compute node for the on-MI300A-CPU number.
//
//   ./bench_dequant_ab [--hidden N] [--ispp N] [--iters N] [--seed N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vkernels/kernels/moe_fused.hpp"

using vkernels::kernels::dequant_weight_tile;
using vkernels::kernels::dequant_weight_tile_ref;

namespace {

constexpr int kBlockN = 64;   // matches BLOCK_N in fused_moe_mxfp4_cpu
constexpr int kBlockK = 64;   // matches BLOCK_K
constexpr int kGroup  = 32;   // group_size for w13 (hidden/2 packed, /32 scale)

uint8_t e2m1_nibble(float f) {
  bool neg = f < 0.0f;
  float af = std::fabs(f);
  if (af == 0.0f || std::isnan(af)) return neg ? 0x8 : 0x0;
  if (std::isinf(af)) return neg ? 0xE : 0x6;
  float vals[5] = {0.25f, 1.0f, 1.5f, 2.0f, 3.0f};
  uint8_t nibs[5] = {1, 2, 3, 4, 5};
  float best_d = std::fabs(af - vals[0]);
  uint8_t best_n = nibs[0];
  for (int i = 1; i < 5; ++i) {
    float d = std::fabs(af - vals[i]);
    if (d < best_d) { best_d = d; best_n = nibs[i]; }
  }
  return neg ? static_cast<uint8_t>(best_n | 0x8) : best_n;
}

uint8_t pack_e2m1_pair(float v0, float v1) {
  return static_cast<uint8_t>(e2m1_nibble(v0) | (e2m1_nibble(v1) << 4));
}

float rnd(uint64_t s) {
  s ^= s >> 33; s *= 0xff51afd7ed558ccdULL; s ^= s >> 33;
  s *= 0xc4ceb9fe1a85ec53ULL; s ^= s >> 33;
  return static_cast<float>(s >> 40) * (1.0f / 16777216.0f);  // 24-bit in [0,1)
}

double median_us(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Walk the tile grid exactly as the oracle does for the gate half (then up),
// calling `fn` for every [BLOCK_N, BLOCK_K] tile.  `packed0`/`scale0` are the
// base of the (gate) weight for a single expert; stride_packed/stride_scale_n
// are the per-row strides the oracle passes.
template <typename Fn>
double time_one_half(const uint8_t* packed0, const uint8_t* scale0,
                     int ispp, int hidden, int iters,
                     std::vector<uint16_t>& tile, Fn fn) {
  const int num_n_blocks = ispp / kBlockN;
  const int num_k_blocks = hidden / kBlockK;
  const int stride_packed = hidden / 2;
  const int stride_scale_n = hidden / kGroup;
  std::vector<double> times;
  times.reserve(static_cast<size_t>(iters));
  for (int it = 0; it < iters; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    for (int nb = 0; nb < num_n_blocks; ++nb) {
      for (int kb = 0; kb < num_k_blocks; ++kb) {
        int k_start = kb * kBlockK;
        const uint8_t* b = packed0 + nb * kBlockN * stride_packed + k_start / 2;
        const uint8_t* s = scale0 + nb * kBlockN * stride_scale_n + k_start / kGroup;
        fn(b, s, tile.data(), kBlockN, kBlockK, kGroup,
           stride_packed, stride_scale_n, 1);
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return median_us(times);
}

}  // namespace

int main(int argc, char** argv) {
  int hidden = 7168, ispp = 512, iters = 7, seed = 7;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](int& j) -> long {
      if (j + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[j]); return 1; }
      return std::strtol(argv[++j], nullptr, 10);
    };
    std::string a = argv[i];
    if      (a == "--hidden") hidden = (int)next(i);
    else if (a == "--ispp")   ispp   = (int)next(i);
    else if (a == "--iters")  iters  = (int)next(i);
    else if (a == "--seed")   seed   = (int)next(i);
    else if (a == "--help" || a == "-h") {
      std::printf("usage: bench_dequant_ab [--hidden 7168] [--ispp 512] "
                  "[--iters 7] [--seed 7]\n");
      return 0;
    }
  }
  if (hidden % 64 != 0 || ispp % 64 != 0) {
    std::fprintf(stderr, "hidden and ispp must be multiples of 64\n");
    return 2;
  }

  // One expert's gate+up MXFP4 weight, packed as in bench_moe_fused_oracle:
  //   packed : [2*ispp, hidden/2] bytes
  //   scale  : [2*ispp, hidden/group] ue8m0
  const std::size_t packed_bytes = static_cast<std::size_t>(2) * ispp * (hidden / 2);
  const std::size_t scale_bytes  = static_cast<std::size_t>(2) * ispp * (hidden / kGroup);
  std::vector<uint8_t> packed(packed_bytes), scale(scale_bytes, 127);
  for (std::size_t i = 0; i < packed_bytes; ++i)
    packed[i] = pack_e2m1_pair(rnd(seed * 101 + i) * 3.0f, rnd(seed * 202 + i) * 3.0f);

  const int num_n_blocks = ispp / kBlockN;
  const int num_k_blocks = hidden / kBlockK;
  const std::size_t tiles_per_expert =
      static_cast<std::size_t>(2) * num_n_blocks * num_k_blocks;  // gate + up

  // ---- bit-exact check on the full grid (both must match) ----
  {
    std::vector<uint16_t> opt(kBlockK * kBlockN), ref(kBlockK * kBlockN);
    const int stride_packed = hidden / 2;
    const int stride_scale_n = hidden / kGroup;
    std::size_t mism = 0, checked = 0;
    const uint8_t* gate_b = packed.data();
    const uint8_t* gate_s = scale.data();
    const uint8_t* up_b   = packed.data() + static_cast<std::size_t>(ispp) * (hidden / 2);
    const uint8_t* up_s   = scale.data()  + static_cast<std::size_t>(ispp) * (hidden / kGroup);
    for (int half = 0; half < 2 && mism == 0; ++half) {
      const uint8_t* b0 = half == 0 ? gate_b : up_b;
      const uint8_t* s0 = half == 0 ? gate_s : up_s;
      for (int nb = 0; nb < num_n_blocks && mism == 0; ++nb) {
        for (int kb = 0; kb < num_k_blocks && mism == 0; ++kb) {
          int k_start = kb * kBlockK;
          const uint8_t* b = b0 + nb * kBlockN * stride_packed + k_start / 2;
          const uint8_t* s = s0 + nb * kBlockN * stride_scale_n + k_start / kGroup;
          dequant_weight_tile    (b, s, opt.data(), kBlockN, kBlockK, kGroup,
                                  stride_packed, stride_scale_n, 1);
          dequant_weight_tile_ref(b, s, ref.data(), kBlockN, kBlockK, kGroup,
                                  stride_packed, stride_scale_n, 1);
          for (std::size_t i = 0; i < opt.size(); ++i) {
            if (opt[i] != ref[i]) { ++mism; break; }
            ++checked;
          }
        }
      }
    }
    if (mism != 0) {
      std::fprintf(stderr, "FAIL: dequant_weight_tile != _ref (%zu mismatch in "
                           "%zu values) — results are NOT bit-identical.\n",
                   mism, checked);
      return 3;
    }
    std::printf("bit-exact check: PASS (%zu bf16 values across %zu tiles)\n",
                checked, tiles_per_expert);
  }

  std::vector<uint16_t> tile(kBlockK * kBlockN);
  const uint8_t* gate_b = packed.data();
  const uint8_t* gate_s = scale.data();
  const uint8_t* up_b   = packed.data() + static_cast<std::size_t>(ispp) * (hidden / 2);
  const uint8_t* up_s   = scale.data()  + static_cast<std::size_t>(ispp) * (hidden / kGroup);

  // Warm + time each half for each implementation.
  double opt_gate = time_one_half(gate_b, gate_s, ispp, hidden, iters, tile,
      [&](const uint8_t* b, const uint8_t* s, uint16_t* o,
          int N, int K, int g, int sp, int ssn, int ssk) {
        dequant_weight_tile(b, s, o, N, K, g, sp, ssn, ssk);
      });
  double opt_up   = time_one_half(up_b, up_s, ispp, hidden, iters, tile,
      [&](const uint8_t* b, const uint8_t* s, uint16_t* o,
          int N, int K, int g, int sp, int ssn, int ssk) {
        dequant_weight_tile(b, s, o, N, K, g, sp, ssn, ssk);
      });
  double ref_gate = time_one_half(gate_b, gate_s, ispp, hidden, iters, tile,
      [&](const uint8_t* b, const uint8_t* s, uint16_t* o,
          int N, int K, int g, int sp, int ssn, int ssk) {
        dequant_weight_tile_ref(b, s, o, N, K, g, sp, ssn, ssk);
      });
  double ref_up   = time_one_half(up_b, up_s, ispp, hidden, iters, tile,
      [&](const uint8_t* b, const uint8_t* s, uint16_t* o,
          int N, int K, int g, int sp, int ssn, int ssk) {
        dequant_weight_tile_ref(b, s, o, N, K, g, sp, ssn, ssk);
      });

  double opt_us = opt_gate + opt_up;
  double ref_us = ref_gate + ref_up;
  double speedup = opt_us > 0 ? ref_us / opt_us : 0.0;

  std::printf("\ndequant A/B  hidden=%d ispp=%d  tiles/expert=%zu  iters=%d\n",
              hidden, ispp, tiles_per_expert, iters);
  std::printf("%-14s %10s %10s %10s\n", "impl", "gate_us", "up_us", "tot_us");
  std::printf("%-14s %10.1f %10.1f %10.1f\n", "ref (orig)",   ref_gate, ref_up, ref_us);
  std::printf("%-14s %10.1f %10.1f %10.1f\n", "opt (LUT)",    opt_gate, opt_up, opt_us);
  std::printf("speedup opt/ref = %.2fx   (opt %.2f us/tile, ref %.2f us/tile)\n",
              speedup, opt_us / tiles_per_expert, ref_us / tiles_per_expert);
  return 0;
}
