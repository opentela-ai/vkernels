// vkernels/kernels/common/heuristics.hpp — launch-config candidates as data.
//
// The per-arch config selectors (gemm_bf16_config_for and friends) encode
// *decisions*; this header holds the *search space* they decide over: the
// tile tables the on-device autotuners swept (bench_gemm_bf16 /
// bench_gemm_bf16.hip, persisted through docs/tuning-cache.md's store).
// Keeping the enumeration next to the selector — the layering DeepGEMM
// uses for csrc/jit_kernels/heuristics — means a new shape class sweeps
// the same table instead of growing a one-off branch.
#pragma once

#include <vector>

#include "vkernels/util/config.hpp"

namespace vkernels::kernels::heuristics {

// One candidate GEMM tile: block M/N/K plus the threads (CUDA) or
// wavefronts (HIP) the kernel runs it at.
struct GemmTile {
  int bm;
  int bn;
  int bk;
  int threads;
};

// The tile set the gemm_bf16 autotuners swept. BK is 64 everywhere (every
// K3 K is a multiple of 64). Ordered small-to-large; selectors return
// entries from this table so the tuned decision and the sweep space stay
// in sync.
//
// CUDA (GB10 / sm_121): (16,16)@32, (16,64)@128, (64,64)@256 — the three
//   points the cp.async double-buffered kernel's matrices in
//   docs/performance/gemm-bf16/gb10.md were regenerated against.
// HIP (MI300A / gfx942): (16,16)@64 and (64,64)@256 — the flat tiles the
//   228-CU autotuner kept (the cross-tile B-reuse schedule lost ~2x and
//   lives behind gemm_bf16_reuse_with_config for offline experiments).
inline std::vector<GemmTile> gemm_bf16_tile_candidates() {
#if VKERNELS_HAS_CUDA
  return {
      {16, 16, 64, 32}, {16, 64, 64, 128}, {64, 64, 64, 256},
  };
#else
  return {
      {16, 16, 64, 64},
      {64, 64, 64, 256},
  };
#endif
}

// Whether (bm, bn, bk, threads) is one of the swept candidates — the
// contract the config selectors must satisfy (a returned tile outside the
// table was never measured).
inline bool is_gemm_bf16_candidate(int bm, int bn, int bk, int threads) {
  for (const GemmTile& tile : gemm_bf16_tile_candidates()) {
    if (tile.bm == bm && tile.bn == bn && tile.bk == bk && tile.threads == threads) {
      return true;
    }
  }
  return false;
}

}  // namespace vkernels::kernels::heuristics
