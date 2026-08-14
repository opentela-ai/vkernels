// vkernels/kernels/gemm_bf16.hpp
//
// bf16 GEMM on AMD gfx942 (CDNA3 / MI300A) via the K16 bf16 MFMA
// (__builtin_amdgcn_mfma_f32_16x16x16bf16_1k), for the Kimi-K3 projection
// shapes (issue #29) that today fall back to AITER's untuned "torch
// solution:0" because bf16_tuned_gemm.csv has no gfx942 entries.
//
//   C[M, N] = alpha * A[M, K] @ B[K, N] + beta * C[M, N]   (bf16 in/out,
//                                                            fp32 accumulate)
//
// Layout. A, B and C are dense, row-major and stored as uint16_t IEEE 754
// bf16 bit patterns:
//   A is [M, K]  -- the input activations,
//   B is [K, N]  -- the *transposed* projection weight W[N, K].T,
//   C is [M, N]  -- the output.
// Each Kimi-K3 projection in the issue is written as "N x K" (the weight's
// [out, in] shape); the serving recipe transposes its [N, K] weight once
// (or keeps it pre-transposed) and calls this kernel with (M, N, K) =
// (batch, N_from_issue, K_from_issue). The K3 serving shapes (M ~ 5-64):
//
//   N x K = 6288x7168 (QKV), 3584x7168, 896x7168, 2112x7168, 1536x7168,
//           7168x1536, 7168x768, 7168x3584, 2304x1536, 3072x512, 1536x128
//
// and the warmup / profiling shapes are the same (N, K) at M = 8192. Every
// K in that list is a multiple of 64 and every N is a multiple of 16, so
// the MFMA tile is K-padded to BK = 64 and N is bounds-checked per tile.
//
// Two-implementation model:
//   gemm_bf16.cpp  -- CPU reference (oracle), always compiled, in
//                     vkernels::kernels. Converts to fp32, accumulates in
//                     fp32, and stores with the same round-to-nearest-even
//                     as the device so host and device agree to the bit.
//   gemm_bf16.hip  -- HIP MFMA implementation (gfx942), compiled with
//                     VKERNELS_HAS_HIP, in vkernels::kernels::hip.
//
// Tuning. hip::gemm_bf16 selects a tile (BM, BN, BK, threads) per shape via
// gemm_bf16_config_for (below). The table is analytically chosen against
// the MI300A roofline (see docs/performance/gemm-bf16/gfx942.md); the
// offline autotuner in meta/benchmarks/bench_gemm_bf16.hip regenerates it
// on device by sweeping the same compile-time tiles.
#include <cstddef>
#include <cstdint>

namespace vkernels::kernels {

// CPU reference (oracle). C = alpha * A @ B + beta * C with A in [M, K],
// B in [K, N], C in [M, N], all bf16 row-major; fp32 accumulation; a single
// round-to-nearest-even on store (matches the MFMA kernel's f32->bf16).
void gemm_bf16_cpu(std::size_t M, std::size_t N, std::size_t K, float alpha,
                   const uint16_t* A, const uint16_t* B, float beta,
                   uint16_t* C);

// Per-shape tuned tile config for the HIP kernel. Writes (bm, bn, bk,
// threads) for the MFMA tile hip::gemm_bf16 should launch for (M, N, K).
// Defaults are chosen against the MI300A roofline; the bench autotuner can
// override them. Both shapes are bf16 memory-bound (serving, M <= 64) and
// bf16 compute-bound (warmup, M >= 1024); BK is fixed at 64 because every
// K3 K is a multiple of 64.
void gemm_bf16_config_for(std::size_t M, std::size_t N, std::size_t K,
                          int* bm, int* bn, int* bk, int* threads);

}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// HIP bf16 GEMM entry point (gfx942). Selects a tuned tile via
// gemm_bf16_config_for and launches the MFMA kernel. Same contract as the
// CPU reference: C = alpha * A @ B + beta * C, bf16 in/out, fp32 accumulate.
void gemm_bf16(std::size_t M, std::size_t N, std::size_t K, float alpha,
               const uint16_t* A, const uint16_t* B, float beta,
               uint16_t* C);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
