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

// Block-FP8 decode GEMM entry point (gfx942, issue #156). Offline-autotuner
// / harness hook (like gemm_bf16_with_config, not routed from gemm_bf16):
//
//   C[M, N] = alpha * A[M, K] @ dequant(W8, scales)^T ... (B-layout [K, N])
//
// with A bf16 [M, K], C bf16 [M, N], and the WEIGHTS stored as block-FP8
// (half the bytes of the bf16 B stream -> 2x bandwidth ceiling):
//   W8     : uint8 [K, N], E4M3FNUZ codes (MI300A-native fnuz encoding,
//            the glm_moe kernels' format; NOT OCP E4M3FN),
//   scales : float32 [ceil(K/128), ceil(N/8)], one scale per
//            (128-row K x 8-col N) block, row-major;
//   dequant(W8, scales)[k][n] = fp8(W8[k*N+n]) * scales[(k/128)][n/8].
// fp8(...) is the branchless fnuz -> fp32 bit decode (glm_moe.hip pattern:
// reinterpret the 7-bit payload as fp32 exponent/mantissa bits, which is
// fnuz_value * 2^-119 uniformly; the 2^119 folds into the scale). The
// dequantization is fused into the split-K sB staging, then the MFMA body,
// split-K grid, fp32 partial planes and fixed-order combine are identical
// to the bf16 split-K path (alpha/beta applied once in the combine).
// Requires N % 8 == 0 && K % 8 == 0 and a compiled tile (bm, bn) in
// {(16,16), (16,64), (32,64), (64,64)}; otherwise prints to stderr and
// leaves C untouched. S is the split count, clamped to [1, ceil(K/64)].
void gemm_fp8_block_splitk_with_config(
    std::size_t M, std::size_t N, std::size_t K, float alpha,
    const uint16_t* A, const uint8_t* W8, const float* scales,
    float beta, uint16_t* C, int bm, int bn, int S);

// Fused split-K entry (single launch; A/B candidate, default OFF): same
// grid / WIDE staging / fp32 partial planes as gemm_bf16_splitk_with_config,
// but the fixed-order combine is folded into the kernel via per-output-tile
// arrival counters -- each block publishes its split partial, bumps the
// counter of its (m, n) tile, and the LAST-arriving block for the tile sums
// all S planes in the SAME fixed ascending-s order as the separate combine
// kernel (alpha / beta*C applied once, single RNE bf16 store), which makes
// the result BIT-EXACT with the two-kernel path. Saves the combine launch
// and its tail latency (docs/kernels-reference.md 3.1: the remaining
// serving gap at 56% of HBM). Gated behind VK_GEMM_SPLITK_FUSED (default 0
// = the proven two-kernel path) until the MI300A A/B lands. Requires the
// WIDE staging alignment (N % 8 == 0 && K % 8 == 0); unaligned shapes fall
// back to the two-kernel path.
void gemm_bf16_splitk_fused_with_config(
    std::size_t M, std::size_t N, std::size_t K, float alpha,
    const uint16_t* A, const uint16_t* B, float beta, uint16_t* C,
    int bm, int bn, int S);

// Decode-GEMV split-K entries (issue #156): tiny-M (M <= 8) kernel with NO
// weight LDS staging and NO per-tile barriers -- each thread owns `tb`
// consecutive N columns (8 or 4; one uint4/uint2 of B per K row), walks the
// split's K range with fp32 FMAs, and S splits at element granularity feed
// the same fixed-order combine as the MFMA split-K path. `threads` = block
// size (64/128/256). Returns false (C untouched) when N % 8 != 0 or M not
// in [1, 8] -- callers fall back to gemm_bf16_splitk_with_config.
bool gemv_decode_bf16_splitk(std::size_t M, std::size_t N, std::size_t K,
                             float alpha, const uint16_t* A,
                             const uint16_t* B, float beta, uint16_t* C,
                             int S, int tb, int threads);
bool gemv_decode_fp8_splitk(std::size_t M, std::size_t N, std::size_t K,
                            float alpha, const uint16_t* A,
                            const uint8_t* W8, const float* scales,
                            float beta, uint16_t* C, int S, int tb,
                            int threads);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP

#if VKERNELS_HAS_CUDA
namespace vkernels::kernels::cuda {

// CUDA bf16 GEMM entry point (NVIDIA, wmma 16x16x16; see gemm_bf16.cu).
// Selects a tuned tile via gemm_bf16_config_for and launches the wmma
// kernel. Same contract as the CPU reference: C = alpha * A @ B + beta * C,
// bf16 in/out, fp32 accumulate.
void gemm_bf16(std::size_t M, std::size_t N, std::size_t K, float alpha,
               const uint16_t* A, const uint16_t* B, float beta,
               uint16_t* C);

}  // namespace vkernels::kernels::cuda
#endif  // VKERNELS_HAS_CUDA
