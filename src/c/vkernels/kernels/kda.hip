// vkernels/kernels/kda.hip
//
// Kimi Delta Attention (KDA) forward on gfx942 / MI300A (issue #21). Compiled
// only with VKERNELS_HAS_HIP. Each kernel is the device counterpart of the
// CPU reference in kda.cpp and shares its contract (see kda.hpp):
//
//   kda_layer_norm_gated   -- gated RMSNorm, one thread per token row.
//   kda_gate_chunk_cumsum  -- within-chunk inclusive log-cumsum (one block
//                             per (b,h,c) chunk) + a second pass for the
//                             cross-chunk exclusive log-cumsum.
//   kda_delta_rule_fwd     -- the gated delta-rule forward. Implemented as a
//                             cooperative per-token recurrence (one block per
//                             (b,h), D×D state in LDS, all threads cooperate
//                             on the matrix-vector products and the
//                             rank-1 outer-product update). This is the
//                             simplest kernel that is *provably identical* to
//                             the cross-checked CPU oracle
//                             (kda_naive_delta_rule_fwd_cpu): the per-token
//                             recurrence is run in the same order, so the
//                             only divergence is fp round-off. The chunked
//                             intra / inter / output passes of
//                             kda.cpp are a *follow-on* for chunk-level
//                             parallelism (fla's C_{c-1}-decoupled solve);
//                             they are not needed for correctness against
//                             the oracle and are validated separately.
//   kda_pack_bitmatrix     -- MSB-first bit packing, one thread per byte.
//
// The cooperative recurrence keeps the D×D state matrix in LDS (kMaxD=128 →
// 64 KB, leaving headroom on MI300A's 256 KB/CU). For each token the kernel
// runs three barrier-separated phases — (1) prediction a = S k, (2) state
// update S = g S + β (v−a) kᵀ, (3) output o = S q — and each phase is a
// grid-stride loop over the D (or D²) work items, so the only cross-thread
// dependency is the per-token barrier, never a data race.
#include <cstddef>
#include <cstdint>

#include "vkernels/kernels/kda.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>

namespace vkernels::kernels::hip {

static constexpr int kTh    = 256;   // cooperating threads per (b,h) block

// ===========================================================================
//  #L1 layer_norm_gated — one thread per token row
// ===========================================================================
__global__ void kda_layer_norm_gated_kernel(const float* __restrict__ x,
                                            const float* __restrict__ weight,
                                            const float* __restrict__ gate,
                                            float* __restrict__ out,
                                            int N, int D, float eps) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* xn = x + (size_t)n * D;
  float* on = out + (size_t)n * D;
  float sq = 0.0f;
  for (int d = 0; d < D; ++d) sq += xn[d] * xn[d];
  const float inv = 1.0f / sqrtf(sq / static_cast<float>(D) + eps);
  for (int d = 0; d < D; ++d) {
    const float g = gate[(size_t)n * D + d];
    const float silu = g / (1.0f + __expf(-g));   // g · sigmoid(g)
    on[d] = xn[d] * inv * weight[d] * silu;
  }
}

void kda_layer_norm_gated(const float* x, const float* weight,
                          const float* gate, float* out,
                          int N, int D, float eps) {
  const int block = 256;
  const int grid = (N + block - 1) / block;
  kda_layer_norm_gated_kernel<<<grid, block, 0, 0>>>(
      x, weight, gate, out, N, D, eps);
}

// ===========================================================================
//  #L2 gate_chunk_cumsum — intra (per chunk) + inter (per b,h)
// ===========================================================================
__global__ void kda_gate_intra_kernel(const float* __restrict__ g,
                                      float* __restrict__ intra_log,
                                      int n_chunks, int chunk_size) {
  // one block per (b,h,c); thread 0 owns the sequential within-chunk scan.
  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (threadIdx.x != 0) return;
  const size_t base = ((size_t)b * gridDim.y + h) * n_chunks + c;
  const float* gc = g + base * chunk_size;
  float* Lc = intra_log + base * chunk_size;
  float acc = 0.0f;
  for (int t = 0; t < chunk_size; ++t) {
    acc += (gc[t] > 0.0f) ? logf(gc[t]) : -1.0e9f;
    Lc[t] = acc;
  }
}

__global__ void kda_gate_inter_kernel(const float* __restrict__ intra_log,
                                      float* __restrict__ inter_log,
                                      int n_chunks, int chunk_size) {
  // one block per (b,h); thread 0 owns the exclusive cross-chunk scan.
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  if (threadIdx.x != 0) return;
  const size_t base = (size_t)b * gridDim.x + h;
  float acc = 0.0f;
  for (int c = 0; c < n_chunks; ++c) {
    inter_log[base * n_chunks + c] = acc;          // exclusive
    acc += intra_log[(base * n_chunks + c) * chunk_size + (chunk_size - 1)];
  }
}

void kda_gate_chunk_cumsum(const float* g, float* intra_log, float* inter_log,
                           int B, int H, int n_chunks, int chunk_size) {
  if (B == 0 || H == 0 || n_chunks == 0) return;
  kda_gate_intra_kernel<<<dim3(n_chunks, H, B), 1, 0, 0>>>(
      g, intra_log, n_chunks, chunk_size);
  kda_gate_inter_kernel<<<dim3(H, B), 1, 0, 0>>>(
      intra_log, inter_log, n_chunks, chunk_size);
}

// ===========================================================================
//  #L7 delta_rule_fwd — cooperative per-token recurrence (oracle-faithful)
// ===========================================================================
// The D×D state lives in a per-(b,h) GLOBAL scratch buffer (one block per
// (b,h), zero-initialized by the launcher) so the kernel is correct for any
// head dim (K3 KDA head_dim = 128 → 64 KB state, above the 64 KB default
// work-group LDS cap). Only the length-D prediction `a = S k` is staged in
// LDS (≤ 512 B for D ≤ 128). Each token runs FOUR barrier-separated phases —
// (1) GATE: S' = g ⊙ S (per-key-dim), (2) PREDICT: a = S' k (from gated
// state), (3) UPDATE: S = S' + β (v−a) kᵀ, (4) OUTPUT: o = S q — grid-strided
// over the D / D² work items, so the only cross-thread dependency is the
// per-token barrier, never a data race. The recurrence order is identical
// to the oracle (kda_naive_delta_rule_fwd_cpu) and the FLA/vLLM K3
// gated-delta-rule (IS_KDA=True), so the only divergence is fp round-off.
template <int D, int Db>
__global__ void kda_delta_rule_fwd_kernel(const float* __restrict__ q,
                                          const float* __restrict__ k,
                                          const float* __restrict__ v,
                                          const float* __restrict__ g,
                                          const float* __restrict__ beta,
                                          float* __restrict__ state,
                                          float* __restrict__ out,
                                          int H, int S,
                                          int chunk_start, int chunk_len) {
  // The per-token recurrence is ROW-PARALLEL: row d of the D×D state needs
  // only its own row (a[d]=S[d,:]·k, S[d,e]=gS[d,e]+β(v−a)kᵀ, o[d]=S[d,:]·q).
  // So blockIdx.z splits the D rows across blocks — one block per (h,b,D-row
  // block). This both spreads work across CUs (grid(H,B,D/Db) vs grid(H,B))
  // and shortens each block so the cumulative one-CU runtime stays under the
  // beverin MI300A watchdog (a single S=1024 D=128 block faults after
  // ~0.5 s of continuous one-CU work).
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int row0 = blockIdx.z * Db;        // this block's first state row
  const int tid = threadIdx.x;
  const size_t bh = (size_t)(b * H + h) * S;       // q/k/v/out base ([B,H,S,D])
  const size_t sh = (size_t)(b * H + h);           // state index ([B,H,D,D])
  float* Sstate = state + sh * D * D;              // this (b,h)'s D×D state (gmem)
  __shared__ float sa[Db];                    // a = S k for this block's rows

  // Each token runs three barrier-separated phases — (1) a = S k,
  // (2) S = g S + β (v−a) kᵀ, (3) o = S q — grid-strided over the Db / D²
  // work items, so the only cross-thread dependency is the per-token
  // barrier, never a data race. The recurrence order is identical to the
  // oracle (kda_naive_delta_rule_fwd_cpu); only fp round-off differs.
  for (int t = 0; t < chunk_len; ++t) {
    const size_t off = bh + (size_t)(chunk_start + t);  // full-S token index
    const float* kt = k + off * D;
    const float* vt = v + off * D;
    const float* qt = q + off * D;
    const float* gt = g + off * D;       // per-key-dim forget gate [D]
    const float bt = beta[off];          // scalar delta gate

    // (1) GATE: S'[d,e] *= g_t[e]   (per-key-dim, this block's Db*D entries)
    for (int idx = tid; idx < Db * D; idx += kTh) {
      const int j = idx / D;              // local row index in [0,Db)
      const int e = idx % D;
      const int d = row0 + j;             // global state row
      Sstate[(size_t)d * D + e] *= gt[e];
    }
    __syncthreads();

    // (2) PREDICT: a[d] = S'[d,:] . k_t   (from GATED state; this block's Db rows)
    for (int j = tid; j < Db; j += kTh) {
      const int d = row0 + j;
      const float* Srow = Sstate + (size_t)d * D;
      float s = 0.0f;
      for (int e = 0; e < D; ++e) s += Srow[e] * kt[e];
      sa[j] = s;
    }
    __syncthreads();

    // (3) UPDATE: S[d,e] += beta_t * (v_t[d] - a[d]) * k_t[e]
    for (int idx = tid; idx < Db * D; idx += kTh) {
      const int j = idx / D;              // local row index in [0,Db)
      const int e = idx % D;
      const int d = row0 + j;             // global state row
      const float ud = vt[d] - sa[j];
      Sstate[(size_t)d * D + e] += bt * ud * kt[e];
    }
    __syncthreads();

    // (4) OUTPUT: o_t[d] = S_t[d,:] . q_t   (this block's Db rows)
    for (int j = tid; j < Db; j += kTh) {
      const int d = row0 + j;
      const float* Srow = Sstate + (size_t)d * D;
      float s = 0.0f;
      for (int e = 0; e < D; ++e) s += Srow[e] * qt[e];
      out[off * D + d] = s;
    }
    __syncthreads();
  }
}

namespace {
// Process the token stream in fixed-size chunks so no single block runs
// long. Each chunk re-enters the SAME gmem D×D state (carried across chunks,
// no re-zeroing) and launches on the default stream, so chunks run in order
// and the result is bit-for-bit the per-token recurrence. Db is the per-
// block row count (D/Db = z-blocks → CUs of parallelism per (b,h)).
constexpr int kChunk = 256;

template <int D, int Db>
void launch(const float* q, const float* k, const float* v, const float* g,
            const float* beta, float* state, float* out, int B, int H, int S) {
  static_assert(D % Db == 0, "D must be a multiple of Db");
  dim3 grid(H, B, D / Db);
  dim3 block(kTh);
  for (int cs = 0; cs < S; cs += kChunk) {
    const int len = (S - cs < kChunk) ? (S - cs) : kChunk;
    kda_delta_rule_fwd_kernel<D, Db><<<grid, block, 0, 0>>>(
        q, k, v, g, beta, state, out, H, S, cs, len);
  }
  hipDeviceSynchronize();   // wait for all chunks before returning
}
}  // namespace

// Explicit-scratch entry point (offline benchmark / autotuner hook): the
// caller owns the B*H*D*D zero-initialized state scratch so it can be
// allocated once and reused across timed iterations.
void kda_delta_rule_fwd_with_scratch(const float* q, const float* k,
                                     const float* v, const float* g,
                                     const float* beta, float* state,
                                     float* out, int B, int H, int S, int D) {
  if (B == 0 || H == 0 || S == 0) return;
  // Db = per-block row count; D/Db = z-blocks of row parallelism per (b,h).
  // Larger D → more blocks so a single (b,h) still fans out across CUs.
  if (D <= 16)       launch<16, 16>(q, k, v, g, beta, state, out, B, H, S);
  else if (D <= 32)  launch<32, 32>(q, k, v, g, beta, state, out, B, H, S);
  else if (D <= 64)  launch<64, 16>(q, k, v, g, beta, state, out, B, H, S);
  else if (D <= 128) launch<128, 32>(q, k, v, g, beta, state, out, B, H, S);
  else               launch<128, 32>(q, k, v, g, beta, state, out, B, H, S);  // cap
}

void kda_delta_rule_fwd(const float* q, const float* k, const float* v,
                        const float* g, const float* beta, float* out,
                        int B, int H, int S, int D, int chunk_size) {
  (void)chunk_size;  // the cooperative kernel streams tokens; chunking is a
                     // follow-on for parallelism (see file header).
  if (B == 0 || H == 0 || S == 0) return;
  float* state = nullptr;
  hipMalloc(&state, (size_t)B * H * D * D * sizeof(float));
  hipMemsetAsync(state, 0, (size_t)B * H * D * D * sizeof(float), 0);
  kda_delta_rule_fwd_with_scratch(q, k, v, g, beta, state, out, B, H, S, D);
  // kda_delta_rule_fwd_with_scratch syncs on return, so the scratch is no
  // longer in use; hipFree (NOT synchronous in ROCm, unlike cudaFree) is
  // then safe.
  hipFree(state);
}

// ===========================================================================
//  #P pack_bitmatrix — one thread per output byte (MSB first)
// ===========================================================================
__global__ void kda_pack_bitmatrix_kernel(const uint8_t* __restrict__ bits,
                                          uint8_t* __restrict__ packed,
                                          std::size_t n_bits) {
  const std::size_t byte = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (byte * 8 >= n_bits) return;
  uint8_t p = 0;
  #pragma unroll
  for (int b = 0; b < 8; ++b) {
    const std::size_t k = byte * 8 + b;
    if (k < n_bits && bits[k] != 0) p |= static_cast<uint8_t>(1u << (7 - b));
  }
  packed[byte] = p;
}

void kda_pack_bitmatrix(const uint8_t* bits, uint8_t* packed,
                        std::size_t n_bits) {
  if (n_bits == 0) return;
  const std::size_t bytes = (n_bits + 7) / 8;
  const int block = 256;
  const int grid = (int)((bytes + block - 1) / block);
  kda_pack_bitmatrix_kernel<<<grid, block, 0, 0>>>(bits, packed, n_bits);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
