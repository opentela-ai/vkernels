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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

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
  float* Sstate = state + sh * D * D;   // gmem owner-of-record (load/store once)
  // Cache this block's Db state rows in LDS across the whole chunk so the
  // per-token GATE / PREDICT / UPDATE / OUTPUT phases touch on-chip state
  // instead of round-tripping HBM ~6× per token. The D×D state was the
  // binding resource (AI≈0.4 ≪ MI300A ridge ~247): a D=128 chunk paid
  // 6×(Db·D)·4 B = 96 KB/token of HBM for the state alone. Caching cuts that
  // to a single load+store per chunk (~375 B/token amortised over kChunk),
  // ~250× less HBM state traffic. Row-independence (each z-block owns
  // disjoint rows) needs no cross-block coordination; the gmem state is only
  // the between-chunk handoff. Math is unchanged from the gmem variant, so
  // results are bit-identical to kda_naive_delta_rule_fwd_cpu.
  __shared__ float srow[Db * D];              // this block's Db state rows (LDS)
  __shared__ float sa[Db];                    // a[d] = S[d,:] . k_t (this block)

  // Load this block's Db state rows from gmem into LDS (once per chunk).
  for (int idx = tid; idx < Db * D; idx += kTh) {
    const int j = idx / D, e = idx % D;
    srow[idx] = Sstate[(size_t)(row0 + j) * D + e];
  }
  __syncthreads();

  // Each token runs four barrier-separated phases — (1) GATE: S' = g⊙S,
  // (2) PREDICT: a = S'k, (3) UPDATE: S = S' + β(v−a)kᵀ, (4) OUTPUT: o = Sq
  // — grid-strided over the Db / D² work items, all on the LDS state cache.
  // The only cross-thread dependency is the per-token barrier, never a data
  // race; the recurrence order is identical to the oracle.
  for (int t = 0; t < chunk_len; ++t) {
    const size_t off = bh + (size_t)(chunk_start + t);  // full-S token index
    const float* kt = k + off * D;
    const float* vt = v + off * D;
    const float* qt = q + off * D;
    const float* gt = g + off * D;       // per-key-dim forget gate [D]
    const float bt = beta[off];          // scalar delta gate

    // (1) GATE: s'[j,e] *= g_t[e]   (LDS only — no HBM state round-trip)
    for (int idx = tid; idx < Db * D; idx += kTh) {
      const int e = idx % D;
      srow[idx] *= gt[e];
    }
    __syncthreads();

    // (2) PREDICT: a[j] = s'[j,:] . k_t   (LDS read; one thread per row)
    for (int j = tid; j < Db; j += kTh) {
      const float* sr = srow + (size_t)j * D;
      float s = 0.0f;
      for (int e = 0; e < D; ++e) s += sr[e] * kt[e];
      sa[j] = s;
    }
    __syncthreads();

    // (3) UPDATE: s[j,e] += beta_t * (v_t[d] - a[j]) * k_t[e]   (LDS only)
    for (int idx = tid; idx < Db * D; idx += kTh) {
      const int j = idx / D, e = idx % D;
      const int d = row0 + j;             // global state row -> v_t index
      const float ud = vt[d] - sa[j];
      srow[idx] += bt * ud * kt[e];
    }
    __syncthreads();

    // (4) OUTPUT: o_t[d] = s[j,:] . q_t   (LDS read; one thread per row)
    for (int j = tid; j < Db; j += kTh) {
      const int d = row0 + j;
      const float* sr = srow + (size_t)j * D;
      float s = 0.0f;
      for (int e = 0; e < D; ++e) s += sr[e] * qt[e];
      out[off * D + d] = s;
    }
    __syncthreads();
  }

  // Store this block's Db state rows back to gmem (once per chunk).
  for (int idx = tid; idx < Db * D; idx += kTh) {
    const int j = idx / D, e = idx % D;
    Sstate[(size_t)(row0 + j) * D + e] = srow[idx];
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
            const float* beta, float* state, float* out, int B, int H, int S,
            hipStream_t stream = nullptr) {
  static_assert(D % Db == 0, "D must be a multiple of Db");
  dim3 grid(H, B, D / Db);
  dim3 block(kTh);
  // stream (issue #69): with a caller stream the chunk launches are
  // ordered BY THE STREAM (bit-identical result) and the caller owns
  // synchronisation — required for graph capture, where a device sync
  // is illegal. Legacy nullptr keeps the historical default-stream +
  // sync-on-return behaviour.
  for (int cs = 0; cs < S; cs += kChunk) {
    const int len = (S - cs < kChunk) ? (S - cs) : kChunk;
    kda_delta_rule_fwd_kernel<D, Db><<<grid, block, 0, stream>>>(
        q, k, v, g, beta, state, out, H, S, cs, len);
  }
  if (stream == nullptr) hipDeviceSynchronize();
}
}  // namespace

// Explicit-scratch entry point (offline benchmark / autotuner hook): the
// caller owns the B*H*D*D zero-initialized state scratch so it can be
// allocated once and reused across timed iterations.
void kda_delta_rule_fwd_with_scratch(const float* q, const float* k,
                                     const float* v, const float* g,
                                     const float* beta, float* state,
                                     float* out, int B, int H, int S, int D,
                                     void* stream) {
  if (B == 0 || H == 0 || S == 0) return;
  // Db = per-block row count; D/Db = z-blocks of row parallelism per (b,h).
  // Larger D → more blocks so a single (b,h) still fans out across CUs.
  const hipStream_t st = static_cast<hipStream_t>(stream);
  if (D <= 16)       launch<16, 16>(q, k, v, g, beta, state, out, B, H, S, st);
  else if (D <= 32)  launch<32, 32>(q, k, v, g, beta, state, out, B, H, S, st);
  else if (D <= 64)  launch<64, 16>(q, k, v, g, beta, state, out, B, H, S, st);
  else if (D <= 128) launch<128, 32>(q, k, v, g, beta, state, out, B, H, S, st);
  else               launch<128, 32>(q, k, v, g, beta, state, out, B, H, S, st);  // cap
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
  // legacy entry: eager default-stream + sync (the stream variant is the
  // graph-capture-safe one)
  kda_delta_rule_fwd_with_scratch(q, k, v, g, beta, state, out, B, H, S, D,
                                  nullptr);
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

// ===========================================================================
//  #L8 chunked WY forward (#70) — the inter-chunk parallelism lever.
//
// The cooperative kernel above is bound by its serial per-token loop (4
// barriers/token; the per-block limit found in docs/kda-lds-optimization.md).
// This path implements the SAME K3 per-key-dim recurrence in the affine WY
// (UT-transform) form derived and CPU-verified against the oracle in
// tests/kernels/attn/test_kda_k3_chunked.cpp (k3_wy_chunked_fwd — bit-close
// to kda_naive_delta_rule_fwd_cpu, incl. the full S=512 D=128 cs=64 shape):
//
//   per chunk (cs=64, local t), L_t[k] = within-chunk per-key log-cumsum:
//     M[t][j] = b_j sum_k exp(L_t-L_j)[k] k_j[k] k_t[k]   (strict tril)
//     N[t][j] = b_j sum_k exp(L_t-L_j)[k] k_j[k] q_t[k]   (incl diag)
//     Ainv = (I + tril(M,-1))^{-1}                       (explicit inverse)
//     Kgw = exp(L_t) k_t;  Qgw = exp(L_t) q_t            (POST-gate: G_{0,t})
//     Kgb = b_t exp(L_end-L_t) k_t;  diagG = exp(L_end)
//     U_v = Ainv v;  W = Ainv Kgw;  T = N U_v;  Opar = Qgw - N W
//   serial state pass (per (b,h), rowblock-splittable over state rows v):
//     u = U_v - W C^T;   o = T + Opar C^T;
//     C = diagG (.) C + Kgb^T u
//
// Three kernels: (1) per-key cumsum — parallel over (b,h,c,k); (2) all
// chunk-local artifacts — parallel over (b,h,c), needs NO inter state; (3)
// the nc-step serial state pass, parallel over (b,h, C-rowblock): each
// state row v only ever meets row v of C, so blockIdx.x splits D rows
// across blocks exactly like the cooperative kernel's row split. The serial
// critical path drops from S token steps (4 barriers each) to nc chunk
// steps (3 barriers each) of GEMM-shaped cooperative work.
//
// CONTRACT (see the CPU test): k is L2-NORMALISED by the caller (the
// production contract, already assumed by bench_kda) so |M| <= 1 by
// Cauchy-Schwarz and the explicit Ainv stays bounded; gates in (0,1],
// beta <= 1. Every exp() argument is <= 0, so every gate factor is in
// (0,1]: no overflow, graceful underflow. S % 64 == 0, D in {16,32,64,128}.
// ===========================================================================
static constexpr int kCs   = 64;   // WY chunk size (must divide S)
static constexpr int kThL  = 512;  // threads for the chunk-local kernel: the
                                   // 35KB of LDS (sM+sN+sTMP) caps residency
                                   // at one block/CU, so doubling the block
                                   // doubles resident warps (8->16) for the
                                   // latency-bound gram loads.

// ---- (1) per-key-dim within-chunk inclusive log-cumsum -------------------
__global__ void kda_k3_cumsum_kernel(const float* __restrict__ g,
                                     float* __restrict__ L,
                                     int nc, int S, int D) {
  const int bh = blockIdx.x;
  const int c  = blockIdx.y;
  const int kk = threadIdx.x;          // one key-dim per thread
  if (kk >= D) return;
  const size_t base  = ((size_t)bh * nc + c) * kCs * D;
  const size_t gbase = ((size_t)bh * S + (size_t)c * kCs) * D;
  float acc = 0.0f;
  for (int t = 0; t < kCs; ++t) {
    const float gg = g[gbase + (size_t)t * D + kk];
    acc += (gg <= 0.0f) ? -1.0e9f : logf(gg);   // log(0) clamp (kda.cpp)
    L[base + (size_t)t * D + kk] = acc;
  }
}

// ---- (2a) gate-weighted grams M/N — LDS-staged, quarter by quarter -----
// The gram dots re-read k/q/L rows ~cs times each; done straight from
// gmem that is L2-latency-bound (measured ~310 us/block even at 512
// threads — dependent loads, poor ILP through the pair decode). Instead
// stage k/q/L tiles for one quarter of the key dim in LDS (3 x cs x D/4
// floats; 24 KB at D=128, fits beside the 32 KB M/N accumulators), so
// every gram read is an LDS hit and the phase becomes SFU (exp) bound.
// M/N land in gmem ([B,H,nc,cs,cs] each) for the (2b) kernel.
template <int D>
__global__ void kda_k3_gram_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ beta, const float* __restrict__ L,
    float* __restrict__ M, float* __restrict__ N,
    int nc, int S) {
  // k and L staged at FULL key width (2 x cs x D = 64 KB at D=128, the
  // whole per-workgroup LDS on gfx942): every gram operand except q is an
  // LDS hit and all 32 lanes carry one float4 each (D=128 -> exactly one
  // 16 B load per lane per row). q is streamed -- one gmem float4 per lane
  // per pair, L2-resident (each q row is re-read ~cs times). Each pair's
  // dot completes in ONE pass, so M/N go straight to gmem: no LDS
  // accumulators, and the only cross-thread step is the shfl reduce.
  static_assert(D % 16 == 0, "D must be a multiple of 16");
  const int bh = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
  const int t0 = c * kCs;
  const size_t tok = (size_t)bh * S + t0;
  const size_t lb  = ((size_t)bh * nc + c) * kCs * D;
  const size_t lbq = ((size_t)bh * nc + c) * kCs * kCs;

  __shared__ float sk[kCs * D];
  __shared__ float sL[kCs * D];

  for (int i = tid; i < kCs * D; i += kThL) {
    const int t = i / D, e = i % D;
    sk[i] = k[(tok + t) * D + e];
    sL[i] = L[lb + (size_t)t * D + e];
  }
  __syncthreads();

  const int warp = tid / 32, lane = tid % 32;
  const int n_pairs = kCs * (kCs + 1) / 2;
  for (int pr = warp; pr < n_pairs; pr += kThL / 32) {
    int t = 0, rem = pr;
    while (rem > t) { rem -= t + 1; ++t; }
    const int j = rem;
    const float4* kj4 = (const float4*)(sk + (size_t)j * D);
    const float4* kt4 = (const float4*)(sk + (size_t)t * D);
    const float4* Lj4 = (const float4*)(sL + (size_t)j * D);
    const float4* Lt4 = (const float4*)(sL + (size_t)t * D);
    const float4* qt4 = (const float4*)(q + (tok + t) * D);
    float mk = 0.0f, nk = 0.0f;
    for (int e4 = lane; e4 < D / 4; e4 += 32) {
      const float4 a = kj4[e4], b = kt4[e4], lj = Lj4[e4], lt = Lt4[e4];
      const float4 cc = qt4[e4];                    // streamed (gmem/L2)
      const float4 gw = {expf(lt.x - lj.x), expf(lt.y - lj.y),
                         expf(lt.z - lj.z), expf(lt.w - lj.w)};
      mk += gw.x * a.x * b.x + gw.y * a.y * b.y + gw.z * a.z * b.z +
            gw.w * a.w * b.w;
      nk += gw.x * a.x * cc.x + gw.y * a.y * cc.y + gw.z * a.z * cc.z +
            gw.w * a.w * cc.w;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      mk += __shfl_xor(mk, off);
      nk += __shfl_xor(nk, off);
    }
    if (lane == 0) {
      const float bj = beta[tok + j];
      N[lbq + (size_t)t * kCs + j] = bj * nk;
      if (j < t) M[lbq + (size_t)t * kCs + j] = bj * mk;
    }
  }
  // zero the strict-upper cells (disjoint from the pair writes; K2b reads
  // full rows and must not see stale scratch)
  for (int i = tid; i < kCs * kCs; i += kThL) {
    const int t = i / kCs, j = i % kCs;
    if (j > t) { M[lbq + i] = 0.0f; N[lbq + i] = 0.0f; }
  }
}

// ---- (2b) chunk-local artifacts (Ainv + U_v/W/T/Opar) ------------------
template <int D>
__global__ void kda_k3_local_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ beta,
    const float* __restrict__ L,
    const float* __restrict__ M, const float* __restrict__ N,
    float* __restrict__ Kgw, float* __restrict__ Qgw,
    float* __restrict__ Kgb, float* __restrict__ diagG,
    float* __restrict__ U_v, float* __restrict__ W,
    float* __restrict__ T, float* __restrict__ Opar,
    int nc, int S, unsigned long long* dbg) {
  const int bh = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
  const unsigned long long t0c = clock64();
  #define KDA_PHASE_CLK(ix)                                              \
    do {                                                                 \
      if (dbg && bh == 0 && c == 0 && tid == 0)                          \
        dbg[ix] = clock64() - t0c;                                       \
    } while (0)
  const int t0 = c * kCs;
  const size_t tok = (size_t)bh * S + t0;              // first chunk token
  const size_t lb  = ((size_t)bh * nc + c) * kCs * D;  // per-chunk base

  const size_t lbq = ((size_t)bh * nc + c) * kCs * kCs;

  __shared__ float sM[kCs * kCs];   // M -> Ainv (in place, upper stays 0)
  __shared__ float sN[kCs * kCs];   // N (beta folded, upper stays 0)
  __shared__ float sTMP[3 * 256];   // off-diagonal inverse X tiles

  // grams arrive from the (2a) kernel via gmem (upper cells are zero)
  for (int i = tid; i < kCs * kCs; i += kThL) {
    sM[i] = M[lbq + i];
    sN[i] = N[lbq + i];
  }
  __syncthreads();

  // phase 0: gated operands (elementwise; L ready from kernel 1)
  for (int i = tid; i < kCs * D; i += kThL) {
    const int t = i / D, e = i % D;
    const float lt   = L[lb + (size_t)t * D + e];
    const float lend = L[lb + (size_t)(kCs - 1) * D + e];
    const float e0 = expf(lt), ee = expf(lend - lt);
    Kgw[lb + i] = e0 * k[(tok + t) * D + e];
    Qgw[lb + i] = e0 * q[(tok + t) * D + e];
    Kgb[lb + i] = beta[tok + t] * ee * k[(tok + t) * D + e];
  }
  for (int e = tid; e < D; e += kThL)
    diagG[((size_t)bh * nc + c) * D + e] =
        expf(L[lb + (size_t)(kCs - 1) * D + e]);
  KDA_PHASE_CLK(0);   // M/N load + phase 0 done

  // (grams now live in the kda_k3_gram_kernel launch above)
  __syncthreads();

  // phase 2a: invert the four 16x16 diagonal blocks of A = I + M.
  // One warp per block, lane = column; the whole column lives in the
  // lane's registers so the forward substitution is barrier-free.
  {
    const int w = tid / 32;
    if (w < 4) {
      const int lane = tid % 32;
      if (lane < 16) {
        const int j = lane, br = w * 16;
        float col[16];
        for (int i = j; i < 16; ++i) {
          float x = (i == j) ? 1.0f : 0.0f;
          const float* arow = sM + (size_t)(br + i) * kCs + br;
          for (int l = j; l < i; ++l) x -= arow[l] * col[l];
          col[i] = x;
        }
        for (int i = j; i < 16; ++i)
          sM[(size_t)(br + i) * kCs + br + j] = col[i];
      }
    }
  }
  KDA_PHASE_CLK(2);   // 2a diagonal inverses done
  __syncthreads();

  // phase 2b: off-diagonal inverse blocks, level by level:
  //   Ainv[r][s] = -Ainv[r][r] @ ( sum_{m=s..r-1} M[r][m] @ Ainv[m][s] )
  // Level r only reads M blocks of row r (never overwritten) and Ainv
  // blocks finalized at earlier levels, so two barrier rounds per level
  // suffice: (a) X tiles into sTMP, (b) the -Ainv[r][r] @ X product back
  // into sM block (r,s) — which destroys M[r][s], unused from here on.
  for (int r = 1; r < 4; ++r) {
    for (int s = 0; s < r; ++s)
      for (int cell = tid; cell < 256; cell += kThL) {
        const int i = cell / 16, jj = cell % 16;
        float x = 0.0f;
        for (int m = s; m < r; ++m)
          for (int l = 0; l < 16; ++l)
            x += sM[(size_t)(r * 16 + i) * kCs + m * 16 + l] *
                 sM[(size_t)(m * 16 + l) * kCs + s * 16 + jj];
        sTMP[s * 256 + cell] = x;
      }
    __syncthreads();
    for (int s = 0; s < r; ++s)
      for (int cell = tid; cell < 256; cell += kThL) {
        const int i = cell / 16, jj = cell % 16;
        float x = 0.0f;
        for (int l = 0; l < 16; ++l)
          x += sM[(size_t)(r * 16 + i) * kCs + r * 16 + l] *
               sTMP[s * 256 + l * 16 + jj];
        sM[(size_t)(r * 16 + i) * kCs + s * 16 + jj] = -x;
      }
    __syncthreads();
  }

  KDA_PHASE_CLK(3);   // 2b off-diagonal inverse done

  // phase 3: U_v = Ainv @ v ; W = Ainv @ Kgw   (dot over j; Ainv row in LDS)
  // Four consecutive output columns per thread step: the Ainv row element
  // is loaded once (LDS broadcast) and reused for a float4 of v/Kgw, 4x
  // fewer load instructions than one-column-per-thread.
  for (int i = tid * 4; i < kCs * D; i += kThL * 4) {
    const int t = i / D;
    const float* arow = sM + (size_t)t * kCs;
    float4 sv = {0, 0, 0, 0}, sw = {0, 0, 0, 0};
    for (int j = 0; j < kCs; ++j) {
      const float a = arow[j];
      const float4 vv = *(const float4*)(v + (tok + j) * D + i % D);
      const float4 ww = *(const float4*)(Kgw + lb + (size_t)j * D + i % D);
      sv.x += a * vv.x; sv.y += a * vv.y; sv.z += a * vv.z; sv.w += a * vv.w;
      sw.x += a * ww.x; sw.y += a * ww.y; sw.z += a * ww.z; sw.w += a * ww.w;
    }
    *(float4*)(U_v + lb + i) = sv;
    *(float4*)(W + lb + i)  = sw;
  }
  KDA_PHASE_CLK(4);   // phase 3 done
  __syncthreads();   // orders the gmem U_v/W writes for phase 4 readers

  // phase 4: T = N @ U_v ; Opar = Qgw - N @ W   (N row in LDS; tril-incl)
  for (int i = tid * 4; i < kCs * D; i += kThL * 4) {
    const int t = i / D;
    const float* nrow = sN + (size_t)t * kCs;
    float4 st = {0, 0, 0, 0}, sp = {0, 0, 0, 0};
    for (int j = 0; j <= t; ++j) {
      const float n = nrow[j];
      const float4 uu = *(const float4*)(U_v + lb + (size_t)j * D + i % D);
      const float4 ww = *(const float4*)(W + lb + (size_t)j * D + i % D);
      st.x += n * uu.x; st.y += n * uu.y; st.z += n * uu.z; st.w += n * uu.w;
      sp.x += n * ww.x; sp.y += n * ww.y; sp.z += n * ww.z; sp.w += n * ww.w;
    }
    const float4 qq = *(const float4*)(Qgw + lb + i);
    *(float4*)(T + lb + i) = st;
    *(float4*)(Opar + lb + i) = {qq.x - sp.x, qq.y - sp.y,
                                 qq.z - sp.z, qq.w - sp.w};
  }
  KDA_PHASE_CLK(5);   // phase 4 done
}

// ---- (3) serial state pass: u/o/C per chunk, rowblock-split over v -------
template <int D, int Rb>
__global__ void kda_k3_state_kernel(
    const float* __restrict__ U_v, const float* __restrict__ W,
    const float* __restrict__ T, const float* __restrict__ Opar,
    const float* __restrict__ Kgb, const float* __restrict__ diagG,
    float* __restrict__ state, float* __restrict__ out,
    int nc, int S) {
  const int bh = blockIdx.y;
  const int v0 = blockIdx.x * Rb;      // this block's state rows [v0, v0+Rb)
  const int tid = threadIdx.x;
  const size_t lb0 = (size_t)bh * nc;
  // Row stride D+4 (padding): with stride D (a multiple of 32), lanes
  // reading sC[vv*D + e] for their own vv all land in LDS bank e%32 — a
  // 32-way conflict on EVERY state read in the u/o phases (measured ~20x
  // slowdown of the whole serial pass). D+4 both spreads lanes across
  // banks (stride%32 = 4) and keeps every row 16B-aligned for float4.
  __shared__ float sC[Rb * (D + 4)];   // C rows v0..v0+Rb (all k)
  __shared__ float sU[kCs * Rb];       // u tile for the current chunk
  float* Cst = state + (size_t)bh * D * D + (size_t)v0 * D;

  for (int i = tid; i < Rb * D; i += kTh) {
    const int vv = i / D, e = i % D;
    sC[vv * (D + 4) + e] = Cst[i];              // initial state
  }
  __syncthreads();

  for (int c = 0; c < nc; ++c) {
    const size_t lb = (lb0 + c) * kCs * D;
    // u = U_v - W C^T.  Each thread step owns FOUR consecutive state rows
    // (vv0..vv0+3): the W row element (gmem, broadcast across lanes that
    // share t) is loaded once and applied to 4 C values -- 4x fewer gmem
    // broadcasts than one-row-per-thread.
    for (int i = tid * 4; i < kCs * Rb; i += kTh * 4) {
      const int t = i / Rb, vv0 = i % Rb;
      const float* wrow = W + lb + (size_t)t * D;
      const float* cr0 = sC + (size_t)vv0 * (D + 4);
      const float* cr1 = cr0 + (D + 4);
      const float* cr2 = cr1 + (D + 4);
      const float* cr3 = cr2 + (D + 4);
      float4 s = {U_v[lb + (size_t)t * D + v0 + vv0],
                  U_v[lb + (size_t)t * D + v0 + vv0 + 1],
                  U_v[lb + (size_t)t * D + v0 + vv0 + 2],
                  U_v[lb + (size_t)t * D + v0 + vv0 + 3]};
      for (int e = 0; e < D; ++e) {
        const float w = wrow[e];
        s.x -= w * cr0[e]; s.y -= w * cr1[e];
        s.z -= w * cr2[e]; s.w -= w * cr3[e];
      }
      sU[i] = s.x; sU[i + 1] = s.y;
      sU[i + 2] = s.z;          sU[i + 3] = s.w;
    }
    __syncthreads();   // sU complete before o/C read it
    // o = T + Opar C^T    (uses the OLD C = C_{c-1})
    for (int i = tid * 4; i < kCs * Rb; i += kTh * 4) {
      const int t = i / Rb, vv0 = i % Rb;
      const float* orow = Opar + lb + (size_t)t * D;
      const float* cr0 = sC + (size_t)vv0 * (D + 4);
      const float* cr1 = cr0 + (D + 4);
      const float* cr2 = cr1 + (D + 4);
      const float* cr3 = cr2 + (D + 4);
      float4 s = {T[lb + (size_t)t * D + v0 + vv0],
                  T[lb + (size_t)t * D + v0 + vv0 + 1],
                  T[lb + (size_t)t * D + v0 + vv0 + 2],
                  T[lb + (size_t)t * D + v0 + vv0 + 3]};
      for (int e = 0; e < D; ++e) {
        const float o = orow[e];
        s.x += o * cr0[e]; s.y += o * cr1[e];
        s.z += o * cr2[e]; s.w += o * cr3[e];
      }
      float* ot = out + ((size_t)bh * S + c * kCs + t) * D + v0 + vv0;
      *(float4*)ot = s;
    }
    __syncthreads();   // all o readers done with old C before C is updated
    // C = diagG (.) C + Kgb^T u
    //    C[v][k] = diagG[k] C[v][k] + sum_t Kgb[t][k] u[t][v]
    //    (float4 over k; sU[t][v] broadcasts across the k-lanes)
    const float4* dg4 = (const float4*)(diagG + (lb0 + c) * D);
    for (int i = tid * 4; i < Rb * D; i += kTh * 4) {
      const int vv = i / D, e4 = (i % D) / 4;
      const float4 d = dg4[e4];
      const float4 cprev = *(const float4*)(sC + (size_t)vv * (D + 4) + e4 * 4);
      float4 s = {d.x * cprev.x, d.y * cprev.y, d.z * cprev.z, d.w * cprev.w};
      for (int t = 0; t < kCs; ++t) {
        const float4 kg = *(const float4*)(Kgb + lb + (size_t)t * D + e4 * 4);
        const float uu = sU[t * Rb + vv];
        s.x += kg.x * uu; s.y += kg.y * uu;
        s.z += kg.z * uu; s.w += kg.w * uu;
      }
      *(float4*)(sC + (size_t)vv * (D + 4) + e4 * 4) = s;
    }
    __syncthreads();   // sC updated before the next chunk's u/o read it
  }

  for (int i = tid; i < Rb * D; i += kTh) {      // final state
    const int vv = i / D, e = i % D;
    Cst[i] = sC[vv * (D + 4) + e];
  }
}

// Scratch float count for the chunked path: L, Kgw, Qgw, Kgb, U_v, W, T,
// Opar are [B,H,nc,cs,D] (= B*H*S*D each); diagG is [B,H,nc,D].
std::size_t kda_chunked_scratch_floats(int B, int H, int S, int D) {
  const size_t ncsd = (size_t)B * H * S * D;
  const size_t nc = (size_t)S / kCs;
  return 8ull * ncsd + (size_t)B * H * nc * D +
         2ull * (size_t)B * H * nc * kCs * kCs;   // M, N grams
}

namespace {

template <int D>
void launch_chunked(const float* q, const float* k, const float* v,
                    const float* g, const float* beta, float* state,
                    float* out, float* scratch, int B, int H, int S,
                    hipStream_t stream) {
  const int nc = S / kCs;
  const size_t ncsd = (size_t)B * H * S * D;      // per-array float count
  float* L    = scratch;
  float* Kgw  = L + ncsd;         float* Qgw = Kgw + ncsd;
  float* Kgb  = Qgw + ncsd;       float* dgG = Kgb + ncsd;
  float* U_v  = dgG + (size_t)B * H * nc * D;
  float* W    = U_v + ncsd;       float* T   = W + ncsd;
  float* Opar = T + ncsd;
  float* M    = Opar + ncsd;
  float* N    = M + (size_t)B * H * nc * kCs * kCs;
  const dim3 gchunk(B * H, nc);
  kda_k3_cumsum_kernel<<<gchunk, kTh, 0, stream>>>(g, L, nc, S, D);
  kda_k3_gram_kernel<D><<<gchunk, kThL, 0, stream>>>(q, k, beta, L, M, N, nc, S);
  kda_k3_local_kernel<D><<<gchunk, kThL, 0, stream>>>(
      q, k, v, beta, L, M, N, Kgw, Qgw, Kgb, dgG, U_v, W, T, Opar, nc, S,
      nullptr);
  constexpr int Rb = (D <= 32) ? D : 32;          // state rows per block
  const dim3 gstate(D / Rb, B * H);
  kda_k3_state_kernel<D, Rb><<<gstate, kTh, 0, stream>>>(
      U_v, W, T, Opar, Kgb, dgG, state, out, nc, S);
  if (stream == nullptr) hipDeviceSynchronize();  // caller owns sync otherwise
}
}  // namespace

void kda_delta_rule_fwd_chunked_with_scratch(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, float* state, float* out, float* scratch,
    int B, int H, int S, int D, int chunk_size, void* stream) {
  if (B == 0 || H == 0 || S == 0) return;
  if (chunk_size != kCs || S % kCs != 0 || D > 128) {
    std::fprintf(stderr,
                 "kda_delta_rule_fwd_chunked_with_scratch: requires "
                 "chunk_size==64, S%%64==0, D<=128 (got cs=%d S=%d D=%d)\n",
                 chunk_size, S, D);
    return;
  }
  const hipStream_t st = static_cast<hipStream_t>(stream);
  if (D <= 16)      launch_chunked<16>(q, k, v, g, beta, state, out, scratch, B, H, S, st);
  else if (D <= 32) launch_chunked<32>(q, k, v, g, beta, state, out, scratch, B, H, S, st);
  else if (D <= 64) launch_chunked<64>(q, k, v, g, beta, state, out, scratch, B, H, S, st);
  else              launch_chunked<128>(q, k, v, g, beta, state, out, scratch, B, H, S, st);
}

void kda_delta_rule_fwd_chunked(const float* q, const float* k,
                                const float* v, const float* g,
                                const float* beta, float* out,
                                int B, int H, int S, int D, int chunk_size) {
  if (B == 0 || H == 0 || S == 0) return;
  float* state = nullptr;
  hipMalloc(&state, (size_t)B * H * D * D * sizeof(float));
  hipMemsetAsync(state, 0, (size_t)B * H * D * D * sizeof(float), 0);
  float* scratch = nullptr;
  const size_t sf = kda_chunked_scratch_floats(B, H, S, D);
  hipMalloc(&scratch, sf * sizeof(float));
  kda_delta_rule_fwd_chunked_with_scratch(q, k, v, g, beta, state, out,
                                          scratch, B, H, S, D, chunk_size,
                                          nullptr);
  hipFree(state);
  hipFree(scratch);
}

// Tuning aid (#70): median wall time of each of the three chunked launches
// (cumsum / chunk-local / state-pass), in microseconds, over `iters`
// timed runs after one warmup of each. out_us[0..2] = K1, K2, K3.
void kda_chunked_phase_times(const float* q, const float* k, const float* v,
                             const float* g, const float* beta, float* state,
                             float* out, float* scratch,
                             int B, int H, int S, int D, int iters,
                             double* out_us) {
  const int nc = S / kCs;
  const size_t ncsd = (size_t)B * H * S * D;
  float* L    = scratch;
  float* Kgw  = L + ncsd;         float* Qgw = Kgw + ncsd;
  float* Kgb  = Qgw + ncsd;       float* dgG = Kgb + ncsd;
  float* U_v  = dgG + (size_t)B * H * nc * D;
  float* W    = U_v + ncsd;       float* T   = W + ncsd;
  float* Opar = T + ncsd;
  float* M    = Opar + ncsd;
  float* N    = M + (size_t)B * H * nc * kCs * kCs;
  const dim3 gchunk(B * H, nc);
  constexpr int Rb128 = 32;
  unsigned long long* dbg = nullptr;
  hipMalloc(&dbg, 8 * sizeof(unsigned long long));
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto time_launch = [&](auto&& launch) -> double {
    launch();                      // warmup
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
      hipEventRecord(e0);
      launch();
      hipEventRecord(e1);
      hipEventSynchronize(e1);
      float ms = 0;
      hipEventElapsedTime(&ms, e0, e1);
      if (ms > 0.0f && ms * 1000.0f < best) best = ms * 1000.0f;
    }
    return best;
  };
  out_us[0] = time_launch([&] {
    kda_k3_cumsum_kernel<<<gchunk, kTh>>>(g, L, nc, S, D);
    hipDeviceSynchronize();
  });
  if (D <= 16)
    out_us[1] = time_launch([&] {
      kda_k3_gram_kernel<16><<<gchunk, kThL>>>(q, k, beta, L, M, N, nc, S);
      hipDeviceSynchronize();
    });
  else if (D <= 32)
    out_us[1] = time_launch([&] {
      kda_k3_gram_kernel<32><<<gchunk, kThL>>>(q, k, beta, L, M, N, nc, S);
      hipDeviceSynchronize();
    });
  else if (D <= 64)
    out_us[1] = time_launch([&] {
      kda_k3_gram_kernel<64><<<gchunk, kThL>>>(q, k, beta, L, M, N, nc, S);
      hipDeviceSynchronize();
    });
  else
    out_us[1] = time_launch([&] {
      kda_k3_gram_kernel<128><<<gchunk, kThL>>>(q, k, beta, L, M, N, nc, S);
      hipDeviceSynchronize();
    });
  if (D <= 16) {
    out_us[2] = time_launch([&] {
      kda_k3_local_kernel<16><<<gchunk, kThL>>>(q, k, v, beta, L, M, N,
                                              Kgw, Qgw, Kgb, dgG, U_v, W, T,
                                              Opar, nc, S, dbg);
      hipDeviceSynchronize();
    });
    for (int i = 0; i < 6; ++i) out_us[4 + i] = (double)dbg[i] / 1000.0;
    out_us[3] = time_launch([&] {
      kda_k3_state_kernel<16, 16><<<dim3(D / 16, B * H), kTh>>>(
          U_v, W, T, Opar, Kgb, dgG, state, out, nc, S);
      hipDeviceSynchronize();
    });
  } else if (D <= 32) {
    out_us[2] = time_launch([&] {
      kda_k3_local_kernel<32><<<gchunk, kThL>>>(q, k, v, beta, L, M, N,
                                              Kgw, Qgw, Kgb, dgG, U_v, W, T,
                                              Opar, nc, S, dbg);
      hipDeviceSynchronize();
    });
    for (int i = 0; i < 6; ++i) out_us[4 + i] = (double)dbg[i] / 1000.0;
    out_us[3] = time_launch([&] {
      kda_k3_state_kernel<32, 32><<<dim3(D / 32, B * H), kTh>>>(
          U_v, W, T, Opar, Kgb, dgG, state, out, nc, S);
      hipDeviceSynchronize();
    });
  } else if (D <= 64) {
    out_us[2] = time_launch([&] {
      kda_k3_local_kernel<64><<<gchunk, kThL>>>(q, k, v, beta, L, M, N,
                                              Kgw, Qgw, Kgb, dgG, U_v, W, T,
                                              Opar, nc, S, dbg);
      hipDeviceSynchronize();
    });
    for (int i = 0; i < 6; ++i) out_us[4 + i] = (double)dbg[i] / 1000.0;
    out_us[3] = time_launch([&] {
      kda_k3_state_kernel<64, 32><<<dim3(D / 32, B * H), kTh>>>(
          U_v, W, T, Opar, Kgb, dgG, state, out, nc, S);
      hipDeviceSynchronize();
    });
  } else {
    out_us[2] = time_launch([&] {
      kda_k3_local_kernel<128><<<gchunk, kThL>>>(q, k, v, beta, L, M, N,
                                               Kgw, Qgw, Kgb, dgG, U_v, W, T,
                                               Opar, nc, S, dbg);
      hipDeviceSynchronize();
    });
    for (int i = 0; i < 6; ++i) out_us[4 + i] = (double)dbg[i] / 1000.0;
    out_us[3] = time_launch([&] {
      kda_k3_state_kernel<128, Rb128><<<dim3(D / Rb128, B * H), kTh>>>(
          U_v, W, T, Opar, Kgb, dgG, state, out, nc, S);
      hipDeviceSynchronize();
    });
  }
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  hipFree(dbg);
}

}  // namespace vkernels::kernels::hip

#endif  // VKERNELS_HAS_HIP
