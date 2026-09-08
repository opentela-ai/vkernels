// vkernels/kernels/dsa.cu -- CUDA wmma implementation (NVIDIA), a port of
// the gfx942 MFMA dsa_topk_logits kernels in dsa.hip (issue #51) for an
// NVIDIA deployment track (verified on GB10 / Grace Blackwell, sm_121).
//
//   out[b, t] = k_scale[page[b,i], j]
//             * Sum_h ( max(0, Sum_d Q[b,h,d] * K[page[b,i],j,d] )
//                       * gate[b, h] )
//   where t = i*block + j, t < seq_lens[b] (else left unwritten).
//
// Three device variants, mirroring dsa.hip exactly:
//   * dsa_topk_logits_kernel        -- fp32-Q scalar GEMV (Q staged once)
//   * dsa_topk_logits_kernel_fp8q   -- fp8-Q scalar GEMV (Q raw, dequant OTF)
//   * dsa_topk_logits_kernel_wmma   -- bf16 wmma (Q transposed once, K-tiled)
//
// The two scalar kernels are near-verbatim translations (64-lane warpfront
// -> 32-lane warp) and, being plain scalar CUDA/HIP-C++, they are SHARED
// with the HIP TU through dsa_topk_device.cuh (one definition, both
// backends -- same discipline as the numeric helpers in device_numeric.cuh).
// The wmma kernel replaces the AMD __builtin_amdgcn_mfma_f32_16x16x16bf16_1k
// (whose fragment layout drove a warp-shuffle gated H-sum) with nvcuda::
// wmma 16x16x16: sK[B][kBK] @ sQt[D][H] = [B,H] per K-tile, staged through
// shared sAcc[B][H] via store_matrix_sync, then a plain per-j gated H-sum
// (the AMD fragment-layout shuffle has no wmma analogue). The binding
// constraint is BLOCK THREADS = (B/16)*(H/16)*32 = B*H/8 <= 1024 (CUDA max
// threads/block; each warp owns ONE [16,16] c_frag, not the kNF
// fragments-per-lane the AMD 64-lane MFMA layout demands).
//
// GB10 vs gfx942 device cap. gfx942 has NO hipFuncSetAttribute opt-in past
// a 64 KB non-optin cap (see the KB note mi300a-dynamic-lds-no-optin), so
// dsa.hip refuses shapes the largest variant won't fit and falls back by
// staging Q as raw fp8. GB10 (sm_121) DOES honour cudaFuncSetAttribute(
// MaxDynamicSharedMemorySize) up to sharedMemPerBlockOptin = 101,376 B
// (verified on ds5), so the wmma variant (74,496 B at H=128) and the fp32-Q
// variant (99,072 B at H=128) both opt in and run there -- no fp8-Q
// fallback is needed on GB10, though the kernel is ported anyway (and
// exercised by the correctness harness) for parity with the HIP path.
//
// Same contract / dispatch as the HIP kernel (see dsa.hpp): cuda::
// dsa_topk_logits selects a variant via dsa_topk_logits_fits_lds{,_fp8q,
// _mfma}; cuda::dsa_topk_logits_with_variant is the explicit-variant hook
// the autotuner/correctness harness use to FORCE a specific kernel.
#include "vkernels/kernels/dsa.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>
#  include <mma.h>
#  include <cstdio>
#  include "vkernels/kernels/dsa_topk_device.cuh"  // shared scalar dsa_topk_logits kernels + numeric helpers

namespace vkernels::kernels::cuda {

using namespace nvcuda::wmma;

static constexpr int kMmaK = 16;   // one bf16 wmma reduces K = 16
static constexpr int kBK   = 64;   // K-tile (fixed); D is a multiple of 64

// ======================================================================
//  bf16 wmma kernel (port of dsa_topk_logits_kernel_mfma)
// ======================================================================
// Replaces the AMD __builtin_amdgcn_mfma_f32_16x16x16bf16_1k (64-lane
// wavefront, fragment-layout warp-shuffle gated H-sum) with nvcuda::wmma
// 16x16x16 (32-lane warp). The [B,H] = sK[B][kBK] @ sQt[D][H] dot matrix
// per K-tile is computed with wmma (A=sK row_major, B=sQt row_major --
// exactly the gemm_bf16.cu pattern), accumulated across the D/kBK K-tiles
// in register fragments, then store_matrix_sync stages the full [B,H]
// result in shared sAcc[B][H]. The per-token gated reduce (ReLU + gate
// mul + H-sum + k_scale) is then a plain fp32 loop over the H heads, one
// row per lane in warp_c==0 (parallel, no redundant work) -- the AMD
// fragment-layout warp-shuffle reduction has no wmma analogue, so the
// shared staging is the bridge.
//
// Storage (mirrors dsa.hip, with sAcc added for the wmma epilogue):
//   sQt[D][H]   bf16  -- Q transposed ONCE per block (reused across every
//                        page in the split). fp8->bf16 is LOSSLESS.
//   sK[B][kBK]  bf16  -- one K-tile, reloaded per K-tile per page
//   sGate[H]    fp32  -- the per-head gate, loaded once per block
//   sKscale[B]  fp32  -- per-token scales, loaded once per page
//   sAcc[B][H]  fp32  -- the wmma output, staged per K-tile (see below)
// At the GLM-5.3 widths (kBK=64, D=128, B=64):
//   H=32:  (4096+4096)*2 + (32+64)*4 + 64*32*4    = 24,960 B
//   H=64:  (8192+4096)*2 + (64+64)*4 + 64*64*4    = 41,472 B
//   H=128: (16384+4096)*2 + (128+64)*4 + 64*128*4 = 74,496 B
// The H=128 case (74,496 B) exceeds the 49,152 B non-optin default, so the
// launcher opts in via cudaFuncSetAttribute when the request exceeds it
// (per launch -- the attribute is per-device state; see launch_wmma).
//
// ----------------------------------------------------------------------
// Thread + shared-memory caps for the wmma kernel (GB10 / sm_121, verified
// on ds5): each warp owns ONE [16,16] output tile (a single c_frag, not
// the kNF fragments-per-lane the AMD 64-lane MFMA layout demands), so the
// binding constraint is BLOCK THREADS = (B/16)*(H/16)*32 = B*H/8 <= 1024
// (CUDA max threads/block), NOT registers. At H=128,B=64 that is 1024
// threads (the max); larger H needs smaller B. Shared is
//   (D*H + B*kBK)*2 + (H+B)*4 + B*H*4
// (the last term is the sAcc[B][H] staging the store_matrix_sync epilogue
// needs -- the AMD fragment-layout warp-shuffle gated reduce has no wmma
// analogue) = 24,960 / 41,472 / 74,496 B at H=32/64/128. At H=128 that is
// 74,496 B < GB10's 101,376 B opt-in
// ceiling, so the launcher opts in via cudaFuncSetAttribute when a shape
// exceeds the 48 KB default (per launch -- the attribute is per-device
// state; see launch_wmma). The accumulators are ZEROED PER PAGE
// -- each page is an independent [H,B]=Q@K^T dot matrix; carrying acc
// across pages would sum prior pages' K into the current output.
template <int H, int D, int B>
__global__ void dsa_topk_logits_kernel_wmma(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int max_table_len, int max_seq_len, int split_kv) {
  static_assert(H % 16 == 0, "H must be a multiple of 16 (wmma fragment)");
  static_assert(D % kBK == 0, "D must be a multiple of kBK (K-tile)");
  static_assert(B % 16 == 0, "B must be a multiple of 16 (wmma fragment)");
  constexpr int kFragsM = B / 16;          // row fragments (KV tokens)
  constexpr int kFragsN = H / 16;          // col fragments (heads)
  constexpr int kWarps  = kFragsM * kFragsN;
  constexpr int kTh     = kWarps * 32;     // = B*H/8 <= 1024

  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int tid  = threadIdx.x;             // 0..kTh-1
  const int warp = tid >> 5;                // 0..kWarps-1
  const int warp_r = warp / kFragsN;        // which 16-row (KV) fragment
  const int warp_c = warp % kFragsN;        // which 16-col (head) fragment
  const int lane = tid & 31;

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  extern __shared__ float smem[];
  uint16_t* sQt = reinterpret_cast<uint16_t*>(smem);            // D * H
  uint16_t* sK = sQt + (size_t)D * H;                           // B * kBK
  float*    sGate = reinterpret_cast<float*>(sK + (size_t)B * kBK);   // H
  float*    sKscale = sGate + H;                               // B
  float*    sAcc = sKscale + B;                                // B * H

  // --- cooperative load Q transposed as bf16 sQt[D][H] once per block ---
  // Q is laid out [H][D] (q[h*D + d] = Q[h,d]); sQt is the TRANSPOSE
  // [D][H] so the wmma B fragment (sQt[K-row=d][N-col=h]) reads Q[h,d]
  // in one bf16 load. fp8 -> fp32 -> bf16 is lossless.
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  for (int idx = tid; idx < H * D; idx += kTh) {
    const int h = idx / D;
    const int d = idx % D;
    sQt[(size_t)d * H + h] = f2bf(fp8e4m3fnuz_to_f32(qp[(size_t)h * D + d]));
  }
  const float* gp = weight + (size_t)b * H;
  for (int h = tid; h < H; h += kTh) sGate[h] = gp[h];
  __syncthreads();

  fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;

  for (int it = 0; it < n_iters; ++it) {
    const int pg = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + pg];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));

    // Per-token scales: load ONCE per page (the trailing B*4 bytes; they do
    // not change across K-tiles).
    for (int j = tid; j < B; j += kTh)
      sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];

    fill_fragment(c_frag, 0.0f);   // per-page: independent dot matrix

    // K-tiles: D/kBK of them, each kBK wide. A K-tile is consumed by
    // kBK/kMmaK = 4 wmma mma_syncs.
    for (int kt = 0; kt < D / kBK; ++kt) {
      // cooperative load sK[B][kBK] for THIS K-tile (bf16). col is the LOCAL
      // K-column; gcol = kt*kBK + col is the global D-column. (D is a
      // multiple of kBK so no D-bound check.)
      for (int idx = tid; idx < B * kBK; idx += kTh) {
        const int row = idx / kBK;          // 0..B-1 (KV token j)
        const int col = idx % kBK;          // 0..kBK-1 (local K-col)
        const int gcol = kt * kBK + col;
        sK[(size_t)row * kBK + col] =
            f2bf(fp8e4m3fnuz_to_f32(kbase[(size_t)row * D + gcol]));
      }
      __syncthreads();

      // --- kBK/kMmaK = 4 wmma mma_syncs consume the K-tile; c_frag is
      //     preserved across K-tiles -> the full D-dot per (j, h). ---
      #pragma unroll
      for (int mf = 0; mf < kBK / kMmaK; ++mf) {
        const int kk = mf * kMmaK;
        load_matrix_sync(a_frag, (const __nv_bfloat16*)&sK[(size_t)(warp_r * 16) * kBK + kk],
                         kBK);
        load_matrix_sync(b_frag, (const __nv_bfloat16*)&sQt[(size_t)(kt * kBK + kk) * H + warp_c * 16],
                         H);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      __syncthreads();                       // before reloading sK next tile
    }

    // Stage this warp's [16,16] fp32 result into sAcc[B][H] (row = warp_r*16
    // + r, col = warp_c*16 + c). The gated reduce below reads sAcc by (j,h):
    // every warp_c stores its 16-col slice, so sAcc[j][0..H) is the COMPLETE
    // dot row after the sync. (The AMD fragment-layout warp-shuffle gated
    // reduce has no wmma analogue -- this shared staging is the bridge.)
    store_matrix_sync(&sAcc[(size_t)warp_r * 16 * H + warp_c * 16], c_frag, H,
                      mem_row_major);
    __syncthreads();

    // --- per-token gated reduce (fp32) + k_scale, then write ---
    // Only warp_c==0 writes (it reads sAcc[j][0..H) -- the complete row,
    // since every warp_c already stored its 16-col slice above). The 16
    // lanes of the warp each own one of the warp's 16 rows, so the reduce
    // is parallel across rows with no redundant work.
    if (warp_c == 0) {
      const int j = warp_r * 16 + (lane & 15);   // 0..B-1 (each lane a row)
      if (lane < 16 && j < B) {
        float partial = 0.0f;
        for (int h = 0; h < H; ++h) {
          float d = sAcc[(size_t)j * H + h];
          if (d < 0.0f) d = 0.0f;              // ReLU (max(0, raw))
          partial += sGate[h] * d;
        }
        const int t = pg * B + j;
        if (t < seq_len)
          out[(size_t)b * max_seq_len + t] = sKscale[j] * partial;
      }
    }
    __syncthreads();                         // before next page's sK load
  }
}

// ---------------------------------------------------------------------------
//  Launchers
// ---------------------------------------------------------------------------
namespace {

// CUDA's default per-block shared budget (static + dynamic). Shapes whose
// request exceeds it must opt in per function; see the launch sites below.
constexpr int kSharedMemDefault = 48 * 1024;

// The opt-in dynamic-shared ceiling of the CURRENT device -- 101,376 B on
// GB10 / sm_121 (sharedMemPerBlockOptin, verified on ds5 / dgx-spark-05;
// larger on datacenter parts). Queried per dispatch instead of baked in: a
// serving process may address several devices with different ceilings, and
// the attribute is read-only device state, so there is nothing to cache or
// synchronise. 0 = no usable device -- every variant is then refused.
//
// With the ceiling queried, admission runs the SAME host-visible arithmetic
// as the HIP path: dsa_topk_logits_fits_lds{,_fp8q,_wmma} from dsa.cpp with
// `lds_cap` passed in (the fp32-Q kernel's largest admitted shape, H=128
// D=128 B=64, stages 99,072 B -- under GB10's ceiling, so GB10 runs the
// fp32-Q kernel at every width gfx942 admits AND larger; the wmma kernel's
// 74,496 B at H=128 also fits).
int device_lds_optin_cap() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  int cap = 0;
  if (cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             dev) != cudaSuccess)
    return 0;
  return cap;
}

template <int H, int D, int B>
void launch_wmma(const uint8_t* q_fp8, const uint8_t* kvcache_u8,
                 const float* weight, const int32_t* seq_lens,
                 const int32_t* page_table, float* out,
                 int batch_size, int max_table_len, int max_seq_len,
                 int split_kv) {
  constexpr int kTh = (B / 16) * (H / 16) * 32;   // = B*H/8 <= 1024
  const int shmem =
      (D * H + B * kBK) * 2 + (H + B) * (int)sizeof(float) +
      B * H * (int)sizeof(float);
  // Opt in only when the shape exceeds CUDA's 48 KB default per-block shared
  // budget (e.g. H=128 -> 74,496 B). The attribute is PER-DEVICE state, so a
  // process-lifetime set -- the static-init this replaces -- silently leaves
  // every other device in a multi-GPU process unconfigured; setting it per
  // launch is a cheap host-side call and small shapes never pay it.
  if (shmem > kSharedMemDefault) {
    cudaFuncSetAttribute((const void*)dsa_topk_logits_kernel_wmma<H, D, B>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
  }
  dim3 block(kTh);
  dim3 grid(batch_size, split_kv, 1);   // x=batch, y=split (mirrors HIP)
  dsa_topk_logits_kernel_wmma<H, D, B><<<grid, block, shmem, 0>>>(
      q_fp8, kvcache_u8, weight, seq_lens, page_table, out,
      max_table_len, max_seq_len, split_kv);
}

}  // namespace

// Dispatch shim shared by dsa_topk_logits (auto) and
// dsa_topk_logits_with_variant (explicit). `variant`: 0 = auto
// (wmma -> fp32q -> fp8q -> refuse); 1 = fp32-Q; 2 = fp8-Q; 3 = wmma.
// An explicit variant that does NOT fit the device's opt-in shared ceiling
// is refused with a stderr diagnostic + no-op (the caller zeroed `out`,
// which is what stays) rather than launch a block the driver silently
// drops -- mirrors the HIP path's contract.
namespace {
void dsa_topk_logits_dispatch(int batch_size, int num_heads, int head_dim,
                              int block, int max_table_len, int max_seq_len,
                              int split_kv, const void* q_fp8,
                              const void* kvcache_u8, const void* weight,
                              const void* seq_lens, const void* page_table,
                              void* out, int variant) {
  if (batch_size <= 0 || max_table_len <= 0) return;  // nothing to write
  if (num_heads <= 0 || head_dim <= 0 || block <= 0) return;  // bad dims

  const auto* q  = reinterpret_cast<const uint8_t*>(q_fp8);
  const auto* kv = reinterpret_cast<const uint8_t*>(kvcache_u8);
  const auto* w  = reinterpret_cast<const float*>(weight);
  const auto* sl = reinterpret_cast<const int32_t*>(seq_lens);
  const auto* pt = reinterpret_cast<const int32_t*>(page_table);
  auto* o        = reinterpret_cast<float*>(out);

  const int H = num_heads, D = head_dim, B = block;

  const bool want_wmma  = (variant == 3) || (variant == 0);
  const bool want_fp32q = (variant == 1) || (variant == 0);
  const bool want_fp8q  = (variant == 2) || (variant == 0);

  // Admission arithmetic shared with the HIP path (dsa.cpp), evaluated
  // against the current device's queried opt-in ceiling.
  const int lds_cap = device_lds_optin_cap();

  // Grid for the scalar kernels (x=batch, y=split). The wmma kernel builds
  // its own grid inside launch_wmma.
  dim3 grid(batch_size, split_kv, 1);

  // Fast path: the bf16 wmma kernel (Matrix-Core). Shapes need H%16==0,
  // D%64==0, B%16==0 and kTh=(B/16)*(H/16)*32 <= 1024 (CUDA max
  // threads/block). The wmma kernel is a template on (H,D,B); only the
  // GLM-5.3 family (H in {16,32,48,64,80,96,112,128}, D=128, B in
  // {16,32,48,64}) is instantiated below. Other fitting shapes fall
  // through to the scalar kernels.
  if (want_wmma && dsa_topk_logits_fits_lds_wmma(H, D, B, lds_cap)) {
    if (H == 32 && D == 128 && B == 64)  { launch_wmma<32,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 64 && D == 128 && B == 64)  { launch_wmma<64,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 128 && D == 128 && B == 64) { launch_wmma<128,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 32 && D == 128 && B == 16)  { launch_wmma<32,128,16>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 32 && D == 128 && B == 32)  { launch_wmma<32,128,32>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 16 && D == 128 && B == 64)  { launch_wmma<16,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 64 && D == 128 && B == 32)  { launch_wmma<64,128,32>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 64 && D == 128 && B == 16)  { launch_wmma<64,128,16>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    // Shape fits the wmma gate but is not instantiated: fall through to the
    // scalar kernels (they admit every shape, so this is never a refuse).
  }

  // fp32-Q kernel (Q dequanted once into shared) -- the default for any
  // shape the wmma kernel doesn't cover, up to the device's opt-in ceiling.
  if (want_fp32q && dsa_topk_logits_fits_lds(H, D, B, lds_cap)) {
    const int shmem = (H * D + H + B * D + B) * (int)sizeof(float);
    if (shmem > kSharedMemDefault) {   // same per-launch opt-in as launch_wmma
      cudaFuncSetAttribute((const void*)dsa_topk_logits_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    dim3 blockdim(B);
    dsa_topk_logits_kernel<<<grid, blockdim, shmem, 0>>>(
        q, kv, w, sl, pt, o, H, D, B, max_table_len, max_seq_len, split_kv);
    return;
  }

  // fp8-Q kernel (Q staged raw, dequantised on the fly in the dot loop with
  // the SAME helper -- bit-identical output to the fp32-Q kernel above) for
  // shapes that fit the smaller fp8-Q footprint but not the fp32-Q one.
  // Same per-launch opt-in as the fp32-Q path (the fp8-Q request also
  // exceeds the 48 KB default at the larger GLM-5.3 widths).
  if (want_fp8q && dsa_topk_logits_fits_lds_fp8q(H, D, B, lds_cap)) {
    const int shmem = (H + B * D + B) * (int)sizeof(float) + H * D;
    if (shmem > kSharedMemDefault) {   // same per-launch opt-in as launch_wmma
      cudaFuncSetAttribute((const void*)dsa_topk_logits_kernel_fp8q,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    dim3 blockdim(B);
    dsa_topk_logits_kernel_fp8q<<<grid, blockdim, shmem, 0>>>(
        q, kv, w, sl, pt, o, H, D, B, max_table_len, max_seq_len, split_kv);
    return;
  }

  // No admitted variant. REFUSE the shape rather than launch a block the
  // driver silently drops; the caller zeroed `out`, which is what stays.
  std::fprintf(stderr,
      "vk_cuda_dsa_topk_logits: indexer (H=%d D=%d B=%d) exceeds the device's "
      "opt-in dynamic-shared ceiling (%d B) under ALL of wmma (%d B), fp32-Q "
      "(%d B) and fp8-Q (%d B). Refusing (output left as the caller "
      "provided).\n",
      num_heads, head_dim, block, lds_cap,
      (D * H + B * kBK) * 2 + (H + B) * 4 + B * H * 4,
      (H * D + H + B * D + B) * 4,
      (H + B * D + B) * 4 + H * D);
}
}  // namespace

void dsa_topk_logits_with_variant(int batch_size, int num_heads, int head_dim,
                                  int block, int max_table_len,
                                  int max_seq_len, int split_kv,
                                  const void* q_fp8, const void* kvcache_u8,
                                  const void* weight, const void* seq_lens,
                                  const void* page_table, void* out,
                                  int variant) {
  if (variant != 0 && variant != 1 && variant != 2 && variant != 3) {
    std::fprintf(stderr,
        "vk_cuda_dsa_topk_logits_with_variant: bad variant=%d "
        "(want 0=auto, 1=fp32q, 2=fp8q, 3=wmma). Refusing.\n", variant);
    return;
  }
  dsa_topk_logits_dispatch(batch_size, num_heads, head_dim, block,
                           max_table_len, max_seq_len, split_kv, q_fp8,
                           kvcache_u8, weight, seq_lens, page_table, out,
                           variant);
}

void dsa_topk_logits(int batch_size, int num_heads, int head_dim, int block,
                     int max_table_len, int max_seq_len, int split_kv,
                     const void* q_fp8, const void* kvcache_u8,
                     const void* weight, const void* seq_lens,
                     const void* page_table, void* out) {
  // Auto: wmma (Matrix-Core, if instantiated) -> fp32-Q -> fp8-Q -> refuse.
  dsa_topk_logits_dispatch(batch_size, num_heads, head_dim, block,
                           max_table_len, max_seq_len, split_kv, q_fp8,
                           kvcache_u8, weight, seq_lens, page_table, out, 0);
}

}  // namespace vkernels::kernels::cuda

#endif  // VKERNELS_HAS_CUDA
