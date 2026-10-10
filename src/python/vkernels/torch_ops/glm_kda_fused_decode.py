"""CUDA KDA fused decode — conv update + delta rule + gated RMSNorm in ONE
kernel (borrow9-kda-cuda follow-up, donor lane).

Vendored from SGLang v0.5.20 ``kernels/ops/attention/kda_fused_decode.py`` +
``kernels/jit/csrc/attention/kda_fused_decode.cuh`` (Apache-2.0; the kernel
body is the NVIDIA x Moonshot Kimi-K3 optimization package's
``kda_decode_fusion_kernel.cu`` many-heads variant, with SGLang's integration
patches already in: row-strided x/g/beta/onorm_g so the fused qkvg GEMM
slices feed it without copies, strided conv/ssm pools, padded graph slots,
PDL, and 1D-TMA state staging). One kernel replaces the three-kernel decode
chain ``causal_conv1d_update -> kda_packed_decode -> rms_norm_gated``: -2
launches x 34 GLM-5.3 KDA layers (~-68 graph nodes + kernel-gap time) on top
of the packed kernel's ~2x state-bandwidth win. Per (token, value-head) CTA:
depthwise conv update (fp32 taps, SiLU in-kernel, conv pool shifted in
place) -> l2-normed q/k + softplus/lower-bound decay + sigmoid beta ->
delta-rule state update over shared-memory-staged [32, 128] state chunks
(cp.async double-buffer, or 1D-TMA + mbarrier staging when the slot pitch is
16B-aligned) -> sigmoid-gated RMSNorm tail.

DEVIATIONS FROM THE DONOR (mechanical, none in the math):
- no sgl_kernel/tvm-ffi stack: the CUDA source is embedded, compiled once
  with the local nvcc (donor flags: ``-O3 --use_fast_math``) into a cached
  .so under the user cache dir, and called through a plain ``extern "C"``
  launcher (raw pointers + the torch current stream) via ctypes — the
  :mod:`~vkernels.torch_ops.glm_kda_packed_decode` house pattern;
- the donor's compile-time head pin ``H = HV in {12, 6, 3}`` (K3 TP8/16/32
  instantiations; head count was a template constant there) is relaxed to
  RUNTIME ``H == HV`` — the K3 static-decode layout maps ``i_h = i_hv``
  (KDA is MHA), so unequal H/HV stays eligibility-REJECTED; GLM-5.3-Flash
  TP4 is ``H = HV = 16`` (K = V = 128, conv width 4, T = 1 — all pinned by
  the kernel and all matching GLM-5.3);
- the donor's C++ ``TensorMatcher`` asserts became the Python eligibility
  checks below (:class:`OpNotEligible`);
- eligibility additionally requires ``state.stride(0) % 4 == 0``: the
  donor's cp.async fallback (selected for non-16B-aligned slot pitches)
  itself issues 16B ``cp.async.cg`` loads from the slot base, which PTX
  requires 16B-aligned — a pitch not divisible by 4 fp32 elements is silent
  UB in the donor; rejected here instead. Every real pool satisfies it
  (dense pitch HV*V*K, or multi-layer envelope pitches);
- PDL opt-in via ``use_pdl`` (sm_90+ PTX griddepcontrol, launch attribute
  set; no-op semantics otherwise). The kernel arg list moved into the
  launcher; the kernel body itself is verbatim.

FLOE CONTRACT MAPPING (GLM-5.3-Flash KDA decode, per rank TP4, H = HV = 16):
- ``mixed_qkv`` ``[B, 3*H*128]`` bf16, contiguous rows (row stride
  arbitrary), q | k | v segments (2048|2048|2048) — the RAW PRE-conv fused
  projection output, i.e. the tensor floe feeds INTO
  :func:`~vkernels.torch_ops.glm_kda_conv_update.kda_conv_update`; the
  sibling :func:`~vkernels.torch_ops.glm_kda_packed_decode.glm_kda_packed_decode`
  takes the conv-update OUTPUT instead — the two kernels bracket the conv;
- ``a`` ``[B, HV*128]`` bf16 — the RAW ``f_b(f_a(x))`` GEMV output (the
  kernel adds ``dt_bias`` and applies the gate branch itself: softplus
  ``-exp(A_log)·softplus(a+dt_bias)`` or the lower-bound sigmoid);
- ``b`` ``[B, HV]`` bf16 — the RAW ``b_proj`` dots (kernel applies sigmoid);
- ``conv_states`` ``[slots, 3, 3*H*128]`` bf16 — the packed mamba conv pool,
  TIME-major ``[w, channel]`` with q|k|v channel segments in mixed_qkv
  order; shifted IN PLACE (``[old w1, old w2, new raw x]`` — pure bf16
  moves, bit-exact). floe's incumbent conv pool is CHANNEL-major
  ``[slots, C, 3]`` — adapt at the boundary with
  :func:`kda_conv_state_cmajor_to_wmajor` / :func:`kda_conv_state_wmajor_to_cmajor`
  (or migrate the pool);
- ``w_q_t``/``w_k_t``/``w_v_t`` ``[4, H*128]`` fp32 — depthwise conv taps,
  TIME-major. The donor checkpoint keeps conv weights fp32; floe's
  ``Conv1d`` weight is ``[C, 1, 4]`` bf16 —
  :func:`kda_conv_weight_to_taps` does the load-time transpose + exact
  bf16->fp32 widen;
- ``conv_bias`` ``[3*H*128]`` fp32 — floe's conv1d is bias-free: pass zeros
  (the donor's ``bias=None`` path);
- ``A_log`` ``[HV]`` fp32, ``dt_bias`` ``[HV*128]`` fp32 (layer params);
- ``onorm_g`` ``[B, HV*128]`` bf16 — the o_norm gate, i.e. the RAW
  ``g_b(g_a(x))`` dots (the g_a handoff); sigmoid applied in-kernel;
- ``onorm_weight`` ``[128]`` fp32 — SHARED ACROSS HEADS: K3's
  ``FusedRMSNormGated`` and floe's ``Glm53RMSNormGated`` are both
  per-head-dim ``[head_dim]`` norms, so this matches floe's
  ``o_norm.weight`` directly;
- ``state`` pool, V-MAJOR: ``[slots, HV, V, K]`` fp32, inner-dense, any
  slot pitch — the SAME pool convention as ``glm_kda_packed_decode`` (a
  born-V-major pool serves both kernels; the adapters
  ``kda_state_kmajor_to_vmajor``/``kda_state_vmajor_to_kmajor`` in that
  module apply verbatim at the boundary). Updated in place; selected rows
  only. NOTE the square-head K/V layout ambiguity documented there;
- ``cache_indices`` ``[B]`` int32 — pool slot ids; ``-1`` marks a padded
  CUDA-graph slot (output row zeroed, BOTH pools untouched — tested).
  Duplicate slot ids in one launch are UB (racing CTAs);
- ``out`` ``[B, HV*128]`` bf16 dense (pass ``None`` to allocate) —
  returned as ``[1, B, HV, V]`` (the donor's view; note the packed sibling
  returns ``[B, 1, HV, V]``).

NUMERICS: everything between the bf16 inputs and the final bf16 store is
fp32 — in particular the conv output is NOT rounded to bf16 before the
recurrence (the incumbent chain's conv store rounds; that bf16-round class
plus the fast-math intrinsics (``__expf``/``log1pf``/``rsqrtf``) put
cross-path agreement at ~1e-2 relative, the packed kernel's documented
class; vs the fp32 oracle below the outputs sit in the bf16-ULP band and
the state at fp32 ULPs). The output ABI is bf16 — the same store rounding
the audited fp32-output decode path rejects, so floe promotion of this
kernel rides the greedy-parity gates exactly like the packed kernel.

CUDA JIT: nvcc via PATH or ``/usr/local/cuda*/bin/nvcc``; cached under
``~/.cache/vkernels/cuda-jit`` keyed by source + nvcc version + arch +
flags. First call compiles (seconds); ``VK_CUDA_JIT=0`` disables the op.
Inference only, no autograd backward.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache

from ._dispatch import OpNotEligible

__all__ = [
    "glm_kda_fused_decode",
    "glm_kda_fused_decode_eligible",
    "glm_kda_fused_decode_reference",
    "kda_conv_weight_to_taps",
    "kda_conv_state_cmajor_to_wmajor",
    "kda_conv_state_wmajor_to_cmajor",
]

_K, _V = 128, 128
_CONV_STATE_W = 3  # kernel width 4 -> 3 cached tokens

# The CUDA source. Kernel body verbatim from the donor .cuh (the NVIDIA x
# Moonshot K3 fusion kernel with SGLang's integration patches); the harness
# around it (device helper shims for sgl_kernel/math|warp|utils, the extern
# "C" launcher, runtime head counts) is rewritten to be dependency-free.
_CUDA_SOURCE = r"""
// KDA fused decode step: causal conv1d update + delta-rule recurrence +
// gated RMSNorm in a single kernel (replaces the three-kernel
// causal_conv1d_update -> kda_packed_decode -> rms_norm_gated decode chain).
//
// Kernel body vendored from the NVIDIA x Moonshot Kimi K3 optimization
// package (KDA_decode/kda_decode_fusion_kernel.cu, many-heads variant) via
// SGLang v0.5.20 kernels/jit/csrc/attention/kda_fused_decode.cuh
// (Apache-2.0), which carried these integration patches: explicit row
// strides for x/g/beta/onorm_g (fused qkvg-projection GEMM slices feed the
// kernel without copies); conv state addressed through cs_slot_stride/
// cs_w_stride (packed [slots, width, conv_dim] mamba pool, updated in
// place); ssm state addressed through state_slot_stride (envelope-strided
// [slots, HV, V, K] pools; int64 slot*stride math avoids the envelope-pitch
// overflow); ssm_state_indices honored (padded cuda-graph slots < 0 zero
// the output row and skip all pool updates); PDL; automatic 1D-TMA bulk
// state load for aligned recurrent-state layouts.
//
// Local port changes (see the module docstring DEVIATIONS): sgl_kernel
// math/warp/PDL helpers replaced by the shims below (same bodies); the
// donor's kFixedHeads/kFixedValueHeads template constants became the
// runtime H/HV kernel params (head-grid decode layout, MHA i_h = i_hv).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

using bf16_t = __nv_bfloat16;

namespace vkda {

// ---- local shims for sgl_kernel/math.cuh / warp.cuh / utils.cuh ---------
namespace device {
namespace math {
__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}
__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
}  // namespace math
namespace warp {
__device__ __forceinline__ float reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}
}  // namespace warp
template <bool kUsePDL>
__device__ __forceinline__ void PDLWaitPrimary() {
#if __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}
template <bool kUsePDL>
__device__ __forceinline__ void PDLTriggerSecondary() {
#if __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    // the "memory" clobber is load-bearing (donor comment): keeps this
    // kernel's stores from sinking past the dependent-grid trigger
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}
}  // namespace device

// ---- local PTX primitives (cp.async / mbarrier / async-proxy fence) -----

namespace ptx {

// Generic ptr -> 32-bit `.shared` address: inline-PTX `.shared` instructions
// take a byte offset in the shared window, not a generic pointer.
template <typename T>
static __device__ __forceinline__ uint32_t to_shared(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ---- non-bulk cp.async (PTX ISA 9.7.9.24) ---------------------------------

// One 16-byte cache-global segment, global -> shared. Both pointers must be
// 16-byte aligned.
static __device__ __forceinline__ void cp_async_cg_16b(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(smem_dst)), "l"(gmem_src) : "memory");
}

static __device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

// Wait until at most N committed cp.async groups remain pending.
template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
  static_assert(N >= 0 && N <= 7, "cp.async wait-group count must be in [0, 7]");
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

static __device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

// ---- bulk 1D TMA (PTX ISA 9.7.9.25) ---------------------------------------

// global -> shared::cluster, completed by an smem mbarrier. Arm `bar` with
// `mbar_arrive_expect_tx(bar, bytes)` before issuing; `bytes` and both
// endpoints must be 16-byte aligned.
static __device__ __forceinline__ void cp_async_bulk_1d_load(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(to_shared(bar))
      : "memory");
}

// ---- mbarrier (PTX ISA 9.7.13.15) -----------------------------------------
//
// Only the `try_wait.parity` waiter is wrapped; the caller owns the phase
// counter and flips it at the stage wrap. After `mbar_init` the bar is at
// parity 0 and each full cycle (count arrivals -> fire -> reset) flips it, so a
// consumer-first waiter starts at 0 and a producer-first waiter (whose first
// wait must be a no-op skip) starts at 1. Getting this backwards deadlocks on
// the second wait.
static __device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}

// Combined arrive + set tx-count, for TMA-load completion.
static __device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}

// Wait for phase `parity` to complete. Looped because the spec allows spurious
// early wakeups. The default `.acquire` semantics make prior `cp.async.bulk`
// writes tracked by this mbarrier visible to later generic-proxy reads on this
// thread with no `fence.proxy.async` (spec 9.7.13.15.16 point 3).
static __device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}

// ---- async-proxy fence (PTX ISA 9.7.13) -----------------------------------

// Make generic-proxy smem writes visible to the async proxy. Required before
// any bulk store that reads smem written by regular ld/st.
static __device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;");
}

}  // namespace ptx

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kKernelWidth = 4;
constexpr int kConvStateWidth = kKernelWidth - 1;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;

__device__ __forceinline__ float bf16_load(const __nv_bfloat16* ptr, int64_t idx) {
  return __bfloat162float(ptr[idx]);
}

__device__ __forceinline__ __nv_bfloat16 bf16_store(float value) {
  return __float2bfloat16(value);
}

template <bool kUseCacheGlobalStore>
__device__ __forceinline__ void store_state_float4(float* ptr, float4 value) {
  if constexpr (kUseCacheGlobalStore) {
    __stcg(reinterpret_cast<float4*>(ptr), value);
  } else {
    *reinterpret_cast<float4*>(ptr) = value;
  }
}

template <int kCopyThreads>
__device__ __forceinline__ void
cp_async_state_chunk_for(float* s_state, const float* state, int slot, int i_hv, int64_t state_slot_stride, int chunk) {
  constexpr int kFloat4PerChunk = kChunkV * kDimK / 4;
  const int tid = threadIdx.x;
  const int stage = chunk & 1;
  const int v_base = chunk * kChunkV;
  const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
  for (int linear4 = tid; linear4 < kFloat4PerChunk; linear4 += kCopyThreads) {
    const int elem = linear4 * 4;
    const int row = elem / kDimK;
    const int k = elem - row * kDimK;
    float* dst = s_state + (stage * kChunkV + row) * kDimK + k;
    const float* src = state + slot_base + ((i_hv * kDimV + v_base + row) * kDimK + k);
    ptx::cp_async_cg_16b(dst, src);
  }
  ptx::cp_async_commit_group();
}

__device__ __forceinline__ void
cp_async_state_chunk(float* s_state, const float* state, int slot, int i_hv, int64_t state_slot_stride, int chunk) {
  cp_async_state_chunk_for<kThreads>(s_state, state, slot, i_hv, state_slot_stride, chunk);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define KDA_FUSED_DECODE_HAS_TMA 1
#else
#define KDA_FUSED_DECODE_HAS_TMA 0
#endif

// 1D TMA bulk copy gmem -> smem; the issuing thread arrives with expect-tx on
// the mbarrier, completion is observed via ptx::mbar_wait_parity.
template <int kStageChunkV>
__device__ __forceinline__ void tma_state_chunk_stage(
    float* s_state,
    const float* state,
    int slot,
    int i_hv,
    int64_t state_slot_stride,
    int chunk,
    int stage,
    uint64_t* bar) {
  constexpr uint32_t kBytes = kStageChunkV * kDimK * sizeof(float);
  float* dst = s_state + stage * kStageChunkV * kDimK;
  // 16B-aligned for any slot iff state_slot_stride*sizeof(float) % 16 == 0
  // (host gates TMA off otherwise); the intra-slot chunk offset is a multiple
  // of kStageChunkV*kDimK*4, always 16B-aligned.
  const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
  const float* src = state + slot_base + ((i_hv * kDimV + chunk * kStageChunkV) * kDimK);
#if KDA_FUSED_DECODE_HAS_TMA
  ptx::fence_async_smem();
  ptx::mbar_arrive_expect_tx(bar, kBytes);
  ptx::cp_async_bulk_1d_load(dst, src, kBytes, bar);
#else
  __trap();
#endif
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float warp_total = device::warp::reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kWarps ? scratch[lane] : 0.0f;
    block_total = device::warp::reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
  return {device::warp::reduce_sum(x), device::warp::reduce_sum(y)};
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_for(float x, float y, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const float warp_x = device::warp::reduce_sum(x);
  const float warp_y = device::warp::reduce_sum(y);
  if (lane == 0) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = device::warp::reduce_sum(block_x);
    block_y = device::warp::reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

__device__ __forceinline__ Sum2 block_reduce_sum2(float x, float y, float* scratch) {
  return block_reduce_sum2_for<kWarps>(x, y, scratch);
}

template <int kReduceWarps>
__device__ __forceinline__ float block_reduce_sum_active_for(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_total = 0.0f;
  if (warp < kReduceWarps) {
    warp_total = device::warp::reduce_sum(value);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_total = device::warp::reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_active_for(float x, float y, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_x = 0.0f;
  float warp_y = 0.0f;
  if (warp < kReduceWarps) {
    warp_x = device::warp::reduce_sum(x);
    warp_y = device::warp::reduce_sum(y);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = device::warp::reduce_sum(block_x);
    block_y = device::warp::reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

__device__ __forceinline__ float fp32_at(const float* ptr, int64_t idx) {
  return ptr[idx];
}

// One-token KDA (Kimi Delta Attention) decode step, fused end to end:
//  1. Causal conv1d update: q/k/v = SiLU(bias + w_{q,k,v}_t (dot) [conv_state, x_{q,k,v}]),
//     shift-registers cs_q/cs_k/cs_v advanced in place (x_q/x_k/x_v are the raw
//     per-token projections, w_*_t the depthwise conv taps).
//  2. Per-head decay gate from a_log/g/dt_bias:
//       decay = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)))   if kUseLowerBound
//             = exp(-exp(a_log) * softplus(g + dt_bias))                otherwise
//  3. q, k are L2-normalized over kDimK (q additionally scaled by `scale`); beta
//     is sigmoid(raw) when kApplyBetaSigmoid, else used as-is.
//  4. Delta-rule recurrent state update per value row v of the [kDimV, kDimK] state h:
//       h_decay = h * decay
//       v_new   = (v - h_decay . k) * beta
//       h'      = h_decay + k (outer) v_new        (written back to `state`)
//       o       = h' . q
//  5. Gated RMSNorm (onorm) over o: out = o * rsqrt(mean(o^2) + onorm_eps)
//     * onorm_weight * sigmoid(onorm_g).
// Grid maps one block per (batch/token, value-head) under the static decode
// layout (kUseHeadGrid; H == HV — KDA is MHA, i_h = i_hv); cu_seqlens/
// ssm_state_indices otherwise resolve the token's batch slot and
// recurrent-state slot. H/HV are RUNTIME params in this port (the donor
// pinned them as template constants {12, 6, 3}).
template <
    bool kApplyOnorm,
    bool kUseStaticDecodeLayout = false,
    bool kUseHeadGrid = false,
    bool kAccumulateOnormSumsq = false,
    bool kUseActiveQkReduction = false,
    bool kUseCacheGlobalStore = false,
    bool kComputeOutputBeforeStore = false,
    bool kSkipWarpSync = false,
    bool kPreloadOnormParams = false,
    bool kPrefetchNextStateChunk = false,
    bool kUseActiveOnormReduction = false,
    bool kUpdateConvState = false,
    bool kUseLowerBound = false,
    bool kApplyBetaSigmoid = true,
    bool kUseTmaLoad = false,
    int kTmaStages = kNumChunks,
    bool kUsePDL = false>
// Shapes use the K3/GLM-5.3 per-rank KDA decode regime: head_dim = 128 ->
// kDimK = kDimV = 128; short_conv kernel width 4 -> kConvStateWidth = 3;
// B is the live decode batch size; slots is the pool capacity, addressed by
// ssm_state_indices, not B.
__global__ __launch_bounds__(kThreads, 2) void kda_decode_fusion_many_heads_kernel(
    const __nv_bfloat16* __restrict__ x_q,  // [B, H*128] row bos*x_row_stride + hk, sliced from mixed_qkv q-segment
    const __nv_bfloat16* __restrict__ x_k,  // [B, H*128] row bos*x_row_stride + hk, sliced from mixed_qkv k-segment
    const __nv_bfloat16* __restrict__ x_v,  // [B, HV*128] row bos*x_row_stride + hvv, sliced from mixed_qkv v-segment
    const float* __restrict__ w_q_t,        // [kKernelWidth=4, H*128] dense conv taps for q, indexed w*hkv_dim + hk
    const float* __restrict__ w_k_t,        // [4, H*128] dense conv taps for k
    const float* __restrict__ w_v_t,        // [4, HV*128] dense conv taps for v
    const float* __restrict__ bias_q,       // [H*128] conv bias for q, sliced from conv_bias
    const float* __restrict__ bias_k,       // [H*128] conv bias for k
    const float* __restrict__ bias_v,       // [HV*128] conv bias for v
    __nv_bfloat16* __restrict__ cs_q,     // [slots, kConvStateWidth=3, H*128] q shift-register, sliced from conv_states
    __nv_bfloat16* __restrict__ cs_k,     // [slots, 3, H*128] k shift-register
    __nv_bfloat16* __restrict__ cs_v,     // [slots, 3, HV*128] v shift-register
    const float* __restrict__ a_log,      // [H] per-head log-decay base, indexed by i_h
    const __nv_bfloat16* __restrict__ g,  // [B, HV*128] raw forget gate, row bos*g_row_stride + i_hv*kDimK + k
    const float* __restrict__ dt_bias,    // [H*128] gate bias added to g before the decay nonlinearity
    const __nv_bfloat16* __restrict__ beta,     // [B, HV] raw beta logit, row bos*beta_row_stride + i_hv
    const __nv_bfloat16* __restrict__ onorm_g,  // [B, HV*128] onorm sigmoid gate, row i_n*onormg_row_stride + i_hv*128
                                                // + v
    const float* __restrict__ onorm_weight,     // [128] onorm RMSNorm scale, shared across all heads
    const int* __restrict__ ssm_state_indices,  // [B] recurrent-state slot per token; <0 marks a padded cuda-graph slot
    const int* __restrict__ cu_seqlens,         // [B+1] token offsets into x_q/x_k/x_v/g/beta; unused under
                                                // kUseStaticDecodeLayout
    float* __restrict__ state,  // [slots, HV, 128, 128] recurrent KDA state h, inner [V,K] contiguous, slot pitch =
                                // state_slot_stride
    __nv_bfloat16* __restrict__ out,  // [B, hv_count*128] fused-decode output, row i_n, col i_hv*128 + v
    int B,                            // live decode batch size (token count for this launch)
    int H,                            // local key/query heads on this TP rank
    int HV,                           // local value heads on this TP rank (H == HV: KDA is MHA)
    float lower_bound,                // linear_attn_config.gate_lower_bound (-5.0 for K3) when kUseLowerBound
    float scale,                      // query scale applied after L2-normalization
    float onorm_eps,                  // onorm RMSNorm epsilon
    int64_t x_row_stride,             // element stride between consecutive tokens' rows in x_q/x_k/x_v (>= 3*H*128)
    int64_t g_row_stride,             // element stride between consecutive tokens' rows in g (>= HV*128)
    int64_t beta_row_stride,          // element stride between consecutive tokens' rows in beta (>= HV)
    int64_t onormg_row_stride,        // element stride between consecutive tokens' rows in onorm_g (>= HV*128)
    int64_t cs_slot_stride,           // element stride between consecutive slots in cs_q/cs_k/cs_v
    int64_t cs_w_stride,              // element stride between shift-register taps (w=0..2) within a slot
    int64_t state_slot_stride) {  // element stride between consecutive slots in state (dense HV*128*128, or larger for
                                  // shared pools)
  device::PDLWaitPrimary<kUsePDL>();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  int i_n;
  int i_hv;
  int i_h;
  int bos;
  int slot;
  if constexpr (kUseStaticDecodeLayout) {
    if constexpr (kUseHeadGrid) {
      i_n = blockIdx.x;
      i_hv = blockIdx.y;
    } else {
      const int nhv = blockIdx.x;
      i_n = nhv / HV;
      i_hv = nhv - i_n * HV;
    }
    i_h = i_hv;
    bos = i_n;
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  } else {
    const int nhv = blockIdx.x;
    i_n = nhv / HV;
    i_hv = nhv - i_n * HV;
    const int hv_per_h = HV / H;
    i_h = i_hv / hv_per_h;

    bos = cu_seqlens == nullptr ? i_n : cu_seqlens[i_n];
    const int eos = cu_seqlens == nullptr ? i_n + 1 : cu_seqlens[i_n + 1];
    if (eos <= bos) {
      device::PDLTriggerSecondary<kUsePDL>();
      return;
    }
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  }

  if (slot < 0) {
    // Padded cuda-graph slot: zero the output row, leave the pools untouched.
    const int hv_count_pad = HV;
    if (tid < kDimV) {
      out[(i_n * hv_count_pad + i_hv) * kDimV + tid] = __float2bfloat16(0.0f);
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const int hk_off = i_h * kDimK;
  const int hv_off = i_hv * kDimV;
  const int h_count = H;
  const int hv_count = HV;
  const int hkv_dim = h_count * kDimK;
  const int hvv_dim = hv_count * kDimV;

  // Dynamic size: 2 cp.async stages (32KB) or kTmaStages TMA stages (16KB per
  // stage; kTmaStages == kNumChunks means no buffer reuse).
  extern __shared__ __align__(16) float s_state[];
  __shared__ float s_q[kDimK];
  __shared__ float s_k[kDimK];
  __shared__ float s_decay[kDimK];
  __shared__ float s_v[kDimV];
  __shared__ float s_o[kDimV];
  __shared__ float s_reduce[kThreads];
  __shared__ float s_beta;
  __shared__ uint64_t s_tma_bar[kNumChunks];
  float pre_onorm_gate = 0.0f;
  float pre_onorm_weight = 0.0f;

  if constexpr (kUseTmaLoad) {
    // Chunk c lives in stage c % kTmaStages behind barrier c % kTmaStages
    // (wait parity (c / kTmaStages) & 1). Chunks 0/1 are issued here; chunk+2
    // is issued at the loop top while its stage is still fresh, and chunks
    // that reuse a stage are issued at the loop bottom behind a
    // __syncthreads(). Barrier-init visibility for waiting threads is
    // covered by the __syncthreads() before the chunk loop.
    if (tid == 0) {
#pragma unroll
      for (int c = 0; c < kTmaStages; ++c) {
        ptx::mbar_init(&s_tma_bar[c], 1);
      }
      tma_state_chunk_stage<kChunkV>(s_state, state, slot, i_hv, state_slot_stride, 0, 0, &s_tma_bar[0]);
      if (kTmaStages > 1 && kNumChunks > 1) {
        tma_state_chunk_stage<kChunkV>(s_state, state, slot, i_hv, state_slot_stride, 1, 1, &s_tma_bar[1]);
      }
    }
  } else {
    cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, 0);
  }

  if constexpr (kUpdateConvState) {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const int64_t cs_base = slot * cs_slot_stride + hk;
      const int64_t xq_idx = bos * x_row_stride + hk;
      const float exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q[hk];
      float k_acc = bias_k[hk];
      __nv_bfloat16 q_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 q_shift1 = __float2bfloat16(0.0f);
      __nv_bfloat16 k_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 k_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __nv_bfloat16 q_state = cs_q[cs_base + w * cs_w_stride];
        const __nv_bfloat16 k_state = cs_k[cs_base + w * cs_w_stride];
        q_acc += __bfloat162float(q_state) * fp32_at(w_q_t, w * hkv_dim + hk);
        k_acc += __bfloat162float(k_state) * fp32_at(w_k_t, w * hkv_dim + hk);
        if (w == 1) {
          q_shift0 = q_state;
          k_shift0 = k_state;
        } else if (w == 2) {
          q_shift1 = q_state;
          k_shift1 = k_state;
        }
      }
      const __nv_bfloat16 q_new = x_q[xq_idx];
      const __nv_bfloat16 k_new = x_k[xq_idx];
      q_acc += __bfloat162float(q_new) * fp32_at(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
      k_acc += __bfloat162float(k_new) * fp32_at(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

      cs_q[cs_base + 0] = q_shift0;
      cs_q[cs_base + cs_w_stride] = q_shift1;
      cs_q[cs_base + 2 * cs_w_stride] = q_new;
      cs_k[cs_base + 0] = k_shift0;
      cs_k[cs_base + cs_w_stride] = k_shift1;
      cs_k[cs_base + 2 * cs_w_stride] = k_new;

      s_q[k] = device::math::silu_fast(q_acc);
      s_k[k] = device::math::silu_fast(k_acc);

      const float g_raw = bf16_load(g, bos * g_row_stride + i_hv * kDimK + k) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] = __expf(lower_bound * device::math::sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * device::math::softplus_fast(g_raw));
      }
    }
  } else {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const float exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q[hk];
      float k_acc = bias_k[hk];
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int64_t cs_idx = slot * cs_slot_stride + hk + w * cs_w_stride;
        q_acc += bf16_load(cs_q, cs_idx) * fp32_at(w_q_t, w * hkv_dim + hk);
        k_acc += bf16_load(cs_k, cs_idx) * fp32_at(w_k_t, w * hkv_dim + hk);
      }
      q_acc += bf16_load(x_q, bos * x_row_stride + hk) * fp32_at(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
      k_acc += bf16_load(x_k, bos * x_row_stride + hk) * fp32_at(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

      s_q[k] = device::math::silu_fast(q_acc);
      s_k[k] = device::math::silu_fast(k_acc);

      const float g_raw = bf16_load(g, bos * g_row_stride + i_hv * kDimK + k) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] = __expf(lower_bound * device::math::sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * device::math::softplus_fast(g_raw));
      }
    }
  }

  if constexpr (kUpdateConvState) {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;
      const int64_t cs_base = slot * cs_slot_stride + hvv;
      const int64_t xv_idx = bos * x_row_stride + hvv;

      float v_acc = bias_v[hvv];
      __nv_bfloat16 v_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 v_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __nv_bfloat16 v_state = cs_v[cs_base + w * cs_w_stride];
        v_acc += __bfloat162float(v_state) * fp32_at(w_v_t, w * hvv_dim + hvv);
        if (w == 1) {
          v_shift0 = v_state;
        } else if (w == 2) {
          v_shift1 = v_state;
        }
      }
      const __nv_bfloat16 v_new = x_v[xv_idx];
      v_acc += __bfloat162float(v_new) * fp32_at(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
      cs_v[cs_base + 0] = v_shift0;
      cs_v[cs_base + cs_w_stride] = v_shift1;
      cs_v[cs_base + 2 * cs_w_stride] = v_new;
      s_v[v] = device::math::silu_fast(v_acc);

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t onorm_idx = i_n * onormg_row_stride + i_hv * kDimV + v;
        pre_onorm_gate = device::math::sigmoid_fast(bf16_load(onorm_g, onorm_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  } else {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;

      float v_acc = bias_v[hvv];
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int64_t cs_idx = slot * cs_slot_stride + hvv + w * cs_w_stride;
        v_acc += bf16_load(cs_v, cs_idx) * fp32_at(w_v_t, w * hvv_dim + hvv);
      }
      v_acc += bf16_load(x_v, bos * x_row_stride + hvv) * fp32_at(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
      s_v[v] = device::math::silu_fast(v_acc);

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t onorm_idx = i_n * onormg_row_stride + i_hv * kDimV + v;
        pre_onorm_gate = device::math::sigmoid_fast(bf16_load(onorm_g, onorm_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  }

  if (tid == 0) {
    const float beta_raw = bf16_load(beta, bos * beta_row_stride + i_hv);
    if constexpr (kApplyBetaSigmoid) {
      s_beta = device::math::sigmoid_fast(beta_raw);
    } else {
      s_beta = beta_raw;
    }
  }
  __syncthreads();

  if constexpr (!kUseTmaLoad && kPrefetchNextStateChunk && kNumChunks > 1) {
    cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, 1);
  }

  const float q_sq = tid < kDimK ? s_q[tid] * s_q[tid] : 0.0f;
  const float k_sq = tid < kDimK ? s_k[tid] * s_k[tid] : 0.0f;
  Sum2 qk_sum;
  if constexpr (kUseActiveQkReduction) {
    qk_sum = block_reduce_sum2_active_for<kDimK / 32>(q_sq, k_sq, s_reduce);
  } else {
    qk_sum = block_reduce_sum2(q_sq, k_sq, s_reduce);
  }
  if (tid < kDimK) {
    s_q[tid] *= rsqrtf(qk_sum.x + 1.0e-6f) * scale;
    s_k[tid] *= rsqrtf(qk_sum.y + 1.0e-6f);
  }
  __syncthreads();

  const int k_base = lane * 4;
  const float4 q4 = *reinterpret_cast<const float4*>(s_q + k_base);
  const float4 k4 = *reinterpret_cast<const float4*>(s_k + k_base);
  const float4 decay4 = *reinterpret_cast<const float4*>(s_decay + k_base);
  float r_q[4] = {q4.x, q4.y, q4.z, q4.w};
  float r_k[4] = {k4.x, k4.y, k4.z, k4.w};
  float r_decay[4] = {decay4.x, decay4.y, decay4.z, decay4.w};
  float o_sumsq = 0.0f;

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    if constexpr (kUseTmaLoad) {
      if (tid == 0 && chunk + 2 < kNumChunks && chunk + 2 < kTmaStages) {
        tma_state_chunk_stage<kChunkV>(
            s_state, state, slot, i_hv, state_slot_stride, chunk + 2, chunk + 2, &s_tma_bar[chunk + 2]);
      }
      ptx::mbar_wait_parity(&s_tma_bar[chunk % kTmaStages], (chunk / kTmaStages) & 1);
    } else if constexpr (kPrefetchNextStateChunk && kNumChunks > 1) {
      if (chunk + 1 < kNumChunks) {
        ptx::cp_async_wait_group<1>();
      } else {
        ptx::cp_async_wait_all();
      }
    } else {
      ptx::cp_async_wait_all();
    }
    if constexpr (!kUseTmaLoad && !kSkipWarpSync) {
      __syncwarp();
    }

    if constexpr (!kUseTmaLoad && !kPrefetchNextStateChunk) {
      if (chunk + 1 < kNumChunks) {
        cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, chunk + 1);
      }
    }

    const float* state_stage = s_state + (kUseTmaLoad ? (chunk % kTmaStages) : (chunk & 1)) * kChunkV * kDimK;

#pragma unroll
    for (int row = 0; row < kRowsPerWarp; row += 2) {
      const int v_row_a = warp + row * kWarps;
      const int v_row_b = warp + (row + 1) * kWarps;
      const int v0 = chunk * kChunkV + v_row_a;
      const int v1 = chunk * kChunkV + v_row_b;
      float h_a_vals[4];
      float h_b_vals[4];
      float dot_hk_a = 0.0f;
      float dot_hk_b = 0.0f;

      const float4 raw_h_a = *reinterpret_cast<const float4*>(state_stage + v_row_a * kDimK + k_base);
      const float4 raw_h_b = *reinterpret_cast<const float4*>(state_stage + v_row_b * kDimK + k_base);
      h_a_vals[0] = raw_h_a.x * r_decay[0];
      h_a_vals[1] = raw_h_a.y * r_decay[1];
      h_a_vals[2] = raw_h_a.z * r_decay[2];
      h_a_vals[3] = raw_h_a.w * r_decay[3];
      h_b_vals[0] = raw_h_b.x * r_decay[0];
      h_b_vals[1] = raw_h_b.y * r_decay[1];
      h_b_vals[2] = raw_h_b.z * r_decay[2];
      h_b_vals[3] = raw_h_b.w * r_decay[3];
      dot_hk_a = h_a_vals[0] * r_k[0] + h_a_vals[1] * r_k[1] + h_a_vals[2] * r_k[2] + h_a_vals[3] * r_k[3];
      dot_hk_b = h_b_vals[0] * r_k[0] + h_b_vals[1] * r_k[1] + h_b_vals[2] * r_k[2] + h_b_vals[3] * r_k[3];

      const Sum2 dot_hk = warp_reduce_sum_pair(dot_hk_a, dot_hk_b);
      const float v_new0 = (s_v[v0] - dot_hk.x) * s_beta;
      const float v_new1 = (s_v[v1] - dot_hk.y) * s_beta;

      float dot_hq_a = 0.0f;
      float dot_hq_b = 0.0f;
      // Writeback mirrors the load addressing: slot pitch from the host stride
      // (int64, envelope-safe), intra-slot offset i_hv*V*K + v*K + k contiguous.
      const int64_t slot_base_wb = static_cast<int64_t>(slot) * state_slot_stride;
      const int64_t state_idx_a = slot_base_wb + ((i_hv * kDimV + v0) * kDimK + k_base);
      const int64_t state_idx_b = slot_base_wb + ((i_hv * kDimV + v1) * kDimK + k_base);
      const float h_a_0 = h_a_vals[0] + r_k[0] * v_new0;
      const float h_a_1 = h_a_vals[1] + r_k[1] * v_new0;
      const float h_a_2 = h_a_vals[2] + r_k[2] * v_new0;
      const float h_a_3 = h_a_vals[3] + r_k[3] * v_new0;
      const float h_b_0 = h_b_vals[0] + r_k[0] * v_new1;
      const float h_b_1 = h_b_vals[1] + r_k[1] * v_new1;
      const float h_b_2 = h_b_vals[2] + r_k[2] * v_new1;
      const float h_b_3 = h_b_vals[3] + r_k[3] * v_new1;
      if constexpr (kComputeOutputBeforeStore) {
        dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
      } else {
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
        dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
      }

      const Sum2 dot_hq = warp_reduce_sum_pair(dot_hq_a, dot_hq_b);
      if (lane == 0) {
        s_o[v0] = dot_hq.x;
        s_o[v1] = dot_hq.y;
        if constexpr (kApplyOnorm && kAccumulateOnormSumsq) {
          o_sumsq += dot_hq.x * dot_hq.x + dot_hq.y * dot_hq.y;
        }
      }
    }

    if constexpr (kUseTmaLoad && kTmaStages < kNumChunks) {
      // chunk + kTmaStages reuses the stage this chunk just read; every warp
      // must be done with it before the single issuing thread overwrites it.
      if (chunk + kTmaStages < kNumChunks) {
        __syncthreads();
        if (tid == 0) {
          tma_state_chunk_stage<kChunkV>(
              s_state,
              state,
              slot,
              i_hv,
              state_slot_stride,
              chunk + kTmaStages,
              chunk % kTmaStages,
              &s_tma_bar[chunk % kTmaStages]);
        }
      }
    } else if constexpr (!kUseTmaLoad && kPrefetchNextStateChunk) {
      if (chunk + 2 < kNumChunks) {
        cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, chunk + 2);
      }
    }
  }
  __syncthreads();

  device::PDLTriggerSecondary<kUsePDL>();

  if constexpr (kApplyOnorm) {
    if constexpr (kAccumulateOnormSumsq) {
      if (lane == 0) {
        s_reduce[warp] = o_sumsq;
      }
      __syncthreads();

      float total_sumsq = 0.0f;
      if (warp == 0) {
        total_sumsq = lane < kWarps ? s_reduce[lane] : 0.0f;
        total_sumsq = device::warp::reduce_sum(total_sumsq);
        if (lane == 0) {
          s_reduce[0] = total_sumsq;
        }
      }
      __syncthreads();

      if (tid < kDimV) {
        const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
        const float raw_o = s_o[tid];
        const float rstd = rsqrtf(s_reduce[0] / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          gate = device::math::sigmoid_fast(bf16_load(onorm_g, i_n * onormg_row_stride + i_hv * kDimV + tid));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    } else {
      const float raw_o = tid < kDimV ? s_o[tid] : 0.0f;
      const float o_sq = raw_o * raw_o;
      float sumsq;
      if constexpr (kUseActiveOnormReduction || kUseActiveQkReduction) {
        sumsq = block_reduce_sum_active_for<kDimV / 32>(o_sq, s_reduce);
      } else {
        sumsq = block_reduce_sum(o_sq, s_reduce);
      }

      if (tid < kDimV) {
        const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
        const float rstd = rsqrtf(sumsq / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          gate = device::math::sigmoid_fast(bf16_load(onorm_g, i_n * onormg_row_stride + i_hv * kDimV + tid));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    }
  } else {
    if (tid < kDimV) {
      const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
      out[out_idx] = bf16_store(s_o[tid]);
    }
  }
}

// K3/GLM-5.3 decode configuration of the many-heads kernel: onorm fused,
// static (B, HV) head grid, onorm params preloaded, next state chunk
// prefetched, active onorm reduction, conv cache updated in place, beta
// sigmoid in-kernel. Both forget-gate variants are compiled (softplus and
// lower-bounded sigmoid) and selected at launch from the model config.
// kUseTmaLoad/kTmaStages select the 1D-TMA state-staging path in place of
// the cp.async fallback. Head counts are RUNTIME params in this port (the
// donor pinned H = HV in {3, 6, 12} as template constants).
template <bool kUseLowerBound, bool kUsePDL, bool kUseTmaLoad = false, int kTmaStages = kNumChunks>
constexpr auto kda_fused_decode_k3_kernel = kda_decode_fusion_many_heads_kernel<
    /*kApplyOnorm=*/true,
    /*kUseStaticDecodeLayout=*/true,
    /*kUseHeadGrid=*/true,
    /*kAccumulateOnormSumsq=*/false,
    /*kUseActiveQkReduction=*/false,
    /*kUseCacheGlobalStore=*/false,
    /*kComputeOutputBeforeStore=*/false,
    /*kSkipWarpSync=*/false,
    /*kPreloadOnormParams=*/true,
    /*kPrefetchNextStateChunk=*/true,
    /*kUseActiveOnormReduction=*/true,
    /*kUpdateConvState=*/true,
    kUseLowerBound,
    /*kApplyBetaSigmoid=*/true,
    kUseTmaLoad,
    kTmaStages,
    kUsePDL>;

using fused_kernel_t = void (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*,
    const float*, const float*, const float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const int*,
    const int*, float*, __nv_bfloat16*, int, int, int, float, float, float,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

template <bool kUsePDL>
static fused_kernel_t select_kda_fused_decode_kernel(bool use_lower_bound, int tma_stages) {
  if (tma_stages == 3) {
    return use_lower_bound ? kda_fused_decode_k3_kernel<true, kUsePDL, true, 3>
                           : kda_fused_decode_k3_kernel<false, kUsePDL, true, 3>;
  }
  if (tma_stages == 4) {
    return use_lower_bound ? kda_fused_decode_k3_kernel<true, kUsePDL, true, 4>
                           : kda_fused_decode_k3_kernel<false, kUsePDL, true, 4>;
  }
  return use_lower_bound ? kda_fused_decode_k3_kernel<true, kUsePDL>
                         : kda_fused_decode_k3_kernel<false, kUsePDL>;
}

}  // namespace vkda

extern "C" int vkda_fused_decode_run(
    const void* mixed_qkv, const void* a, const void* b, void* conv_states,
    const void* w_q_t, const void* w_k_t, const void* w_v_t, const void* conv_bias,
    const void* A_log, const void* dt_bias, const void* onorm_g, const void* onorm_weight,
    void* state, const void* indices, void* out,
    unsigned B, unsigned H, unsigned HV,
    double scale, double onorm_eps, double lower_bound, int use_lower_bound,
    long long x_row_stride, long long g_row_stride, long long beta_row_stride,
    long long onormg_row_stride, long long cs_slot_stride, long long cs_w_stride,
    long long state_slot_stride,
    long long stream, int use_pdl) {
  if (B == 0) {
    return 0;  // nothing live: no launch, pools untouched (donor early-out)
  }
  using bf16_t = __nv_bfloat16;
  const int64_t kSeg = static_cast<int64_t>(H) * vkda::kDimK;  // q, k and v segment width (H == HV)
  const bf16_t* mixed_ptr = static_cast<const bf16_t*>(mixed_qkv);
  bf16_t* cs_ptr = static_cast<bf16_t*>(conv_states);
  const float* bias_ptr = static_cast<const float*>(conv_bias);
  // Real per-slot pitch of the ssm/temporal pool (elements): dense HV*V*K
  // for a local pool, the multi-layer envelope for unified / page-major
  // pools. Threaded into every ssm-state read/write; int64 avoids the
  // envelope-pitch overflow.
  int tma_stages = 0;
  // TMA 1D-bulk needs the per-slot source address (state + slot*stride) 16B
  // aligned for every slot. state.data_ptr() is torch-aligned and each chunk
  // offset is a multiple of kChunkV*kDimK*4 B, so alignment holds iff the slot
  // pitch itself is 16B-aligned, i.e. state_slot_stride % 4 == 0 (fp32). The
  // K3 envelope pitch and the dense pitch both satisfy this; a pathological
  // stride falls back to cp.async (still fully fused, just no TMA) rather
  // than silently mis-addressing the descriptor. (The Python eligibility
  // gate additionally rejects pitches the cp.async fallback could not load
  // 16B-aligned either.)
  const bool tma_slot_stride_aligned = (state_slot_stride % 4) == 0;
  if (tma_slot_stride_aligned) {
    // Full-state staging (4 stages, 64KB, sync-free) wins while the grid
    // is small enough that occupancy isn't the limiter; past that the
    // 48KB 3-stage variant (one sync for the single stage reuse) benches
    // fastest.
    tma_stages = static_cast<int>(B) * static_cast<int>(HV) >= 512 ? 3 : 4;
  }
  const vkda::fused_kernel_t kernel =
      use_pdl ? vkda::select_kda_fused_decode_kernel<true>(use_lower_bound != 0, tma_stages)
              : vkda::select_kda_fused_decode_kernel<false>(use_lower_bound != 0, tma_stages);
  const int smem_stages = tma_stages == 0 ? 2 : tma_stages;
  const size_t smem_bytes = static_cast<size_t>(smem_stages) * vkda::kChunkV * vkda::kDimK * sizeof(float);
  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  if (err != cudaSuccess) {
    return (int)err;
  }
  const dim3 grid(B, HV);
  const dim3 block(vkda::kThreads);
  const cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
  if (use_pdl) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = cs;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    return (int)cudaLaunchKernelEx(
        &cfg,
        kernel,
        /*x_q=*/mixed_ptr,
        /*x_k=*/mixed_ptr + kSeg,
        /*x_v=*/mixed_ptr + 2 * kSeg,
        static_cast<const float*>(w_q_t),
        static_cast<const float*>(w_k_t),
        static_cast<const float*>(w_v_t),
        /*bias_q=*/bias_ptr,
        /*bias_k=*/bias_ptr + kSeg,
        /*bias_v=*/bias_ptr + 2 * kSeg,
        /*cs_q=*/cs_ptr,
        /*cs_k=*/cs_ptr + kSeg,
        /*cs_v=*/cs_ptr + 2 * kSeg,
        static_cast<const float*>(A_log),
        /*g=*/static_cast<const bf16_t*>(a),
        static_cast<const float*>(dt_bias),
        /*beta=*/static_cast<const bf16_t*>(b),
        static_cast<const bf16_t*>(onorm_g),
        static_cast<const float*>(onorm_weight),
        static_cast<const int*>(indices),
        /*cu_seqlens=*/static_cast<const int*>(nullptr),
        static_cast<float*>(state),
        static_cast<bf16_t*>(out),
        static_cast<int>(B),
        /*H=*/static_cast<int>(H),
        /*HV=*/static_cast<int>(HV),
        static_cast<float>(lower_bound),
        static_cast<float>(scale),
        static_cast<float>(onorm_eps),
        x_row_stride,
        g_row_stride,
        beta_row_stride,
        onormg_row_stride,
        cs_slot_stride,
        cs_w_stride,
        state_slot_stride);
  } else {
    kernel<<<grid, block, smem_bytes, cs>>>(
        /*x_q=*/mixed_ptr,
        /*x_k=*/mixed_ptr + kSeg,
        /*x_v=*/mixed_ptr + 2 * kSeg,
        static_cast<const float*>(w_q_t),
        static_cast<const float*>(w_k_t),
        static_cast<const float*>(w_v_t),
        /*bias_q=*/bias_ptr,
        /*bias_k=*/bias_ptr + kSeg,
        /*bias_v=*/bias_ptr + 2 * kSeg,
        /*cs_q=*/cs_ptr,
        /*cs_k=*/cs_ptr + kSeg,
        /*cs_v=*/cs_ptr + 2 * kSeg,
        static_cast<const float*>(A_log),
        /*g=*/static_cast<const bf16_t*>(a),
        static_cast<const float*>(dt_bias),
        /*beta=*/static_cast<const bf16_t*>(b),
        static_cast<const bf16_t*>(onorm_g),
        static_cast<const float*>(onorm_weight),
        static_cast<const int*>(indices),
        /*cu_seqlens=*/static_cast<const int*>(nullptr),
        static_cast<float*>(state),
        static_cast<bf16_t*>(out),
        static_cast<int>(B),
        /*H=*/static_cast<int>(H),
        /*HV=*/static_cast<int>(HV),
        static_cast<float>(lower_bound),
        static_cast<float>(scale),
        static_cast<float>(onorm_eps),
        x_row_stride,
        g_row_stride,
        beta_row_stride,
        onormg_row_stride,
        cs_slot_stride,
        cs_w_stride,
        state_slot_stride);
    return (int)cudaGetLastError();
  }
}
"""

_NVCC_FLAGS = ["-O3", "--use_fast_math"]  # donor extra_cuda_cflags, kept for numeric parity


def _find_nvcc() -> str | None:
    import shutil

    exe = shutil.which("nvcc")
    if exe:
        return exe
    import glob

    for pat in ("/usr/local/cuda*/bin/nvcc", os.path.expanduser("~/.local/cuda*/bin/nvcc")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _arch_flag(device) -> str:
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _compile_or_load(device) -> ctypes.CDLL:
    """Compile (once, cached) and dlopen the fused-decode .so."""
    nvcc = _find_nvcc()
    if nvcc is None:
        raise OpNotEligible("nvcc not found (need CUDA toolkit to JIT glm_kda_fused_decode)")
    arch = _arch_flag(device)
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    tag = hashlib.sha256(
        (_CUDA_SOURCE + ver + arch + " ".join(_NVCC_FLAGS)).encode()
    ).hexdigest()[:16]
    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vkernels", "cuda-jit",
    )
    so = os.path.join(cache_dir, f"kda_fused_decode-{tag}.so")
    if not os.path.exists(so):
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            src = os.path.join(tmp, "kda_fused_decode.cu")
            with open(src, "w") as f:
                f.write(_CUDA_SOURCE)
            obj = os.path.join(tmp, "kda_fused_decode.so")
            cmd = [
                nvcc, *_NVCC_FLAGS, "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
                f"-arch={arch}", "-DNDEBUG", src, "-o", obj,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise OpNotEligible(
                    "nvcc failed for glm_kda_fused_decode "
                    f"({arch}): {proc.stderr[-2000:]}"
                )
            os.replace(obj, so)  # atomic publish
    lib = ctypes.CDLL(so)
    lib.vkda_fused_decode_run.restype = ctypes.c_int
    lib.vkda_fused_decode_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ctypes.c_longlong, ctypes.c_int,
    ]
    return lib


@lru_cache(maxsize=4)
def _lib_for(device_index: int):
    import torch

    return _compile_or_load(torch.device("cuda", device_index))


def _validate(
    mixed_qkv, a, b, conv_states, w_q_t, w_k_t, w_v_t, conv_bias, A_log,
    dt_bias, onorm_g, onorm_weight, state, indices, out, lower_bound,
):
    """Shared contract check; returns (B, H, HV, lower_bound|None)."""
    import torch

    if os.environ.get("VK_CUDA_JIT", "1") == "0":
        raise OpNotEligible("VK_CUDA_JIT=0 disables the CUDA-JIT KDA fused decode")
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1 or mixed_qkv.shape[1] % 3:
        raise OpNotEligible("mixed_qkv must be 2-D [B, 3*H*128] with contiguous rows")
    b_sz = mixed_qkv.shape[0]
    seg = mixed_qkv.shape[1] // 3
    if seg % _K:
        raise OpNotEligible("mixed_qkv segments must be multiples of 128 (H*128 q|k|v rows)")
    hv = seg // _K
    if state.ndim != 4 or state.dtype != torch.float32:
        raise OpNotEligible("state pool must be FP32 [slots, HV, V, K]")
    if state.shape[-3] != hv or state.shape[-2:] != (_V, _K):
        raise OpNotEligible(
            f"state must be [slots, HV={hv}, 128, 128] (K = V = 128; H == HV required, "
            f"the K3 static decode layout is MHA)"
        )
    if state.stride(-1) != 1 or state.stride(-2) != _K or state.stride(-3) != _V * _K:
        raise OpNotEligible("state inner layout must be dense V-major [HV, V, K]")
    if state.stride(0) % 4:
        # both staging paths (cp.async.cg / 1D-TMA) issue 16B loads from the
        # slot base — a slot pitch not divisible by 4 fp32 elements cannot be
        # loaded 16B-aligned (silent UB in the donor; rejected here)
        raise OpNotEligible("state slot pitch (stride(0)) must be a multiple of 4 elements")
    if a.ndim != 2 or a.shape != (b_sz, seg) or a.stride(-1) != 1:
        raise OpNotEligible("a must be [B, HV*128] with contiguous rows")
    if b.ndim != 2 or b.shape != (b_sz, hv) or b.stride(-1) != 1:
        raise OpNotEligible("b must be [B, HV] with contiguous rows")
    if onorm_g.ndim != 2 or onorm_g.shape != (b_sz, seg) or onorm_g.stride(-1) != 1:
        raise OpNotEligible("onorm_g must be [B, HV*128] with contiguous rows")
    if conv_states.ndim != 3 or conv_states.shape[-2:] != (_CONV_STATE_W, 3 * seg) or conv_states.stride(-1) != 1:
        raise OpNotEligible(
            f"conv_states must be [slots, 3, {3 * seg}] bf16, TIME-major (w, q|k|v channels)"
        )
    for name, t, shape in (
        ("w_q_t", w_q_t, (4, seg)), ("w_k_t", w_k_t, (4, seg)), ("w_v_t", w_v_t, (4, seg)),
    ):
        if t.shape != shape or t.dtype != torch.float32 or t.stride(-1) != 1 or t.stride(-2) != seg:
            raise OpNotEligible(f"{name} must be dense FP32 [4, {seg}] time-major conv taps")
    if conv_bias.numel() != 3 * seg or conv_bias.dtype != torch.float32 or conv_bias.stride(-1) != 1:
        raise OpNotEligible(f"conv_bias must be FP32 [{3 * seg}]")
    if A_log.numel() != hv or A_log.dtype != torch.float32:
        raise OpNotEligible(f"A_log must be FP32 [{hv}]")
    if dt_bias.numel() != seg or dt_bias.dtype != torch.float32 or dt_bias.stride(-1) != 1:
        raise OpNotEligible(f"dt_bias must be FP32 [{seg}]")
    if onorm_weight.numel() != _V or onorm_weight.dtype != torch.float32 or onorm_weight.stride(-1) != 1:
        raise OpNotEligible("onorm_weight must be dense FP32 [128] (shared across heads)")
    for name, t, dt in (
        ("mixed_qkv", mixed_qkv, torch.bfloat16),
        ("a", a, torch.bfloat16),
        ("b", b, torch.bfloat16),
        ("conv_states", conv_states, torch.bfloat16),
        ("onorm_g", onorm_g, torch.bfloat16),
    ):
        if t.dtype != dt:
            raise OpNotEligible(f"{name} must be {dt}, got {t.dtype}")
    if indices.dtype != torch.int32 or indices.ndim != 1 or indices.shape[0] != b_sz or not indices.is_contiguous():
        raise OpNotEligible("cache_indices must be contiguous int32 [B]")
    if out.dtype != torch.bfloat16 or out.shape != (b_sz, seg) or not out.is_contiguous():
        raise OpNotEligible(f"out must be dense BF16 [B, {seg}]")
    dev = mixed_qkv.device
    if not dev.type == "cuda":
        raise OpNotEligible("glm_kda_fused_decode requires CUDA tensors")
    for name, t in (
        ("a", a), ("b", b), ("conv_states", conv_states),
        ("w_q_t", w_q_t), ("w_k_t", w_k_t), ("w_v_t", w_v_t), ("conv_bias", conv_bias),
        ("A_log", A_log), ("dt_bias", dt_bias), ("onorm_g", onorm_g),
        ("onorm_weight", onorm_weight), ("state", state), ("indices", indices), ("out", out),
    ):
        if t.device != dev:
            raise OpNotEligible(f"{name} on {t.device}, expected {dev}")
    if lower_bound is not None:
        lower_bound = float(lower_bound)
    return b_sz, hv, hv, lower_bound


def kda_conv_weight_to_taps(conv_weight):
    """floe's ``Conv1d`` depthwise weight ``[C, 1, 4]`` -> time-major fp32 taps.

    Returns ``(w_q_t, w_k_t, w_v_t)``, each FP32 ``[4, C/3]`` dense — the
    kernel's conv-tap contract (the donor checkpoint keeps conv weights
    fp32; the bf16 -> fp32 widen is exact). Mirrors the K3 loader's
    ``w.t().contiguous()`` + per-segment slices, done once at weight load.
    """
    if conv_weight.ndim != 3 or conv_weight.shape[1] != 1 or conv_weight.shape[0] % 3 or conv_weight.shape[2] != 4:
        raise OpNotEligible("expected the floe Conv1d depthwise weight [C, 1, 4] with 3 | C")
    seg = conv_weight.shape[0] // 3
    wt = conv_weight.squeeze(1).t().float().contiguous()  # [4, C] time-major
    return wt[:, :seg].contiguous(), wt[:, seg : 2 * seg].contiguous(), wt[:, 2 * seg :].contiguous()


def kda_conv_state_cmajor_to_wmajor(conv_state):
    """floe's conv pool ``[slots, C, 3]`` -> the kernel's ``[slots, 3, C]``."""

    if conv_state.ndim != 3:
        raise OpNotEligible("expected a 3-D conv-state pool [slots, C, w]")
    return conv_state.transpose(1, 2).contiguous()


def kda_conv_state_wmajor_to_cmajor(conv_state):
    """The kernel's ``[slots, 3, C]`` conv pool -> floe's ``[slots, C, 3]``."""

    if conv_state.ndim != 3:
        raise OpNotEligible("expected a 3-D conv-state pool [slots, w, C]")
    return conv_state.transpose(1, 2).contiguous()


def glm_kda_fused_decode_eligible(
    mixed_qkv, a, b, conv_states, w_q_t, w_k_t, w_v_t, conv_bias, A_log,
    dt_bias, onorm_g, onorm_weight, state, cache_indices, out=None, lower_bound=None,
) -> bool:
    """Contract check for :func:`glm_kda_fused_decode` (no device sync)."""
    import torch

    if out is None:
        out = torch.empty(
            mixed_qkv.shape[0], state.shape[-3] * _V,
            dtype=torch.bfloat16, device=mixed_qkv.device,
        )
    try:
        _validate(mixed_qkv, a, b, conv_states, w_q_t, w_k_t, w_v_t, conv_bias,
                  A_log, dt_bias, onorm_g, onorm_weight, state, cache_indices,
                  out, lower_bound)
    except OpNotEligible:
        return False
    return True


def glm_kda_fused_decode(
    mixed_qkv, a, b, conv_states, w_q_t, w_k_t, w_v_t, conv_bias, A_log,
    dt_bias, onorm_g, onorm_weight, state, cache_indices, scale, onorm_eps,
    lower_bound=None, out=None, use_pdl=False,
):
    """One-launch fused KDA decode step: conv update (pool shifted in place)
    + delta-rule state update (pool updated in place) + gated RMSNorm.

    Consumes the RAW PRE-conv fused-projection rows (``mixed_qkv`` q|k|v),
    the RAW ``f_b``/``b_proj`` dots and the RAW o-norm gate dots; returns
    the gated-normed attention output ``[1, B, HV, V]`` bf16 (allocated when
    ``out is None``; pass a dense ``[B, HV*128]`` buffer to replay into).
    Inputs per the FLOE CONTRACT MAPPING in the module docstring;
    ``-1`` cache indices are padded graph slots (output zeroed, both pools
    untouched). Raises :class:`~vkernels.torch_ops._dispatch.OpNotEligible`
    outside the contract so the caller falls back to the unfused chain.
    """
    import torch

    if out is None:
        out = torch.empty(
            mixed_qkv.shape[0], state.shape[-3] * _V,
            dtype=torch.bfloat16, device=mixed_qkv.device,
        )
    b_sz, h, hv, lower_bound = _validate(
        mixed_qkv, a, b, conv_states, w_q_t, w_k_t, w_v_t, conv_bias, A_log,
        dt_bias, onorm_g, onorm_weight, state, cache_indices, out, lower_bound,
    )
    if b_sz == 0:
        return out.view(1, 0, hv, _V)
    lib = _lib_for(mixed_qkv.device.index or 0)
    stream = torch.cuda.current_stream(mixed_qkv.device).cuda_stream
    rc = lib.vkda_fused_decode_run(
        mixed_qkv.data_ptr(), a.data_ptr(), b.data_ptr(), conv_states.data_ptr(),
        w_q_t.data_ptr(), w_k_t.data_ptr(), w_v_t.data_ptr(), conv_bias.data_ptr(),
        A_log.data_ptr(), dt_bias.data_ptr(), onorm_g.data_ptr(), onorm_weight.data_ptr(),
        state.data_ptr(), cache_indices.data_ptr(), out.data_ptr(),
        b_sz, h, hv,
        float(scale), float(onorm_eps),
        lower_bound if lower_bound is not None else 0.0,
        int(lower_bound is not None),
        mixed_qkv.stride(0), a.stride(0), b.stride(0), onorm_g.stride(0),
        conv_states.stride(0), conv_states.stride(1), state.stride(0),
        stream, int(bool(use_pdl)),
    )
    if rc != 0:
        raise RuntimeError(f"kda_fused_decode launch failed: CUDA error {rc}")
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize(mixed_qkv.device)  # surface launch errors eagerly
    return out.view(1, b_sz, hv, _V)


def glm_kda_fused_decode_reference(
    mixed_qkv, a, b, conv_states, w_q_t, w_k_t, w_v_t, conv_bias, A_log,
    dt_bias, onorm_g, onorm_weight, state, cache_indices, scale, onorm_eps,
    lower_bound=None,
):
    """Eager fp32 torch oracle (the kernel's exact op order, CPU/GPU).

    Returns ``(out [1, B, HV, V] bf16, next_conv_states [slots, 3, C] bf16,
    next_state [slots, HV, V, K] fp32)`` — FRESH tensors (the kernel mutates
    the pools in place; the oracle copies so both can be compared after the
    fact). ``-1`` cache indices are padded slots: zero output row, pool rows
    passed through. The conv-state shift is pure bf16 moves (bit-exact, as
    in the kernel); everything else is fp32 in the kernel's op order.
    """
    import torch

    b_sz = mixed_qkv.shape[0]
    hv, v_dim, k_dim = state.shape[-3:]
    next_conv = conv_states.clone()
    next_state = state.clone()
    out = torch.zeros(b_sz, hv, v_dim, dtype=torch.bfloat16, device=mixed_qkv.device)

    mq = mixed_qkv.float()
    a_f = a.float().view(b_sz, hv, k_dim)
    b_f = b.float().view(b_sz, hv)
    a_log = A_log.float()
    dt = dt_bias.float().view(hv, k_dim)
    og = onorm_g.float().view(b_sz, hv, v_dim)
    ow = onorm_weight.float()
    w_all = torch.cat([w_q_t, w_k_t, w_v_t], dim=1).float()  # [4, 3*seg]
    bias_all = conv_bias.float()

    for n in range(b_sz):
        sidx = int(cache_indices[n].item())
        if sidx < 0:
            continue
        # --- conv update (one token): bias + w0*s0 + w1*s1 + w2*s2 + w3*x ---
        cs = next_conv[sidx]  # [3, 3*seg] bf16
        window = torch.cat([cs.float(), mq[n].unsqueeze(0)], 0)  # [4, 3*seg]
        acc = bias_all + window[0] * w_all[0] + window[1] * w_all[1] \
            + window[2] * w_all[2] + window[3] * w_all[3]
        y = acc * torch.sigmoid(acc)  # SiLU, fp32, never rounded to bf16
        # state shift: [old w1, old w2, new raw x] — pure bf16 moves
        next_conv[sidx] = torch.cat([cs[1:3], mixed_qkv[n].unsqueeze(0)], 0)

        yv = y.view(3, hv, 128)
        for j in range(hv):
            q = yv[0, j]
            k = yv[1, j]
            v = yv[2, j]
            q = q / torch.sqrt((q * q).sum() + 1e-6) * scale
            k = k / torch.sqrt((k * k).sum() + 1e-6)
            xx = a_f[n, j] + dt[j]
            exp_a = torch.exp(a_log[j])
            if lower_bound is None:
                sp = torch.where(xx <= 20.0, torch.log1p(torch.exp(torch.clamp(xx, max=20.0))), xx)
                decay = torch.exp(-exp_a * sp)
            else:
                decay = torch.exp(lower_bound / (1.0 + torch.exp(-exp_a * xx)))
            beta = torch.sigmoid(b_f[n, j])

            s = next_state[sidx, j] * decay[None, :]  # [V, K]
            t = (s * k[None, :]).sum(dim=1)
            delta = (v - t) * beta
            s = s + delta[:, None] * k[None, :]
            next_state[sidx, j] = s
            o = (s * q[None, :]).sum(dim=1)
            rstd = torch.rsqrt((o * o).mean() + onorm_eps)
            y_o = o * rstd * ow * torch.sigmoid(og[n, j])
            out[n, j] = y_o.to(torch.bfloat16)
    return out.view(1, b_sz, hv, v_dim), next_conv, next_state
