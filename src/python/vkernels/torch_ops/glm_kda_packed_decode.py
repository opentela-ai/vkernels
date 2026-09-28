"""CUDA KDA packed-decode — row-streaming state update, ~2x the Triton
state-bandwidth class (borrow9-kda-cuda, donor lane).

Vendored from SGLang v0.5.20 ``kernels/ops/attention/kda_packed_decode.py``
+ ``kernels/jit/csrc/attention/kda_packed_decode.cuh`` (Apache-2.0; the CUDA
body itself is from the NVIDIA x Moonshot Kimi-K3 optimization package).
The kernel is a row-streaming port of the triton fused-recurrent decode:
the triton kernel keeps a whole [BV, K] fp32 state tile in one warp's
registers and tops out at ~5 TB/s; this kernel streams the state one 512B
row at a time (one warp per V-row group: float4 load -> warp-reduced dot ->
decayed delta-rule update -> 512B store) and reaches the in-place
read+write bandwidth of the part (~9.6 TB/s on H100). Outputs match the
triton kernel to ULPs (warp-shuffle reduction order), not bits.

DEVIATIONS FROM THE DONOR (all mechanical, none in the math):
- no sgl_kernel/tvm-ffi dependency stack: the CUDA source is embedded,
  compiled once with the local nvcc into a cached .so under the user cache
  dir, and called through a plain ``extern "C"`` launcher (raw pointers +
  the torch current stream) via ctypes;
- the tensor contracts moved from the C++ ``TensorMatcher`` asserts to the
  Python eligibility checks below (:class:`OpNotEligible` — the vkernels
  torch_ops dispatch contract);
- the donor's ``covered()`` head-count gate ``H in {12, 6, 3}`` (K3's
  TP8/16/32 instantiations) is relaxed to any ``H`` with ``HV % H == 0`` —
  the kernel takes head counts at RUN time; GLM-5.3-Flash TP4 is
  ``H = HV = 16`` (K = V = 128, conv width 4, T = 1);
- the donor's ``B >= 8`` perf gate is a documented note, not eligibility
  (the launch cost floor it guards against only matters for production
  graphs); PDL is opt-in via ``use_pdl`` (sm_90+; no-op otherwise).

FLOE CONTRACT MAPPING (GLM-5.3-Flash KDA decode, per rank TP4):
- ``mixed_qkv`` ``[B, 2*H*K + HV*V]`` bf16, contiguous rows — the conv
  output (post conv+SiLU), laid out q | k | v exactly like floe's
  ``glm_kda_conv_update`` product (H=16: ``[B, 6144]`` = 2048|2048|2048);
- ``a`` ``[B, HV*K]`` bf16 — the RAW ``f_b(h)`` GEMV output (the kernel
  adds ``dt_bias`` and applies the gate branch itself: softplus
  ``-exp(A_log)·softplus(a+dt_bias)`` or ``lower_bound·sigmoid(exp(A_log)·
  (a+dt_bias))``); floe's equivalent chain is ``Glm53ForgetGate``;
- ``b`` ``[B, HV]`` bf16 — the RAW ``b_proj`` dots (the kernel applies the
  sigmoid; floe's model applies it at the beta site);
- ``A_log`` ``[HV]`` fp32, ``dt_bias`` ``[HV*K]`` fp32;
- ``scale`` = ``D**-0.5`` (the l2-normed q scale);
- ``state`` pool, V-MAJOR: ``[slots, HV, V, K]`` fp32 with dense inner
  layout (any slot pitch — the kernel reads ``stride(0)``); NOTE floe's
  incumbent Triton pool is K-major ``[slots, H, K, V]`` — with square
  128x128 heads the two layouts are STRIDE-IDENTICAL, so eligibility
  cannot detect a wrong-layout pool: the convention is a caller contract,
  enforced once at the wiring site. Use
  :func:`kda_state_kmajor_to_vmajor` / :func:`kda_state_vmajor_to_kmajor`
  at the pool boundary (a new V-major pool makes them free; a live
  K-major pool pays one transpose per state load/store round trip);
- ``ssm_state_indices`` ``[B]`` int32 — slot ids; ``-1`` marks a padded
  CUDA-graph slot (output zeroed, pool row untouched — tested);
- ``out`` ``[B, HV, V]`` bf16 (pass ``[B, 1, HV, V]``; the wrapper views).

The state update is IN PLACE on the selected pool rows; the output is
written to a caller-allocated (or wrapper-allocated) tensor so a decode
graph replays into it. Outputs match :func:`~vkernels.torch_ops.glm_kda_decode.glm_kda_decode`
to ULPs, not bits (warp-shuffle vs tl.sum reduction order; validated here
against a torch fp32 oracle and the Triton kernel with tolerance).

CUDA JIT: nvcc is located via PATH or ``/usr/local/cuda*/bin/nvcc``; the
compiled module is cached under ``~/.cache/vkernels/cuda-jit`` keyed by the
source hash + nvcc version + target arch. First call compiles (seconds);
``VK_CUDA_JIT=0`` disables the op (always ``OpNotEligible``). Inference
only, no autograd backward.
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
    "glm_kda_packed_decode",
    "glm_kda_packed_decode_eligible",
    "glm_kda_packed_decode_reference",
    "kda_state_kmajor_to_vmajor",
    "kda_state_vmajor_to_kmajor",
]

_K, _V = 128, 128
_WARPS = 8  # donor constant: one warp per V-row group (V/warps = 16 rows)

# The CUDA source. Kernel body verbatim from the donor .cuh; the harness
# around it (types, PDL, launcher) is rewritten to be dependency-free.
_CUDA_SOURCE = r"""
// CUDA port of the triton fused_recurrent_kda_packed_decode_kernel for
// batched decode. The triton kernel holds a whole [BV, K] fp32 state tile in
// the registers of a single warp, which caps it at ~5 TB/s of the ~9.6 TB/s
// this in-place read+write stream can reach (probe: torch inplace mul_).
// This kernel streams the state row by row instead: one warp per V-row group,
// each row is a 512B float4 load -> warp-reduced dot -> decayed delta-rule
// update -> 512B store, so loads pipeline across rows and nothing holds a
// tile. Setup (l2norm'd q/k, per-K decay, beta) is computed redundantly per
// warp - the kernel has no __syncthreads at all.
//
// Math follows the triton kernel exactly (fp32 throughout, same op order):
//   h *= exp(g);  t = <h, k>;  v = (v - t) * sigmoid(b);  h += v * k;
//   o = <h, q>
// with g = -exp(A_log) * softplus(a + dt_bias) (no lower bound) or
// lower_bound * sigmoid(exp(A_log) * (a + dt_bias)). Warp-shuffle reduction
// order differs from tl.sum, so outputs match to ULPs, not bits.
//
// Vendored from SGLang v0.5.20 (Apache-2.0): sglang/kernels/jit/csrc/
// attention/kda_packed_decode.cuh, NVIDIA x Moonshot K3 optimization
// package. sgl_kernel/tvm-ffi machinery replaced by a plain extern "C"
// launcher (see borrow9-kda-cuda DEVIATIONS in the module docstring).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

using bf16_t = __nv_bfloat16;

__device__ __forceinline__ float bf2f(bf16_t v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16_t f2bf(float v) { return __float2bfloat16_rn(v); }

template <bool kUsePDL>
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}
template <bool kUsePDL>
__device__ __forceinline__ void pdl_trigger() {
#if __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    // the "memory" clobber is load-bearing (donor comment): keeps this
    // kernel's stores from sinking past the dependent-grid trigger
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

namespace vkda {

struct KdaPackedDecodeParams {
  const bf16_t* __restrict__ mixed_qkv;  // [B, 2*H*K + HV*V]
  const bf16_t* __restrict__ a;          // [B, HV*K]
  const bf16_t* __restrict__ b;          // [B, HV]
  const float* __restrict__ A_log;       // [HV]
  const float* __restrict__ dt_bias;     // [HV*K]
  bf16_t* __restrict__ o;                // [B, HV*V] (contiguous view)
  float* __restrict__ state;             // pool, row stride = stride_state
  const int32_t* __restrict__ indices;   // [B]
  int64_t stride_mixed;
  int64_t stride_a;
  int64_t stride_b;
  int64_t stride_state;  // elements per pool slot
  uint32_t H;
  uint32_t HV;
  float scale;
  float lower_bound;
  int32_t use_lower_bound;
};

__device__ __forceinline__ float warp_allreduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// K = V = 128 specialization: one lane owns 4 consecutive K-elements (16B).
template <int kWarps, bool kUsePDL>
__global__
__launch_bounds__(kWarps * 32) void kda_packed_decode_kernel(const KdaPackedDecodeParams __grid_constant__ params) {
  constexpr int K = 128;
  constexpr int V = 128;
  constexpr int kElems = 4;  // K / 32 lanes

  const uint32_t i_nh = blockIdx.x;
  const uint32_t n = i_nh / params.HV;
  const uint32_t hv = i_nh % params.HV;
  const uint32_t i_h = hv / (params.HV / params.H);
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;

  pdl_wait<kUsePDL>();

  bf16_t* o_ptr = params.o + (static_cast<int64_t>(n) * params.HV + hv) * V;
  const int64_t sidx = params.indices[n];
  if (sidx < 0) {
    // Padded cuda-graph slot: zero the output, leave the pool untouched.
    for (uint32_t i = threadIdx.x; i < V; i += kWarps * 32) {
      o_ptr[i] = f2bf(0.0f);
    }
    pdl_trigger<kUsePDL>();
    return;
  }

  // --- per-warp redundant setup (no cross-warp synchronization) ---
  const bf16_t* mixed = params.mixed_qkv + n * params.stride_mixed;
  const uint32_t e0 = lane * kElems;

  float q[kElems], k[kElems];
  float q_sq = 0.0f, k_sq = 0.0f;
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    q[e] = bf2f(mixed[i_h * K + e0 + e]);
    k[e] = bf2f(mixed[params.H * K + i_h * K + e0 + e]);
    q_sq += q[e] * q[e];
    k_sq += k[e] * k[e];
  }
  // tl: q / sqrt(sum(q*q) + 1e-6), then * scale
  const float q_inv = 1.0f / sqrtf(warp_allreduce_sum(q_sq) + 1e-6f);
  const float k_inv = 1.0f / sqrtf(warp_allreduce_sum(k_sq) + 1e-6f);
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    q[e] = q[e] * q_inv * params.scale;
    k[e] = k[e] * k_inv;
  }

  const float exp_A = expf(params.A_log[hv]);
  float decay[kElems];
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    const float x = bf2f(params.a[n * params.stride_a + hv * K + e0 + e]) + params.dt_bias[hv * K + e0 + e];
    float g;
    if (params.use_lower_bound) {
      g = params.lower_bound / (1.0f + expf(-exp_A * x));
    } else {
      const float sp = (x <= 20.0f) ? logf(1.0f + expf(x)) : x;
      g = -exp_A * sp;
    }
    decay[e] = expf(g);
  }
  const float beta = 1.0f / (1.0f + expf(-bf2f(params.b[n * params.stride_b + hv])));

  const bf16_t* v_ptr = mixed + 2 * params.H * K + hv * V;
  float* h_base = params.state + sidx * params.stride_state + static_cast<int64_t>(hv) * V * K;

  // --- stream this warp's V-rows: 512B load -> update -> 512B store ---
  constexpr int kRowsPerWarp = V / kWarps;
#pragma unroll 4
  for (int r = warp * kRowsPerWarp; r < (int)(warp + 1) * kRowsPerWarp; ++r) {
    float4 h4 = *reinterpret_cast<const float4*>(h_base + r * K + e0);
    float h[kElems] = {h4.x, h4.y, h4.z, h4.w};
    float t = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      h[e] *= decay[e];
      t += h[e] * k[e];
    }
    t = warp_allreduce_sum(t);
    const float v_new = (bf2f(v_ptr[r]) - t) * beta;
    float o_acc = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      h[e] += v_new * k[e];
      o_acc += h[e] * q[e];
    }
    o_acc = warp_allreduce_sum(o_acc);
    *reinterpret_cast<float4*>(h_base + r * K + e0) = make_float4(h[0], h[1], h[2], h[3]);
    if (lane == 0) {
      o_ptr[r] = f2bf(o_acc);
    }
  }

  pdl_trigger<kUsePDL>();
}

}  // namespace vkda

extern "C" int vkda_packed_decode_run(
    const void* mixed_qkv, const void* a, const void* b, const void* A_log,
    const void* dt_bias, void* o, void* state, const void* indices,
    long long stride_mixed, long long stride_a, long long stride_b,
    long long stride_state, unsigned H, unsigned HV, unsigned B,
    double scale, double lower_bound, int use_lower_bound,
    long long stream, int use_pdl) {
  vkda::KdaPackedDecodeParams params{
      static_cast<const bf16_t*>(mixed_qkv),
      static_cast<const bf16_t*>(a),
      static_cast<const bf16_t*>(b),
      static_cast<const float*>(A_log),
      static_cast<const float*>(dt_bias),
      static_cast<bf16_t*>(o),
      static_cast<float*>(state),
      static_cast<const int32_t*>(indices),
      stride_mixed, stride_a, stride_b, stride_state,
      H, HV,
      static_cast<float>(scale),
      static_cast<float>(lower_bound),
      use_lower_bound ? 1 : 0,
  };
  const dim3 grid(B * HV);
  const dim3 block(VKDA_WARPS * 32);
  const cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
  if (use_pdl) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = cs;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    return (int)cudaLaunchKernelEx(&cfg, vkda::kda_packed_decode_kernel<VKDA_WARPS, true>, params);
  } else {
    vkda::kda_packed_decode_kernel<VKDA_WARPS, false><<<grid, block, 0, cs>>>(params);
    return (int)cudaGetLastError();
  }
}
"""


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
    """Compile (once, cached) and dlopen the packed-decode .so."""
    nvcc = _find_nvcc()
    if nvcc is None:
        raise OpNotEligible("nvcc not found (need CUDA toolkit to JIT glm_kda_packed_decode)")
    arch = _arch_flag(device)
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    tag = hashlib.sha256(
        (_CUDA_SOURCE + ver + arch + str(_WARPS)).encode()
    ).hexdigest()[:16]
    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vkernels", "cuda-jit",
    )
    so = os.path.join(cache_dir, f"kda_packed_decode-{tag}.so")
    if not os.path.exists(so):
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            src = os.path.join(tmp, "kda_packed_decode.cu")
            with open(src, "w") as f:
                f.write(f'#define VKDA_WARPS {_WARPS}\n' + _CUDA_SOURCE)
            obj = os.path.join(tmp, "kda_packed_decode.so")
            cmd = [
                nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
                f"-arch={arch}", "-DNDEBUG", src, "-o", obj,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise OpNotEligible(
                    "nvcc failed for glm_kda_packed_decode "
                    f"({arch}): {proc.stderr[-2000:]}"
                )
            os.replace(obj, so)  # atomic publish
    lib = ctypes.CDLL(so)
    lib.vkda_packed_decode_run.restype = ctypes.c_int
    lib.vkda_packed_decode_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_int,
    ]
    return lib


# module-level torch proxy removed: torch imports stay lazy inside functions


@lru_cache(maxsize=4)
def _lib_for(device_index: int):
    import torch

    return _compile_or_load(torch.device("cuda", device_index))


def _validate(
    mixed_qkv, a, b, A_log, dt_bias, state, indices, num_q_heads, out, lower_bound
):
    import torch

    if os.environ.get("VK_CUDA_JIT", "1") == "0":
        raise OpNotEligible("VK_CUDA_JIT=0 disables the CUDA-JIT KDA decode")
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise OpNotEligible("mixed_qkv must be 2-D [B, 2*H*K + HV*V] with contiguous rows")
    b_sz = mixed_qkv.shape[0]
    if a.ndim != 2 or a.shape[0] != b_sz or a.stride(-1) != 1:
        raise OpNotEligible("a must be [B, HV*K] with contiguous rows")
    if b.ndim != 2 or b.shape[0] != b_sz or b.stride(-1) != 1:
        raise OpNotEligible("b must be [B, HV] with contiguous rows")
    if state.ndim != 4 or state.dtype != torch.float32:
        raise OpNotEligible("state pool must be FP32 [slots, HV, V, K]")
    hv, v_dim, k_dim = state.shape[-3:]
    if (k_dim, v_dim) != (_K, _V):
        raise OpNotEligible(f"glm_kda_packed_decode is specialized for K = V = 128, got K={k_dim} V={v_dim}")
    if state.stride(-1) != 1 or state.stride(-2) != _K or state.stride(-3) != _V * _K:
        raise OpNotEligible("state inner layout must be dense V-major [HV, V, K]")
    if a.shape[1] != hv * _K or dt_bias.numel() != hv * _K:
        raise OpNotEligible("a/dt_bias must be [*, HV*K]")
    if b.shape[1] != hv or A_log.numel() != hv:
        raise OpNotEligible("b must be [B, HV] and A_log [HV]")
    h = int(num_q_heads)
    if h < 1 or hv % h:
        raise OpNotEligible(f"H={h} must divide HV={hv}")
    if mixed_qkv.shape[1] != 2 * h * _K + hv * _V:
        raise OpNotEligible("mixed_qkv last dim must be 2*H*K + HV*V (q | k | v)")
    for name, t, dt in (
        ("mixed_qkv", mixed_qkv, torch.bfloat16),
        ("a", a, torch.bfloat16),
        ("b", b, torch.bfloat16),
        ("A_log", A_log, torch.float32),
        ("dt_bias", dt_bias, torch.float32),
    ):
        if t.dtype != dt:
            raise OpNotEligible(f"{name} must be {dt}, got {t.dtype}")
    if indices.dtype != torch.int32 or indices.ndim != 1 or indices.shape[0] != b_sz:
        raise OpNotEligible("ssm_state_indices must be int32 [B]")
    if out.dtype != torch.bfloat16 or out.shape != (b_sz, 1, hv, _V) or not out.is_contiguous():
        raise OpNotEligible("out must be contiguous BF16 [B, 1, HV, V]")
    dev = mixed_qkv.device
    if not dev.type == "cuda":
        raise OpNotEligible("glm_kda_packed_decode requires CUDA tensors")
    for name, t in (
        ("a", a), ("b", b), ("A_log", A_log), ("dt_bias", dt_bias),
        ("state", state), ("indices", indices), ("out", out),
    ):
        if t.device != dev:
            raise OpNotEligible(f"{name} on {t.device}, expected {dev}")
    if lower_bound is not None:
        lower_bound = float(lower_bound)
    return b_sz, h, hv, lower_bound


def kda_state_kmajor_to_vmajor(state):
    """[slots, H, K, V] (floe's incumbent Triton pool) -> [slots, H, V, K]."""
    import torch

    if state.ndim != 4 or state.dtype != torch.float32:
        raise OpNotEligible("expected an FP32 [slots, H, K, V] state pool")
    return state.transpose(-1, -2).contiguous()


def kda_state_vmajor_to_kmajor(state):
    """[slots, H, V, K] -> [slots, H, K, V] (back to the incumbent layout)."""
    import torch

    if state.ndim != 4 or state.dtype != torch.float32:
        raise OpNotEligible("expected an FP32 [slots, H, V, K] state pool")
    return state.transpose(-1, -2).contiguous()


def glm_kda_packed_decode_eligible(
    mixed_qkv, a, b, A_log, dt_bias, state, ssm_state_indices, num_q_heads,
    out=None, lower_bound=None,
) -> bool:
    """Contract check for :func:`glm_kda_packed_decode` (no device sync)."""
    import torch

    if out is None:
        out = torch.empty(
            mixed_qkv.shape[0], 1, state.shape[-3], _V,
            dtype=torch.bfloat16, device=mixed_qkv.device,
        )
    try:
        _validate(mixed_qkv, a, b, A_log, dt_bias, state, ssm_state_indices,
                  num_q_heads, out, lower_bound)
    except OpNotEligible:
        return False
    return True


def glm_kda_packed_decode(
    mixed_qkv, a, b, A_log, dt_bias, scale, state, ssm_state_indices,
    num_q_heads, out=None, lower_bound=None, use_pdl=False,
):
    """One-launch batched KDA decode: in-place V-major state update + output.

    Updates the ``state`` pool rows selected by ``ssm_state_indices`` (the
    delta rule, fp32 in-kernel) and writes the attention output into
    ``out`` ``[B, 1, HV, V]`` bf16 (allocated when ``None``). Returns
    ``out``. Inputs per the FLOE CONTRACT MAPPING in the module docstring;
    ``-1`` indices are padded graph slots (zero output, pool untouched).
    Raises :class:`~vkernels.torch_ops._dispatch.OpNotEligible` outside the
    contract so the caller falls back to the Triton
    :func:`~vkernels.torch_ops.glm_kda_decode.glm_kda_decode`.
    """
    import torch

    if out is None:
        out = torch.empty(
            mixed_qkv.shape[0], 1, state.shape[-3], _V,
            dtype=torch.bfloat16, device=mixed_qkv.device,
        )
    b_sz, h, hv, lower_bound = _validate(
        mixed_qkv, a, b, A_log, dt_bias, state, ssm_state_indices,
        num_q_heads, out, lower_bound,
    )
    try:
        lib = _lib_for(mixed_qkv.device.index or 0)
    except OpNotEligible as e:
        raise e
    stream = torch.cuda.current_stream(mixed_qkv.device).cuda_stream
    rc = lib.vkda_packed_decode_run(
        mixed_qkv.data_ptr(), a.data_ptr(), b.data_ptr(), A_log.data_ptr(),
        dt_bias.data_ptr(), out.data_ptr(), state.data_ptr(),
        ssm_state_indices.data_ptr(),
        mixed_qkv.stride(0), a.stride(0), b.stride(0), state.stride(0),
        h, hv, b_sz,
        float(scale), lower_bound if lower_bound is not None else 0.0,
        int(lower_bound is not None),
        stream, int(bool(use_pdl)),
    )
    if rc != 0:
        raise RuntimeError(f"kda_packed_decode launch failed: CUDA error {rc}")
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize(mixed_qkv.device)  # surface launch errors eagerly
    return out


def glm_kda_packed_decode_reference(
    mixed_qkv, a, b, A_log, dt_bias, scale, state, ssm_state_indices,
    num_q_heads, lower_bound=None,
):
    """Eager fp32 torch oracle (the kernel's exact op order, CPU/GPU).

    Returns ``(out [B, 1, HV, V] bf16, next_state [slots, HV, V, K] fp32)``
    with ``next_state`` a FRESH tensor (the kernel mutates in place; the
    oracle copies so both can be compared after the fact). ``-1`` indices
    are padded slots: zero output, pool row passed through.
    """
    import torch

    b_sz = mixed_qkv.shape[0]
    hv, v_dim, k_dim = state.shape[-3:]
    h = int(num_q_heads)
    next_state = state.clone()
    out = torch.zeros(b_sz, 1, hv, v_dim, dtype=torch.bfloat16, device=mixed_qkv.device)

    mq = mixed_qkv.float()
    q_all = mq[:, : h * k_dim].view(b_sz, h, k_dim)
    k_all = mq[:, h * k_dim : 2 * h * k_dim].view(b_sz, h, k_dim)
    v_all = mq[:, 2 * h * k_dim :].view(b_sz, hv, v_dim)
    dt = dt_bias.float().view(hv, k_dim)
    a_f = a.float().view(b_sz, hv, k_dim)
    b_f = b.float().view(b_sz, hv)
    a_log = A_log.float()

    for n in range(b_sz):
        sidx = int(ssm_state_indices[n].item())
        if sidx < 0:
            continue
        for j in range(hv):
            i_h = j // (hv // h)
            q = q_all[n, i_h]
            k = k_all[n, i_h]
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
            delta = (v_all[n, j] - t) * beta
            s = s + delta[:, None] * k[None, :]
            next_state[sidx, j] = s
            out[n, 0, j] = (s * q[None, :]).sum(dim=1).to(torch.bfloat16)
    return out, next_state
