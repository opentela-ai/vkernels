"""gfx942 HIP device bindings for the GLM-5.3-Flash attention stack.

Torch-tensor ctypes wrappers over the ``libvkernels_hip.so`` C ABI for the
three GLM-5.3-Flash ops whose tilelang forms fault on MI300A (issue #51):

* :func:`dsa_sparse_fwd` — DeepseekSparseAttn sparse-MLA forward
  (``vk_hip_dsa_sparse_fwd`` / ``vk_hip_dsa_sparse_fwd_split``), the
  absorbed-latent form floe's ``Glm53DSA`` uses (``dim = kv_lora_rank``,
  ``tail_dim = 0``).
* :func:`mhc_pre_gemm_sqrsum` / :func:`mhc_post` — the multi-head
  hyper-connection pre-GEMM + compose kernels.
* :func:`kda_delta_rule_fwd` / :func:`kda_delta_rule_fwd_with_scratch` —
  the per-key-dim gated delta-rule forward (``g`` is ``[B, H, S, D]``,
  the shape GLM-5.3-Flash needs and the scalar-per-head Python CPU
  binding in :mod:`vkernels.kernels` cannot express).

This is an OPTIONAL integration module: importing it requires ``torch``;
everything else is lazy (the shared library is loaded on first use, so
``import vkernels.hip_dsa_mhc`` works on a torch-free or HIP-free host and
only the call sites raise). :func:`available` reports whether the device
path is usable without raising.

Stream discipline (issue #69): every function takes ``stream=None`` which
resolves to torch's CURRENT stream — during graph capture that is the
capturing stream, so these calls are capture-safe with a current library
(the ``*_stream`` C ABI symbols; a pre-#69 library falls back to the
legacy eager entry points and raises if an explicit stream is passed).
The KDA ``with_scratch``/``chunked`` variants do NOT synchronise on the
stream path — ordering and sync are the caller's (stream order or graph
replay).
"""

from __future__ import annotations

import ctypes
import math
from typing import Optional

__all__ = [
    "available",
    "dsa_sparse_fwd",
    "dsa_sparse_fwd_split",
    "dsa_split_for",
    "mhc_pre_gemm_sqrsum",
    "mhc_post",
    "kda_delta_rule_fwd",
    "kda_delta_rule_fwd_with_scratch",
    "kda_delta_rule_fwd_chunked",
    "kda_chunked_scratch_floats",
]

# MI300A / gfx942 CU count (hipDeviceProp_t::multiProcessorCount, verified on
# a CSCS beverin node) — the documented constant behind dsa_split_for.
NUM_CU_GFX942 = 228

import torch

_LIB = None
_LIB_FAILED = False

# bf16 as a ctypes storage type: torch bf16 tensors are passed to the C ABI
# by data_ptr, so the kernels see their native uint16 storage. These ctypes
# types exist purely for prototype declarations.
_VOIDP = ctypes.c_void_p
_INT = ctypes.c_int
_FLOAT = ctypes.c_float


def _load():
    """Load and cache ``libvkernels_hip.so``; returns None when absent."""
    global _LIB, _LIB_FAILED
    if _LIB is not None:
        return _LIB
    if _LIB_FAILED:
        return None
    try:
        from vkernels.vllm_experts import load_libvkernels_hip

        _LIB = load_libvkernels_hip()
    except (RuntimeError, OSError, ImportError):
        _LIB_FAILED = True
        return None
    return _LIB


def _stream_ptr(stream) -> int:
    """Resolve a ctypes-usable hipStream_t: explicit arg or torch's current
    stream (the graph-capturing stream during capture — issue #69)."""
    import torch
    if stream is None:
        return torch.cuda.current_stream().cuda_stream
    if isinstance(stream, torch.cuda.Stream):
        return stream.cuda_stream
    return int(stream)


def _device_guard(t):
    """Context manager running the enclosed HIP launch on t's device.

    HIP launches target the CURRENT device: without this guard, tensors
    living on a non-current GPU (multi-GPU serving) launch on the wrong
    device with foreign pointers — garbage or a fault (#69 non-current
    device contract). No-op when t is already on the current device;
    restores the previous device on exit. torch.cuda.device requires a
    torch Tensor (device index), which every caller here has.
    """
    import contextlib
    return contextlib.nullcontext() if t is None else torch.cuda.device(t.device)


def _launch(f_stream, f_legacy, args, stream):
    """Call the _stream symbol when present (returning the launch error so a
    partial failure is never silent); fall back to the legacy symbol for
    pre-#69 libraries (stream must be None there)."""
    if f_stream is not None:
        rc = f_stream(*args, _stream_ptr(stream))
        if rc != 0:
            raise RuntimeError(f"HIP launch failed: error {rc}")
        return
    if stream is not None:
        raise RuntimeError("this libvkernels_hip build predates the #69 "
                           "stream ABI; pass stream=None or rebuild")
    f_legacy(*args)


def _sym(lib, name):
    return getattr(lib, name, None)


def stream_abi() -> bool:
    """True when the loaded library carries the #69 ``*_stream`` symbols
    (graph-capture-safe launches). Floe uses this to decide whether its
    device dispatch may run under capture or must fall back to torch."""
    lib = _load()
    if lib is None:
        return False
    return hasattr(lib, "vk_hip_kda_delta_rule_fwd_chunked_with_scratch_stream")


def available() -> bool:
    """True when ``libvkernels_hip.so`` is loadable (device path usable)."""
    return _load() is not None


def _dptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _check_device(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a device tensor (got {t.device})")
    if torch.cuda.is_current_stream_capturing() and not stream_abi():
        # Pre-#69 library: the launches would go to the legacy default
        # stream, which cannot be captured. With the stream ABI the calls
        # resolve torch's CURRENT (capturing) stream and are safe.
        raise RuntimeError(
            "this libvkernels_hip build predates the #69 stream ABI; its "
            "launches go to the legacy default stream and cannot be "
            "captured into a CUDA/HIP graph"
        )


def _contig(t: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    _check_device(t, name)
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype} (got {t.dtype})")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return t


# ---------------------------------------------------------------------------
# DSA sparse-MLA forward
# ---------------------------------------------------------------------------
def dsa_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    dim: int,
    tail_dim: int,
    topk: Optional[int] = None,
    kv_group: int = 1,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    stream=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sparse-MLA forward over indexer-selected keys (device, bf16 ABI).

    ``stream`` (issue #69): a torch.cuda.Stream, raw stream int, or None
    (default: torch's current stream — the capturing stream during graph
    capture, making this call capture-safe).

    Mirrors :func:`vkernels.kernels.dsa_sparse_fwd` (the fp32 CPU oracle)
    with the device ABI's storage dtypes:

    * ``q``       bf16 ``(1, S_q, H, dim + tail_dim)``
    * ``kv``      bf16 ``(1, S_kv, kv_group, dim + tail_dim)`` (kv_group==1)
    * ``indices`` int32 ``(1, S_q, kv_group, topk)`` — entries ``< 0`` masked
    * ``out``     bf16 ``(1, S_q, H, dim - tail_dim)``
    * ``lse``     fp32 ``(1, S_q, H)`` (base-2; required when
      ``return_lse``)

    ``sm_scale`` must already fold ``log2(e)`` (the kernel applies
    ``score = sm_scale * q·k`` and softmaxes in base 2). Batch is 1 — for a
    B>1 decode batch call once per request (floe does).
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")

    B, S_q, H, W = q.shape
    Bk, S_kv, kvg, W2 = kv.shape
    Bi, S_q2, kvg2, tk = indices.shape
    if B != 1:
        raise ValueError(f"dsa_sparse_fwd is batch-1 (got B={B}); loop per request")
    if W != dim + tail_dim or W2 != dim + tail_dim:
        raise ValueError(f"q/kv last dim {W}/{W2} != dim + tail_dim = {dim + tail_dim}")
    if kvg != kv_group or kvg2 != kv_group or kv_group != 1:
        raise ValueError("kv_group must be 1 everywhere")
    if S_q != S_q2:
        raise ValueError(f"S_q mismatch: q={S_q} indices={S_q2}")
    if topk is None:
        topk = tk
    elif int(topk) != tk:
        raise ValueError(f"topk mismatch: arg={topk} indices={tk}")
    d_v = dim - tail_dim
    if d_v <= 0:
        raise ValueError(f"dim - tail_dim must be > 0 (got {d_v})")

    q = _contig(q, "q", torch.bfloat16)
    kv = _contig(kv, "kv", torch.bfloat16)
    indices = _contig(indices, "indices", torch.int32)
    if out is None:
        out = torch.empty(1, S_q, H, d_v, dtype=torch.bfloat16, device=q.device)
    else:
        _contig(out, "out", torch.bfloat16)
    if return_lse:
        if lse is None:
            lse = torch.empty(1, S_q, H, dtype=torch.float32, device=q.device)
        else:
            _contig(lse, "lse", torch.float32)

    if sm_scale is None:
        sm_scale = (1.0 / math.sqrt(dim + tail_dim)) * math.log(math.e)
    # The host tile selector (pure math, no GPU): the HIP launcher
    # recomputes tiles internally via dsa_config_for, but the C ABI still
    # validates topk % (block_I * inner_iter) == 0 against OURS.
    from vkernels.kernels import dsa_config

    _bq, _th, block_i, inner_iter = dsa_config(int(S_q), int(H), int(dim), int(topk))

    fs = _sym(lib, "vk_hip_dsa_sparse_fwd_stream")
    if fs is not None:
        fs.restype = _INT
        fs.argtypes = [
            _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _FLOAT,
            _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP,
        ]
        with _device_guard(q):
            rc = fs(
                int(S_q), int(S_kv), int(H), int(dim), int(tail_dim), int(topk),
                int(kv_group), int(block_i), int(inner_iter),
                ctypes.c_float(sm_scale), 1 if return_lse else 0,
                _dptr(q), _dptr(kv), _dptr(indices), _dptr(out),
                _dptr(lse) if return_lse else None, _stream_ptr(stream),
            )
        if rc != 0:
            raise RuntimeError(f"vk_hip_dsa_sparse_fwd failed with rc={rc}")
    else:
        fn = lib.vk_hip_dsa_sparse_fwd
        fn.restype = _INT
        fn.argtypes = [
            _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _FLOAT,
            _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP,
        ]
        with _device_guard(q):
            rc = fn(
                int(S_q), int(S_kv), int(H), int(dim), int(tail_dim), int(topk),
                int(kv_group), int(block_i), int(inner_iter),
                ctypes.c_float(sm_scale), 1 if return_lse else 0,
                _dptr(q), _dptr(kv), _dptr(indices), _dptr(out),
                _dptr(lse) if return_lse else None,
            )
        if rc != 0:
            raise RuntimeError(f"vk_hip_dsa_sparse_fwd failed with rc={rc}")
    return (out, lse) if return_lse else out


def dsa_split_for(s_q: int, h: int, topk: int, block_i: int = 1) -> int:
    """Recommended ``split_kv`` (dsa.hpp, issue #51 follow-up, MI300A):

    ``1`` when the plain grid already fills the CUs (prefill — splitting
    would only add combine traffic), else ``ceil(sqrt(2*topk))`` (lands on
    the measured best or nearest neighbour on all four swept decode
    shapes: 2048→64 = measured best 19.7x, 256→23 (best 16), 128→16).
    Pure arithmetic on documented device constants.
    """
    if -(-s_q // block_i) * h >= NUM_CU_GFX942:
        return 1
    return min(int(math.ceil(math.sqrt(2 * topk))), int(topk))


def dsa_sparse_fwd_split(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    dim: int,
    tail_dim: int,
    topk: Optional[int] = None,
    kv_group: int = 1,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    split_kv: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    stream=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Split-key sparse-MLA forward (decode occupancy fix).

    Same contract as :func:`dsa_sparse_fwd` but the per-query ``topk``
    index range is divided across ``split_kv`` partial blocks and merged
    in the log2 domain (flash-decoding). PERF-ONLY variant of the plain
    forward — bf16-tolerant vs the CPU oracle. ``split_kv <= 1`` behaves
    exactly like :func:`dsa_sparse_fwd`; the recommended value comes from
    :func:`dsa_split_for`. The fp32 partial scratch (``[S_q, H, split_kv,
    d_v]`` + ``[S_q, H, split_kv]``) is allocated internally.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")

    B, S_q, H, W = q.shape
    _Bk, S_kv, kvg, W2 = kv.shape
    _Bi, S_q2, kvg2, tk = indices.shape
    if B != 1:
        raise ValueError(f"dsa_sparse_fwd_split is batch-1 (got B={B}); loop per request")
    if W != dim + tail_dim or W2 != dim + tail_dim:
        raise ValueError(f"q/kv last dim {W}/{W2} != dim + tail_dim = {dim + tail_dim}")
    if kvg != kv_group or kvg2 != kv_group or kv_group != 1:
        raise ValueError("kv_group must be 1 everywhere")
    if S_q != S_q2:
        raise ValueError(f"S_q mismatch: q={S_q} indices={S_q2}")
    if topk is None:
        topk = tk
    elif int(topk) != tk:
        raise ValueError(f"topk mismatch: arg={topk} indices={tk}")
    d_v = dim - tail_dim
    if d_v <= 0:
        raise ValueError(f"dim - tail_dim must be > 0 (got {d_v})")
    if split_kv is None:
        from vkernels.kernels import dsa_config

        _bq, _th, block_i, _ii = dsa_config(int(S_q), int(H), int(dim), int(topk))
        split_kv = dsa_split_for(int(S_q), int(H), int(topk), block_i)
    if not 1 <= split_kv <= topk:
        raise ValueError(f"split_kv {split_kv} out of [1, {topk}]")

    q = _contig(q, "q", torch.bfloat16)
    kv = _contig(kv, "kv", torch.bfloat16)
    indices = _contig(indices, "indices", torch.int32)
    if out is None:
        out = torch.empty(1, S_q, H, d_v, dtype=torch.bfloat16, device=q.device)
    else:
        _contig(out, "out", torch.bfloat16)
    if return_lse:
        if lse is None:
            lse = torch.empty(1, S_q, H, dtype=torch.float32, device=q.device)
        else:
            _contig(lse, "lse", torch.float32)
    if split_kv > 1:
        partial_out = torch.empty(S_q, H, split_kv, d_v, dtype=torch.float32, device=q.device)
        partial_lse = torch.empty(S_q, H, split_kv, dtype=torch.float32, device=q.device)
    else:
        partial_out = partial_lse = None

    if sm_scale is None:
        sm_scale = (1.0 / math.sqrt(dim + tail_dim)) * math.log(math.e)
    from vkernels.kernels import dsa_config

    _bq, _th, block_i, inner_iter = dsa_config(int(S_q), int(H), int(dim), int(topk))

    fs = _sym(lib, "vk_hip_dsa_sparse_fwd_split_stream")
    if fs is not None:
        fs.restype = _INT
        fs.argtypes = [
            _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _FLOAT,
            _INT, _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP,
            _VOIDP, _VOIDP,
        ]
        with _device_guard(q):
            rc = fs(
                int(S_q), int(S_kv), int(H), int(dim), int(tail_dim), int(topk),
                int(kv_group), int(block_i), int(inner_iter),
                ctypes.c_float(sm_scale), 1 if return_lse else 0,
                int(split_kv), _dptr(q), _dptr(kv), _dptr(indices),
                _dptr(out), _dptr(lse) if return_lse else None,
                _dptr(partial_out) if partial_out is not None else None,
                _dptr(partial_lse) if partial_lse is not None else None,
                _stream_ptr(stream),
            )
    else:
        fn = lib.vk_hip_dsa_sparse_fwd_split
        fn.restype = _INT
        fn.argtypes = [
            _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _FLOAT,
            _INT, _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP,
            _VOIDP,
        ]
        with _device_guard(q):
            rc = fn(
                int(S_q), int(S_kv), int(H), int(dim), int(tail_dim),
                int(topk), int(kv_group), int(block_i), int(inner_iter),
                ctypes.c_float(sm_scale), 1 if return_lse else 0,
                int(split_kv), _dptr(q), _dptr(kv), _dptr(indices),
                _dptr(out), _dptr(lse) if return_lse else None,
                _dptr(partial_out) if partial_out is not None else None,
                _dptr(partial_lse) if partial_lse is not None else None,
            )
    if rc != 0:
        raise RuntimeError(f"vk_hip_dsa_sparse_fwd_split failed with rc={rc}")
    return (out, lse) if return_lse else out


# ---------------------------------------------------------------------------
# MHC pre / post
# ---------------------------------------------------------------------------
def mhc_pre_gemm_sqrsum(
    x: torch.Tensor,
    fn: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    sqrsum: Optional[torch.Tensor] = None,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``out = x @ fn.T`` (fp32) + per-token ``sum(x^2)`` (device).

    * ``x``      bf16 ``[num_tokens, hc_hidden_size]``
    * ``fn``     fp32 ``[hc_mult3, hc_hidden_size]`` (hc_mult3 <= 32)
    * ``out``    fp32 ``[num_tokens, hc_mult3]``
    * ``sqrsum`` fp32 ``[num_tokens]``

    The RMS rescale ``rsqrt(sqrsum/(hc_mult*hidden) + eps)`` stays with the
    caller (floe applies it in torch, exactly like the CPU path).
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")
    n, hc_hidden = x.shape
    hc_mult3 = fn.shape[0]
    if fn.shape[1] != hc_hidden:
        raise ValueError(f"fn width {fn.shape[1]} != x width {hc_hidden}")
    if hc_mult3 > 32:
        raise ValueError(f"hc_mult3 {hc_mult3} > 32 (kernel limit)")
    x = _contig(x, "x", torch.bfloat16)
    fn = _contig(fn, "fn", torch.float32)
    if out is None:
        out = torch.empty(n, hc_mult3, dtype=torch.float32, device=x.device)
    else:
        _contig(out, "out", torch.float32)
    if sqrsum is None:
        sqrsum = torch.empty(n, dtype=torch.float32, device=x.device)
    else:
        _contig(sqrsum, "sqrsum", torch.float32)
    fs = _sym(lib, "vk_hip_mhc_pre_gemm_sqrsum_stream")
    if fs is not None:
        fs.restype = _INT
        fs.argtypes = [_INT, _INT, _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP]
        with _device_guard(x):
            rc = fs(int(n), int(hc_mult3), int(hc_hidden), _dptr(x),
                    _dptr(fn), _dptr(out), _dptr(sqrsum),
                    _stream_ptr(stream))
        if rc != 0:
            raise RuntimeError(f"vk_hip_mhc_pre_gemm_sqrsum failed: rc={rc}")
    else:
        f = lib.vk_hip_mhc_pre_gemm_sqrsum
        f.restype = None
        f.argtypes = [_INT, _INT, _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP]
        f(int(n), int(hc_mult3), int(hc_hidden), _dptr(x), _dptr(fn),
          _dptr(out), _dptr(sqrsum))
    return out, sqrsum


def mhc_post(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    stream=None,
) -> torch.Tensor:
    """``out[n,j,:] = c[n,j]·d[n,:] + Σ_k a[n,k,j]·b[n,k,:]`` (device).

    * ``a`` (comb)     fp32 ``[num_tokens, hc, hc]``
    * ``b`` (residual) bf16 ``[num_tokens, hc, hidden]``
    * ``c`` (post)     fp32 ``[num_tokens, hc]``
    * ``d`` (subout)   bf16 ``[num_tokens, hidden]``
    * ``out``          bf16 ``[num_tokens, hc, hidden]`` (hc <= 63)
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")
    n, hc, hc2 = a.shape
    hidden = d.shape[1]
    if hc != hc2 or b.shape[1] != hc or c.shape[1] != hc or b.shape[2] != hidden:
        raise ValueError("mhc_post shape mismatch")
    if hc > 63:
        raise ValueError(f"hc {hc} > 63 (kernel limit)")
    a = _contig(a, "a", torch.float32)
    b = _contig(b, "b", torch.bfloat16)
    c = _contig(c, "c", torch.float32)
    d = _contig(d, "d", torch.bfloat16)
    if out is None:
        out = torch.empty(n, hc, hidden, dtype=torch.bfloat16, device=d.device)
    else:
        _contig(out, "out", torch.bfloat16)
    fs = _sym(lib, "vk_hip_mhc_post_stream")
    if fs is not None:
        fs.restype = _INT
        fs.argtypes = [_INT, _INT, _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP,
                       _VOIDP, _VOIDP]
        with _device_guard(d):
            rc = fs(int(n), int(hc), int(hidden), _dptr(a), _dptr(b),
                    _dptr(c), _dptr(d), _dptr(out), _stream_ptr(stream))
        if rc != 0:
            raise RuntimeError(f"vk_hip_mhc_post failed: rc={rc}")
    else:
        f = lib.vk_hip_mhc_post
        f.restype = None
        f.argtypes = [_INT, _INT, _INT, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP]
        f(int(n), int(hc), int(hidden), _dptr(a), _dptr(b), _dptr(c),
          _dptr(d), _dptr(out))
    return out


# ---------------------------------------------------------------------------
# KDA per-key-dim gated delta-rule forward
# ---------------------------------------------------------------------------
def _kda_check(q, k, v, g, beta):
    B, H, S, D = q.shape
    for t, name in ((k, "k"), (v, "v"), (g, "g")):
        if tuple(t.shape) != (B, H, S, D):
            raise ValueError(f"{name} shape {tuple(t.shape)} != {(B, H, S, D)}")
    if tuple(beta.shape) != (B, H, S):
        raise ValueError(f"beta shape {tuple(beta.shape)} != {(B, H, S)}")
    tensors = [_contig(t, n, torch.float32) for t, n in ((q, "q"), (k, "k"), (v, "v"), (g, "g"), (beta, "beta"))]
    return B, H, S, D, tensors


def kda_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-key-dim gated delta-rule forward (device, fp32 ABI).

    ``g`` is ``[B, H, S, D]`` in the NORMAL space ``(0, 1]`` — pass
    ``g_log.exp()`` when your reference keeps log-space gates (floe's
    ``_kda_chunk`` does). ``k`` must be L2-normalised and ``q``
    L2-normalised and scaled by ``D**-0.5`` caller-side (matching the
    kernel contract; ``_kda_chunk`` applies the query scale internally).

    NOTE (parity-pinned on MI300A, 2025-09, bench_hip_parity.py): the
    kernel writes ``out`` in ``[B, S, H, D]`` orientation and the scratch
    state as ``[B, H, v_dim, k_dim]`` — BOTH transposed relative to this
    docstring/ABI's ``[B, H, S, D]`` / ``[B, H, k, v]`` (the C ABI header
    is stale; verified against ``_kda_chunk`` to 1e-6 rel). Read outputs
    accordingly.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")
    B, H, S, D, (q, k, v, g, beta) = _kda_check(q, k, v, g, beta)
    if out is None:
        out = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
    else:
        _contig(out, "out", torch.float32)
    f = lib.vk_hip_kda_delta_rule_fwd
    f.restype = None
    f.argtypes = [_VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _INT, _INT, _INT, _INT, _INT]
    f(_dptr(q), _dptr(k), _dptr(v), _dptr(g), _dptr(beta), _dptr(out), B, H, S, D, int(chunk_size))
    return out


def kda_delta_rule_fwd_with_scratch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """As :func:`kda_delta_rule_fwd` but the caller owns the state scratch.

    ``state`` is fp32 ``[B, H, D, D]`` — pre-fill with the gathered initial
    state (zeros for a first-turn prefill); the final state is written back
    into the SAME buffer, so multi-turn decode reads it after the call.
    Caller-owned storage keeps the address stable across calls (the
    capture-safety convention of this library's device path). NOTE: the
    buffer is read/written in the kernel's ``[B, H, v_dim, k_dim]``
    orientation (see the NOTE on :func:`kda_delta_rule_fwd`) — transpose
    when exchanging states with a ``[B, H, k, v]`` cache.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")
    B, H, S, D, (q, k, v, g, beta) = _kda_check(q, k, v, g, beta)
    if tuple(state.shape) != (B, H, D, D):
        raise ValueError(f"state shape {tuple(state.shape)} != {(B, H, D, D)}")
    state = _contig(state, "state", torch.float32)
    if out is None:
        out = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
    else:
        _contig(out, "out", torch.float32)
    fs = _sym(lib, "vk_hip_kda_delta_rule_fwd_with_scratch_stream")
    if fs is not None:
        fs.restype = _INT
        fs.argtypes = [_VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP,
                       _INT, _INT, _INT, _INT, _VOIDP]
        with _device_guard(q):
            rc = fs(_dptr(q), _dptr(k), _dptr(v), _dptr(g), _dptr(beta),
                    _dptr(state), _dptr(out), B, H, S, D,
                    _stream_ptr(stream))
        if rc != 0:
            raise RuntimeError(f"vk_hip_kda_delta_rule_fwd_with_scratch "
                               f"failed: rc={rc}")
    else:
        if stream is not None:
            raise RuntimeError("stream requires the #69 stream ABI "
                               "(rebuild libvkernels_hip)")
        f = lib.vk_hip_kda_delta_rule_fwd_with_scratch
        f.restype = None
        f.argtypes = [_VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP, _VOIDP,
                      _INT, _INT, _INT, _INT]
        f(_dptr(q), _dptr(k), _dptr(v), _dptr(g), _dptr(beta), _dptr(state),
          _dptr(out), B, H, S, D)
    return out, state


def kda_chunked_scratch_floats(B: int, H: int, S: int, D: int) -> int:
    """WY scratch size in float32s for :func:`kda_delta_rule_fwd_chunked`."""
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")
    f = lib.vk_hip_kda_chunked_scratch_floats
    f.restype = ctypes.c_ulonglong
    f.argtypes = [_INT, _INT, _INT, _INT]
    return int(f(B, H, S, D))


def kda_delta_rule_fwd_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    scratch: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked WY per-key-dim delta-rule forward (device, fp32 ABI, #70).

    Same contract and layouts as :func:`kda_delta_rule_fwd_with_scratch`
    (normal-space ``g`` in ``(0, 1]``, L2-normalised ``k``, caller-scaled
    ``q``, caller-owned ``state [B, H, D, D]`` read and written in place),
    computed in the affine WY form: gate cumsum -> per-chunk grams +
    explicit triangular inverse + U_v/W/T/Opar precompute -> an nc-step
    serial state pass split over D-row blocks. 1.1-4.0x over the
    cooperative kernel on gfx942 (beverin, docs/kda-lds-optimization.md).

    CONTRACT (stricter than the cooperative kernel): ``k`` must be
    L2-normalised (bounds the explicit inverse); ``chunk_size`` must be
    64 and ``S % 64 == 0`` (pad at the call site); ``D <= 128`` and
    ``D % 16 == 0``; eager (the launcher syncs).

    ``scratch`` is the WY buffer of
    :func:`kda_chunked_scratch_floats` (B, H, S, D) float32s — pass
    it to reuse storage across calls; when None a fresh buffer is
    allocated per call (contents clobbered either way).
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libvkernels_hip.so not found")
    B, H, S, D, (q, k, v, g, beta) = _kda_check(q, k, v, g, beta)
    if chunk_size != 64 or S % 64 != 0:
        raise ValueError(
            f"chunked KDA requires chunk_size==64 and S%64==0 "
            f"(got chunk_size={chunk_size}, S={S}); pad S at the call site")
    if D > 128 or D % 16 != 0:
        raise ValueError(f"chunked KDA requires D<=128 and D%16==0 (got D={D})")
    if tuple(state.shape) != (B, H, D, D):
        raise ValueError(f"state shape {tuple(state.shape)} != {(B, H, D, D)}")
    state = _contig(state, "state", torch.float32)
    if out is None:
        out = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
    else:
        _contig(out, "out", torch.float32)
    if scratch is None:
        scratch = torch.empty(
            kda_chunked_scratch_floats(B, H, S, D),
            dtype=torch.float32, device=q.device)
    else:
        want = kda_chunked_scratch_floats(B, H, S, D)
        if scratch.numel() < want:
            raise ValueError(f"scratch needs {want} floats "
                             f"(got {scratch.numel()})")
        scratch = _contig(scratch, "scratch", torch.float32)
    fs = _sym(lib, "vk_hip_kda_delta_rule_fwd_chunked_with_scratch_stream")
    if fs is not None:
        fs.restype = _INT
        fs.argtypes = ([_VOIDP] * 8 + [_INT] * 5 + [_VOIDP])
        with _device_guard(q):
            rc = fs(_dptr(q), _dptr(k), _dptr(v), _dptr(g), _dptr(beta),
                    _dptr(state), _dptr(out), _dptr(scratch), B, H, S, D,
                    int(chunk_size), _stream_ptr(stream))
        if rc != 0:
            raise RuntimeError(f"vk_hip_kda_delta_rule_fwd_chunked "
                               f"failed: rc={rc}")
    else:
        if stream is not None:
            raise RuntimeError("stream requires the #69 stream ABI "
                               "(rebuild libvkernels_hip)")
        f = lib.vk_hip_kda_delta_rule_fwd_chunked_with_scratch
        f.restype = None
        f.argtypes = ([_VOIDP] * 8 + [_INT] * 5)
        f(_dptr(q), _dptr(k), _dptr(v), _dptr(g), _dptr(beta), _dptr(state),
          _dptr(out), _dptr(scratch), B, H, S, D, int(chunk_size))
    return out, state
