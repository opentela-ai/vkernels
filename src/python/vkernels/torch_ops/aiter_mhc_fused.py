"""AMD AITER fused mHC post+pre bridge (gfx942) — ``aiter.ops.mhc.mhc_fused_post_pre``.

Fuses the mHC compose (``mhc_post``: ``streams[j] = post[j]·x + Σ_k comb[k,j]·residual[k]``)
with the NEXT section's pre (``mhc_pre``: mix GEMM + Σx² + RMS rescale + sigmoid gates +
Sinkhorn comb + stream collapse) into ONE kernel launch, removing one launch and one
``[n, hc, D]`` intermediate round-trip per mHC section boundary. vLLM dispatches this op
on its GLM-5.3 ROCm path (``vllm/model_executor/layers/mhc.py`` ``MHCFusedPostPreOp``;
wrappers in ``vllm/model_executor/kernels/mhc/aiter.py`` and ``vllm/_aiter_ops.py``).

AVAILABILITY (evidence, beverin 2026-09-20): the sglang ROCm image's bundled aiter
(commit 7a8ff7dd4, /sgl-workspace/aiter) does NOT ship the symbol —
``from aiter.ops.mhc import mhc_fused_post_pre`` raises ImportError (mi300 in-container
probe, srun step of job 644852) and the module file lists only ``mhc_pre`` /
``mhc_post`` / ``mhc_pre_big_fuse*`` (file grep, srun job 644860). This bridge probes
the symbol per-process and declines with ``OpNotEligible`` until a newer aiter is
present. Fallback path: build aiter from source (~15 min) in its own job with
``AITER_JIT_DIR`` set for JIT-building jobs ONLY — do not shadow the image aiter that
the promoted ``aiter_mhc`` (two-launch) path resolves without re-running that A/B
(aiter-mhc-promotion KB, jobs 644821/644832).

Call contract (mirrors ``vllm/_aiter_ops.py`` ``mhc_fused_post_pre``; floe argument
convention):

* ``x`` ``[n, D]`` bf16 — the sublayer output (tokens flattened).
* ``residual`` ``[n, hc, D]`` bf16 — the current mHC streams.
* ``post`` ``[n, hc]`` (or ``[n, hc, 1]``) fp32, ``comb`` ``[n, hc, hc]`` fp32 — the
  CURRENT section's mixes as produced by this section's pre.
* ``fn`` ``[2hc + hc², hc·D]`` fp32, ``scale`` ``[3]`` fp32, ``base`` ``[2hc + hc²]``
  fp32 — the NEXT section's pre weights (floe: the next ``Glm53HyperConnection``'s
  ``(fn, base, scale)`` in the ``[pre | post | comb]`` split order).
* aiter returns ``(post_mix, comb_mix, layer_input, next_residual)``; this bridge
  returns floe order: ``(streams_new [n, hc, D] bf16, post_new [n, hc] fp32,
  comb_new [n, hc, hc] fp32, collapsed [n, D] bf16)``.
* ``norm_weight`` / ``norm_eps`` optionally fuse the next pre's learned RMSNorm (floe's
  pre uses an unweighted RMSNorm + rescale → pass ``norm_weight=None``; aiter computes
  Σx² in-kernel either way).
* ``hidden % 256 == 0`` (aiter contract, asserted by the kernel).

Every wrapper follows the torch_ops calling convention: an eligibility miss — aiter
unavailable, symbol absent, non-CUDA tensor, wrong dtype/shape/contiguity — raises
``OpNotEligible`` and callers catch it and keep their two-launch/eager path; genuine
aiter kernel failures are NOT swallowed and propagate. Probes are cached; importing
this module stays dependency-free (aiter resolves lazily).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

from ._dispatch import OpNotEligible

__all__ = ["available", "report", "aiter_fused_mhc_post_pre"]


@lru_cache(maxsize=1)
def _mhc_module():
    """The image's ``aiter.ops.mhc`` module when importable, else None (cached)."""
    try:
        import aiter.ops.mhc as mhc  # noqa: F401
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    return mhc


@lru_cache(maxsize=1)
def _fused_fn():
    """The ``mhc_fused_post_pre`` symbol when present, else None (cached)."""
    mhc = _mhc_module()
    if mhc is None:
        return None
    fn = getattr(mhc, "mhc_fused_post_pre", None)
    return fn if callable(fn) else None


@lru_cache(maxsize=1)
def _gfx() -> Optional[str]:
    mhc = _mhc_module()
    if mhc is None:
        return None
    try:
        import aiter

        return aiter.get_gfx()
    except Exception:  # noqa: BLE001
        return None


def available() -> bool:
    """True when aiter imports, reports gfx942, AND ships ``mhc_fused_post_pre``.

    The public availability probe: callers use it to skip pre-work before
    discovering the op itself would decline with ``OpNotEligible``. On the
    current beverin image this is False by design (symbol absent) — see the
    module docstring for the evidence trail.
    """
    return _gfx() == "gfx942" and _fused_fn() is not None


def report() -> dict:
    """Availability table for dispatch reports / bench logs (non-raising)."""
    return {
        "aiter": _mhc_module() is not None,
        "gfx": _gfx(),
        "mhc_fused_post_pre": _fused_fn() is not None,
    }


def _check_bf16(t: torch.Tensor, ndim: int, name: str) -> None:
    if t.dim() != ndim or t.dtype is not torch.bfloat16 or not t.is_cuda or not t.is_contiguous():
        raise OpNotEligible(
            f"aiter_fused_mhc_post_pre: {name} must be contiguous CUDA bf16 "
            f"with {ndim} dims, got dim={t.dim()} dtype={t.dtype} "
            f"cuda={t.is_cuda} contig={t.is_contiguous()}"
        )


def _check_f32(t: torch.Tensor, shape: tuple, name: str) -> None:
    if tuple(t.shape) != shape or t.dtype is not torch.float32 or not t.is_cuda or not t.is_contiguous():
        raise OpNotEligible(
            f"aiter_fused_mhc_post_pre: {name} must be contiguous CUDA fp32 "
            f"of shape {shape}, got {tuple(t.shape)} dtype={t.dtype}"
        )


def aiter_fused_mhc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    sinkhorn_repeat: int,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
):
    """Fused mHC compose (current section) + pre (next section) in one launch.

    Returns ``(streams_new [n, hc, D] bf16, post_new [n, hc] fp32,
    comb_new [n, hc, hc] fp32, collapsed [n, D] bf16)`` — algebraically
    ``mhc_post(x, residual, post, comb)`` followed by
    ``mhc_pre(streams_new, fn, scale, base, ...)``.
    Raises ``OpNotEligible`` on any contract miss (availability, dtype,
    shape, device, contiguity); aiter kernel failures propagate.
    """
    op = _fused_fn()
    if op is None:
        raise OpNotEligible(
            "aiter_fused_mhc_post_pre: aiter.ops.mhc.mhc_fused_post_pre absent "
            f"(image aiter 7a8ff7dd4 predates the fused op); report={report()}"
        )
    _check_bf16(x, 2, "x")
    _check_bf16(residual, 3, "residual")
    n, hc, hidden = residual.shape
    if x.shape[0] != n or x.shape[1] != hidden:
        raise OpNotEligible(
            f"aiter_fused_mhc_post_pre: x {tuple(x.shape)} inconsistent with "
            f"residual {tuple(residual.shape)} (want [n, D], [n, hc, D])"
        )
    if hidden % 256 != 0:
        raise OpNotEligible(f"aiter_fused_mhc_post_pre: hidden {hidden} not divisible by 256")
    mix = 2 * hc + hc * hc
    _check_f32(fn, (mix, hc * hidden), "fn")
    _check_f32(scale, (3,), "scale")
    _check_f32(base, (mix,), "base")
    if post.dim() == 3 and post.shape[-1] == 1:
        post = post.squeeze(-1)
    _check_f32(post, (n, hc), "post")
    _check_f32(comb, (n, hc, hc), "comb")
    if norm_weight is not None and (
        norm_weight.dim() != 1 or norm_weight.dtype is not torch.float32
        or not norm_weight.is_cuda or not norm_weight.is_contiguous()
    ):
        raise OpNotEligible(
            f"aiter_fused_mhc_post_pre: norm_weight must be contiguous CUDA fp32 1-D, "
            f"got dim={norm_weight.dim()} dtype={norm_weight.dtype}"
        )
    if n == 0:
        return (
            torch.empty_like(residual),
            torch.empty(n, hc, dtype=torch.float32, device=residual.device),
            torch.empty(n, hc, hc, dtype=torch.float32, device=residual.device),
            torch.empty(n, hidden, dtype=torch.bfloat16, device=residual.device),
        )

    post_flat = post if post.is_contiguous() else post.contiguous()
    args = (
        x,
        residual,
        post_flat,
        comb,
        fn,
        scale,
        base,
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        int(sinkhorn_repeat),
    )
    with torch.device(residual.device):
        if norm_weight is None:
            post_mix, comb_mix, layer_input, next_residual = op(*args)
        else:
            post_mix, comb_mix, layer_input, next_residual = op(*args, norm_weight, float(norm_eps))
    return next_residual, post_mix, comb_mix, layer_input
