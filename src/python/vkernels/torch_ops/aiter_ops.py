"""AMD AITER op bridge (gfx942) for the mHC / grouped-fp8-MoE dispatch sites.

vLLM's GLM-5.3-Flash ROCm serving path (recipes.vllm.ai, ``VLLM_ROCM_USE_AITER=1``,
integration PR #53906) dispatches the same mHC pre/compose and grouped-fp8-MoE
sites that floe owns to AMD's AITER kernels. This module wraps the ops that
exist in the sglang ROCm image's bundled aiter (verified in-container on
beverin: aiter ``7a8ff7dd4``, ROCm 7.0.51831, MI300A/gfx942, triton 3.4.0)
behind availability probes so each site can A/B them against the vkernels
HIP/Triton kernels and flip per-site knobs (``aiter_mhc`` / ``aiter_moe``).

Contracts (shapes follow ``aiter.ops.mhc`` and floe's ``Glm53`` modules):

* :func:`aiter_mhc_pre` — ``residual [n, hc, D]`` bf16 (the mHC streams,
  batch-flattened), ``fn [2*hc + hc*hc, hc*D]`` fp32 (floe's ``(fn, base,
  scale)`` triple rows in the ``[hc | hc | hc*hc]`` split order that matches
  ``logits.split([hc, hc, hc*hc], dim=-1)``). Returns
  ``(post_mix [n, hc, 1] fp32, comb_mix [n, hc, hc] fp32, layer_input [n, D]
  bf16)`` — the raw mix GEMM, the sigmoid gating, the Sinkhorn comb
  projection and (with ``norm_weight``) the learned RMSNorm fused into two
  launches (``mhc_pre_gemm_sqrsum`` + ``mhc_pre_big_fuse[_rmsnorm]``).
  NUMERICS GATE: floe's comb normalizes ``softmax(logits) + eps`` and then
  runs ``iters - 1`` alternating row/col divisions; aiter's Sinkhorn variant
  and gate rounding are aiter's own — the ``aiter_mhc`` knob stays opt-in
  until ``bench_aiter_ab.py`` shows parity against floe's eager reference.
* :func:`aiter_mhc_post` — ``out[j] = post[j]·x + Σ_k comb[k,j]·residual[k]``
  with fp32 mixes and bf16 data: identical algebra to vkernels' HIP
  ``mhc_post`` (vkernels #69 audit), different argument order.
* :func:`fp8_blockscale_experts` — grouped fp8-blockscale expert GEMMs via
  ``moe_align_block_size`` + ``ck_moe_stage1_fwd`` + ``ck_moe_stage2_fwd``
  with ``activation=None``: the aiter fused ``fmoe_fp8_blockscale_g1u1``
  applies plain SiLU and cannot express GLM's ``swiglu_limit`` clamp, so the
  caller injects its swiglu between the stages instead. Weights must be
  ``float8_e4m3fnuz`` on CDNA3 (convert once per weight version with
  :func:`glm_fp8_blockwise_gemm.e4m3fn_to_fnuz`; aiter's ``dtypes.fp8`` is
  fnuz on gfx942, the checkpoint flavour is e4m3fn).

Every wrapper follows the torch_ops calling convention (``docs/torch-ops.md``
§1 / ``_dispatch.py``): an eligibility miss — aiter unavailable, non-CUDA
tensor, wrong dtype/shape/contiguity — raises ``OpNotEligible`` and callers
catch it and keep their eager path; genuine aiter kernel failures are NOT
swallowed and propagate. ``aiter_available()`` / ``available()`` / ``report()``
stay non-raising availability probes. Importing this module stays
dependency-free (aiter resolves lazily, probes are cached).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import torch

from ._dispatch import OpNotEligible

__all__ = [
    "available",
    "report",
    "aiter_mhc_pre",
    "aiter_mhc_post",
    "aiter_available",
    "moe_align_block_size",
    "ck_moe_stage1",
    "ck_moe_stage2",
    "fp8_blockscale_experts",
]


@lru_cache(maxsize=1)
def _aiter():
    """The aiter module when importable, else None (cached)."""
    try:
        import aiter  # noqa: F401
        import aiter.ops.mhc  # noqa: F401  (presence probe for the jit core)
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    return aiter


def aiter_available() -> bool:
    """Whether the AMD aiter kernels are importable on this host (cached).

    The public availability probe for the ``aiter_*`` ops: ROCm-only, so
    callers use it to skip expensive pre-work (weight conversions, caches)
    before discovering the op itself would decline with ``OpNotEligible``.
    """
    return _aiter() is not None


@lru_cache(maxsize=1)
def _gfx() -> Optional[str]:
    a = _aiter()
    if a is None:
        return None
    try:
        return a.get_gfx()
    except Exception:  # noqa: BLE001
        return None


def available() -> bool:
    """True when aiter imports and reports a gfx942 device."""
    return _gfx() == "gfx942"


def report() -> dict:
    """Availability table for the dispatch report / bench logs."""
    a = _aiter()
    if a is None:
        return {"aiter": False}
    out: dict = {"aiter": True, "gfx": _gfx()}
    import aiter.ops.mhc as mhc
    out["mhc"] = {
        "mhc_pre": hasattr(mhc, "mhc_pre"),
        "mhc_post": hasattr(mhc, "mhc_post"),
        "mhc_pre_gemm_sqrsum": hasattr(mhc, "mhc_pre_gemm_sqrsum"),
        "mhc_pre_big_fuse": hasattr(mhc, "mhc_pre_big_fuse"),
        "mhc_pre_big_fuse_rmsnorm": hasattr(mhc, "mhc_pre_big_fuse_rmsnorm"),
    }
    out["moe"] = {
        name: hasattr(a, name)
        for name in ("moe_align_block_size", "ck_moe_stage1_fwd", "ck_moe_stage2_fwd",
                     "fmoe_fp8_blockscale_g1u1", "per_token_quant_hip")
    }
    try:
        import aiter.ops.quant as quant
        out["moe"]["per_group_quant_hip"] = hasattr(quant, "per_group_quant_hip")
    except ImportError:
        out["moe"]["per_group_quant_hip"] = False
    return out


# ---------------------------------------------------------------------------
# mHC
# ---------------------------------------------------------------------------

def aiter_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
):
    """Fused mHC pre via ``aiter.ops.mhc.mhc_pre``.

    ``residual`` [n, hc, D] bf16 contiguous (floe's ``streams`` flattened to
    [B*S, hc, D]); ``fn`` [2*hc + hc*hc, hc*D] fp32 contiguous (pre rows,
    post rows, comb mixer rows — the ``[hc | hc | hc*hc]`` split order).
    Returns ``(post_mix [n, hc, 1] fp32, comb_mix [n, hc, hc] fp32,
    layer_input [n, D] bf16)``. Raises ``OpNotEligible`` outside the
    contract (aiter unavailable, CPU tensor, wrong dtype/shape/contiguity);
    genuine aiter kernel failures propagate.

    ``hc_post_mult_value`` is floe's post gate multiplier (``2 * sigmoid``),
    ``sinkhorn_repeat`` floe's ``hc_sinkhorn_iters``. With ``norm_weight``
    the learned RMSNorm is fused (``mhc_pre_big_fuse_rmsnorm``); without it
    the plain big-fuse runs. NOTE: aiter's layer_input is the *unnormalized*
    (input-RMS-rescaled) stream collapse; floe applies its learned norm at
    the consumer when one exists.
    """
    if residual.dtype is not torch.bfloat16:
        raise OpNotEligible(
            f"aiter_mhc_pre: residual must be bf16, got {residual.dtype}")
    if residual.dim() != 3:
        raise OpNotEligible(
            "aiter_mhc_pre: residual must be [n, hc, D], got shape "
            f"{tuple(residual.shape)}")
    if not residual.is_contiguous() or not fn.is_contiguous():
        raise OpNotEligible(
            "aiter_mhc_pre: residual and fn must be contiguous")
    n, hc, d = residual.shape
    if fn.dim() != 2 or fn.shape[0] != 2 * hc + hc * hc or fn.shape[1] != hc * d:
        raise OpNotEligible(
            f"aiter_mhc_pre: fn must be [2*hc + hc*hc, hc*D] = "
            f"[{2 * hc + hc * hc}, {hc * d}] for residual {tuple(residual.shape)}, "
            f"got shape {tuple(fn.shape)}")
    if fn.dtype is not torch.float32:
        raise OpNotEligible(f"aiter_mhc_pre: fn must be fp32, got {fn.dtype}")
    a = _aiter()
    if a is None:
        raise OpNotEligible(
            "aiter_mhc_pre: aiter is not importable on this host "
            "(ROCm-only gfx942 bridge); not eligible")
    if not residual.is_cuda:
        raise OpNotEligible(
            "aiter_mhc_pre: residual must be a CUDA tensor, got device "
            f"{residual.device}")
    mhc = a.ops.mhc
    return mhc.mhc_pre(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, norm_weight, norm_eps,
    )


def aiter_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: Optional[torch.Tensor] = None,
):
    """``streams[j] = post[j]·x + Σ_k comb[k,j]·residual[k]`` via ``aiter.ops.mhc.mhc_post``.

    ``x`` [n, D] bf16 (sublayer output), ``residual`` [n, hc, D] bf16,
    ``post_layer_mix`` [n, hc(, 1)] fp32, ``comb_res_mix`` [n, hc, hc] fp32.
    Returns the composed streams ``out`` [n, hc, D] bf16 (allocated when not
    given — the FULL stream tensor, matching vLLM's ``mhc_post`` wrapper:
    ``empty_like(residual_flat)`` — not just the sublayer shape). Raises
    ``OpNotEligible`` outside the contract (aiter unavailable, CPU tensor,
    wrong dtype/shape/contiguity); genuine aiter kernel failures propagate.
    Same algebra as vkernels' HIP ``mhc_post``.
    """
    if residual.dtype is not torch.bfloat16:
        raise OpNotEligible(
            f"aiter_mhc_post: residual must be bf16, got {residual.dtype}")
    if x.dtype is not torch.bfloat16:
        raise OpNotEligible(f"aiter_mhc_post: x must be bf16, got {x.dtype}")
    if not x.is_contiguous() or not residual.is_contiguous():
        raise OpNotEligible("aiter_mhc_post: x and residual must be contiguous")
    if residual.dim() != 3:
        raise OpNotEligible(
            "aiter_mhc_post: residual must be [n, hc, D], got shape "
            f"{tuple(residual.shape)}")
    if x.shape[0] != residual.shape[0]:
        raise OpNotEligible(
            f"aiter_mhc_post: x has {x.shape[0]} rows but residual has "
            f"{residual.shape[0]}")
    n, hc, d = residual.shape
    if d % 256:
        raise OpNotEligible(
            f"aiter_mhc_post: aiter asserts hidden_size % 256 == 0, got D={d}")
    if post_layer_mix.dtype is not torch.float32 or comb_res_mix.dtype is not torch.float32:
        raise OpNotEligible(
            "aiter_mhc_post: post_layer_mix and comb_res_mix must be fp32, "
            f"got {post_layer_mix.dtype} / {comb_res_mix.dtype}")
    post = post_layer_mix.reshape(n, hc, 1) if post_layer_mix.dim() == 2 else post_layer_mix
    if tuple(post.shape) != (n, hc, 1) or tuple(comb_res_mix.shape) != (n, hc, hc):
        raise OpNotEligible(
            f"aiter_mhc_post: mixes must be [n, hc(, 1)]=[{n}, {hc}(, 1)] and "
            f"[n, hc, hc]=[{n}, {hc}, {hc}], got "
            f"{tuple(post.shape)} / {tuple(comb_res_mix.shape)}")
    if out is None:
        out = torch.empty_like(residual)
    elif out.shape != residual.shape or out.dtype is not torch.bfloat16:
        raise OpNotEligible(
            "aiter_mhc_post: out must be [n, hc, D] bf16 like residual, got "
            f"{tuple(out.shape)} {out.dtype}")
    a = _aiter()
    if a is None:
        raise OpNotEligible(
            "aiter_mhc_post: aiter is not importable on this host "
            "(ROCm-only gfx942 bridge); not eligible")
    if not residual.is_cuda:
        raise OpNotEligible(
            "aiter_mhc_post: residual must be a CUDA tensor, got device "
            f"{residual.device}")
    a.ops.mhc.mhc_post(out, x.reshape(n, d), residual, post, comb_res_mix)
    return out


# ---------------------------------------------------------------------------
# grouped fp8-blockscale MoE (stage split so the caller can apply swiglu_limit)
# ---------------------------------------------------------------------------

def per_group_quant_fp8(x: torch.Tensor, group_size: int = 128):
    """Per-token-group fp8 (fnuz on gfx942) quantization along the last dim.

    aiter's ``per_group_quant_hip`` — the activation quant format the CK
    blockscale MoE kernels expect (``QuantType.per_1x128``): scale is
    ``[..., K/group_size]`` fp32 (NOT the whole-row per-token scalar the
    per-token quant returns — that one is for per_Token kernels and gives
    wrong results with per_1x128 GEMMs). Returns ``(x_q, scale)``. Raises
    ``OpNotEligible`` outside the contract (aiter unavailable, CPU tensor,
    non-contiguous input, last dim not a multiple of ``group_size``);
    genuine aiter kernel failures propagate.
    """
    if x.shape[-1] % group_size:
        raise OpNotEligible(
            f"per_group_quant_fp8: last dim {x.shape[-1]} must be a multiple "
            f"of group_size {group_size}")
    if not x.is_contiguous():
        raise OpNotEligible("per_group_quant_fp8: x must be contiguous")
    a = _aiter()
    if a is None:
        raise OpNotEligible(
            "per_group_quant_fp8: aiter is not importable on this host "
            "(ROCm-only gfx942 bridge); not eligible")
    if not x.is_cuda:
        raise OpNotEligible(
            f"per_group_quant_fp8: x must be a CUDA tensor, got device {x.device}")
    import aiter.ops.quant as quant
    return quant.per_group_quant_hip(
        x, quant_dtype=a.dtypes.fp8, group_size=group_size,
        transpose_scale=False)


def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """aiter's ``moe_align_block_size`` — signature (ground truth from
    aiter/ops/moe_op.py): ``(topk_ids, num_experts, block_size,
    sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad)``.
    ``sorted_token_ids`` holds flattened ``token*topk + slot`` indices
    padded to block multiples with the ``T*topk`` sentinel; ``experts_ids``
    is per-block expert ids. Returns ``(sorted_token_ids, experts_ids,
    token_nums, num_tokens_post_pad)``. Raises ``OpNotEligible`` outside
    the contract (aiter unavailable, non-CUDA ``topk_ids``, ``topk_ids``
    not [T, topk]); genuine aiter kernel failures propagate."""
    if topk_ids.dim() != 2:
        raise OpNotEligible(
            "moe_align_block_size: topk_ids must be [T, topk], got shape "
            f"{tuple(topk_ids.shape)}")
    a = _aiter()
    if a is None:
        raise OpNotEligible(
            "moe_align_block_size: aiter is not importable on this host "
            "(ROCm-only gfx942 bridge); not eligible")
    if not topk_ids.is_cuda:
        raise OpNotEligible(
            "moe_align_block_size: topk_ids must be a CUDA tensor, got device "
            f"{topk_ids.device}")
    t, k = topk_ids.shape
    dev = topk_ids.device
    max_padded = t * k + num_experts * (block_size - 1)
    m_blocks = (max_padded + block_size - 1) // block_size
    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
    experts_ids = torch.empty(m_blocks, dtype=torch.int32, device=dev)
    token_nums = torch.empty(num_experts, dtype=torch.int32, device=dev)
    num_post = torch.empty(1, dtype=torch.int32, device=dev)
    a.moe_align_block_size(topk_ids.to(torch.int32), num_experts, block_size,
                           sorted_token_ids, experts_ids, token_nums, num_post)
    return sorted_token_ids, experts_ids, token_nums, num_post


def ck_moe_stage1(x_q, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, *,
                  topk: int, out=None, w1_scale=None, a1_scale=None, block_m: int = 32):
    """Grouped stage-1 GEMM (gate|up), activation DISABLED — the caller
    applies floe's clamp-swiglu between the stages (aiter's in-kernel
    ``Swiglu`` hardcodes alpha=1.702 and the +1 up-bias, which does NOT
    match GLM-5.3's ``silu(clamp(g))·clamp(u)``).

    Ground-truth raw signature (aiter/ops/moe_op.py + fused_moe.py call):
    ``(hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids,
    num_valid_ids, out, topk, kernelName, w1_scale, a1_scale, block_m,
    sorted_weights, quant_type, activation, splitk, use_non_temporal_load,
    dst_type)`` — BOTH weight stacks are passed (JIT metadata). The kernel
    SCATTERS results to token-major ``out`` rows via the sorted_ids values.
    ``w1`` [E, 2I, H] fp8-fnuz with per-128x128 ``w1_scale`` [E, 2I/128,
    H/128]; ``a1_scale`` [T, H/128] fp32 (per_1x128). Returns ``out``
    [T*topk, 2I] bf16. Raises ``OpNotEligible`` outside the contract
    (aiter unavailable, wrong w1/w2 rank or odd 2I); genuine aiter kernel
    failures propagate."""
    if w1.dim() != 3 or w2.dim() != 3 or w1.shape[1] % 2:
        raise OpNotEligible(
            "ck_moe_stage1: w1/w2 must be 3-D [E, 2I, H] / [E, H, I] with "
            f"an even gate|up dim, got {tuple(w1.shape)} / {tuple(w2.shape)}")
    a = _aiter()
    if a is None:
        raise OpNotEligible(
            "ck_moe_stage1: aiter is not importable on this host "
            "(ROCm-only gfx942 bridge); not eligible")
    if not x_q.is_cuda:
        raise OpNotEligible(
            f"ck_moe_stage1: x_q must be a CUDA tensor, got device {x_q.device}")
    if out is None:
        out = torch.empty((x_q.shape[0] * topk, w1.shape[1]),
                          dtype=torch.bfloat16, device=x_q.device)
    a.ck_moe_stage1_fwd(
        x_q, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out,
        topk, "", w1_scale, a1_scale, block_m, None,
        a.QuantType.per_1x128, a.ActivationType.No)
    return out


def ck_moe_stage2(inter_q, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, *,
                  topk: int, out=None, w2_scale=None, a2_scale=None, block_m: int = 32,
                  sorted_weights=None):
    """Grouped stage-2 GEMM (down) with in-kernel routed-weight multiply
    (``sorted_weights`` per sorted slot, fp32) and token-major accumulation
    into ``out`` [T, H] — the caller zero-initializes it. Raw signature:
    ``(inter_states, w1, w2, sorted_token_ids, sorted_expert_ids,
    num_valid_ids, out, topk, kernelName, w2_scale, a2_scale, block_m,
    sorted_weights, quant_type, activation, use_non_temporal_load)``.
    Raises ``OpNotEligible`` when aiter is unavailable; genuine aiter
    kernel failures propagate."""
    a = _aiter()
    if a is None:
        raise OpNotEligible(
            "ck_moe_stage2: aiter is not importable on this host "
            "(ROCm-only gfx942 bridge); not eligible")
    if not inter_q.is_cuda:
        raise OpNotEligible(
            f"ck_moe_stage2: inter_q must be a CUDA tensor, got device {inter_q.device}")
    if out is None:
        out = torch.zeros((inter_q.shape[0] // topk, w2.shape[1]),
                          dtype=torch.bfloat16, device=inter_q.device)
    a.ck_moe_stage2_fwd(
        inter_q, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out,
        topk, "", w2_scale, a2_scale, block_m, sorted_weights,
        a.QuantType.per_1x128, a.ActivationType.No)
    return out


def fp8_blockscale_experts(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down: torch.Tensor,
    down_scale: torch.Tensor,
    topk_index: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    swiglu: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    swiglu_limit: Optional[float] = None,
    block_m: int = 32,
) -> torch.Tensor:
    """Full grouped fp8-blockscale expert pipeline (EXPERIMENTAL — parity gate
    in bench_aiter_ab.py must pass before any knob flips).

    BLOCKED (2026-09-20, beverin container): the CK ck2stages codegen in the
    sglang image cannot emit an activation-free stage1 — gen_instances.py's
    argparse accepts only silu/gelu and the C++ dispatch enum is
    ``ActivationType { Silu=0, Gelu=1 }`` (aiter_enum.h; no No/identity
    value exists). GLM-5.3 requires silu(clamp(g))·clamp(u) with
    swiglu_limit=10 — aiter's in-kernel Swiglu is the gpt-oss formula
    (α=1.702, +1 bias) and plain silu drops the clamp, so neither is
    exact. This wrapper stays for a future container that adds a No
    epilogue (or for gfx950's flydsl SEPARATED-gate path); do not spend
    time re-diagnosing the JIT retry loop it currently hits
    ('gemm_moe_ck2stages_lookup.h' not found ⇒ codegen gap, see the
    aiter-jit build dir + ninja log).

    Flow (ground truth from aiter fused_moe.py ``fused_moe_2stages``):
      x [T, H] bf16 → per_1x128 group quant → a1 [T, H] fp8-fnuz,
      a1_scale [T, H/128] fp32
      → moe_align (sorted_ids hold flattened token*topk+slot; T*topk sentinel)
      → CK stage1 (activation=No) SCATTERS gate|up to token-major
        a2 [T*topk, 2I] bf16
      → floe's clamp-swiglu on bf16 (in torch — aiter's in-kernel Swiglu is
        alpha=1.702 + up-bias, NOT GLM-5.3's silu-clamp form)
      → group requant → a2q [T*topk, I], a2_scale [T*topk, I/128]
      → CK stage2 with in-kernel routed-weight multiply (sorted_weights
        gathered per sorted slot, sentinel-masked) → out [T, H] bf16.
    No unsorted-restore is needed: both kernels scatter/gather token-major
    via the sorted_ids values themselves.

    ``gate_up`` [E, 2I, H], ``down`` [E, H, I] fp8-fnuz with per-128x128
    block scales; ``swiglu(gate, up)`` applies the caller's activation.
    Returns ``[T, H]`` bf16 (routing weights applied, tokens summed).
    Raises ``OpNotEligible`` outside the contract (aiter unavailable, CPU
    tensor, wrong dtype/shape); genuine aiter kernel failures propagate —
    including the ``OpNotEligible`` of the internal quant/align/stage ops.
    """
    if x.dtype is not torch.bfloat16:
        raise OpNotEligible(
            f"fp8_blockscale_experts: x must be bf16, got {x.dtype}")
    if topk_index.dim() != 2:
        raise OpNotEligible(
            "fp8_blockscale_experts: topk_index must be [T, topk], got shape "
            f"{tuple(topk_index.shape)}")
    t, k = topk_index.shape
    e = gate_up.shape[0]
    if down.shape[0] != e or topk_weights.shape != topk_index.shape:
        raise OpNotEligible(
            f"fp8_blockscale_experts: down must have E={e} experts matching "
            f"gate_up and topk_weights must be [T, topk] like topk_index, "
            f"got down E={down.shape[0]}, topk_weights "
            f"{tuple(topk_weights.shape)}")
    if 2 * x.shape[-1] % 128 or down.shape[1] % 128:
        raise OpNotEligible(
            "fp8_blockscale_experts: gate|up (2*H) and down I must be "
            f"multiples of 128, got H={x.shape[-1]}, I={down.shape[1]}")
    a = _aiter()
    if a is None:
        raise OpNotEligible(
            "fp8_blockscale_experts: aiter is not importable on this host "
            "(ROCm-only gfx942 bridge); not eligible")
    if not x.is_cuda:
        raise OpNotEligible(
            f"fp8_blockscale_experts: x must be a CUDA tensor, got device {x.device}")
    # 1) activation quant (per_1x128 — the CK blockscale format)
    a1, a1_scale = per_group_quant_fp8(x)
    # 2) routing alignment
    sorted_ids, expert_ids, _token_nums, num_post = moe_align_block_size(
        topk_index, e, block_m)
    # 3) stage 1: gate|up, no activation, scattered token-major [T*topk, 2I]
    s1 = ck_moe_stage1(a1, gate_up, down, sorted_ids, expert_ids, num_post,
                       topk=k, w1_scale=gate_up_scale, a1_scale=a1_scale,
                       block_m=block_m)
    # 4) floe's clamp-swiglu on the bf16 intermediate
    gate, up = s1.view(t, k, -1).chunk(2, dim=-1)
    act = swiglu(gate, up).contiguous().view(t * k, -1)
    # 5) requant the activated intermediate
    act_q, a2_scale = per_group_quant_fp8(act)
    # 6) stage 2 with in-kernel routed weights; sentinel slots contribute 0
    flat_w = topk_weights.reshape(-1).to(torch.float32)
    valid = sorted_ids < t * k
    sorted_w = torch.where(
        valid, flat_w[sorted_ids.clamp(0, t * k - 1)],
        torch.zeros((), dtype=torch.float32, device=x.device))
    out = torch.zeros((t, x.shape[-1]), dtype=torch.bfloat16, device=x.device)
    ck_moe_stage2(act_q, gate_up, down, sorted_ids, expert_ids, num_post,
                  topk=k, out=out, w2_scale=down_scale, a2_scale=a2_scale,
                  block_m=block_m, sorted_weights=sorted_w)
    return out
