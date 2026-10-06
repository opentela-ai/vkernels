"""DeepGEMM masked grouped-GEMM borrow — the W8A8 MoE decode band (lane 31).

Wraps DeepGEMM's JIT ``m_grouped_fp8_gemm_nt_masked`` (the donor SGLang
v0.5.20 glm5 entrypoint behind ``--enable-dp-lm-head``-era decode MoE on
sm90) as a vkernels op shaped like the sgl_moe adoption: same blockwise
fp8-e4m3fn weight stacks, same per-token-group-128 activation quant, same
swiglu/combine seams — but the two GEMM stages run as DeepGEMM masked
grouped kernels instead of the Triton fused-MoE tiles.

Why masked-grouped (donor argument): DeepGEMM's SM90 kernels are the
WGMMA fp8 paths SGLang serves glm5 decode with; the masked layout gives
each expert a contiguous ``[max_m, K]`` operand window whose live row
count lives in a DEVICE ``masked_m`` int32 tensor — no host sync, no
padded-tile launch fan-out (the sgl fused kernel's known decode loss is
exactly a 64-row-tile problem), and the fp8 WGMMA dots at the k-block
granularity the checkpoint's 128x128 scales already provide.

HARD CAPABILITY GATE (this lane's central finding): DeepGEMM dispatches
on ``device.get_arch_major()`` and implements only 9 (sm90, wgmma fp8)
and 10 (sm100, tcgen05). Any other major hits
``csrc/utils/layout.hpp: DG_HOST_UNREACHABLE("Unknown recipe")`` at the
first kernel call. GB10 (this lane's bench box) reports capability
(12, 1) — sm_121 — and ptxas rejects BOTH ISA families there:

  * ``wgmma.mma_async.f32.e4m3`` -> "Instruction 'wgmma.mma_async with
    FP8 types' not supported on .target 'sm_121'"
  * ``tcgen05.alloc.cta_group::1`` -> "Instruction 'tcgen05.alloc' not
    supported on .target 'sm_121'"

so no DeepGEMM kernel can JIT for sm_121, and the eligibility checks
below return False BEFORE any deep_gemm call on such devices. The
module therefore imports cleanly everywhere (stdlib+torch at module
import; deep_gemm lazily inside the gated paths) and every consumer
keeps its landed fallback (sgl_fused_moe -> grouped -> expert_gemv).

Capture contract (the inherited 16384-shape 15-50-min JIT warmup makes
this non-negotiable): deep_gemm compiles at FIRST CALL per (arch,
shape-class, config). :func:`warmup_deepgemm_moe` compiles every stage
a serve will capture EAGERLY, before any graph capture; the op itself
never triggers a cold compile mid-capture because the tuning-cache
config resolution (:mod:`vkernels.tuning.cache`) serves the memo or the
declared default while a stream captures, and the JIT cache (disk) is
populated by the warmup call. Elapsed warmup time is logged once.

Tuning-cache integration (lane-30 op tier, ``vkernels/tuning/cache.py``):
op ``"deepgemm_moe"``, decode-band shape classes ``t2``/``t4``/``t8``
(the tree's decode T<=8 convention; top_k=8 so slots == 8T), config
vocabulary ``{"expected_m_mult": float}`` — DeepGEMM's own heuristic
sizes its block schedule from the caller-supplied ``expected_m``, so
the multiplier is the one caller-owned knob. Seeded sm90 prior in
:data:`SM90_SEED` (donor defaults; a locally tuned record outranks it).

Stdlib+torch at import; deep_gemm and triton lazily inside the gated
paths (CPU-only hosts and GB10 import this module fine).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple

import torch

__all__ = [
    "SM90_SEED",
    "deepgemm_grouped_moe",
    "deepgemm_moe_eligible",
    "deepgemm_unavailable_reason",
    "seed_deepgemm_defaults",
    "warmup_deepgemm_moe",
]

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn

# The donor prior (status "seeded"): DeepGEMM's own masked-GEMM test suite
# and SGLang's call both size expected_m at ~1.2x the mean group load.
SM90_SEED = {
    "t2": {"config": {"expected_m_mult": 1.2}, "origin": "lane31-deepgemm-borrow (donor default)"},
    "t4": {"config": {"expected_m_mult": 1.2}, "origin": "lane31-deepgemm-borrow (donor default)"},
    "t8": {"config": {"expected_m_mult": 1.2}, "origin": "lane31-deepgemm-borrow (donor default)"},
}


# ---------------------------------------------------------------------------
# capability gate (cheap; NO deep_gemm import on ineligible archs)
# ---------------------------------------------------------------------------
def deepgemm_unavailable_reason(device=None) -> Optional[str]:
    """None when DeepGEMM masked-grouped GEMMs can run on ``device``'s arch,
    else a one-line human reason (logged once by the first eligibility
    miss). Import failure and arch rejection are both reported here —
    never raised from the hot path."""
    try:
        import deep_gemm  # noqa: F401
    except Exception as exc:
        return f"deep_gemm import failed: {exc!r}"
    if device is None:
        if not torch.cuda.is_available():
            return "no CUDA device"
        device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    if major == 9:
        return None  # sm90: the donor path (wgmma fp8, fp32 scales)
    if major == 10:
        return None  # sm100: tcgen05 path (int ue8m0 scales)
    return (
        f"DeepGEMM dispatches only on arch major 9/10; this device is "
        f"capability ({major}, {minor}) — sm_{major}{minor} supports neither "
        f"wgmma fp8 (sm90) nor tcgen05 (sm100) MMA, so the JIT cannot emit "
        f"any DeepGEMM kernel here (ptxas-rejected; lane-31 evidence)"
    )


def deepgemm_available(device=None) -> bool:
    return deepgemm_unavailable_reason(device) is None


# ---------------------------------------------------------------------------
# contract (the sgl_fused_moe_eligible shape/dtype contract + the gate)
# ---------------------------------------------------------------------------
def deepgemm_moe_eligible(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_scale: Optional[torch.Tensor],
    down_proj: torch.Tensor,
    down_scale: Optional[torch.Tensor],
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> bool:
    """Same weight-stack contract as :func:`sgl_fused_moe.sgl_fused_moe_eligible`
    (fp8-e4m3fn [E, 2I, H] / [E, H, I] with fp32 128-block scales, bf16
    activations, int routing), PLUS the DeepGEMM capability gate and the
    decode-band limits: T <= 8 (top_k 8 -> slots <= 64 == max_m cap of
    the masked layout this wrapper builds) and the fp32-scale variant the
    sm90 kernels consume."""
    if not deepgemm_available(x.device if x.is_cuda else None):
        return False
    e = gate_up_proj.shape[0] if gate_up_proj.dim() == 3 else 0
    return bool(
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and 0 < x.shape[0] <= 8
        and gate_up_proj.dim() == 3
        and down_proj.dim() == 3
        and gate_up_proj.dtype == _FP8
        and down_proj.dtype == _FP8
        and gate_up_proj.is_contiguous()
        and down_proj.is_contiguous()
        and gate_up_scale is not None
        and down_scale is not None
        and gate_up_scale.dtype == torch.float32
        and down_scale.dtype == torch.float32
        and gate_up_proj.shape[0] == down_proj.shape[0] == e
        and e > 0
        and gate_up_proj.shape[2] == x.shape[1]
        and down_proj.shape[1] == x.shape[1]
        and down_proj.shape[2] * 2 == gate_up_proj.shape[1]
        and gate_up_proj.shape[1] % 128 == 0
        and gate_up_proj.shape[2] % 128 == 0
        and down_proj.shape[1] % 128 == 0
        and down_proj.shape[2] % 128 == 0
        and gate_up_scale.shape
        == (e, gate_up_proj.shape[1] // 128, gate_up_proj.shape[2] // 128)
        and down_scale.shape == (e, down_proj.shape[1] // 128, down_proj.shape[2] // 128)
        and top_k_index.shape == top_k_weights.shape
        and top_k_index.shape[0] == x.shape[0]
        and not top_k_index.dtype.is_floating_point
        and top_k_weights.is_floating_point()
        and top_k_index.shape[1] > 0
        and x.shape[0] * top_k_index.shape[1] <= 64
    )


# ---------------------------------------------------------------------------
# deep_gemm call shims (signature drift between deep_gemm 1.x/2.x)
# ---------------------------------------------------------------------------
def _masked_fp8_gemm(a, sfa, b, sfb, d, masked_m, expected_m):
    """One versioned call into ``m_grouped_fp8_gemm_nt_masked``.

    Donor (SGLang v0.5.20 + deep_gemm 1.x-era): positional
    ``(a, sfa), (b, sfb), d, masked_m, expected_m`` — kept primary. The
    2.x tree added recipe kwargs; the fp32-scale sm90 legacy format is
    still the default there, so the plain call stands. A TypeError (a
    renamed positional) falls through to the keyword spelling so a newer
    donor image keeps working; anything else propagates (wrong numerics
    must never be hidden behind a fallback).
    """
    import deep_gemm

    try:
        return deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (a, sfa), (b, sfb), d, masked_m, expected_m
        )
    except TypeError as exc:
        if "positional" not in str(exc) and "argument" not in str(exc):
            raise
        return deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a=a, b=b, out=d, masked_m=masked_m, expected_m=expected_m,
            disable_ue8m0_cast=True,
        )


def _tma_aligned(sf: torch.Tensor) -> torch.Tensor:
    """sm90 masked GEMMs consume TMA-aligned (column-major-padded) scale
    factors; deep_gemm ships the transform. Absent (renamed) -> pass
    through and let deep_gemm's own layout check speak."""
    import deep_gemm

    fn = getattr(deep_gemm, "get_col_major_tma_aligned_tensor", None)
    if fn is None:
        return sf
    return fn(sf)


# ---------------------------------------------------------------------------
# the op
# ---------------------------------------------------------------------------
def deepgemm_grouped_moe(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_proj: torch.Tensor,
    down_scale: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    swiglu_limit: float = math.inf,
) -> torch.Tensor:
    """Routed-expert MLP via two DeepGEMM masked grouped GEMMs.

    Layout (decode band, T <= 8, top_k = 8): each expert j gets a
    ``[max_m = T, K]`` operand window in a statically-shaped ``[E, T, K]``
    tensor; ``masked_m[j]`` (device int32) is its live row count and
    ``expected_m`` a host int. Routing prep is the static-shape
    ``_route`` below: stable argsort of the flat expert ids, a histogram
    for masked_m, and a within-group rank — all device ops on static
    shapes, no bincount, no ``.item()``, no host syncs (CUDA-graph
    capturable; buffers ride the graph pool).

    Stage 1 ``[E,T,2I] = masked_m-masked (a, w13)`` -> grouped swiglu ->
    re-quant -> Stage 2 ``[E,T,H]`` -> inverse-gather to the [T, topk, H]
    slot order -> :func:`moe_combine.moe_weighted_sum`. The between-stages
    activation and the combine reuse the package's landed ops
    (``elementwise.swiglu_limit`` / ``moe_combine.moe_weighted_sum``) —
    the same seams the sgl_moe adoption uses, so the two donors differ
    ONLY in the two GEMMs.

    Raises :class:`OpNotEligible` on any contract miss (callers gate with
    :func:`deepgemm_moe_eligible` and fall through to sgl_fused_moe /
    the landed grouped + expert_gemv ladder).
    """
    from ._dispatch import OpNotEligible

    if not deepgemm_moe_eligible(x, gate_up_proj, gate_up_scale, down_proj,
                                 down_scale, top_k_index, top_k_weights):
        raise OpNotEligible("deepgemm_grouped_moe: contract miss (see deepgemm_moe_eligible)")

    from .sgl_moe import per_token_group_quant_fp8

    e, _, h = down_proj.shape
    i = down_proj.shape[2]
    t, topk = top_k_index.shape
    max_m = t  # distinct experts per token -> expert j holds <= T rows
    dev = x.device

    # ---- routing prep (static shapes, device-only) ------------------------
    flat = top_k_index.reshape(-1).to(torch.int64)              # [S]
    order = torch.argsort(flat, stable=True)                    # [S]
    sorted_experts = flat[order]                                # [S]
    # masked_m[j] = |{slots routed to j}| via a static-size histogram
    masked_m = torch.zeros(e, device=dev, dtype=torch.int32)
    masked_m.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts, dtype=torch.int32))
    # within-group rank and group id per sorted slot
    starts = torch.zeros(e, device=dev, dtype=torch.int64)
    starts[1:] = torch.cumsum(masked_m.to(torch.int64), 0)[:-1]
    group = sorted_experts                                      # [S]
    rank = torch.arange(order.numel(), device=dev, dtype=torch.int64) \
        - starts[sorted_experts]                                # [S]
    # token each sorted slot came from
    src_token = order // topk                                   # [S]

    # ---- stage 1 operands --------------------------------------------------
    x_rows = x[src_token]                                       # [S, H]
    q1, s1 = per_token_group_quant_fp8(x_rows)                  # [S,H], [S,H/128]
    a1 = torch.zeros(e, max_m, h, device=dev, dtype=_FP8)
    a1[group, rank] = q1
    sfa1 = torch.zeros(e, max_m, h // 128, device=dev, dtype=torch.float32)
    sfa1[group, rank] = s1
    sfb1 = _tma_aligned(gate_up_scale)
    cfg, _status = _deepgemm_config(t, x, gate_up_proj, down_proj,
                                    gate_up_scale, down_scale, top_k_index)
    expected_m = max(1, min(max_m, int(round(t * cfg["expected_m_mult"]))))
    d1 = torch.empty(e, max_m, 2 * i, device=dev, dtype=torch.bfloat16)
    _masked_fp8_gemm(a1, sfa1, gate_up_proj, sfb1, d1, masked_m, expected_m)

    # ---- between-stages activation (grouped, then re-quant) ---------------
    from .elementwise import swiglu_limit as _swiglu

    gate, up = d1[..., :i], d1[..., i:]
    act = _swiglu(gate, up, swiglu_limit)                       # [E, max_m, I]
    act_rows = act[group, rank]                                 # [S, I]
    q2, s2 = per_token_group_quant_fp8(act_rows)
    a2 = torch.zeros(e, max_m, i, device=dev, dtype=_FP8)
    a2[group, rank] = q2
    sfa2 = torch.zeros(e, max_m, i // 128, device=dev, dtype=torch.float32)
    sfa2[group, rank] = s2
    sfb2 = _tma_aligned(down_scale)

    d2 = torch.empty(e, max_m, h, device=dev, dtype=torch.bfloat16)
    _masked_fp8_gemm(a2, sfa2, down_proj, sfb2, d2, masked_m, expected_m)

    # ---- inverse gather + weighted combine --------------------------------
    slot_rows = d2[group, rank]                                 # [S, H] in sorted order
    slots = torch.empty(order.numel(), h, device=dev, dtype=torch.bfloat16)
    slots[order] = slot_rows                                    # unsort -> [S= T*topk, H]
    out = slots.view(t, topk, h)
    from .moe_combine import moe_weighted_sum

    return moe_weighted_sum(out, top_k_weights.to(torch.float32))


def _deepgemm_config(t: int, x, gate_up_proj, down_proj, gate_up_scale,
                     down_scale, idx):
    """Cache-first launch config for this decode-band bucket (lane-30 op
    tier). Shape class ``tN`` by token count; the only caller-owned knob
    is the expected_m multiplier (DeepGEMM sizes its block schedule from
    it). Capture-safe: op_config serves the memo/declared default while a
    stream captures; the sweep, when it ever runs, happens in floe's
    eager warmup pass."""
    global _RESOLVING
    from ..tuning.cache import op_config

    hit = _CONFIG_MEMO.get((t, x.shape[0], x.shape[1], gate_up_proj.shape))
    if hit is not None:
        return {"expected_m_mult": hit}, "memo"
    if _RESOLVING:
        # bench-closure re-entry (the sweep benches the whole op): serve the
        # declared default so the sweep measures the default-config path —
        # op_config's own lock is NOT reentrant.
        return {"expected_m_mult": 1.2}, "default"

    shape_class = f"t{min(8, max(2, t))}"
    default = {"expected_m_mult": 1.2}
    cands = [{"expected_m_mult": m} for m in (1.0, 1.2, 1.5, 2.0)]

    weights = torch.full((t, 8), 1.0 / 8, device=x.device)
    down_shape = down_proj.shape

    def bench(cfg):
        # Sweep the whole two-stage op at the LIVE shape: compile-first via
        # a warmup call, then a do_bench of the launch path.
        _warm_stage(gate_up_proj.shape, x.device)
        _warm_stage(down_shape, x.device)
        from triton.testing import do_bench

        return do_bench(
            lambda: deepgemm_grouped_moe(
                x, gate_up_proj, gate_up_scale, down_proj, down_scale,
                idx, weights,
            ),
            return_mode="median",
        )

    _RESOLVING = True
    try:
        cfg, _ = op_config(
            "deepgemm_moe", shape_class,
            default=default, candidates=cands, bench=bench,
            source_files=(__file__,),
        )
    finally:
        _RESOLVING = False
    mult = float(cfg["expected_m_mult"])
    _CONFIG_MEMO[(t, x.shape[0], x.shape[1], gate_up_proj.shape)] = mult
    return {"expected_m_mult": mult}, _


_RESOLVING = False
_CONFIG_MEMO: dict = {}


def _warm_stage(weight_shape, device):
    """JIT-compile one masked stage for these shapes (dummy operands)."""
    e, n, k = weight_shape
    a = torch.zeros(e, 1, k, device=device, dtype=_FP8)
    sfa = torch.zeros(e, 1, k // 128, device=device, dtype=torch.float32)
    b = torch.zeros(e, n, k, device=device, dtype=_FP8)
    sfb = torch.zeros(e, n // 128, k // 128, device=device, dtype=torch.float32)
    d = torch.empty(e, 1, n, device=device, dtype=torch.bfloat16)
    masked_m = torch.ones(e, device=device, dtype=torch.int32)
    _masked_fp8_gemm(a, sfa, b, sfb, d, masked_m, 1)


# ---------------------------------------------------------------------------
# eager warmup (the capture-guard's production half)
# ---------------------------------------------------------------------------
def warmup_deepgemm_moe(
    stage_shapes: Sequence[Tuple[int, int, int]],
    device=None,
) -> int:
    """Compile every DeepGEMM stage a decode serve will capture, EAGERLY.

    ``stage_shapes`` lists the (E, N, K) weight-stack shapes of both stages
    (for glm53 tp4: (288, 1024, 4096) and (288, 4096, 512)). Runs the full
    masked GEMM once per shape on dummy operands at the capture-time max
    token count, so the per-(arch, shape, config) JIT entries are hot on
    disk before the first graph capture. The inherited warning stands:
    big-N shapes JIT for minutes on first ever call; the disk cache makes
    later boots cheap. Returns the number of stages compiled THIS call
    (0 when the capability gate says no, or the cache was already hot —
    indistinguishable cheaply, so this is best-effort telemetry only).
    """
    reason = deepgemm_unavailable_reason(device)
    if reason is not None:
        logger.info("deepgemm warmup skipped: %s", reason)
        return 0
    if device is None:
        device = torch.cuda.current_device()
    n_compiled = 0
    for e, n, k in stage_shapes:
        if (n % 128) or (k % 128) or e <= 0:
            continue
        _warm_stage((e, n, k), device)
        n_compiled += 1
    logger.info("deepgemm warmup: %d stage shape(s) compiled/hot", n_compiled)
    return n_compiled


# ---------------------------------------------------------------------------
# tuning-cache population
# ---------------------------------------------------------------------------
def seed_deepgemm_defaults(*, store_dir=None, force=False) -> dict:
    """Land the sm90 donor prior (:data:`SM90_SEED`) into the op-tier
    store (status "seeded"; a local tune outranks it). Sourceless seed —
    the store fingerprint still binds to the adapter module, so an edit
    here re-tunes over it. Returns the persisted record map (auditable)."""
    from ..tuning.cache import seed

    return seed(
        "deepgemm_moe", SM90_SEED,
        source_files=(__file__,),
        producer="lane31-deepgemm-borrow",
        store_dir=store_dir,
        device={"capability": (9, 0), "sm_count": 0, "name": "sm90-seed"},
        force=force,
    )
