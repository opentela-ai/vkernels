"""Fused-epilogue variant of the selected-expert FP8 decode GEMV.

Companion to :mod:`vkernels.torch_ops.glm_expert_gemv` (issues #64/#65):
same block-FP8 weight decode, same fp32 dot, same bf16 rounding points —
but the SwiGLU activation is applied IN-KERNEL and only ``act[T,K,IA]`` is
written, instead of the two-launch chain

    gu = expert_gemv(x, w, s, idx)              # [T,K,O=2*IA] bf16 gate|up
    act = silu_mul(gu[..., :IA], gu[..., IA:])  # elementwise silu_mul

which round-trips O bf16 columns per (token, slot) through HBM and pays a
second launch. This is the GLM decode seam named in
docs/glm53-decode-kernels.md ("further gains need a different design —
fusing the top-k gather / the activation"), minus the router plumbing.

Shapes: the stacked gate/up GEMV has ``weights[E, O, I]`` with ``O = 2*IA``
(gate rows [0,IA), up rows [IA,2I)), activations x[T,I] or x[T,K,I], and
the fused output is act[T,K,IA] — the GLM serving shape is
``weights[E, 4096, 4096]`` (IA=2048, I_in=4096), so the input width and
the act width are DISTINCT dimensions and neither is derived from the
other beyond ``O == 2*IA``.

Numerics contract (bit-exactness with the unfused chain, not just
tolerance): the gate/up dot results are rounded to bf16 *in registers*
before the activation — exactly the store-then-reload the two-kernel chain
performs — then ``silu`` is evaluated in fp32, rounded to bf16 (the
``silu_mul`` storage-dtype boundary), and the product is stored bf16. The
silu expression ``(g / (1 + exp(-g)))`` is copied verbatim from
``elementwise._swiglu_limit`` (the ``limit=+inf`` member ``silu_mul``
shares), so on a given device the fused epilogue is bit-identical to the
unfused chain, NaN propagation included. Weight decode, scale gather,
weight bf16 rounding, dot order and launch geometry (``ROWS=4``,
``COLS=next_pow2(I)``, ``num_warps=4``, ``enable_fp_fusion=False``) are
inherited unchanged from ``glm_expert_gemv`` — the ``tl.sum`` reduction
order depends only on ``COLS``, so gate/up rows compute the identical dot
whether they are produced by their own program (unfused, grid over
``O=2*IA`` rows) or paired inside one program (fused).

``storage="e4m3fnuz"`` (issue #71) takes the manual bit-decode on both
backends (CUDA has no fnuz dtype); plain e4m3fn on NVIDIA uses the native
``float8e4nv`` bitcast path, mirroring ``_expert_gemv_native``.

The old path stays the default: this module is opt-in per call site, and
:func:`silu_fused_enabled` (env ``GLM53_MOE_SILU_FUSED``) is the serving
knob — ``0`` (default) keeps ``expert_gemv`` + ``silu_mul``.

Torch and Triton load lazily. Inputs are read-only; inference-only, no
autograd backward. Graph capture: warm up eagerly per shape first.
"""

from ._dispatch import OpNotEligible
import os
from functools import lru_cache

__all__ = ["expert_gemv_silu", "expert_gemv_silu_reference", "silu_fused_enabled"]


def silu_fused_enabled() -> bool:
    """Serving knob: run the fused-epilogue GEMV instead of the two-launch
    ``expert_gemv`` + ``silu_mul`` chain. Defaults to OFF (the proven
    path); ``GLM53_MOE_SILU_FUSED=1`` opts in. MI300A A/B before default
    flip: see NOTES-fusion-glm-decode-fused.md."""
    return os.environ.get("GLM53_MOE_SILU_FUSED", "0") == "1"


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_fp8(raw, FNUZ: tl.constexpr):
        """Branchless block-FP8 byte decode (shared with glm_expert_gemv).

        e4m3fnuz: bias 8, max finite 240, NaN ONLY at 0x80, subnormals
        m*2^-10, DOUBLED stored scales (issue #71). e4m3fn: bias 7, max
        finite 448, NaN at 0x7F/0xFF, subnormals m*2^-9.
        """
        raw = raw.to(tl.int32)
        exponent, mantissa = (raw >> 3) & 15, raw & 7
        if FNUZ:
            bits = ((exponent + 119) << 23) | (mantissa << 20)
            value = tl.where(
                exponent == 0,
                mantissa.to(tl.float32) * 0.0009765625,
                bits.to(tl.float32, bitcast=True),
            )
            value = tl.where(raw == 128, float("nan"), value)
        else:
            bits = ((exponent + 120) << 23) | (mantissa << 20)
            value = tl.where(
                exponent == 0,
                mantissa.to(tl.float32) * 0.001953125,
                bits.to(tl.float32, bitcast=True),
            )
            value = tl.where((raw & 127) == 127, float("nan"), value)
        return (value.to(tl.int32, bitcast=True) | ((raw & 128) << 24)).to(
            tl.float32, bitcast=True
        )

    @triton.jit
    def _expert_gemv_silu(
        X,
        W,
        S,
        IDX,
        Y,
        IA: tl.constexpr,
        I: tl.constexpr,
        K: tl.constexpr,
        BROADCAST: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        FNUZ: tl.constexpr,
    ):
        """Stacked gate/up GEMV (O = 2*IA, input width I) with in-kernel
        silu(gate)*up.

        Program (selected, block) owns act columns [block*ROWS, ...): it
        loads the gate weight rows AND the matching up weight rows (row
        offset +IA), dots both against x, and applies the activation.
        Weight traffic is unchanged versus the unfused pair (the same rows
        are read exactly once either way); the [T,K,2*IA] bf16 gate/up
        round trip and the second launch disappear.
        """
        selected = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        expert = tl.load(IDX + selected).to(tl.int64)
        inbounds = (row[:, None] < IA) & (col[None, :] < I)
        base = W + expert * (2 * IA * I)
        x_row = selected // K if BROADCAST else selected
        x = tl.load(X + x_row * I + col, col < I, 0).to(tl.float32)

        raw_gate = tl.load(
            base + row[:, None] * I + col[None, :], inbounds, 0
        )
        value = _decode_fp8(raw_gate, FNUZ)
        # Scale rows: gate occupies O-rows [0,IA), up [IA,2*IA); the
        # per-expert scale stride is (O//128) = (2*IA)//128 row-blocks,
        # each row-block spanning 128 input columns (I//128 of them).
        scale = tl.load(
            S
            + (expert * (2 * IA // 128) + row[:, None] // 128) * (I // 128)
            + col[None, :] // 128,
            inbounds,
            0,
        )
        weight = (value * scale).to(tl.bfloat16).to(tl.float32)
        gate = tl.sum(weight * x[None, :], axis=1)

        raw_up = tl.load(
            base + (row[:, None] + IA) * I + col[None, :], inbounds, 0
        )
        value = _decode_fp8(raw_up, FNUZ)
        scale = tl.load(
            S
            + (expert * (2 * IA // 128) + (row[:, None] + IA) // 128)
            * (I // 128)
            + col[None, :] // 128,
            inbounds,
            0,
        )
        weight = (value * scale).to(tl.bfloat16).to(tl.float32)
        up = tl.sum(weight * x[None, :], axis=1)

        # Bit-exact epilogue vs expert_gemv(bf16 store) + silu_mul:
        # round each dot to the storage dtype (the store-then-reload the
        # two-kernel chain performs), silu in fp32, round the silu result
        # (silu_mul's storage boundary), product stored bf16. The silu
        # expression is verbatim elementwise._swiglu_limit (limit=+inf is
        # an exact identity there; NaN propagates through both forms).
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + selected * IA + row, act * up, row < IA)

    @triton.jit
    def _expert_gemv_silu_native(
        X,
        W,
        S,
        IDX,
        Y,
        IA: tl.constexpr,
        I: tl.constexpr,
        K: tl.constexpr,
        BROADCAST: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        """CUDA variant of ``_expert_gemv_silu``: hardware float8e4nv
        decode (see ``_expert_gemv_native`` in glm_expert_gemv), compact
        scale load with the reshape broadcast. Numerically identical to
        the manual path (e4m3FN == e4m3nv semantics on CUDA)."""
        selected = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        expert = tl.load(IDX + selected).to(tl.int64)
        inbounds = (row[:, None] < IA) & (col[None, :] < I)
        base = W + expert * (2 * IA * I)
        scol = tl.arange(0, COLS // 128)
        x_row = selected // K if BROADCAST else selected
        x = tl.load(X + x_row * I + col, col < I, 0).to(tl.float32)

        raw_gate = tl.load(
            base + row[:, None] * I + col[None, :], inbounds, 0
        )
        scale = tl.load(
            S
            + (expert * (2 * IA // 128) + row[:, None] // 128) * (I // 128)
            + scol[None, :],
            (row[:, None] < IA) & (scol[None, :] < I // 128),
            0,
        )
        weight = (
            (raw_gate.to(tl.float8e4nv, bitcast=True).to(tl.float32)
             .reshape(ROWS, COLS // 128, 128) * scale[:, :, None])
            .reshape(ROWS, COLS)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        gate = tl.sum(weight * x[None, :], axis=1)

        raw_up = tl.load(
            base + (row[:, None] + IA) * I + col[None, :], inbounds, 0
        )
        scale = tl.load(
            S
            + (expert * (2 * IA // 128) + (row[:, None] + IA) // 128)
            * (I // 128)
            + scol[None, :],
            (row[:, None] < IA) & (scol[None, :] < I // 128),
            0,
        )
        weight = (
            (raw_up.to(tl.float8e4nv, bitcast=True).to(tl.float32)
             .reshape(ROWS, COLS // 128, 128) * scale[:, :, None])
            .reshape(ROWS, COLS)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        up = tl.sum(weight * x[None, :], axis=1)

        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + selected * IA + row, act * up, row < IA)

    return _expert_gemv_silu, _expert_gemv_silu_native


def expert_gemv_silu(x, weights, scales, indices, storage="e4m3fn", t_cap=None):
    """Return BF16 ``act[T,K,IA] = silu(gate) * up`` for BF16 x[T,I] or
    x[T,K,I], T <= t_cap (default 2 — floe passes its
    ``moe_decode_max_tokens`` knob), where ``gate``/``up`` are the two halves
    of the stacked GEMV output over ``weights[E, 2*IA, I]``.

    Bit-identical (same device) to::

        gu = glm_expert_gemv.expert_gemv(x, weights, scales, indices, storage,
                                         t_cap=t_cap)
        act = elementwise.silu_mul(gu[..., :IA], gu[..., IA:])

    but writes only [T,K,IA] and launches one kernel instead of two.
    Validation mirrors ``expert_gemv`` plus the stacked-layout requirement
    ``weights.shape[1] == 2 * (weights.shape[1] // 2)`` with the act width
    taken as ``IA = weights.shape[1] // 2``.
    """
    import torch

    if storage not in ("e4m3fn", "e4m3fnuz"):
        raise OpNotEligible(f"unknown weight storage {storage!r}")
    fnuz = storage == "e4m3fnuz"
    want = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    cap = max(2, int(t_cap)) if t_cap is not None else 2
    if weights.ndim != 3 or indices.ndim != 2:
        raise OpNotEligible("expected weights [E,O,I] and indices [T,K]")
    e, o, i = weights.shape
    ia = o // 2
    t, k = indices.shape
    if not e or not o or not i or o % 128 or o != 2 * ia or i % 128 or t > cap:
        raise OpNotEligible(
            f"requires stacked gate/up weights [E,2*IA,I] with 2*IA%128==0, "
            f"I%128==0 and T<={cap}"
        )
    if x.shape not in ((t, i), (t, k, i)):
        raise OpNotEligible("expected x[T,I] or x[T,K,I]")
    if scales.shape != (e, o // 128, i // 128):
        raise OpNotEligible("expected scales [E,O/128,I/128]")
    if (
        x.dtype != torch.bfloat16
        or weights.dtype != want
        or scales.dtype != torch.float32
        or indices.dtype != torch.int64
    ):
        raise OpNotEligible(
            "requires BF16 activations, "
            f"{'E4M3FNUZ' if fnuz else 'E4M3FN'} weights, FP32 scales, int64 indices"
        )
    if any(not v.is_cuda or v.device != weights.device for v in (x, scales, indices)):
        raise OpNotEligible("inputs must share a GPU device")
    if any(not v.is_contiguous() for v in (x, weights, scales, indices)):
        raise OpNotEligible("inputs must be contiguous")
    out = torch.empty((t, k, ia), device=x.device, dtype=torch.bfloat16)
    if t and k:
        import triton  # lazy: validation above needs only torch

        silu_gemv, silu_gemv_native = _kernel()
        with torch.cuda.device(x.device):
            # Backend gate copied from expert_gemv: HIP tensors report
            # is_cuda and gfx942's native fp8 is FNUZ, so fnuz STORAGE
            # always takes the manual kernel; NVIDIA e4m3fn decodes via
            # the float8e4nv bitcast.
            if fnuz:
                silu_gemv[(t * k, triton.cdiv(ia, 4))](
                    x,
                    weights.view(torch.uint8),
                    scales,
                    indices,
                    out,
                    ia,
                    i,
                    k,
                    x.ndim == 2,
                    4,
                    triton.next_power_of_2(i),
                    True,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                kernel = silu_gemv if torch.version.hip else silu_gemv_native
                # manual kernel takes a trailing FNUZ constexpr, the
                # native one does not (same 11-vs-12-arg rule as
                # expert_gemv; see the d5678ec regression note there).
                extra_fnuz = [False] if torch.version.hip else []
                kernel[(t * k, triton.cdiv(ia, 4))](
                    x,
                    weights.view(torch.uint8),
                    scales,
                    indices,
                    out,
                    ia,
                    i,
                    k,
                    x.ndim == 2,
                    4,
                    triton.next_power_of_2(i),
                    *extra_fnuz,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
    return out


def expert_gemv_silu_reference(x, weights, scales, indices, t_cap=None):
    """Unfused oracle: ``expert_gemv_reference`` + the eager silu chain.

    Delegates the FP8 gather-dequant dot to
    :func:`glm_expert_gemv.expert_gemv_reference` (same rounding contract:
    scaled weight rounded to BF16 before the fp32 reduction) and applies
    the eager ``F.silu(gate) * up`` — the exact chain the fused kernel
    replaces, so parity here plus the kernel-chain bit-equality check pins
    the whole path.
    """
    import torch
    import torch.nn.functional as F

    from .glm_expert_gemv import expert_gemv_reference

    gu = expert_gemv_reference(x, weights, scales, indices, t_cap=t_cap)
    ia = weights.shape[1] // 2
    return (F.silu(gu[..., :ia].float()) * gu[..., ia:].float()).to(
        torch.bfloat16
    )
