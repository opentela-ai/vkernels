"""Selected-expert E4M3FN block-FP8 decode GEMV (issues #64 / #65).

Decode selected checkpoint experts directly in a BF16 GEMV: no gathered or
dequantized weight tensor is allocated. Each selected (token, expert-slot)
pair's weight is rounded to BF16 after block scaling, matching
gather-dequant followed by GEMV; the product reduction is FP32 and its
order can differ from rocBLAS.

Adopted from floe's ``engine/runner/kernels/glm5_fp8_gemv.py`` (the
decode-validated implementation; measured ~1.0 TB/s effective at the GLM
top-8 dispatch on MI300A — faster than a per-matrix native kernel there).
floe imports this back through a thin adapter so vkernels owns the kernel
(#65's thin-adapter model) while floe keeps a local fallback.

Torch and Triton load lazily. Inputs are read-only; inference-only, no
autograd backward. Graph capture: warm up eagerly per shape first (the
autotune-free kernel still compiles on first launch).
"""

from ._dispatch import OpNotEligible
import os
from functools import lru_cache


def _t_cap() -> int:
    """Row cap for the batched GEMV: 2 (the decode-validated limit) unless
    GLM53_MOE_DECODE_MAX_TOKENS widens it for speculative-decoding blocks
    (e.g. 8 for DFlash2 verify/replay). The kernel body is shape-generic —
    one program per (token, expert-slot) pair — so only the wrapper guard
    needs lifting; parity at the raised T is checked by the caller's job."""
    try:
        return max(2, int(os.environ.get("GLM53_MOE_DECODE_MAX_TOKENS", "2")))
    except ValueError:
        return 2


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _expert_gemv(
        X,
        W,
        S,
        IDX,
        Y,
        O: tl.constexpr,
        I: tl.constexpr,
        K: tl.constexpr,
        BROADCAST: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        FNUZ: tl.constexpr,
    ):
        selected = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        expert = tl.load(IDX + selected).to(tl.int64)
        raw = tl.load(
            W + expert * (O * I) + row[:, None] * I + col[None, :],
            (row[:, None] < O) & (col[None, :] < I),
            0,
        ).to(tl.int32)
        exponent, mantissa = (raw >> 3) & 15, raw & 7
        if FNUZ:
            # e4m3fnuz storage (AMD-native fp8; issue #71 in-place
            # conversion): bias 8, max finite 240, NaN ONLY at 0x80 (there
            # is no -0), subnormals m*2^-10. The stored scales are the
            # DOUBLED scales from e4m3fn_to_fnuz_inplace, so value*scale
            # reproduces the original e4m3fn weight exactly (verified
            # exhaustive over all 256 bytes). No hardware shortcut: CUDA
            # has no fnuz dtype and this Triton rejects direct fp8-pointer
            # loads, so both backends take the manual bit-decode.
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
        value = (value.to(tl.int32, bitcast=True) | ((raw & 128) << 24)).to(
            tl.float32, bitcast=True
        )
        scale = tl.load(
            S
            + (expert * (O // 128) + row[:, None] // 128) * (I // 128)
            + col[None, :] // 128,
            (row[:, None] < O) & (col[None, :] < I),
            0,
        )
        weight = (value * scale).to(tl.bfloat16).to(tl.float32)
        x_row = selected // K if BROADCAST else selected
        x = tl.load(X + x_row * I + col, col < I, 0).to(tl.float32)
        result = tl.sum(weight * x[None, :], axis=1)
        tl.store(Y + selected * O + row, result, row < O)

    @triton.jit
    def _expert_gemv_native(
        X,
        W,
        S,
        IDX,
        Y,
        O: tl.constexpr,
        I: tl.constexpr,
        K: tl.constexpr,
        BROADCAST: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        """CUDA variant: bitcast the raw e4m3FN bytes (uint8 load — direct
        fp8-pointer loads are rejected by this Triton) to tl.float8e4nv and
        let the hardware converter decode. Numerically identical to
        _expert_gemv (e4m3FN is the IEEE variant on CUDA; the manual
        bit-decode only exists because gfx942's native fp8 is FNUZ) but
        ~8x faster: the bit-twiddle path costs ~10 ALU ops per weight element
        and caps the kernel at ~390 GB/s effective on GH200; the
        native-cast path is bandwidth-bound."""
        selected = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        expert = tl.load(IDX + selected).to(tl.int64)
        raw = tl.load(
            W + expert * (O * I) + row[:, None] * I + col[None, :],
            (row[:, None] < O) & (col[None, :] < I),
            0,
        )
        value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # compact scale load: [ROWS, COLS//128] gathered once, broadcast over
        # the 128-wide groups via reshape (the per-element scale gather costs
        # more bytes than the fp8 weights themselves when uncoalesced).
        scol = tl.arange(0, COLS // 128)
        scale = tl.load(
            S
            + (expert * (O // 128) + row[:, None] // 128) * (I // 128)
            + scol[None, :],
            (row[:, None] < O) & (scol[None, :] < I // 128),
            0,
        )
        weight = (
            (value.reshape(ROWS, COLS // 128, 128) * scale[:, :, None])
            .reshape(ROWS, COLS)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        x_row = selected // K if BROADCAST else selected
        x = tl.load(X + x_row * I + col, col < I, 0).to(tl.float32)
        result = tl.sum(weight * x[None, :], axis=1)
        tl.store(Y + selected * O + row, result, row < O)

    return _expert_gemv, _expert_gemv_native


def expert_gemv(x, weights, scales, indices, storage="e4m3fn"):
    """Return BF16 [T,K,O] for BF16 x[T,I] or x[T,K,I], T <= _t_cap().

    Contiguous GPU inputs and valid expert IDs are required. On NVIDIA the
    FN bytes are hardware-decoded (e4m3FN == e4m3nv semantics there); on
    gfx942 they are decoded explicitly, because its native fp8 is FNUZ
    (different bias/NaN encodings).

    ``storage="e4m3fnuz"`` (issue #71) reads the IN-PLACE converted stacks
    produced by ``e4m3fn_to_fnuz_inplace`` — fnuz bytes with DOUBLED fp32
    scales, sharing the checkpoint's storage. The manual bit-decode handles
    both backends there (CUDA has no fnuz dtype).
    """
    import torch

    if storage not in ("e4m3fn", "e4m3fnuz"):
        raise OpNotEligible(f"unknown weight storage {storage!r}")
    fnuz = storage == "e4m3fnuz"
    want = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    cap = _t_cap()
    if weights.ndim != 3 or indices.ndim != 2:
        raise OpNotEligible("expected weights [E,O,I] and indices [T,K]")
    e, o, i = weights.shape
    t, k = indices.shape
    if not e or not o or not i or o % 128 or i % 128 or t > cap:
        raise OpNotEligible(f"requires positive block-128 dimensions and T<={cap}")
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
    out = torch.empty((t, k, o), device=x.device, dtype=torch.bfloat16)
    if t and k:
        import triton  # lazy: validation above needs only torch

        gemv, gemv_native = _kernel()
        with torch.cuda.device(x.device):
            # Gate on the BACKEND, not is_cuda: HIP tensors report is_cuda,
            # and gfx942's native fp8 is FNUZ (different bias/NaN encodings
            # from checkpoint e4m3FN) — it needs the manual bit-decode.
            # NVIDIA decodes e4m3FN natively: uint8 load + in-kernel bitcast
            # to float8e4nv (direct fp8-pointer loads are rejected by this
            # Triton), ~8x faster than the ~10-ALU-ops-per-element decode.
            # fnuz STORAGE always takes the manual kernel (FNUZ=1): CUDA has
            # no fnuz dtype to bitcast to, and the doubled-scale convention
            # is part of the in-place conversion contract (issue #71).
            if fnuz:
                gemv[(t * k, triton.cdiv(o, 4))](
                    x,
                    weights.view(torch.uint8),
                    scales,
                    indices,
                    out,
                    o,
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
                kernel = gemv if torch.version.hip else gemv_native
                # _expert_gemv (HIP/fnuz) takes a trailing FNUZ constexpr;
                # _expert_gemv_native (CUDA) does NOT — binding 12 args
                # against its 11-param signature raised
                # "dynamic_func() takes 11 positional arguments but 12 were
                # given" (regression from the fnuz-storage commit d5678ec).
                extra_fnuz = [False] if torch.version.hip else []
                kernel[(t * k, triton.cdiv(o, 4))](
                    x,
                    weights.view(torch.uint8),
                    scales,
                    indices,
                    out,
                    o,
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


def expert_gemv_reference(x, weights, scales, indices):
    """Gather-dequant + einsum reference (fp32 dot, BF16 output).

    Mirrors the kernel's rounding contract: the scaled weight is rounded
    to BF16 *before* the fp32 product reduction (matching gather-dequant
    followed by GEMV). The decode runs GATHER-FIRST and chunked — both
    bit-identical (the byte decode is elementwise) — so the oracle's fp32
    temporaries stay bounded: the un-chunked form materialized ~6x the
    full [E,O,I] fp32 stack and OOMed a 128 GiB MI300A through the
    plugin-shape acceptance test even after its memory guard passed
    (beverin job 644666). Gathering through the uint8 view also keeps the
    oracle working on builds where CPU fp8 fancy-indexing is
    unimplemented.
    """
    import torch

    cap = _t_cap()
    e, o, i = weights.shape
    t, k = indices.shape
    if t > cap:
        raise OpNotEligible(f"T={t} exceeds the cap {cap}")
    sel_w8 = weights.view(torch.uint8)[indices]  # [T,K,O,I] raw bytes
    sel_scale = scales[indices]  # [T,K,O/128,I/128]
    expand = sel_scale.repeat_interleave(128, dim=2).repeat_interleave(128, dim=3)
    flat = sel_w8.reshape(-1)
    vals = torch.empty(flat.shape, dtype=torch.float32, device=flat.device)
    chunk = 1 << 26  # 64 Mi elements: ~1.5 GiB of fp32 temporaries per slice
    for beg in range(0, flat.numel(), chunk):
        raw = flat[beg:beg + chunk].to(torch.int32)
        exponent, mantissa = (raw >> 3) & 15, raw & 7
        bits = (((exponent + 120) << 23) | (mantissa << 20)).to(torch.int32)
        value = bits.view(torch.float32)
        value = torch.where(
            exponent == 0, mantissa.to(torch.float32) * (1.0 / 512.0), value
        )
        value = torch.where((raw & 127) == 127, float("nan"), value)
        vals[beg:beg + chunk] = value * torch.where((raw & 128) != 0, -1.0, 1.0)
    weight = (vals.reshape(t, k, o, i) * expand).to(torch.bfloat16)
    x3 = x if x.ndim == 3 else x[:, None, :].expand(t, k, i)
    out = torch.einsum("tki,tkoi->tko", x3.float(), weight.float())
    return out.to(torch.bfloat16)
