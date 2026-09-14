"""FP64-direct BF16 rounding oracle and argmax-ID recording (#67).

Torch's ``float64 -> float32 -> bfloat16`` chain (e.g. ``x.to(torch.float32)
.to(torch.bfloat16)``) can double-round near midpoints: a float64 value just
below a bfloat16 midpoint may round *up* to the midpoint in float32 and then
round up again, while direct round-to-nearest-even from float64 rounds down.
Quality comparisons must therefore derive the BF16 oracle directly from FP64
with explicit round-to-nearest-even, never through an FP32 hop.

Argmax evidence records the **actual argmax IDs** (and the top-1/top-2
margins, flagging ties) rather than top-k tie order, so a changed prediction
is never an artifact of tie ordering.
"""

import torch


def bf16_round_fp64(x):
    """Round a float64 tensor to bfloat16 with true round-to-nearest-even.

    The rounding decision is made once, on the IEEE-754 FP64 bit pattern: the
    52-bit mantissa is rounded straight to bfloat16's 7 explicit bits (drop
    45, ties-to-even). The chosen value is then embedded as a float32 bit
    pattern whose low 16 mantissa bits are zero — exactly representable, so
    the final float32->bfloat16 step is an exact relabel and no double
    rounding can occur. Overflow saturates to +-inf, NaNs map to quiet NaNs,
    and FP64 subnormals (< 2^-1022) round to zero.
    """
    if x.dtype != torch.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    bits = x.contiguous().view(torch.int64)
    sign32 = (bits >> 32) & 0x80000000  # sign bit, positioned for a float32 pattern
    magnitude = bits & ((1 << 63) - 1)
    exponent = (magnitude >> 52) & 0x7FF
    mantissa = magnitude & ((1 << 52) - 1)

    exp32 = exponent - 1023 + 127  # shared FP32/bfloat16 exponent bias
    is_nan = (exponent == 0x7FF) & (mantissa != 0)
    is_inf = ((exponent == 0x7FF) & (mantissa == 0)) | (exp32 >= 0xFF)
    # Zero and FP64 subnormals: values < 2^-1022 are far below half the
    # smallest bf16 subnormal (2^-133.5), so they all round to zero.
    is_zero = exponent == 0
    # Normal bf16: exp32 in [1, 254]. exp32 <= 0 lands in the subnormal grid.
    normal = (exponent != 0) & (exp32 >= 1) & (exp32 <= 0xFE) & ~is_inf
    subnormal = (exponent != 0) & (exp32 <= 0) & ~is_inf

    # Normals: keep the top 7 mantissa bits (drop 45), RNE on the dropped bits.
    keep = mantissa >> 45
    remainder = mantissa & ((1 << 45) - 1)
    half = 1 << 44
    round_up = (remainder > half) | ((remainder == half) & ((keep & 1) == 1))
    keep = torch.where(round_up, keep + 1, keep)          # 0..128
    carry = keep >> 7                                      # 1 if mantissa overflowed
    keep = keep & 0x7F
    normal_bits = sign32 | ((exp32 + carry) << 23) | (keep << 16)

    # Subnormals: value = s * 2^-133 for integer s; s = virtual * 2^(exp32-46)
    # where virtual = (1 << 52) | mantissa. For exp32 <= -8 the whole grid
    # index is below 0.5 ulp of the smallest subnormal and rounds to zero.
    virtual = mantissa | (1 << 52)
    shift_true = 46 - exp32                              # >= 46 in this branch
    representable = shift_true <= 53                     # else s_exact < 0.5
    shift = shift_true.clamp(0, 53)                      # keep int64 shifts safe
    sub_keep = virtual >> shift
    sub_remainder = virtual & ((1 << shift) - 1)
    sub_half = 1 << (shift - 1)
    sub_round_up = (sub_remainder > sub_half) | \
        ((sub_remainder == sub_half) & ((sub_keep & 1) == 1))
    sub_keep = torch.where(sub_round_up, sub_keep + 1, sub_keep)
    sub_keep = torch.where(representable, sub_keep, torch.zeros_like(sub_keep))
    # sub_keep == 128 carries into the smallest normal (2^-126).
    sub_bits = torch.where(sub_keep >= 128, sign32 | (1 << 23),
                           sign32 | (sub_keep << 16))

    inf_nan_bits = torch.where(is_nan, sign32 | 0x7FC00000, sign32 | (0xFF << 23))

    out = torch.where(normal, normal_bits,
                      torch.where(subnormal, sub_bits,
                                  torch.where(is_zero, sign32, inf_nan_bits)))
    # Reinterpret the float32 bit pattern; the value is exactly representable
    # in bfloat16 (mantissa low 16 bits are zero), so the final conversion is
    # an exact relabel, not a round.
    return out.to(torch.int32).view(torch.float32).to(torch.bfloat16).reshape(x.shape)


def _fraction_rne_bf16(value):
    """Round a python float (exact as Fraction) to bfloat16 RNE, for tests."""
    from fractions import Fraction

    value = float(value)
    if value != value:
        return float("nan")
    if value in (float("inf"), float("-inf")):
        return value
    sign = -1.0 if value < 0 else 1.0
    value = abs(value)
    if value == 0:
        return 0.0
    # Largest power of two <= value
    import math
    exponent = math.floor(math.log2(value))
    # values >= 2^128 overflow bf16
    if exponent > 127 or (exponent == 128 and Fraction(value) >= Fraction(2) ** 128):
        return sign * float("inf")
    if exponent >= -126:  # normal path: 8-bit significand in [1, 2)
        grid = Fraction(2) ** (exponent - 7)
    else:  # subnormal: significand is m/2^7 * 2^-126
        grid = Fraction(2) ** -133
    exact = Fraction(value) / grid
    floor = int(exact)
    remainder = exact - floor
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and floor % 2 == 1):
        floor += 1
    return sign * float(floor * grid)


def qkv_oracle_fp64(x, weights):
    """BF16 Q/K/V computed in FP64 and rounded directly from FP64.

    ``x`` is [...,4096] and ``weights`` three [8192,4096] BF16 tensors (BF16
    -> FP64 is exact, so the oracle sees precisely the caller's values).
    Returns BF16 [...,24576], ordered Q, K, V.
    """
    x64 = x.to(torch.float64)
    out = torch.cat([
        torch.nn.functional.linear(x64, weight.to(torch.float64))
        for weight in weights], dim=-1)
    return bf16_round_fp64(out)


def argmax_evidence(reference, candidate):
    """Record actual argmax IDs plus margins for a reference/candidate pair.

    Returns a JSON-safe dict: agreement over rows, per-row changed IDs with
    top-1/top-2 margins, and tie flags (top-1 margin == 0 means the reference
    had an exact tie — a changed prediction there is not evidence of quality
    loss in either direction by itself).
    """
    ref64, cand64 = reference.to(torch.float64), candidate.to(torch.float64)

    def ids_and_margin(y):
        values, ids = torch.max(y, dim=-1)
        if y.shape[-1] > 1:
            top2 = torch.topk(y, 2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
        else:
            margin = values
        return ids, margin

    ref_ids, ref_margin = ids_and_margin(ref64)
    cand_ids, cand_margin = ids_and_margin(cand64)
    changed = (ref_ids != cand_ids).nonzero(as_tuple=False)
    rows = [{
        "index": [int(i) for i in index],
        "reference_argmax_id": int(ref_ids[tuple(index)]),
        "candidate_argmax_id": int(cand_ids[tuple(index)]),
        "reference_margin": float(ref_margin[tuple(index)]),
        "candidate_margin": float(cand_margin[tuple(index)]),
        "reference_tie": float(ref_margin[tuple(index)]) == 0.0,
    } for index in changed.tolist()]
    total = ref_ids.numel()
    return {
        "rows": total,
        "agreement": (total - len(rows)) / total if total else 1.0,
        "changed_count": len(rows),
        "changed": rows,
    }
