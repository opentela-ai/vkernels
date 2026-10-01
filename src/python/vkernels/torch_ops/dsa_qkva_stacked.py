"""Stacked DSA q_a | kv_a fp8 GEMV — one launch over the concatenated rows
(node-cuts F).

The GLM-5.3 DSA decode step runs two fp8 dense GEMVs per layer against the
same activation: ``q_a`` [1536, 4096] (scales [12, 32]) and
``kv_a_with_mqa`` [512, 4096] (scales [4, 32]) — both projections REPLICATE
under TP, so their rows can be concatenated at load time into one
[2048, 4096] weight with a [16, 32] scale grid: 1536 and 512 are
128-multiples, so the block-scale rows concatenate EXACTLY on the
scale-block boundary ([0:12] = q_a's, [12:16] = kv_a's).

The GEMV itself is :func:`~vkernels.torch_ops.glm_dense_fp8_gemv.dense_gemv_fp8`
unchanged — same kernel, same per-row BLOCK_I reduction, same scale
application — so each output row is BIT-IDENTICAL to running the two GEMVs
separately *when the tile configs match* (the no-pin default; or forced via
``VK_FP8_DENSE_GEMV_TILES``). Caveat: the H100 pin table gives the two
sub-shapes different ``(ROWS, num_warps)`` — ``(1536,4096)`` rides the
heuristic ``(4, ·, 4)`` the stacked ``(2048,4096)`` also takes (bit-identical
half), while ``(512,4096)`` is pinned ``(2, ·, 8)``: its warp split reorders
the fp32 accumulation, so under the production pins the kv_a half lands in
the adopted perf-only reassociation class (bf16-rounding band), not
bit-identity. Deployment choice: either pin ``(2048,4096)`` alongside the
kv_a pin or accept the band — the win is the node cut either way. Only the
launch count changes: 2 -> 1 per layer (−11 nodes, −0.02 ms kernel per the
node-cuts census; the win is the node-gap and one fewer x read).

Output slicing (zero-copy views): ``q_a_out = out[:, :o_q]``,
``kv_a_out = out[:, o_q:]``.

Torch loads lazily. Inference-only, no autograd backward.
"""

from ._dispatch import OpNotEligible

__all__ = [
    "dsa_qkva_stacked",
    "dsa_qkva_stacked_eligible",
    "dsa_qkva_stacked_reference",
    "stack_dsa_qkva",
]


def stack_dsa_qkva(qa_w8, qa_scales, kva_w8, kva_scales):
    """Concatenate the q_a | kv_a fp8 weight + block-scale grid at load time.

    Any device (the loader bakes this). Raises :class:`OpNotEligible` unless
    both halves are 128-row-multiples with matching widths and scale grids —
    the exactness condition for the stacked scale concat.
    """
    import torch

    for name, w, s in (("q_a", qa_w8, qa_scales), ("kv_a", kva_w8, kva_scales)):
        if w.ndim != 2 or s.ndim != 2:
            raise OpNotEligible(f"{name} weight must be 2-D and scales 2-D")
        if w.shape[0] % 128 or w.shape[1] % 128:
            raise OpNotEligible(f"{name} shape {tuple(w.shape)} not on the 128 block grid")
        if s.shape != (w.shape[0] // 128, w.shape[1] // 128):
            raise OpNotEligible(f"{name} scales {tuple(s.shape)} != {(w.shape[0]//128, w.shape[1]//128)}")
    if qa_w8.shape[1] != kva_w8.shape[1]:
        raise OpNotEligible("q_a/kv_a widths differ; no shared activation read")
    if qa_w8.device != kva_w8.device or qa_scales.device != kva_w8.device:
        raise OpNotEligible("stack inputs must share one device")
    return (
        torch.cat([qa_w8, kva_w8], dim=0).contiguous(),
        torch.cat([qa_scales, kva_scales], dim=0).contiguous(),
    )


def dsa_qkva_stacked_eligible(x, w8, scales, m_cap=None) -> bool:
    """Contract check for :func:`dsa_qkva_stacked` (no device sync)."""
    try:
        _peek(x, w8, scales, m_cap)
    except OpNotEligible:
        return False
    return True


def _peek(x, w8, scales, m_cap):
    """Shape/dtype/device/contiguity gate mirroring dense_gemv_fp8's."""
    import torch

    if x.ndim not in (1, 2) or w8.ndim != 2 or scales.ndim != 2:
        raise OpNotEligible("expected x [M, I] (or [I]), w8 [O, I], scales [O//128, I//128]")
    m = 1 if x.ndim == 1 else x.shape[0]
    o, i = w8.shape
    if x.shape[-1] != i:
        raise OpNotEligible(f"shape mismatch x{tuple(x.shape)} w8{tuple(w8.shape)}")
    cap = max(2, int(m_cap)) if m_cap is not None else 2
    if m < 1 or m > cap:
        raise OpNotEligible(f"M={m} outside the 1..{cap} decode-GEMV cap")
    if i % 128 or o % 128 or i <= 0 or o <= 0:
        raise OpNotEligible("O and I must be positive multiples of 128 (block-scale grid)")
    if scales.shape != (o // 128, i // 128):
        raise OpNotEligible(f"scale shape {tuple(scales.shape)} != {(o // 128, i // 128)}")
    if x.dtype != torch.bfloat16 or w8.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        raise OpNotEligible("requires bf16 activations, e4m3fn weights, fp32 scales")
    if not (x.is_cuda and w8.is_cuda and x.device == w8.device and scales.device == w8.device):
        raise OpNotEligible("inputs must share a GPU device")
    if not (x.is_contiguous() and w8.is_contiguous() and scales.is_contiguous()):
        raise OpNotEligible("inputs must be contiguous")


def dsa_qkva_stacked(x, w8, scales, m_cap=None):
    """Return bf16 ``[M, O]`` — the stacked q_a | kv_a fp8 GEMV in one launch.

    ``w8``/``scales`` come from :func:`stack_dsa_qkva` (or an equivalent
    loader-side cat). Row ``r < o_q`` is bit-identical to
    ``dense_gemv_fp8(x, qa_w8, qa_scales)`` row ``r``; rows ``>= o_q`` to the
    kv_a call — same kernel, same reduction, concatenated storage only.
    ``m_cap`` mirrors ``dense_gemv_fp8``'s (floe passes its
    ``moe_decode_max_tokens``/gate8 knob). Raises
    :class:`~vkernels.torch_ops._dispatch.OpNotEligible` outside the
    contract so the caller falls back to the two separate GEMVs.
    """
    from .glm_dense_fp8_gemv import dense_gemv_fp8

    return dense_gemv_fp8(x, w8, scales, m_cap=m_cap)


def dsa_qkva_stacked_reference(x, qa_w8, qa_scales, kva_w8, kva_scales):
    """Eager oracle: the two separate dense_gemv_fp8 calls, concatenated.

    Any device CUDA-only (the oracle reuses the per-matrix reference); used
    by the bit-identity gate in the tests.
    """
    from .glm_dense_fp8_gemv import dense_gemv_fp8_reference

    import torch

    q = dense_gemv_fp8_reference(x, qa_w8, qa_scales)
    kv = dense_gemv_fp8_reference(x, kva_w8, kva_scales)
    return torch.cat([q, kv], dim=-1)
