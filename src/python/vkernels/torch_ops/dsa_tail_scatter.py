"""Fused DSA pool tail-scatter (optional): gather + pad-zero + index_copy in ONE launch.

The batched stage writeback (:class:`floe...graphs._StageWriteback.body`) scatters each DSA
layer's window-tail column into its persistent stage pool with a four-kernel eager chain
per (layer, pool):

    v = src.gather(1, win.expand(Bb, 1, D)).squeeze(1)   # SGATHER (the tail column)
    v = torch.where(valid[:, None], v, torch.zeros_like(v))  # FILL + WHERE (pad guard)
    dst.index_copy_(0, slots, v)                         # IDXCOPY

i.e. 4 launches x 2 pools x 11 DSA layers = 88 mostly-launch-bound nodes per decode step
(measured: ~0.14 ms of the g29 writeback graph at bs=1). This op folds the chain into one
kernel:

    dst[slots[b], d] = valid[b] ? src[b, win, d] : 0

Bit-identity: the chain is pure data movement — gather copies, the pad guard selects
between the loaded value and exact zero, ``index_copy_`` copies again — so the fused kernel
reproduces every stored byte exactly (bf16/fp32 values are only widened for the select and
narrowed back, an exact round-trip). Slot collisions are safe under the same contract the
eager form relies on: live rows' slots are allocator-unique, and every writer that collides
on the scratch page writes identical zeros (order-independent). Padded rows therefore must
pass ``valid=False`` (the caller already routes them to the scratch page; this kernel writes
the zeros that keep that page zero).

``win`` arrives as a device int64 scalar (a captured-graph staging buffer — reading it
in-kernel keeps the op capture-safe: the graph updates the buffer between replays and the
kernel dereferences the same pointer each time).

Torch and Triton load lazily. Inference-only, in-place on ``dst``, no autograd.
"""

from ._dispatch import OpNotEligible, same_gpu_contiguous
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _tail_scatter(
        SRC,
        DST,
        WIN,
        SLOTS,
        VALID,
        src_row_stride,
        src_win_stride,
        dst_row_stride,
        D: tl.constexpr,
        BD: tl.constexpr,
    ):
        b = tl.program_id(0)
        cb = tl.program_id(1)
        cols = cb * BD + tl.arange(0, BD)
        cmask = cols < D
        live = tl.load(VALID + b) != 0
        # The window column is a device scalar: dereferenced in-kernel so a
        # captured graph can update it between replays (capture-safe).
        w = tl.load(WIN)
        x = tl.load(
            SRC + b * src_row_stride + w * src_win_stride + cols,
            mask=cmask,
            other=0,
        ).to(tl.float32)
        val = tl.where(live, x, 0.0)
        slot = tl.load(SLOTS + b)
        tl.store(
            DST + slot * dst_row_stride + cols,
            val.to(DST.dtype.element_ty),
            mask=cmask,
        )

    return _tail_scatter


def dsa_tail_scatter(dst, src, win, slots, valid):
    """Scatter ``src[:, win, :]`` into ``dst[slots]``, zeroing padded rows, in one launch.

    ``dst`` ``[S, D]`` contiguous pool rows (written in place, returned for fluency);
    ``src`` ``[Bb, cap, D]`` contiguous, same dtype as ``dst``; ``win`` device int64
    0-dim/[1] scalar (the window tail column); ``slots`` ``[Bb]`` int64 destination
    rows; ``valid`` ``[Bb]`` bool live-row mask. Every element of ``dst`` touched by a
    program is overwritten (pads write exact zeros), matching the eager
    gather -> zeros_like -> where -> index_copy_ chain bit for bit under the
    collision contract documented in the module docstring. Inference only.
    """
    import torch

    if dst.ndim != 2 or src.ndim != 3:
        raise OpNotEligible("expected dst [S, D] and src [Bb, cap, D]")
    bb, cap, dim = src.shape
    if dst.shape[1] != dim:
        raise OpNotEligible("dst/src feature dims must match")
    if cap <= 0:
        raise OpNotEligible("src window axis is empty")
    if slots.shape != (bb,) or valid.shape != (bb,):
        raise OpNotEligible("slots/valid must be [Bb]")
    if slots.dtype != torch.int64:
        raise OpNotEligible("slots must be int64")
    if valid.dtype != torch.bool:
        raise OpNotEligible("valid must be bool")
    if win.dtype != torch.int64 or win.numel() != 1 or not win.is_cuda:
        raise OpNotEligible("win must be a single int64 device scalar")
    if src.dtype != dst.dtype or dst.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise OpNotEligible("dst/src must share one BF16/FP16/FP32 dtype")
    same_gpu_contiguous(dst, src, slots, valid)
    if bb and dim and cap:
        import triton  # lazy: validation above needs only torch

        with torch.cuda.device(dst.device):
            _kernel()[(bb, triton.cdiv(dim, 512))](
                src,
                dst,
                win.reshape(1),
                slots,
                valid,
                src.stride(0),
                src.stride(1),
                dst.stride(0),
                dim,
                512,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return dst


def dsa_tail_scatter_reference(dst, src, win, slots, valid):
    """Eager oracle mirroring the incumbent four-kernel chain (no mutation)."""
    import torch

    bb, _, dim = src.shape
    v = src.gather(1, win.reshape(1, 1, 1).expand(bb, 1, dim)).squeeze(1)
    v = torch.where(valid.unsqueeze(1), v, torch.zeros_like(v))
    return dst.index_copy(0, slots, v)
