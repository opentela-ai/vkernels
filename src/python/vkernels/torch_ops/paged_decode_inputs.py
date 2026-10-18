"""Stage integer inputs for a one-token paged decode graph in one launch.

The caller owns graph buffers and leased slot IDs. This operator copies
metadata only; it grants no slots, reads no KV values, and changes no model
arithmetic. Padded graph rows/columns retain the eager staging contract.
"""

from __future__ import annotations

import functools

import torch

from ._dispatch import require


def _outputs(outputs):
    ids, positions, lengths, table, pick = outputs
    require(ids.ndim == 2 and ids.shape[1] == 1, "only one-token decode graphs are supported")
    batch = ids.shape[0]
    require(0 < batch <= 256 and table.ndim == 2 and table.shape[0] == batch
            and table.shape[1] > 0, "invalid decode graph geometry")
    require(positions.shape == ids.shape and lengths.shape == pick.shape == (batch,),
            "graph metadata shapes must agree")
    require(all(t.is_cuda and t.device == ids.device and t.is_contiguous() for t in outputs),
            "graph inputs must be contiguous on one GPU")
    require(all(t.dtype == torch.int64 for t in (ids, positions, lengths, pick))
            and table.dtype == torch.int32, "unsupported graph input dtypes")
    return batch, table.shape[1]


@functools.lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["B", "W", "SOURCE_STRIDE"])
    def stage(IDS, POS, LENS, TABLE, PICK, OUT_IDS, OUT_POS, OUT_LENS, OUT_TABLE,
              OUT_PICK, B, W, SOURCE_STRIDE, WIDTH: tl.constexpr,
              SCRATCH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(1)
        column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(TABLE + row * SOURCE_STRIDE + column,
                        mask=(row < B) & (column < W) & (column < WIDTH), other=SCRATCH)
        tl.store(OUT_TABLE + row * WIDTH + column, value, mask=column < WIDTH)
        if tl.program_id(0) == 0:
            token = tl.load(IDS + row, mask=row < B, other=0)
            position = tl.load(POS + row, mask=row < B, other=0)
            length = tl.load(LENS + row, mask=row < B, other=0)
            choice = tl.load(PICK + row, mask=row < B, other=0)
            tl.store(OUT_IDS + row, token)
            tl.store(OUT_POS + row, position)
            tl.store(OUT_LENS + row, length)
            tl.store(OUT_PICK + row, choice)

    return stage


def _arguments(inputs, outputs, scratch_slot):
    ids, _, _, table, _ = inputs
    return (*inputs, *outputs, ids.shape[0], table.shape[1], table.stride(0),
            outputs[3].shape[1], int(scratch_slot), 256)


def stage_paged_decode_inputs(ids, positions, lengths, table, pick, outputs, *, scratch_slot):
    """Copy a live batch into static graph buffers, padding with zeros/scratch.

    Source block-table rows may be strided (a bounded view of a wider engine
    table). Every other input is contiguous int64. Source/output storage
    must not overlap; the graph's prior use must be ordered on this stream.
    All eligibility checks precede writes. Contract misses raise OpNotEligible.
    """
    inputs = (ids, positions, lengths, table, pick)
    graph_batch, graph_width = _outputs(outputs)
    batch = ids.shape[0] if ids.ndim == 2 else 0
    require(0 < batch <= graph_batch and ids.shape == positions.shape == (batch, 1)
            and lengths.shape == pick.shape == (batch,), "invalid live decode metadata shapes")
    require(table.ndim == 2 and table.shape[0] == batch and table.shape[1] > 0
            and table.stride(1) == 1 and table.stride(0) >= table.shape[1],
            "source table must have contiguous, nonoverlapping rows")
    require(all(t.is_cuda and t.device == outputs[0].device for t in inputs),
            "source metadata must be on the graph's GPU")
    require(all(t.is_contiguous() and t.dtype == torch.int64
                for t in (ids, positions, lengths, pick)) and table.dtype == torch.int32,
            "unsupported live metadata layout/dtypes")
    # Compare address ranges rather than storage identities: managed graph
    # buffers can be disjoint suballocations of one activation arena.
    source_ranges = [(t.data_ptr(), t.data_ptr() + t.element_size() *
                      (sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1))
                     for t in inputs]
    destination_ranges = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in outputs]
    require(all(a1 <= b0 or b1 <= a0 for a0, a1 in source_ranges
                for b0, b1 in destination_ranges), "live and graph metadata must not overlap")
    require(all(a1 <= b0 or b1 <= a0 for index, (a0, a1) in enumerate(destination_ranges)
                for b0, b1 in destination_ranges[index + 1:]), "graph buffers must not overlap")
    _kernel()[((graph_width + 255) // 256, graph_batch)](
        *_arguments(inputs, outputs, scratch_slot), num_warps=4)


def warmup_paged_decode_inputs(outputs, *, scratch_slot):
    """Compile staging for a captured bucket without launching or writing.

    B, source width and row stride are runtime arguments, so one compiled
    kernel covers every live batch and table view fitting this graph bucket.
    """
    batch, width = _outputs(outputs)
    _kernel().warmup(*_arguments(outputs, outputs, scratch_slot),
                     grid=((width + 255) // 256, batch), num_warps=4)
