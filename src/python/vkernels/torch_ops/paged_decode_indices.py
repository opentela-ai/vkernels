"""Compact caller-owned slot metadata into CSR order in one GPU launch.

This copies integer slot IDs, never KV values. The caller supplies validated
CSR offsets on the same ordered stream: offsets start at zero, end at the
output size, and each row length is within the block-table width.
"""
from __future__ import annotations

import functools

import torch

from ._dispatch import require


@functools.lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def compact(TABLE, INDPTR, OUT, STRIDE: tl.constexpr, WIDTH: tl.constexpr,
                BLOCK: tl.constexpr):
        row = tl.program_id(0)
        begin = tl.load(INDPTR + row)
        end = tl.load(INDPTR + row + 1)
        column = tl.arange(0, BLOCK)
        valid = (column < WIDTH) & (column < end-begin)
        value = tl.load(TABLE + row*STRIDE + column, mask=valid, other=0)
        tl.store(OUT + begin + column, value, mask=valid)

    return compact


def pack_paged_decode_indices(table, indptr, output):
    """Copy each row's valid prefix to a nonoverlapping CSR output buffer.

    Shapes/dtypes/aliasing are checked before writes. CSR value invariants
    are the caller's responsibility; validating GPU values here would add a
    device synchronization to every decode step. Contract misses raise
    OpNotEligible so callers may retain an eager metadata path.
    """
    require(table.ndim == 2 and 0 < table.shape[0] <= 256
            and 0 < table.shape[1] <= 8192, 'invalid CSR table geometry')
    batch, width = table.shape
    require(indptr.shape == (batch+1,) and output.ndim == 1
            and 0 < output.numel() <= batch*width, 'invalid CSR offset/output shapes')
    require(all(t.is_cuda and t.device == table.device and t.dtype == torch.int32
                for t in (table, indptr, output)), 'CSR metadata must be int32 on one GPU')
    require(table.stride(1) == 1 and table.stride(0) >= width
            and indptr.is_contiguous() and output.is_contiguous(), 'unsupported CSR metadata layout')
    output_start, output_end = output.data_ptr(), output.data_ptr()+output.numel()*4
    for tensor in (table, indptr):
        start = tensor.data_ptr()
        end = start+(sum((size-1)*stride for size, stride in zip(tensor.shape, tensor.stride()))+1)*4
        require(output_end <= start or end <= output_start, 'CSR output must not overlap inputs')
    block = 1 << (width-1).bit_length()
    _kernel()[(batch,)](table, indptr, output, table.stride(0), width, block,
                       num_warps=8 if block > 4096 else 4)
