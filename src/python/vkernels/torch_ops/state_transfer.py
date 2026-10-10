"""Bit-preserving transfers between row pools and fixed-address stage tensors.

The caller owns and validates the pointer tables and their referenced storage.
No allocation, host scalar reads, or model arithmetic occurs during a launch.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def transfer(ptrs, strides, layers, pool, live, rows, SIZE: tl.constexpr,
                 POOL_STRIDE: tl.constexpr, POOL_ROWS: tl.constexpr,
                 BITS: tl.constexpr, GATHER: tl.constexpr, BLOCK: tl.constexpr):
        chunk, layer, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        active = b < tl.load(live)
        row = tl.load(rows + b, active, other=-1)
        valid = active & (row >= 0) & (row < POOL_ROWS)
        dtype = tl.uint16 if BITS == 16 else tl.uint32
        stage = tl.load(ptrs + layer).to(tl.pointer_type(dtype))
        stride = tl.load(strides + layer)
        pool_layer = tl.load(layers + layer)
        off = chunk * BLOCK + tl.arange(0, BLOCK)
        stage_addr = stage + b * stride + off
        pool_addr = pool + row * POOL_STRIDE + pool_layer * SIZE + off
        if GATHER:
            value = tl.load(pool_addr, valid & (off < SIZE), other=0)
            tl.store(stage_addr, value, off < SIZE)
        else:
            value = tl.load(stage_addr, valid & (off < SIZE), other=0)
            tl.store(pool_addr, value, valid & (off < SIZE))

    return transfer


def masked_state_transfer(ptrs, strides, layers, pool, live, rows, *, size,
                          pool_stride, pool_rows, bits, batch, gather=False):
    """Launch a prepared transfer; inactive gathered rows become zero.

    Stage rows may have gaps between them; each individual row is contiguous.
    Scatter destinations must be unique among the live request rows.
    """
    import triton

    _kernel()[(triton.cdiv(size, 512), layers.numel(), batch)](
        ptrs, strides, layers, pool, live, rows, size, pool_stride, pool_rows,
        bits, gather, 512, num_warps=4)
