"""Shared tile constants and region geometry."""
from __future__ import annotations
from ..operator_ir import Region, TensorValue
from ..task_ir import TileDomain
GEMM_TILE_M = 16

GEMM_TILE_N = 16

ELEM_TILE = 1024

EMBED_TILE_C = 256

THREADS_PER_WORKER = 256

def _whole(view: TensorValue) -> Region:
    return Region.whole(view)

def _tile_region(view: TensorValue, box) -> Region:
    return Region.tile(view, box)

def _gemm_domain(x: TensorValue, w: TensorValue) -> TileDomain:
    m = x.shape[0]
    n = w.shape[1]
    return TileDomain(((m, GEMM_TILE_M), (n, GEMM_TILE_N)))

def _pair_box(domain: TileDomain, coords):
    m_idx, n_idx = coords
    (m_extent, m_tile), (n_extent, n_tile) = domain.dims
    m0, n0 = m_idx * m_tile, n_idx * n_tile
    return (m0, min(m0 + m_tile, m_extent)), (n0, min(n0 + n_tile, n_extent))

GEMV_FP8_TILE_N = 16

def _flat_box(view: TensorValue, lo: int, hi: int):
    """[lo, hi) flat-element box on a contiguous row-major view."""
    if not view.is_contiguous():
        raise ValueError(f"elementwise tiling requires contiguous views ({view.name})")
    shape = view.shape
    if len(shape) == 1:
        return ((lo, hi),)
    cols = shape[-1]
    r0, c0 = divmod(lo, cols)
    r1, c1 = divmod(hi - 1, cols)
    return ((r0, r1 + 1), (c0 if r0 == r1 else 0, (c1 + 1) if r0 == r1 else cols))

def _gdn_tile(C: int) -> int:
    """Channel-tile width for gdn_conv: ELEM_TILE when it divides C, else the
    largest power-of-two divisor that does (the device template requires an
    exact tiling, no channel masking)."""
    tile = ELEM_TILE
    while C % tile:
        tile //= 2
    return tile

INDEXER_TILE_M = 64

INDEXER_TILE_M = 64
