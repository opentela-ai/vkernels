#!/usr/bin/env python3
"""Exhaustive + SMT equivalence check for the tiled-GEMM index map.

A tiled GEMM reorders the naive ``(i, j, k)`` loop nest but must visit exactly
the same set of operand/accumulator indices and write every output element
once.  This is the property that mechanical translation of a tiled kernel
(`src/c/vkernels/kernels/gemm_bf16.hip` and the tile loops in `dsa`/`mla`)
must preserve; an off-by-one or wrong stride here produces a kernel that is
numerically wrong only near tile boundaries -- exactly the class of bug
sampled testing tends to miss.

Two layers:

* **Exhaustive** (always runs, no dependencies): enumerate the naive and the
  tiled traversal over a matrix of shapes and tile sizes and compare the
  resulting index sets and write multiplicities.
* **Symbolic** (only if ``z3`` is importable): prove, for symbolic tile-local
  indices, that the tile flattening is injective onto its range, so no two
  local iterations collide on the same accumulator slot.

Exit status: 0 on success, 1 on a failed check, 77 if the symbolic layer is
requested with ``--require-z3`` and Z3 is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product

Index3 = tuple[int, int, int]
Index2 = tuple[int, int]


def naive_triples(m: int, n: int, k: int) -> list[Index3]:
    """The reference loop order: i, then j, then k."""
    return [(i, j, kk) for i in range(m) for j in range(n) for kk in range(k)]


def tiled_triples(m: int, n: int, k: int, bm: int, bn: int, bk: int) -> list[Index3]:
    """A blocked traversal in tile order: (ti, tj, tk, ii, jj, kk)."""
    out: list[Index3] = []
    for ti in range(0, m, bm):
        for tj in range(0, n, bn):
            for tk in range(0, k, bk):
                for ii in range(ti, min(ti + bm, m)):
                    for jj in range(tj, min(tj + bn, n)):
                        for kk in range(tk, min(tk + bk, k)):
                            out.append((ii, jj, kk))
    return out


def check_shape(m: int, n: int, k: int, bm: int, bn: int, bk: int) -> list[str]:
    errors: list[str] = []
    naive = naive_triples(m, n, k)
    tiled = tiled_triples(m, n, k, bm, bn, bk)

    # Same read set, same number of reads (multiset equality).
    if sorted(naive) != sorted(tiled):
        missing = sorted(set(naive) - set(tiled))
        extra = sorted(set(tiled) - set(naive))
        errors.append(
            f"[{m}x{n}x{k} tile {bm}x{bn}x{bk}] read set differs: "
            f"missing={missing[:4]} extra={extra[:4]}"
        )

    # Every output (i, j) is written exactly once -- count accumulator
    # updates per output slot in the tiled order.
    writes: dict[Index2, int] = {}
    for i, j, _ in tiled:
        writes[(i, j)] = writes.get((i, j), 0) + 1
    bad = {ij: c for ij, c in writes.items() if c != k}
    if bad:
        sample = list(bad.items())[:4]
        errors.append(f"[{m}x{n}x{k} tile {bm}x{bn}x{bk}] bad write counts: {sample}")
    if len(writes) != m * n:
        errors.append(
            f"[{m}x{n}x{k} tile {bm}x{bn}x{bk}] covered {len(writes)}/{m * n} outputs"
        )
    return errors


def exhaustive() -> list[str]:
    # Divisible and non-divisible shapes, square and ragged tiles, to exercise
    # the min() clamping at the tails.
    shapes = [(4, 4, 4), (6, 4, 8), (8, 8, 8), (5, 7, 3)]
    tiles = [(2, 2, 2), (4, 4, 4), (3, 2, 3), (4, 4, 8)]
    errors: list[str] = []
    checked = 0
    for (m, n, k), (bm, bn, bk) in product(shapes, tiles):
        errors += check_shape(m, n, k, bm, bn, bk)
        checked += 1
    print(f"exhaustive: {checked} shape/tile combinations checked")
    return errors


def symbolic() -> list[str]:
    """Prove tile-flattening injectivity for symbolic tile-local indices.

    For a tile of shape (BM, BN, BK) the element (ii, jj, kk) maps to the
    flattened offset ((ii*BN)+jj)*BK+kk.  Proving injectivity of that map over
    the tile guarantees the tiled kernel never aliases two local iterations
    onto one accumulator/operand slot.
    """
    try:
        import z3  # type: ignore
    except ImportError:
        return ["z3-not-available"]

    bm, bn, bk = z3.Ints("bm bn bk")
    i1, j1, k1, i2, j2, k2 = z3.Ints("i1 j1 k1 i2 j2 k2")
    solver = z3.Solver()
    tile_bounds = [
        bm >= 1, bn >= 1, bk >= 1,
        i1 >= 0, i1 < bm, j1 >= 0, j1 < bn, k1 >= 0, k1 < bk,
        i2 >= 0, i2 < bm, j2 >= 0, j2 < bn, k2 >= 0, k2 < bk,
    ]
    flat1 = ((i1 * bn) + j1) * bk + k1
    flat2 = ((i2 * bn) + j2) * bk + k2
    # Look for a collision between two distinct local indices.
    solver.add(*tile_bounds)
    solver.add(flat1 == flat2)
    solver.add(z3.Or(i1 != i2, j1 != j2, k1 != k2))
    if solver.check() == z3.sat:
        model = solver.model()
        return [f"symbolic: tile flattening collides under {model}"]
    print("symbolic: tile flattening proven injective for all BM,BN,BK >= 1")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-z3",
        action="store_true",
        help="exit 77 instead of skipping the symbolic layer when z3 is absent",
    )
    args = parser.parse_args()

    errors = exhaustive()
    sym = symbolic()
    if sym == ["z3-not-available"]:
        msg = "z3 not installed; symbolic layer skipped (pip install z3-solver)"
        if args.require_z3:
            print(msg)
            return 77
        print(f"NOTE: {msg}")
    else:
        errors += sym

    if errors:
        print("FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("OK: tiled index map is equivalent to the naive map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
