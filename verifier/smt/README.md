# smt — index/tiling equivalence

`gemm_index_equivalence.py` checks that a **blocked** GEMM traversal is a
refinement of the naive `(i, j, k)` loop nest:

1. **Exhaustive** (always runs, no dependencies): the naive and tiled
   traversals visit the same multiset of `(i, j, k)` reads, and every output
   `(i, j)` is written exactly `K` times, across divisible and
   non-divisible shapes so the tail `min()` clamping is exercised.
2. **Symbolic** (needs `z3-solver`): proves tile flattening
   `((ii·BN)+jj)·BK+kk` is injective over a symbolic tile, i.e. no two local
   iterations alias the same accumulator slot.

```bash
./run.sh                 # exhaustive always; symbolic if z3 present
./run.sh --require-z3    # exit 77 if z3 is absent
pip install z3-solver    # enable the symbolic layer
```

The same checker applies to any `TILE_*` loop nest in
`src/c/vkernels/kernels/*.hip`: point `check_shape` at that kernel's tile
shape. It is the SMT/brute-force complement to the CBMC value proofs —
those cover arithmetic, this covers *which* memory the tile loops touch.
