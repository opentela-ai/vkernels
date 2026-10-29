# Indexed shared-latent MLA for training

`vkernels.torch_ops.indexed_mla_train` supplies a torch oracle and Triton
forward/backward for per-query, unstructured key selection with a learned
additive gate. It is an experimental training primitive. The first measured
implementation saves memory but is slower than the trainer's efficient SDPA.

## Contract

| Input | Shape | Meaning |
| --- | --- | --- |
| `q` | `[B,S,H,R]` | Query with the per-head K projection absorbed |
| `qr` | `[B,S,H,P]` | Rotated query tail; `P=0` is supported |
| `kv` | `[B,S,R]` | Shared normalized key/value latent |
| `kr` | `[B,S,P]` | Shared rotated key tail |
| `indices` | `[B,S,K]` | Selected key positions, int32/int64 |
| `gate` | `[B,S,K]` | Additive attention bias, differentiable |

For each query/head, logits are
`scale * (dot(q, kv[key]) + dot(qr, kr[key])) + gate[key]`.
Softmax is over valid selected entries. The output `[B,S,H,R]` is the
probability-weighted shared latent. Negative and out-of-range indices are
ignored. Empty rows return zero output and gradients. Duplicate valid indices
are separate softmax entries: the caller must deduplicate a set union.

Device inputs must be contiguous on one CUDA/ROCm device; Q/K/V and their
RoPE tails share BF16 or FP16 dtype. The CPU oracle uses FP64 for FP64 input
and otherwise FP32. Only BF16 on ROCm/gfx942 has been measured here.

`indexed_mla_forward(...)` returns `(out, out_fp32, lse_fp32)`.
`indexed_mla_backward(dout, ..., out_fp32, lse_fp32, scale)` returns gradients
for `(q, qr, kv, kr, gate)`. The trainer wraps these in a first-order autograd
Function and owns projection absorption/expansion. The primitive does not
select keys or implement the indexer.

Forward streams selected keys with online softmax and never expands the
shared latent into `[B,S,H,K,R]`. Backward recomputes probabilities and sums
shared-key gradients with FP32 atomics. The optional `grad_splits` argument
partitions query-row accumulation; default `min(8, ceil(S/256))`. The final
reduction and output casts happen after accumulation. Atomics make the
reduction order nondeterministic. Dot operands, including probabilities and
score derivatives, are rounded to the model dtype for tensor-core GEMMs.

## Validation and performance

Host tests cover FP64 gradcheck, ignored indices and empty rows. Device
tests compare all five derivatives against the oracle, including empty rows,
two batches, zero-length RoPE and a split accumulation. The predeclared gate
is relative L2 error below 1% for outputs and every gradient.

Trainer integration tests also compare the compact top-k/window/sink union
and learned gates under bidirectional, causal and block-causal layouts.
FP64 absorbed projections match the unabsorbed computation and all weight
gradients. BF16 absorption changes intermediate rounding points and therefore
is an explicit numeric mode.

Final trained-layer measurement (beverin job 693832, torch 2.9.1+rocm6.3,
Triton 3.5.1, BF16, B=1, S=4096, H=16, R=512, P=64):

| Implementation | Forward + backward median | Peak allocated |
| --- | ---: | ---: |
| Original sparse math SDPA | 22.36 ms | 4.62 GiB |
| Zero-padded efficient SDPA | 16.52 ms | 1.23 GiB |
| Gathered latent MLA | 27.91 ms | 1.04 GiB |

Every trained-layer parameter/input gradient passed the 1% gate; maximum
relative error was below 0.46%. The final primitive, its two-batch/empty-row
tests and the trainer's backend/Liger checks all passed (46 tests). The
first version used one key-gradient accumulator and measured 28.39 ms with
0.98 GiB peak (job 693819). Kernel-only tuning at K=773 found about 2.7 ms forward and
18.6–19.0 ms backward with four warps and 16-key tiles; using 4–16
accumulators changes total time by less than 2%. Larger tiles/warp counts
were slower. This is not yet the recommended throughput path.

Real training from the frozen step-7500 checkpoint also passed six optimizer
steps with gradient checkpointing and 32-to-four-rank optimizer resharding
(job 693824): maximum paired CE difference 0.0017 and gradient-norm relative
difference 0.65%. It measured 7,645 vs 7,820 reference tokens/s at a
65,536-token global batch; peaks were 68.84 vs 77.36 GiB on rank 0.

Run the focused tests with torch and Triton already installed:

```bash
PYTHONPATH=src/python python -m pytest -q tests/python/test_indexed_mla_train.py
```

The trainer's `scripts/beverin/bench_sparse_layer.py` reproduces the trained
layer comparison. See its `docs/sparse-throughput.md` for full-model results,
source branches and experimental-job provenance.
