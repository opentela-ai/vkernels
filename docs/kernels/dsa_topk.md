# dsa_topk — GLM-5.3 DSA pool-level top-k transform (gfx942 / MI300A)

The pool-level top-k transform that sits **between** the DSA indexer logits
and the sparse-attention forward (issue #58). An external indexer has
already produced, per query row, a vector of pool-group logits; this
transform selects `token_topk / pool_size` pool groups per row, expands
each selected group to `pool_size` token ids, and optionally remaps the raw
token indices through a page table or a per-row ragged offset. It is the
native (gfx942 HIP) reimplementation of sglang's kpool transform, replacing
the Triton kernel that JIT-aborts on MI300A.

```
score : [batch_size, score_stride]  float32, pool-group logits (row-major)
lengths : [batch_size]              int32, valid group count per row
dst_token_indices : [batch_size, out_cols]  int32 (the selected token ids)
```

Per row the transform selects `group_topk = token_topk / pool_size` pool
groups, expands each winner to `pool_size` token ids, and writes them
contiguously into `dst_token_indices[row, :]`. The remaining arguments are
all optional remaps:

- `page_table` (`[batch_size, page_table_stride]` int32) + `page_table_row_index`
  (`[batch_size]` int32): map each score row to a page-table row, then map
  raw token indices through `page_table[page_row, token]`. `page_table` and
  `topk_indices_offset` are mutually exclusive.
- `topk_indices_offset` (`[batch_size]` int32): a per-row ragged offset added
  to every raw token index.
- `row_starts` (`[batch_size]` int32): offsets the valid score range within
  each row.
- `seq_lens` (`[batch_size]` int32): when present, `out_cols` must equal
  `token_topk + pool_size - 1` and the final partial pool
  (`seq_len % pool_size`) is appended after the selected history tokens.

## Two-implementation model

| Operation | CPU (`dsa_topk.cpp`) | HIP (`dsa_topk.hip`) |
|---|---|---|
| pool-level top-k transform | `dsa_topk_transform_cpu` | `dsa_topk_transform` |
| group-top-k support query | `dsa_topk_transform_group_topk_supported` | — |

The HIP kernel runs one 1024-thread workgroup per score row, performing the
same two-stage radix selection as sglang's reference kernel. All pointers
are device pointers; the launch uses the default stream.

- **Source (CPU)**: `src/c/vkernels/kernels/dsa_topk.cpp` (oracle, always
  compiled)
- **Source (HIP)**: `src/c/vkernels/kernels/dsa_topk.hip`
  (`VKERNELS_HAS_HIP`); shared device helpers in `dsa_topk_device.cuh`
- **Header**: `src/c/vkernels/kernels/dsa_topk.hpp`
- **C ABI**: `vk_dsa_topk_transform`, `vk_dsa_topk_group_topk_supported`
  (host, `src/c/vkernels/capi/capi_attn.cpp`); `vk_hip_dsa_topk_transform`
  (device, `src/c/vkernels/capi/hip_capi.cpp`)
- **Python**: `vkernels.kernels.dsa_topk_transform(score, lengths, *,
  pool_size, token_topk, out_cols=None, page_table=None,
  page_table_row_index=None, topk_indices_offset=None, row_starts=None,
  seq_lens=None, out=None)`

## Tests & benchmark

- **Host**: `tests/kernels/attn/test_dsa_topk.cpp` — `dsa_topk_transform_cpu`
  vs the expected expansion/remap, including the `page_table`,
  `topk_indices_offset`, and `seq_lens` tail cases.
- **Device**: `meta/benchmarks/test_dsa_topk_correct.cu` /
  `test_dsa_topk_correct.hip` (`hip::dsa_topk_transform` vs the CPU oracle),
  `meta/benchmarks/test_capi_dsa_topk.hip` (the C ABI wrapper).
- **Benchmark**: `meta/benchmarks/bench_dsa_topk.hip`.

This is the stage that feeds the sparse forward ([dsa.md](dsa.md)) and the
kpool-cache path ([dsa_kpool.md](dsa_kpool.md)); see
[../performance/dsa-topk/gfx942.md](../performance/dsa-topk/gfx942.md) for
measured throughput.
