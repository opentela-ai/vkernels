# `torch_ops` — device-native Triton inference operators

`vkernels.torch_ops` is the Python-side operator layer: hand-written Triton
kernels (plus a few HIP/AITER bridges) that replace eager-torch glue on the
decode/prefill hot paths of the served models — **GLM-5.3-Flash**,
**DeepSeek-V4/V4.1**, **Kimi-K3** (MLA + gated-delta attention), and
**Qwen3.5** (GDN). The kernels are *owned* here and floe imports them back
through a thin adapter (the #64/#65 model), so the same numerics contract
runs in both codebases.

- **Source**: `src/python/vkernels/torch_ops/`
- **Tests**: `tests/python/test_*.py` (CPU-oracle parity suites)
- **Benchmarks**: `bench/` (device A/B + parity drivers)
- **Harness/MI300A validation**: [torch-ops-mi300.md](torch-ops-mi300.md)
- Torch and Triton load **lazily** — importing `vkernels.torch_ops` loads
  neither. All ops are inference-only (no autograd backward).

---

## 1. The calling convention: ops validate themselves

Every public op owns its **full eligibility check** — device/arch, dtypes,
shapes, contiguity, backend availability — so model-side call sites never
pre-gate on hardware or layout. The convention (implemented in
[`_dispatch.py`](../src/python/vkernels/torch_ops/_dispatch.py)):

* **Contract miss → `OpNotEligible`.** The op cannot take the fused path for
  this input (CPU tensor, unexpected dtype, shape outside the validated
  envelope, missing Triton/AITER, non-degenerate group config, …). This is
  an *expected, routine* outcome: callers catch it and fall back to their
  eager torch path (or the op's `*_reference` oracle). `OpNotEligible`
  subclasses both `ValueError` and `TypeError`, so older call sites and
  tests that pinned either historical flavor keep working unchanged.
* **Anything else is a caller bug** — `TypeError` for structurally wrong
  arguments no fallback can fix, `RuntimeError` for operational failures
  (e.g. an unconfigured tuned backend) — and propagates.

This keeps model-side call sites to one line (`if knob: return op(...)`)
with the hardware/backend knowledge living next to the kernels it describes.

---

## 2. Module inventory

### elementwise (`elementwise.py`)

Pointwise/norm Triton ops with matching `*_reference` oracles:

| Op | Computation |
|---|---|
| `rms_norm` | RMSNorm (+ optional residual add) |
| `rms_norm_unweighted` / `rms_norm_gated` | RMSNorm without weight / with a per-head gate |
| `qk_norm` | per-head RMSNorm on q and k |
| `rotary` | GPT-J-style rotary half-rotation on q/k |
| `qk_norm_rope` | **fused** QK-RMSNorm + rotary in one kernel — the rounding sequence matches the two-kernel chain exactly (norm result rounded to input dtype before the rope multiply, each rope half rounded before the add); removes one launch and one full read+write of q/k per layer on the decode hot path |
| `silu_mul` | `SiLU(gate) * up` in one launch |
| `store_kv` | append new k/v rows into the paged KV cache via the block table |

### triton_attn (`triton_attn.py`)

Triton paged single-token decode attention (fp32 accumulation, one program
per (batch, query head), online softmax over the block-table-gathered paged
KV cache). Adopted from floe's `triton_attn.py`; the fallback attention
backend when FlashInfer is unavailable.

| Entry point | Strategy |
|---|---|
| `decode_attention` | one program per (batch, head); optional fused **new-token KV store** (`k_new/v_new` appended to the cache inside the attention pass, `scratch_slot` for the in-flight page) |
| `decode_attention_gqa` | GQA-grouped: **per-KV-head programs** walk all query heads of the group, tensor-core dots over head-dim tiles |
| `decode_attention_split` | **flash-decoding** split-KV: stage 1 computes per-split partial (max, sum, accumulator) triples into a scratch buffer, stage 2 reduces across splits — long-context decode on small batches |
| `decode_attention_reference` | eager oracle for all of the above |

### GLM-5.3-Flash ops

| Op | What it fuses |
|---|---|
| `glm_router.fused_router` | sigmoid + bias + group mask + triple `topk` + weight gather + `norm_topk_prob` + scaling in one launch over fp32 `[T, E]` logits (shipped `n_group == 1` config; other configs raise `OpNotEligible` so the grouped eager path runs) |
| `glm_kda_decode` | single-token FP32 per-dimension-gated KDA step; bf16/fp16 ABI with in-kernel widening (bit-identical to eager widening, minus 5 cast kernels per layer per step) |
| `glm_expert_gemv`, `glm_expert_gather_dequant`, `glm_fp8_blockwise_gemm` (+ `_glm_fp8_sm90_gemm`) | FP8 block-scaled MoE GEMV/GEMM path (issue #65) |
| `glm_mhc_mix`, `mhc_projection` | mHC residual-stream mixing / projection GEMV |
| `mhc_compose.mhc_collapse` / `mhc_compose` | mHC stream collapse (`Σ pre·streams`) and compose (`combᵀ·streams + post⊙out`) with an exact-fp32 matmul oracle |
| `qkv_projection` (+ `qkv_tuned_blas`) | chunked q/k/v projection GEMV, dispatch-tuned |
| `glm_projection` | narrow GLM projection GEMVs (honest negative vs tuned BLAS — see [glm53-decode-kernels.md](glm53-decode-kernels.md)) |

### DeepSeek-V4.1 ops (`v41_*.py`)

FP8 blockwise GEMM, MXFP4 dequant/GEMV, DSA indexer top-k, sparse attention
— the DeepSeek-V4.1 serving vertical (see [ORIENTATION.md](ORIENTATION.md)
§4 and the kernels-reference table for measured numbers).

### Vendored vLLM kernels

| Module | Provenance | Role |
|---|---|---|
| `vllm_kda.py` | vLLM `third_party/flash_linear_attention` (upstream flash-linear-attention, MIT; fetched 2026-02) | fla **KDA chunked prefill** (per-dim-gated delta rule) — the portable, any-GPU, graph-capture-safe tier where the HIP WY kernel (`kda_chunk_hip`) is unavailable (e.g. NVIDIA clariden), and an A/B peer on gfx942 |
| `vllm_kda_amd.py` | same lineage, AMD path | ROCm-tuned KDA prefill variants |
| `vllm_sparse_mla.py` | vLLM `rocm_aiter_mla_sparse.py` (fetched 2026-01) | Triton **sparse-MLA (DSA)** prefill/decode: `build_ragged_indices_from_dense`, `sparse_attn_prefill[_ragged]`, `sparse_attn_decode` (partial + reduce, auto split count), `dsa_sparse_fwd` |

These are vendored self-contained (no vLLM machinery); re-diff against
upstream before upgrading — the fetch commit pins are recorded in the
module docstrings.

### AITER bridge (`aiter_ops.py`)

gfx942 wrappers behind availability probes for the AMD AITER ops vLLM
dispatches on the GLM-5.3-Flash ROCm path: `aiter_mhc_pre`/`aiter_mhc_post`,
`fp8_blockscale_experts` (grouped fp8-blockscale expert GEMMs), and
`per_group_quant_fp8` / `moe_align_block_size` helpers. Each dispatch site
can A/B AITER against the vkernels HIP/Triton kernel via per-site knobs
(`aiter_mhc` / `aiter_moe`); `available()` / `report()` expose what the
bundled aiter (verified: `aiter 7a8ff7dd4`, ROCm 7.0, MI300A) provides.

---

## 3. Validation & benchmarking

- Every op ships a `*_reference` oracle; parity suites live in
  `tests/python/` and run on CPU in host CI plus device tests on the
  self-hosted runners and gfx942 boxes.
- `bench/` drivers:
  - `bench_vllm_sparse_mla.py`, `bench_sparse_mla_hip_ab.py` — sparse-MLA
    Triton vs HIP A/B
  - `vllm_kda_parity.py` — vendored KDA vs the HIP WY kernel / eager oracle
  - `kda_fault_bisect.py` — fault localization for the KDA prefill path
- MI300A harness + validation history: [torch-ops-mi300.md](torch-ops-mi300.md).
