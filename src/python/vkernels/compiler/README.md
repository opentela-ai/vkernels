# CuTe DSL Megakernel Model Compiler

Implementation of the compiler proposed in
[`cutedsl_megakernel_compiler_design.md`](./cutedsl_megakernel_compiler_design.md):
a GPT-2 single-token decode step is captured as an operator graph, lowered
into logical-tile **task families**, and scheduled as **one phase-synchronous
persistent kernel** — one launch for the whole model instead of one launch per
operator.

```python
from vkernels.compiler import GPT2Config, KVCache, compile_model, random_weights

cfg = GPT2Config()  # B=1, C=128, H=4, F=512, L=2 (29 phases)
exe = compile_model(model_config=cfg, weights=random_weights(cfg), workers=4)
cache = KVCache(cfg)
for p in range(cfg.cache_capacity):  # sequential single-token decode
    logits, trace = exe.run(ids[p : p + 1], cache, p)  # mode="reference"
    assert trace.kernel_launches == 1 and trace.grid_barriers == 29
```

## Module map (design §14 → code)

| Module | Design sections | Milestone |
|---|---|---|
| `capture.py` | §4 recording backend, symbolic values, cache effects | M1 |
| `operator_ir.py` | §5 tensors, views, memory regions, RAW/WAR/WAW hazards | M1 |
| `model_gpt2.py` | §4.1 GPT-2 body + independent NumPy oracle | M1 |
| `model_qwen3.py` | dense-Qwen3 body (RMSNorm/RoPE/QK-norm/GQA/SwiGLU) + oracle + `weights_from_hf` | M1 |
| `legality.py` | §3.3 supported-subset checks and diagnostics | — |
| `task_ir.py` | §6 TaskFamily contract, tile domains | — |
| `lowerings/` | §3.1, §6.2, §6.4 task-template registry (16×16 GEMM tiles, …) | M2 |
| `schedule_phase.py` | §8 operator-ordered persistent schedule, coverage/barrier invariants | M3 |
| `memory.py` | §9 workspace lifetime reuse, §7.2 scratch rule | M4 |
| `schedule_static.py` | §10 tile-region dependencies, §10.4 static worker orders | M5 |
| `codegen_cute.py` | §8.1 CuTe DSL megakernel source emission | M3 |
| `device_triton.py` | §6 task templates + §8 schedule on GPU (Triton backend, Milestone-0-verified grid barrier) | M0/M3 |
| `reference_exec.py` | §15.1 CPU validation of the compiled schedule | — |
| `compile.py` | §13 `compile_model` API + capability report | — |
| `runtime/` | §7.4 launch, §8.4 grid-barrier contract, §11 ready-task spec | M0/M5 |

## What is validated (and what deliberately is not)

Validated on CPU (`tests/python/test_megakernel_compiler.py` (this repo), no GPU required):

- capture fidelity: 29 unfused arithmetic/cache phases, aliasing (tied head),
  symbolic position, cache-effect ordering (§5.2 hazards);
- tile counts of the §6.4 worked example (QKV 24, MLP up 32, MLP down 8);
- coverage/ownership for any worker count P; uniform barrier participation;
- task-precise regions: one attention head depends only on its own QKV tiles;
- workspace reuse with a lifetime soundness proof + NaN-canary protocol;
- whole-model logits and KV contents vs. an independent NumPy oracle across
  the full cache capacity, batch sizes, and P ∈ {1,…,9};
- one simulated kernel launch per invocation (§15.2 event counting);
- strict-mode refusal: the device path raises until the Milestone-0 grid-sync
  primitive is verified.

**Not claimed:** CuTe execution. Per design §8.4, no `cute.arch.grid_sync()`
API is established on the pinned CuTe version; the generated CuTe kernel
routes all grid synchronization through a single `grid_sync()` seam that
raises until the Milestone-0 producer–barrier–consumer microprogram passes
on the target CuTe stack.

**What IS executed on GPU:** the Triton device backend
(`device_triton.py`) implements the same §6 task templates and §8 phase
schedule with a Milestone-0-verified monotonic-counter grid barrier
(GPU-scope acq_rel atomics; `verify_milestone0()` runs the §8.4
microprogram — repeated barriers, cross-block visibility, idle workers,
repeated invocations — and flips the strict-mode capability flag). On this
GB10 it serves dense-Qwen3 decode in exactly one kernel launch per step,
validated against the HF-checked oracle (7.6e-7 logits at the real 0.6B
dims). Benchmarks: `bench/bench_megakernel_qwen3.py` (floe repo).

## Lightning-indexer lane (issue #97)

DSA/GLM-indexer decode support (`indexer_scores` + `index_topk`), the
static-task-grid + runtime-indirection pattern of #94: selection is
data-dependent but the decode output count is static (`k = index_topk`),
and the i32 indirection table orders the attention scores/values tasks
(#95/#96) through the existing RAW-hazard phase barrier.

- `ops.indexer_scores(q, entries, mix_w)` — per-(batch, head) ReLU scoring
  of the compressed entries (`scale = head_dim**-0.5`), f32 fused head mix;
  q/entries may be stored bf16 (`.cg` streamed on device).
- `ops.index_topk(scores, valid_counts, k)` — fixed-count selection over
  the fused scores, masked by per-row valid candidate counts: descending
  score, deterministic lowest-index tie-break, NaN scores in the valid
  prefix excluded; outputs the i32 indirection table `[B, k]` plus the
  normalized block_bias `[B, k]` (`s_j / ||s_valid||_2`); `-1/0.0` slots
  beyond a row's valid count (ragged candidates, `k > valid_count` ok).
  HCA variant (no indexer): `k = capacity` — same ops, selection trivial.
- Device templates `_t_indexer_scores` (one task per batch × 64-entry tile)
  and `_t_index_topk` (one task per batch row; rank-by-comparison-counting
  sweep, no sort/scratch, M ≤ ~1k). CPU oracle suite:
  `tests/python/test_megakernel_indexer_topk.py`. The Triton templates are
  CUDA-gated and unverified on CPU-only stacks.

## kvaas integration

The device backend addresses its KV cache through the **kvaas data
plane**: token-slot-major pool buffers (one slot = one token's K or V for
one layer, `n_kv_heads × head_dim × dtype` bytes) indexed by a **slot
table** — the identity table locally, `Admission.block_table()` under a
lease. `attach_megakernel_pool()` requests the pool from a live kvaasd as
one packed buffer per side (daemon-placed per-layer buffers have
non-uniform VMM strides and are rejected with a diagnostic), imports it
via CUDA IPC, and `TritonMegakernel(k_cache=..., v_cache=...)` decodes
straight into daemon-owned memory. Admit/commit stay on the host around
the single launch, matching the design doc's boundary (§1.1/§13.4).

Validated (`tests/test_megakernel_kvaas.py` (floe repo)): permuted leases decode
identically; a full `admit → block_table → decode → commit` lifecycle
against `FakeResidency` gets a content-addressed prefix hit on turn 2 and
continues turn-1's KV with zero re-prefill drift; a live kvaasd decode
matches the oracle. Measured at 0.6B dims: the daemon-owned pool costs
~0.02 ms/step vs local storage (106.4 vs 106.7 tok/s), bf16-cache logits
within 4.8e-3 of the fp64 oracle.

## Qwen3-0.6B

The dense-Qwen3 frontend compiles the published 0.6B dims (C=1024, L=28,
16Q/8KV heads, D=128, F=3072, SwiGLU, RMSNorm, tied head) into a
479-phase / ~34k-task schedule; `weights_from_hf` adapts an HF checkpoint.
The oracle is validated against HF `Qwen3ForCausalLM` itself (residual =
HF's fp32 RoPE tables). Measured on this GB10 (`bench/bench_megakernel_qwen3.py` (floe repo),
bf16, B=1): eager 14.2 ms/step (2,054 launches), CUDA-graph replay
10.6 ms/step, **megakernel 9.2 ms/step (108.7 tok/s) in exactly one
launch** — the single persistent launch beats the graph baseline by
~1.4 ms/step net of 478 in-kernel grid barriers, because it also reclaims
inter-kernel gaps; the remaining ~4.8 ms gap to the 4.4 ms weight-read
floor is device-side (memory-efficiency of the 16-wide GEMV tiles), not
launch overhead.

## B>1 batching and cross-request prefix hits

The backend decodes **B sequences per launch** — the design doc's §1.1
"each sequence supplies one token" contract. Task domains grow by B where
per-row or per-head (row norms, per-(b, h) QK-norm/RoPE/attention, per-(b,
kv-head) append); projection tiles compute B rows per task. Every row
carries its **own runtime position and slot-table row**, so batches are
naturally ragged: rows at different lengths, some continuing committed
kvaas prefixes, some cold — one persistent launch serves them all.

Measured (bf16, real 0.6B dims, one launch per step): B=1 108 tok/s →
B=2 194 → B=4 331 → B=8 459 tok/s — the weight-read cost amortizes across
the batch as expected. **Cross-request prefix sharing**: R=4 requests
sharing a 4-token prompt — cold serving costs 24 launches / 239 ms; with
the committed prefix found by content and the continuations batched,
**6 launches / 73 ms (3.3×), 12 prefill token-steps skipped, logits
bit-identical** (`bench/bench_megakernel_qwen3.py` (floe repo) rows f/g,
`tests/test_megakernel_batched.py` (floe repo)).

Two robustness notes from this work, both diagnosed and fixed with
reproducers: per-row slot leases must be *disjoint* (independent
permutations of one slot universe alias — a real daemon's free list never
does), and the per-step token/position H2D transfers must be blocking on
this stack (async pinned copies raced the kernel queue under deep
pipelines); cross-phase workspace/pool loads use `.cg` (L2-coherent) as
belt-and-braces for the barrier protocol.

## Qwen3.8-27B-FP8 + DFlash2: full hybrid megakernel (validated)

The 27B target is a Qwen3.5-family hybrid (64 layers: 48 GDN linear
attention + 16 full attention, gated attention, DeepSeek-style 128x128
block-FP8 weights). It is now served as **one persistent launch per
decode step** by `HybridMegakernel`
(`floe/engine/compiler/device_triton_hybrid.py`) — the full model, not
just building blocks. The task graph embeds token, runs 48 GDN layers
(11 barriers each: input-norm, fp8 qkv/z/a/b, conv, per-head delta rule,
fp8 out-proj, residual, post-norm, fp8 gate/up, SwiGLU, fp8 down, residual)
and 16 full-attention layers (14 barriers each: input-norm, fp8 q/k/v,
QK-norm, RoPE+KV-append, paged scores, softmax, values+output-gate, fp8
o-proj, residual, post-norm, fp8 gate/up, SwiGLU, fp8 down, residual), then
final-norm + lm-head GEMV — **754 in-kernel grid barriers, one launch**.

Validated against the repo's own reference (`Qwen35ForCausalLM`, bf16
weights dequantized from the SAME block-fp8 checkpoints) on the real
248 320-vocab, 64-layer, 5120-hidden target (`bench/bench_megakernel_27b.py` (floe repo),
`tests/python/test_megakernel_27b.py` (this repo)):

* logits across a 4-token decode: **1.4–3.4e-3 max rel err** (top-5
  overlap 5/5, argmax match);
* state evolution after 4 steps: ssm 8.6e-3, conv 4.8e-2, KV-cache 4.8e-2
  (absolute — the fp8-vs-bf16 projection error accumulates differently
  per channel; the logits are the true gate);
* **§15.2 launch check: exactly 1 CUDA kernel per 64-layer decode step**;
* **122 ms/step → 8.2 tok/s** (min of 20, B=1, fp8 weights GPU-resident
  at 29.7 GB — well within the 128 GB unified budget; reference freed
  before megakernel build).

Two bugs found and fixed during integration, both with reproducers:
the **V pool was never populated** (the RoPE-append task wrote k but not
v to the KV cache — attention output was zero at all but the last cached
position), and the **input/post/final RMSNorms and QK-norms used the
standard `×w` instead of the model's gemma `×(1+w)`** convention (every
layer was ~18% off). The out-projection GEMV and the residual add are in
*separate* barrier phases (GDN 6→7, FA 9→10) to avoid the within-phase
read-after-write race that fused them.

## fp8-blockwise GEMV linear in the compiler IR (issue #91)

The compiler lane gained a first-class fp8 decode projection:
`ops.linear_fp8(x, w_fp8, scale)` records the `linear_fp8` op variant
(`weight_layout="fp8_block"`, the scale tensor as the second weight
external — DeepSeek-style 128×128 block-FP8, e4m3 weights [N, K]
row-major, fp32 scales [ceil(N/128), K/128] block-major; ragged trailing
N blocks allowed). The lowering (`gemv_fp8` kind) reuses the dense
linear's GEMV tile domain — one task per (m, 16-column) output tile with
a full-K sweep — and each task reads exactly one scale row (a 16-column
tile always lies inside one 128-wide N block) plus one fp32 scale per
128-deep k-block. The reference body dequantizes tile-exactly and runs
the matmul in fp64; the device side is the already-validated
`_t_gemv_fp8` / `_h_gemv_fp8` Triton template (dequant-in-register, fp32
accumulate, no intermediate dequantized tensor). The CPU oracle suite
(`tests/python/test_linear_fp8_compiler.py`) proves reference-executor ↔
fp64-oracle equivalence (1e-12), the worker-stride device-template
mirror (1e-5, odd m / ragged N / block boundaries), and the bf16-reference
vs fp8 GEMV rel-err bound (measured 2.6–3.1e-2 for a single K=512 GEMV;
the 27B end-to-end logits gate is 1.4–3.4e-3 across 64 layers). MXFP4
expert weights (glm/deepseek checkpoints) are a dtype follow-up on the
same task shape.
