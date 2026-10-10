# Coordinated serving consolidation, 10 October 2026

Use branch `consolidate/serving-stack-20261010` in the sibling Floe, vkernels
and kvaas repositories. Floe's `.github/dependencies.env` pins the exact
coordinated vkernels/kvaas commits. Preserve the sibling layout.

This branch combines remote main `edf0a5c`, the campaign kernel lineage ending
at `bd9abf6`, and the extra local work explicitly requested for consolidation.
Snapshot `7efb396` preserves all 34 modified and 99 untracked source, test,
benchmark, verifier and compact experiment-report files from the original
checkout. The original checkout was left unchanged. Generated build products,
checkpoints and raw profiler captures are not part of this consolidation.

Included work covers kernel capability/signature registries and dispatch,
benchmark and numerical tooling, compilation monitoring, recurrent model
compiler/lowerings, router/indexer/DSA/KDA/MLA kernels, native grouped GEMM and
MQA logits, shared launch/epilogue helpers, verification harnesses and CI.
The campaign adds backend-aware grouped-MoE geometry and the optional
owner-held SGLang push-reduction plan, with its upstream license/provenance.

Integration preserves both registry-selected expert GEMV and the pre-existing
CUDA-native zero-route API. A new test checks active/skipped routes and changed
inputs under graph replay, and rejects portable dispatch for the zero-route
contract. Offline import tests disable installed editable finders. The MLA
oracle now computes attention independently for each query head; sharing K/V
does not make distinct query heads produce identical attention.

Validation counts and exclusions are recorded in
[`serving-stack-20261010-validation.json`](serving-stack-20261010-validation.json).
Local hardware is NVIDIA GB10, Torch2.13/CUDA13.0. Host and CUDA native builds
are tested separately. Multi-GPU/NCCL, HIP, external integration and selected
checkpoint tests retain explicit runtime/dependency skips where applicable.

The full GPU Python sweep has **1558 passes,91 skips and one failure**:
`test_sgl_fused_moe_matches_eager_serving_shape[8]`. Its58/32768 mismatches
reproduce the previously documented GB10 quant-aware elementwise gate failure;
the isolated module also reproduces it. The assertion and rtol/atol are
unchanged. This branch does not claim a fully green GPU suite or B8 numerical
qualification. Baseline confirmation is recorded in the validation JSON.

The earlier Clariden campaign measured the Floe candidate `962580b` with
vkernels `bd9abf6`: repeated B1 104.202→116.283–116.539 tok/s and B4
330.717→359.368–362.449 tok/s on GH200 TP4/EP1. **Those are historical results,
not a benchmark of this combined tree**, which additionally includes the local
kernel/compiler work. No consolidated-tree GLM quality or throughput claim is
made. The earlier mean NLL1.183960393 still fails the immutable1.149878794
ceiling. Keep optional Floe orchestration switches default-off and do not infer
precision/DeepGEMM qualification from the merge or from operator tests.
