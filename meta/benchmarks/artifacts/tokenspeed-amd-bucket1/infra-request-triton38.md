# Infra request: triton ≥ 3.8 in a beverin (MI300A) container env

**To:** CSCS service desk / owners of the `glm5-smoke` environment
**Repo:** vkernels (AMD kernel bring-up for the serving stack)
**Status:** ready to send — drafted 2026-10-10 after the measurement run in
`meta/benchmarks/artifacts/tokenspeed-amd-bucket1/`

## Ask

Please provide a container environment on the `mi300` partition (beverin)
whose `triton` is **≥ 3.8.0** (rocm7.0-compatible, x86_64, python 3.10),
either by bumping `glm5-smoke` or by adding a variant (e.g. `glm5-smoke-triton38`).
Everything else in the env can stay as-is: torch 2.9.0a0+git7bcbafe
(hip 7.0.51831-a3e329ad8), aiter, pytest.

## Why

Our AMD kernel work targets Triton **Gluon** (the new layout-aware kernel
API that AMD's recent open-source kernels — e.g. lightseekorg/tokenspeed —
are written against). The env's stock `triton==3.4.0` ships the first
experimental gluon and is feature-incomplete for every kernel in that
family; specifically missing:

- `triton.experimental.gluon.constexpr_function` (needed by any kernel with
  layout-helper functions)
- `gl.histogram`, `gl.associative_scan`-era stdlib surface
- the whole `triton.experimental.gluon.language.amd` namespace
  (`amd.cdna4` buffer ops / MFMA layouts)

With 3.4, all such kernels fail at import or compile — verified on
nid00266x (job 675311, results in
`meta/benchmarks/artifacts/tokenspeed-amd-bucket1/results-stock-triton34.json`).

## Proof it works

We installed this exact wheel into scratch and re-ran everything green
(29/29 tests incl. GPU parity; full benchmark in
`results-triton38.json`, jobs 675350/675351):

- wheel: `triton-3.8.0+gitdf3f91dd-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- index: `https://download.pytorch.org/whl/nightly/rocm7.0/` (package `triton`)
- install: `pip install --no-deps --target <scratch>/triton-38 <wheel>`,
  then prepend to `PYTHONPATH` (no torch rebuild needed)

Headline measurements on MI300A with it: fused softmax top-k routing
1.3–3× over the torch chain, 128-wide Hadamard at 829 GB/s, mHC
pre-reduce ~30× over torch, sampling argmax 2× at 262k vocab.

## Alternatives considered

- Building triton from source in the env: slower to iterate, needs the
  same rocm7.0 toolchain anyway.
- Staying on 3.4 and porting kernels to pre-gluon Triton: throws away the
  layout-explicit programming model these kernels are built on; not viable.
