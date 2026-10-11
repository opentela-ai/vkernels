"""TokenSpeed-Kernel-AMD ops, vendored for the portable AMD paths.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel-amd`` subpackage,
``main`` branch, fetched 2026-11 from
``tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/``
(MIT, Copyright (c) 2026 LightSeek Foundation).

This is "bucket 1" of the borrowing plan in
``docs/borrowing-tokenspeed-deepgemm.md``: the arch-independent host
machinery plus the generic-Gluon small ops. Everything CDNA4-specific
(the mxfp4/mxfp8 stage kernels, MFMA layouts) stays upstream.

Vendored (verbatim bodies; see the per-file markers):

  ``moe/mxfp4/scale_layout.py``    — CDNA4 MXFP4 e8m0 scale swizzle (pure torch)
  ``moe/mxfp4/weight_preprocess.py`` — MXFP4 MoE weight/scale preprocessing
                                       (gate/up interleave, W2 block-N padding,
                                       gdot128 preshuffle; pure torch +
                                       inlined ``fused/_common`` helpers)
  ``moe/mxfp4/n16_weights.py``     — N16 packed-cell MXFP4 weight bank (pure
                                       torch shuffles + gluon offset helpers)
  ``moe/mxfp4/routing.py``         — fused dense top-k routing (gluon softmax
                                       kernel + CDNA4 sigmoid kernels)
  ``moe/mxfp4/moe_sorting.py``     — block-aligned expert sort, four generic
                                       gluon stages (no host sync)
  ``sampling/argmax.py``           — vocabulary argmax (CDNA4 gluon paths +
                                       torch fallback)
  ``transform/hadamard.py``        — length-128 fast Hadamard transform
                                       (generic gluon, register butterfly)
  ``mhc.py``                       — hyperconnection pre reduce/apply for
                                       decode-shaped batches (generic gluon)

Adapted (import points only, plus availability gating):

* ``tokenspeed_kernel_amd._triton`` imports became the local shim
  :mod:`vkernels.torch_ops.tokenspeed_amd._gluon`, which binds stock
  ``triton.experimental.gluon`` instead of LightSeek's vendored Triton and
  degrades to inert stubs when Gluon is absent (this repo must import on
  CPU-only hosts for registry metadata).
* ``gl.amd.cdna4`` is resolved lazily (stock Triton < 3.5 has no
  ``gluon.language.amd`` namespace): the sigmoid-bias routing kernels and the
  argmax gluon paths need it, and refuse cleanly where it is missing. On
  MI300A/gfx942 with triton 3.4 this leaves the softmax routing, moe sorting,
  hadamard, mhc, and all torch-side preprocessing runnable.
* ``weight_preprocess`` takes an optional ``attach_prefill_aliases_hook``
  instead of importing the CDNA4 stage-kernel package.

Runnability matrix (measured on MI300A — see
``meta/benchmarks/artifacts/tokenspeed-amd-bucket1``; the gluon kernels need
triton ≥ 3.8 — stock 3.4 gluon lacks ``constexpr_function``, ``gl.histogram``
and the ``amd`` namespace, see :func:`_gluon.gluon_features`):

=========================  ==============  =====================================
component                  needs           notes
=========================  ==============  =====================================
scale_layout, n16 weights, torch only      any device, incl. CPU
weight_preprocess          torch only      (n16 gluon offset helpers need gluon)
routing (softmax top-k)    gluon ≥3.8      generic layouts, no cdna4
moe_sorting                gluon ≥3.8      generic layouts (gl.histogram)
hadamard_128               gluon ≥3.8      constexpr_function
mhc pre reduce/apply       gluon ≥3.8      generic layouts
routing (sigmoid bias)     gluon+cdna4     gfx950-class Triton
argmax (gluon paths)       gluon+cdna4     gfx950-class Triton (torch fallback
                                           otherwise)
=========================  ==============  =====================================

Kernels register through
:mod:`vkernels.torch_ops.tokenspeed_amd_registry` (priority bands + format
signatures) instead of tokenspeed's package-level dispatch.
"""

from vkernels.torch_ops.tokenspeed_amd._gluon import (
    cdna4_available,
    cdna4_module,
    gluon_available,
    gluon_features,
)

__all__ = [
    "cdna4_available",
    "cdna4_module",
    "gluon_available",
    "gluon_features",
]
