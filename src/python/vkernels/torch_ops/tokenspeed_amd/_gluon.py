"""Import shim for stock-Triton Gluon, with graceful degradation.

The vendored TokenSpeed-Kernel-AMD kernels (see this package's ``__init__``)
were written against ``tokenspeed_triton`` — LightSeek's Triton vendor build
whose Gluon always ships the ``triton.experimental.gluon.language.amd.cdna4``
namespace. This repo runs stock Triton, where:

* ``triton.experimental.gluon`` (generic layouts, ``gluon.jit``) landed first;
* the ``gluon.language.amd`` / ``amd.cdna4`` namespaces (CDNA4 MFMA layouts,
  ``buffer_load``/``buffer_store``) only exist on newer releases.

So every vendored module imports ``gl``/``gluon``/``triton`` from here, and
callers gate kernels on :func:`gluon_available` / :func:`cdna4_available`:

* pure-Triton hosts without Gluon still import the package (registries and the
  torch-side preprocessing stay usable);
* Gluon hosts without ``amd.cdna4`` (e.g. triton 3.4 on MI300A/gfx942) run
  every kernel that only uses generic ``gl`` ops and cleanly refuse the
  CDNA4-specific ones (sigmoid routing, argmax).

When Gluon is absent the shim substitutes inert stand-ins: ``@gluon.jit``
returns the function unchanged (launching it then fails loudly), and ``gl``
attribute access yields inert placeholders so module-level constants such as
``_ROUTE_GL_DTYPE`` or ``gl.constexpr(...)`` do not explode at import time.
"""

from __future__ import annotations

from types import SimpleNamespace


class _Inert:
    """Attribute/placeholder sink: never fails at import, fails on use."""

    def __getattr__(self, item: str) -> "_Inert":
        return _Inert()

    def __call__(self, *args, **kwargs) -> "_Inert":
        return _Inert()

    def __getitem__(self, grid) -> None:
        raise RuntimeError(
            "Gluon is unavailable on this host (triton without "
            "triton.experimental.gluon); the vendored tokenspeed-amd "
            "device kernels cannot launch here"
        )


__all__ = [
    "cdna4_available",
    "cdna4_module",
    "gl",
    "gluon",
    "gluon_available",
    "triton",
]

try:  # pragma: no cover - exercised only on hosts with stock triton
    import triton
    from triton import language as tl
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except Exception:  # noqa: BLE001 — any triton shape can be missing on a host
    triton = None
    tl = None
    gluon = None
    gl = None


def _identity_decorator(fn=None, **launch_kwargs):
    def wrap(f):
        return f

    return wrap if fn is None else wrap(fn)


_CONSTEXPR_FUNCTION_NATIVE = gluon is not None and hasattr(
    gluon, "constexpr_function"
)

if gluon is not None and not _CONSTEXPR_FUNCTION_NATIVE:
    # triton ~3.4 shipped gluon without ``constexpr_function``; the kernels
    # that rely on it (hadamard, argmax) may still fail to *compile* there,
    # but the module must import so availability can be probed and reported.
    gluon = SimpleNamespace(
        jit=gluon.jit,
        constexpr_function=_identity_decorator,
    )


if triton is None:
    tl = _Inert()
    triton = SimpleNamespace(
        cdiv=lambda a, b: -(-a // b),
        next_power_of_2=lambda n: 1 << max(0, n - 1).bit_length() if n > 0 else 1,
    )

if gluon is None or gl is None:

    def _identity_jit(fn=None, **launch_kwargs):
        def wrap(f):
            return _Inert()

        return wrap if fn is None else wrap(fn)

    gluon = SimpleNamespace(
        jit=_identity_jit,
        constexpr_function=_identity_jit,
    )
    gl = _Inert()


def gluon_available() -> bool:
    """Whether real (non-stub) Gluon is importable on this host."""
    return not isinstance(gl, _Inert)


def gluon_features() -> dict:
    """Feature-level Gluon probe for benchmark reports.

    Kernels need more than ``gluon.jit``: ``constexpr_function`` (hadamard,
    argmax), ``gl.associative_scan`` (moe sorting stage 3), and
    ``gl.amd.cdna4`` (sigmoid routing, argmax loads). Recording each flag
    separately keeps "unavailable" honest on partial hosts like triton 3.4.
    """
    return {
        "gluon": gluon_available(),
        "constexpr_function": _CONSTEXPR_FUNCTION_NATIVE,
        "associative_scan": hasattr(gl, "associative_scan"),
        "amd_cdna4": cdna4_available(),
    }


def cdna4_module():
    """The ``gluon.language.amd.cdna4`` module, or ``None`` when absent."""
    if not gluon_available():
        return None
    amd = getattr(gl, "amd", None)
    return getattr(amd, "cdna4", None)


def cdna4_available() -> bool:
    """Whether CDNA4-specific Gluon ops (``buffer_load`` et al.) exist."""
    return cdna4_module() is not None
