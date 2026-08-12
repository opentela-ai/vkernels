"""Backend selection for the vkernels Python bindings.

The public API in :mod:`vkernels.kernels`, :mod:`vkernels.comm` and
:mod:`vkernels.core` is backend-agnostic. Two backends implement it:

* **compiled** — the pybind11 extension :mod:`vkernels._core` (built from
  ``src/python/vkernels/_core.cpp`` by CMake when ``VKERNELS_BUILD_PYTHON=ON``).
  Every call crosses into the C++ library under ``src/c/vkernels/``; this is
  the fast path and the one that exercises the real kernels.
* **fallback** — a pure-Python (numpy) reference in :mod:`vkernels._fallback`
  that mirrors the CPU reference implementations, so the package stays
  importable and testable on machines where the extension was not built
  (mirroring the repository's "CPU reference always works" philosophy).

``load_extension()`` returns the compiled module when it can be found and
``None`` otherwise. Resolution order:

1. an already-importable ``vkernels._core`` (installed wheel copy or
   ``PYTHONPATH`` pointing at a build tree), then
2. a ``_core*.so`` sitting directly next to this file (wheel-installed
   extension that failed step 1 — e.g. ABI mismatch), then
3. a freshly built ``build/*/python/vkernels/_core*.so`` under the
   repository root (dev / editable-install CMake layout), newest first, then
4. ``None`` (the fallback backend is used).
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path | None:
    """Repository root (a directory containing ``src/c/vkernels``), or None."""
    for candidate in (_PKG_DIR, *_PKG_DIR.parents):
        if (candidate / "src" / "c" / "vkernels").is_dir():
            return candidate
    return None


def _find_next_to_package() -> Path | None:
    """A ``_core*.so`` in the same directory as this file (wheel install)."""
    candidates = sorted(_PKG_DIR.glob("_core*.so"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_built_extension() -> Path | None:
    """Newest ``build/*/python/vkernels/_core*.so`` under the repo root."""
    root = _repo_root()
    if root is None:
        return None
    candidates = sorted(root.glob("build/*/python/vkernels/_core*.so"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_from_path(path: Path) -> object | None:
    """Try to load a .so from a filesystem path; return the module or None."""
    try:
        spec = importlib.util.spec_from_file_location("vkernels._core", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_loaded: object | None = None
_tried = False


def load_extension():
    """Import and return the compiled ``vkernels._core`` module, or None.

    The result is cached, so repeated calls are cheap. Failures (missing
    extension, ABI mismatch, numpy unavailable) degrade gracefully to None.
    """
    global _loaded, _tried
    if _tried:
        return _loaded
    _tried = True

    # 1. Already importable (installed wheel, or build dir on PYTHONPATH).
    try:
        _loaded = importlib.import_module("vkernels._core")
        return _loaded
    except ImportError:
        pass

    # 2. Next to this file (wheel-installed, but step 1 failed — e.g.
    #    the extension was built for a different Python or glibc).
    path = _find_next_to_package()
    if path is not None:
        _loaded = _load_from_path(path)
        if _loaded is not None:
            return _loaded

    # 3. Built by CMake into build/<preset>/python/vkernels/.
    path = _find_built_extension()
    if path is not None:
        _loaded = _load_from_path(path)
        return _loaded

    # 4. No compiled backend.
    _loaded = None
    return None


def backend_name() -> str:
    """``"compiled"`` when the extension is loaded, else ``"fallback"``."""
    return "compiled" if load_extension() is not None else "fallback"
