"""vkernels — Python access to the vkernels kernel library.

The package has two layers:

* **Discovery / CLI** — :mod:`vkernels.discovery` scans the C++/CUDA sources
  under ``csrc/vkernels`` and :mod:`vkernels.cli` exposes ``vkl``, the
  command-line inspector. These need no third-party dependencies.

* **Bindings** — :mod:`vkernels.kernels`, :mod:`vkernels.comm` and
  :mod:`vkernels.core` are the documented Python interface to the kernels
  and communication primitives. They dispatch to the compiled backend
  (:mod:`vkernels._core`, a pybind11 extension built from
  ``src/vkernels/_core.cpp`` when ``VKERNELS_BUILD_PYTHON=ON``) when it is
  available and otherwise fall back to pure-Python reference
  implementations (:mod:`vkernels._fallback`). The backend in use is exposed
  as :data:`backend`.

Example:
    >>> import numpy as np
    >>> from vkernels import kernels
    >>> kernels.add([1.0, 2.0], [3.0, 4.0])
    array([4., 6.], dtype=float32)
    >>> from vkernels import comm
    >>> comm.ring_allreduce([np.array([1.0, 2.0], dtype=np.float32)] * 2)
    [array([2., 4.], dtype=float32), array([2., 4.], dtype=float32)]

See ``docs/python-bindings.md`` for the full documentation.
"""

from __future__ import annotations

import importlib

from vkernels import _backend, discovery

#: Version of the C++ library, read from ``csrc/vkernels/util/version.hpp``.
__version__ = discovery.version() or "0.1.0"

#: ``"compiled"`` when the pybind11 extension is loaded, else ``"fallback"``.
backend: str = _backend.backend_name()

_Lazy = ("core", "kernels", "comm")


def __getattr__(name: str):
    """Lazily import the binding submodules so ``import vkernels`` stays
    dependency-free (the discovery/CLI layer needs no numpy)."""
    if name in _Lazy:
        module = importlib.import_module(f"vkernels.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
