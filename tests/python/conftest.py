"""Test-session path hygiene for vkernels (issue #99, discovery during mHC work).

Envs that carry a *pip-installed* vkernels (e.g. the serving-stack floe venv,
which installs it editable from the main checkout) silently bind the
installed package the first time any test module does ``import vkernels``.
Worktree branches under test must run against their OWN src tree, otherwise:

* modules whose tests import lazily (inside functions) get a *different*
  ``vkernels.compiler`` identity than the one collected at module import —
  two distinct exception classes, so ``pytest.raises(CaptureError)`` misses;
* a branch's new subpackages (e.g. mHC capture recorders) are simply absent
  from the installed copy.

This conftest runs before any test module import: it pins ``sys.path`` to the
repo's ``src/python`` and evicts any pre-bound vkernels modules that do not
come from this checkout. In bare envs (no installed vkernels) it is a no-op.
Modules that already imported the foreign package before conftest (none, at
collection start) are unaffected; pytest imports conftest first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "python"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

for _name in [
    m for m in list(sys.modules) if m == "vkernels" or m.startswith("vkernels.")
]:
    _file = getattr(sys.modules[_name], "__file__", None) or ""
    if str(_SRC) not in _file:
        del sys.modules[_name]
