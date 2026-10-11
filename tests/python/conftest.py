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


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="run external serving-stack oracle tests")
    parser.addoption("--run-checkpoint", action="store_true", help="run tests using VKERNELS_TEST_CHECKPOINT")


def pytest_configure(config):
    import os
    # Op-config cache (vkernels.tuning) defaults OFF for the whole test
    # session: integrated ops keep their exact pre-cache launch behavior,
    # no sweep runs, and the developer's real ~/.cache/vkernels is never
    # touched by a test run. Tests that exercise the cache re-enable it
    # per-test against a tmp store via monkeypatch.setenv.
    os.environ.setdefault("VKERNELS_CACHE", "off")
    for name, description in {
        "integration": "requires explicitly installed external dependencies",
        "checkpoint": "requires an explicitly selected model checkpoint",
        "gpu": "requires a GPU runtime",
    }.items():
        config.addinivalue_line("markers", f"{name}: {description}")
    if config.getoption("--run-integration"):
        os.environ["VKERNELS_TEST_INTEGRATION"] = "1"


def pytest_collection_modifyitems(config, items):
    import pytest
    for item in items:
        for marker, option in (("integration", "--run-integration"), ("checkpoint", "--run-checkpoint")):
            if item.get_closest_marker(marker) and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"requires {option}"))
