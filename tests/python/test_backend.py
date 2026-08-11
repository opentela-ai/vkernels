"""Backend resolution tests for the vkernels Python bindings.

These run without numpy and pin the loader contract: ``vkernels.backend``
names the active backend, the submodules are importable lazily, and the
compiled extension (when present) is the one actually used.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import vkernels  # noqa: E402
from vkernels import _backend  # noqa: E402


class BackendTest(unittest.TestCase):
    def test_backend_name_is_known(self):
        self.assertIn(vkernels.backend, ("compiled", "fallback"))

    def test_extension_loading_is_cached(self):
        self.assertIs(_backend.load_extension(), _backend.load_extension())

    def test_extension_matches_backend_flag(self):
        self.assertEqual(
            vkernels.backend,
            "compiled" if _backend.load_extension() is not None else "fallback",
        )

    def test_version(self):
        import re
        self.assertRegex(vkernels.__version__, r"^\d+\.\d+\.\d+$")

    def test_submodules_import_lazily(self):
        self.assertIn("kernels", vkernels.__getattr__("kernels").__name__)
        self.assertIn("comm", vkernels.__getattr__("comm").__name__)
        self.assertIn("core", vkernels.__getattr__("core").__name__)

    def test_attribute_access_imports_submodules(self):
        # `vkernels.kernels` etc. resolve through __getattr__ and stay cached.
        self.assertIs(vkernels.kernels, vkernels.kernels)
        self.assertIs(vkernels.comm, vkernels.comm)
        self.assertIs(vkernels.core, vkernels.core)

    def test_unknown_attribute_raises(self):
        with self.assertRaises(AttributeError):
            vkernels.definitely_not_a_module  # noqa: B018


if __name__ == "__main__":
    unittest.main()
