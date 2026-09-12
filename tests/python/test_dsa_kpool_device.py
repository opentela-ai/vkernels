"""Tests for the ctypes device loader (vkernels.dsa_kpool_device, issue #60).

The device library only exists on GPU boxes, so these tests pin the parts
that run anywhere:

* :func:`find_libvkernels` resolution order (env override wins; graceful
  None when nothing is found).
* :func:`load_libvkernels` binds the four ``vk_hip_dsa_kpool_*`` prototypes
  (skipped when no library is present on the machine).
* the torch-facing wrappers' dtype/contiguity enforcement — in particular
  that IN-PLACE arguments (the decode live tail) are rejected rather than
  silently cast (a cast would fork the buffer and drop the kernel's tail
  update). CPU tensors suffice; no kernel is launched.

On CI (numpy-only, no torch) the torch tests are skipped.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vkernels import dsa_kpool_device as dev  # noqa: E402

try:  # torch is optional for the package; required for the wrapper checks
    import torch
except ImportError:  # pragma: no cover
    torch = None


class FindLibTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("VKERNELS_LIB")
        dev._lib_cache.pop("lib", None)
        dev._lib_cache.pop("path", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("VKERNELS_LIB", None)
        else:
            os.environ["VKERNELS_LIB"] = self._saved
        dev._lib_cache.pop("lib", None)
        dev._lib_cache.pop("path", None)

    def test_env_override_wins(self):
        with tempfile.NamedTemporaryFile(suffix=".so") as f:
            os.environ["VKERNELS_LIB"] = f.name
            self.assertEqual(dev.find_libvkernels(), f.name)

    def test_env_override_missing_path_is_ignored(self):
        os.environ["VKERNELS_LIB"] = "/nonexistent/vkernels.so"
        found = dev.find_libvkernels()
        if found is not None:
            # A build tree on this machine satisfied a later step; the env
            # path itself must not have been returned.
            self.assertNotEqual(found, "/nonexistent/vkernels.so")
        else:
            self.assertIsNone(found)

    def test_missing_library_returns_none(self):
        # Only valid when no dev build exists next to the repo; otherwise a
        # real library is legitimately found (and available() is True).
        if dev.find_libvkernels() is not None:
            self.skipTest("a vkernels device library exists on this machine")
        self.assertIsNone(dev.find_libvkernels())
        self.assertFalse(dev.available())


@unittest.skipIf(dev.find_libvkernels() is None, "no device library")
class PrototypeTest(unittest.TestCase):
    def test_load_binds_kpool_prototypes(self):
        lib = dev.load_libvkernels()
        for name in (
            "vk_hip_dsa_kpool_assemble",
            "vk_hip_dsa_kpool_assemble_fp8",
            "vk_hip_dsa_kpool_decode_update",
            "vk_hip_dsa_kpool_decode_update_fp8",
        ):
            fn = getattr(lib, name)
            self.assertTrue(fn.argtypes, name)
            self.assertEqual(fn.restype, None, name)

    def test_available_is_true_and_cached(self):
        self.assertTrue(dev.available())
        self.assertIs(dev.load_libvkernels(), dev.load_libvkernels())


@unittest.skipIf(torch is None, "torch not installed")
class PrepContractTest(unittest.TestCase):
    """The _prep dtype/contiguity contract (CPU tensors; no kernel launch)."""

    def test_input_cast_is_allowed(self):
        t = torch.zeros(4, 128, dtype=torch.float32)
        out = dev._prep(t, torch.bfloat16, "chunk_k")
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(out.is_contiguous())

    def test_in_place_args_reject_wrong_dtype(self):
        t = torch.zeros(2, 4, 128, dtype=torch.float32)
        with self.assertRaises(TypeError):
            dev._prep(t, torch.bfloat16, "tail_k", in_place_ok=True)

    def test_in_place_args_reject_non_contiguous(self):
        t = torch.zeros(4, 2, 128, dtype=torch.bfloat16).transpose(0, 1)
        self.assertFalse(t.is_contiguous())
        with self.assertRaises(TypeError):
            dev._prep(t, torch.bfloat16, "tail_k", in_place_ok=True)

    def test_contiguous_bf16_in_place_passes_through(self):
        t = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
        self.assertIs(dev._prep(t, torch.bfloat16, "tail_k", in_place_ok=True), t)


if __name__ == "__main__":
    unittest.main()
