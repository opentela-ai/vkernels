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
_SRC = _REPO / "src" / "python"  # the package lives at src/python/vkernels
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


class Bf16StorageContractTest(unittest.TestCase):
    """The bf16-storage / no-fp8e4nv contract (issue #60), pinned at source level.

    The whole point of the native kpool kernels is that SM80 (A100) Triton
    cannot declare ``*fp8e4nv`` in a kernel signature, so the cache ships as
    **bf16** and nothing in the device path may depend on a native FP8 type
    or hardware FP8 intrinsic. The fp8-layout entries
    (``*_fp8``) still exist for the legacy uint8 cache, but their fp8
    quantization is pure IEEE bit manipulation (``f32_to_fp8e4m3fn_rne``),
    which compiles on any arch. These tests scan the device sources so a
    future edit cannot silently reintroduce a ``fp8e4nv``-class dependency
    (a Triton type, a hardware fp8 conversion intrinsic, or an fp8 header)
    and break the SM80 port again. Runs anywhere (no GPU, no torch).
    """

    @classmethod
    def setUpClass(cls):
        def strip_comments(src: str) -> str:
            # Drop // line comments (the design notes legitimately *mention*
            # fp8e4nv; the contract is that no CODE does). Good enough for a
            # contract guard: these sources use no // inside string literals.
            return "\n".join(
                line.split("//")[0] if "//" in line else line
                for line in src.splitlines()
            )

        cls.kpool_hip_raw = (_REPO / "src" / "c" / "vkernels" / "kernels"
                             / "dsa_kpool.hip").read_text(encoding="utf-8")
        cls.kpool_hip = strip_comments(cls.kpool_hip_raw)
        compat = (_REPO / "src" / "c" / "vkernels" / "kernels" / "cuda_compat")
        cls.compat_sources = strip_comments("".join(
            p.read_text(encoding="utf-8") for p in sorted(compat.rglob("*.h"))
        ))

    def test_no_fp8e4nv_in_code(self):
        # The Triton type that JIT-fails on SM80 must not appear in CODE (it
        # may still be named in the design-note comments, which setUpClass
        # strips).
        self.assertNotIn("fp8e4nv", self.kpool_hip)
        self.assertNotIn("fp8e4nv", self.compat_sources)
        self.assertNotIn("tl.", self.kpool_hip)  # no Triton in a CUDA TU

    def test_no_hardware_fp8_intrinsics_or_headers(self):
        # No native-FP8 dependency: no CUDA/HIP fp8 header, no hardware
        # fp8 conversion intrinsic, no fp8 vector type. The fp8-layout
        # entries quantize in software (f32_to_fp8e4m3fn_rne), which is
        # architecture-neutral.
        forbidden = (
            "__nv_fp8", "__hip_fp8", "cvt.rn.f8e4m3", "cvt.rn.satfinite",
            "cuda_fp8.h", "hip_fp8.h", "__fp8", ".e4m3", "f8e4m3fnx",
        )
        for tok in forbidden:
            self.assertNotIn(tok, self.kpool_hip, tok)
            self.assertNotIn(tok, self.compat_sources, tok)

    def test_bf16_storage_helpers_present(self):
        # The bf16 cache convention: raw-uint16 round-trip helpers are what
        # every store/load goes through (see dsa_kpool.hpp "Storage") —
        # bf16_to_f32 on load, f2bf (RNE) on store.
        self.assertIn("bf16_to_f32", self.kpool_hip)
        self.assertIn("f2bf", self.kpool_hip)

    def test_device_abi_exports_bf16_entries(self):
        # When a device library is present, it must export BOTH storage
        # flavours: the bf16 entries (the SM80 serving path) and the legacy
        # fp8-layout entries (the gfx942 drop-in). Same names on the CUDA
        # build (libvkernels_c.so, cuda_capi_kpool.cpp) and the HIP build
        # (libvkernels_hip.so, hip_capi.cpp) — one ABI, two backends.
        if dev.find_libvkernels() is None:
            self.skipTest("no device library")
        lib = dev.load_libvkernels()
        for name in (
            "vk_hip_dsa_kpool_assemble",          # bf16 store
            "vk_hip_dsa_kpool_decode_update",     # bf16 store
            "vk_hip_dsa_kpool_assemble_fp8",      # legacy uint8 layout
            "vk_hip_dsa_kpool_decode_update_fp8",
        ):
            self.assertTrue(getattr(lib, name).argtypes, name)


if __name__ == "__main__":
    unittest.main()
