"""Tests for the optional vLLM integration (vkernels.vllm_experts).

These exercise the reusable, torch-only pieces:

* :class:`CaptureSafeScratch` — persistent, capture-safe device scratch
  (issue #41, item 1): buffers are sized ONCE on the eager warmup and
  reused forever; growth is refused during CUDA-graph capture.
* :func:`moe_align_block_size_with_map` — the ``expert_map``-aware CPU
  helper; checked against the canonical
  :func:`vkernels.kernels.moe_align_block_size` for the no-map case.
* :func:`find_libvkernels_hip` — the ctypes loader's env resolution.

The :class:`VkernelFusedExperts` vLLM expert layer is built lazily (it
imports vLLM); its import test is skipped when vLLM is not installed.

On CI (numpy-only, fallback backend) the torch tests are skipped; they run
on any box with torch (CPU is sufficient — no GPU needed).
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

import numpy as np
from vkernels import kernels

try:  # torch is optional for the package; required for the integration
    import torch
except ImportError:  # pragma: no cover
    torch = None

from vkernels.vllm_experts import (
    CaptureSafeScratch,
    find_libvkernels_hip,
    max_em_count,
    max_num_tokens_hint,
    moe_align_block_size_with_map,
)


class MaxEmCountTest(unittest.TestCase):
    """Pure-Python helpers (no torch)."""

    def test_max_em_pads_to_multiple_of_64(self):
        # M*top_k + local_n, rounded up to a multiple of 64.
        self.assertEqual(max_em_count(1, 16, 896), ((16 + 896 + 63) // 64) * 64)
        self.assertEqual(max_em_count(8192, 16, 896) % 64, 0)
        # No rounding needed when already aligned.
        self.assertEqual(max_em_count(0, 1, 0), 0)

    def test_max_num_tokens_hint_env(self):
        key = "MAX_NUM_BATCHED_TOKENS"
        old = os.environ.pop(key, None)
        old2 = os.environ.pop("VLLM_MAX_NUM_BATCHED_TOKENS", None)
        try:
            os.environ[key] = "12345"
            self.assertEqual(max_num_tokens_hint(), 12345)
        finally:
            os.environ.pop(key, None)
            os.environ.pop("VLLM_MAX_NUM_BATCHED_TOKENS", None)
            if old is not None:
                os.environ[key] = old
            if old2 is not None:
                os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"] = old2

    def test_max_num_tokens_hint_falls_back_when_unset(self):
        for k in ("MAX_NUM_BATCHED_TOKENS", "VLLM_MAX_NUM_BATCHED_TOKENS"):
            os.environ.pop(k, None)
        self.assertEqual(max_num_tokens_hint(), 8192)

    def test_max_num_tokens_hint_ignores_garbage(self):
        key = "MAX_NUM_BATCHED_TOKENS"
        old = os.environ.pop(key, None)
        old2 = os.environ.pop("VLLM_MAX_NUM_BATCHED_TOKENS", None)
        try:
            os.environ[key] = "not-a-number"
            self.assertEqual(max_num_tokens_hint(), 8192)
        finally:
            os.environ.pop(key, None)
            os.environ.pop("VLLM_MAX_NUM_BATCHED_TOKENS", None)
            if old is not None:
                os.environ[key] = old
            if old2 is not None:
                os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"] = old2


class MoeAlignWithMapTest(unittest.TestCase):
    """moe_align_block_size_with_map parity vs the canonical kernels helper."""

    def _ids(self, M, top_k, E, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, E, size=(M, top_k), dtype=np.int32)

    def test_no_map_matches_canonical(self):
        # expert_map=None (1:1 global→local) must produce the identical
        # sorted_ids / expert_ids / EM as vkernels.kernels.moe_align
        # _block_size (the global, 2-D reference) for several shapes.
        for M, top_k, E, bs in [
            (1, 1, 8, 16),
            (4, 2, 8, 16),
            (7, 16, 896, 16),
            (32, 16, 896, 64),
            (50, 8, 32, 64),
        ]:
            with self.subTest(M=M, top_k=top_k, E=E, bs=bs):
                ids = self._ids(M, top_k, E, seed=M + top_k)
                ref_sids, ref_eids, ref_EM = kernels.moe_align_block_size(
                    ids, E, bs
                )
                got_sids, got_eids, got_EM = moe_align_block_size_with_map(
                    ids.ravel(), E, bs, expert_map=None
                )
                self.assertEqual(got_EM, ref_EM)
                np.testing.assert_array_equal(got_sids, ref_sids)
                np.testing.assert_array_equal(got_eids, ref_eids)

    def test_identity_map_matches_no_map(self):
        ids = self._ids(12, 4, 16, seed=7)
        no_map = moe_align_block_size_with_map(
            ids.ravel(), 16, 16, expert_map=None
        )
        ident_map = np.arange(16, dtype=np.int32)
        with_map = moe_align_block_size_with_map(
            ids.ravel(), 16, 16, expert_map=ident_map
        )
        self.assertEqual(with_map[2], no_map[2])
        np.testing.assert_array_equal(with_map[0], no_map[0])
        np.testing.assert_array_equal(with_map[1], no_map[1])

    def test_skip_map_excludes_unmapped_experts(self):
        # Only experts {0, 2} are mapped to local {0, 1}; expert 1 is
        # skipped (-1). The local layout must contain ONLY local experts
        # 0 and 1, in sorted order.
        ids = np.array([[0, 2], [2, 2], [1, 0]], dtype=np.int32)  # M=3 tk=2
        expert_map = np.array([0, -1, 1], dtype=np.int32)
        sids, eids, _EM = moe_align_block_size_with_map(
            ids.ravel(), 3, 16, expert_map=expert_map
        )
        # expert 1 (global) is skipped entirely -> only local 0 and 1.
        local_present = {int(e) for e in eids if e != -1}
        self.assertEqual(local_present, {0, 1})
        # Padding sentinel is M*top_k = 6.
        self.assertTrue(np.all((sids >= 0) & (sids <= 6)))
        # Flat indices of routed tokens: 0,1,2,3,5 (global-1 token at idx4
        # is the only skipped one).
        real = sorted(int(s) for s in sids if s != 6)
        self.assertEqual(real, [0, 1, 2, 3, 5])

    def test_empty_routing_yields_at_least_block_size(self):
        # Every token maps to a skipped expert -> EM must still be >= bs
        # (avoids zero-length arrays downstream).
        ids = np.array([[0, 1], [0, 1]], dtype=np.int32)
        expert_map = np.array([-1, -1], dtype=np.int32)
        sids, eids, EM = moe_align_block_size_with_map(
            ids.ravel(), 2, 16, expert_map=expert_map
        )
        self.assertGreaterEqual(EM, 16)
        self.assertEqual(EM % 16, 0)
        # All padding.
        np.testing.assert_array_equal(sids, np.full(EM, 4, dtype=np.int32))
        np.testing.assert_array_equal(eids, np.full(EM // 16, -1, dtype=np.int32))


class FindLibTest(unittest.TestCase):
    """find_libvkernels_hip env resolution (no .so required)."""

    def test_vkernel_lib_path_is_returned_when_it_exists(self):
        with tempfile.TemporaryDirectory() as d:
            lib = os.path.join(d, "libvkernels_hip.so")
            Path(lib).write_bytes(b"\x7fELF")  # plausible enough
            old = os.environ.get("VKERNELS_LIB")
            try:
                os.environ["VKERNELS_LIB"] = lib
                self.assertEqual(find_libvkernels_hip(), lib)
            finally:
                if old is None:
                    os.environ.pop("VKERNELS_LIB", None)
                else:
                    os.environ["VKERNELS_LIB"] = old

    def test_vkernel_lib_nonexistent_is_skipped(self):
        old = os.environ.get("VKERNELS_LIB")
        try:
            os.environ["VKERNELS_LIB"] = "/definitely/not/here/libvkernels_hip.so"
            # Falls through to K3 / VKERNELS_DIR / LD_LIBRARY_PATH; on CI
            # none of those have it, so the result is None (or a real
            # build if present — accept both, just not the bogus path).
            got = find_libvkernels_hip()
            self.assertNotEqual(got, os.environ["VKERNELS_LIB"])
        finally:
            if old is None:
                os.environ.pop("VKERNELS_LIB", None)
            else:
                os.environ["VKERNELS_LIB"] = old


@unittest.skipIf(torch is None, "torch is required for CaptureSafeScratch")
class CaptureSafeScratchTest(unittest.TestCase):
    """The reusable, capture-safe persistent-scratch fix (torch, CPU)."""

    def _dev(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_first_call_allocates_at_capacity(self):
        s = CaptureSafeScratch()
        dev = self._dev()
        out = s.get("out", 8, dtype=torch.float32, device=dev, capacity=100)
        self.assertEqual(out.numel(), 8)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.data_ptr(), s.get("out", 8, dtype=torch.float32,
                                              device=dev).data_ptr())

    def test_second_call_reuses_same_storage(self):
        s = CaptureSafeScratch()
        dev = self._dev()
        a = s.get("out", 8, dtype=torch.float32, device=dev, capacity=100)
        b = s.get("out", 8, dtype=torch.float32, device=dev)
        self.assertEqual(a.data_ptr(), b.data_ptr())
        # And a smaller slice reuses the same base storage.
        c = s.get("out", 5, dtype=torch.float32, device=dev)
        self.assertEqual(a.data_ptr(), c.data_ptr())

    def test_different_keys_get_different_storage(self):
        s = CaptureSafeScratch()
        dev = self._dev()
        a = s.get("out", 8, dtype=torch.float32, device=dev, capacity=100)
        act = s.get(("act", 32), 8, dtype=torch.bfloat16, device=dev,
                    capacity=100)
        self.assertNotEqual(a.data_ptr(), act.data_ptr())
        self.assertEqual(act.dtype, torch.bfloat16)

    def test_growth_refused_during_capture(self):
        events = []

        def probe():
            return len(events) % 2 == 1  # odd calls = "capturing"

        s = CaptureSafeScratch(capture_probe=probe)
        dev = self._dev()
        # First call (not capturing): allocate a small buffer.
        events.append(1)
        s.get("out", 4, dtype=torch.float32, device=dev, capacity=4)
        # Second call (capturing) needing MORE than the pinned cap: refuse.
        events.append(2)
        with self.assertRaises(RuntimeError) as cm:
            s.get("out", 64, dtype=torch.float32, device=dev)
        self.assertIn("capture", str(cm.exception).lower())

    def test_growth_allowed_when_not_capturing(self):
        probe = {"capturing": False}
        s = CaptureSafeScratch(capture_probe=lambda: probe["capturing"])
        dev = self._dev()
        s.get("out", 4, dtype=torch.float32, device=dev, capacity=4)
        # Not capturing: reallocate to a larger buffer.
        big = s.get("out", 64, dtype=torch.float32, device=dev, capacity=64)
        self.assertEqual(big.numel(), 64)
        # Subsequent smaller slice reuses the new (larger) storage.
        self.assertEqual(big.data_ptr(),
                         s.get("out", 8, dtype=torch.float32,
                               device=dev).data_ptr())

    def test_within_capacity_slice_succeeds_during_capture(self):
        # A call whose count fits the pinned capacity must NOT raise even
        # while capturing (it only slices existing storage).
        probe = {"capturing": True}
        s = CaptureSafeScratch(capture_probe=lambda: probe["capturing"])
        dev = self._dev()
        # Pin a large capacity BEFORE capture begins.
        probe["capturing"] = False
        s.get("out", 4, dtype=torch.float32, device=dev, capacity=200)
        # Now capture; smaller slices are fine.
        probe["capturing"] = True
        out = s.get("out", 50, dtype=torch.float32, device=dev)
        self.assertEqual(out.numel(), 50)

    def test_reset_drops_buffers(self):
        s = CaptureSafeScratch()
        dev = self._dev()
        s.get("out", 8, dtype=torch.float32, device=dev, capacity=8)
        self.assertEqual(len(s.keys()), 1)
        s.reset()
        self.assertEqual(len(s.keys()), 0)

    def test_negative_count_raises(self):
        s = CaptureSafeScratch()
        with self.assertRaises(ValueError):
            s.get("out", -1, dtype=torch.float32, device=self._dev())


class VkernelFusedExpertsLazyBuildTest(unittest.TestCase):
    """The vLLM expert layer is built lazily (imports vLLM on first use)."""

    @staticmethod
    def _run(code: str) -> int:
        # Fresh interpreter so the lazy-import guarantee is real (this
        # module's own `from vkernels.vllm_experts import ...` already
        # populated vkernels.__dict__).
        import subprocess

        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-c", code], env=env, check=False
        ).returncode

    def test_import_vkernels_does_not_pull_vllm(self):
        # `import vkernels` must stay dependency-free: neither vllm nor
        # the torch integration is forced by the core package.
        rc = self._run(
            "import sys, vkernels; "
            "sys.exit(0 if 'vllm' not in sys.modules and "
            "'vkernels.vllm_experts' not in sys.modules else 1)"
        )
        self.assertEqual(rc, 0, "import vkernels pulled in vllm/vllm_experts")

    def test_capture_safe_scratch_import_does_not_pull_vllm(self):
        # The reusable CaptureSafeScratch (and numpy helpers) are usable
        # without vLLM; importing them must not import vllm.
        rc = self._run(
            "import sys; from vkernels.vllm_experts import "
            "CaptureSafeScratch, moe_align_block_size_with_map; "
            "sys.exit(0 if 'vllm' not in sys.modules else 1)"
        )
        self.assertEqual(rc, 0, "CaptureSafeScratch import pulled in vllm")

    def test_vllm_experts_class_builds_when_vllm_present(self):
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm is not installed")
        import inspect

        from vkernels.vllm_experts import VkernelFusedExperts
        self.assertTrue(inspect.isclass(VkernelFusedExperts))


if __name__ == "__main__":
    unittest.main()
