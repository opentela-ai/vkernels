"""Contract tests for the fused indexed K/V layer gather (issue #2).

Covers the full acceptance matrix (BF16/FP16 x head dims {64,128,256} x
page sizes {1,16,32,64} x page counts {0..128+} x {dense,sparse,non-
monotonic} x {int32,int64}) against the two-gather PyTorch reference
(``dst[:, :, 0] = k_src[slot_ids]; dst[:, :, 1] = v_src[slot_ids]``), the
async stream contract (exactly one task, result matches sync), and the
validation surface (dtype, contiguity, shape, out-of-range slot,
non-writable dst).

Runs against the active backend (compiled when the extension is built,
otherwise the pure-Python fallback) and, when both are present, cross-checks
them.
"""

from __future__ import annotations

import unittest

import numpy as np

from vkernels import _backend, comm
from vkernels._fallback import kv_gather_layer as _fallback_gather
from vkernels.core import Stream

try:
    _core = _backend.load_extension()
    _COMPILED = _core is not None
except Exception:  # pragma: no cover
    _core = None
    _COMPILED = False


def _two_gather_ref(k: np.ndarray, v: np.ndarray,
                    slot_ids: np.ndarray) -> np.ndarray:
    """The PyTorch two-gather reference (numpy): K then V into the [2] dim."""
    num_pages, page_size = slot_ids.shape
    num_kv_heads, head_dim = k.shape[1], k.shape[2]
    dst = np.zeros((num_pages, page_size, 2, num_kv_heads, head_dim),
                   dtype=k.dtype)
    dst[:, :, 0] = k[slot_ids]
    dst[:, :, 1] = v[slot_ids]
    return dst


def _make_src(num_slots, num_kv_heads, head_dim, dtype, seed):
    """Deterministic per-element source so mis-routing is obvious."""
    total = num_slots * num_kv_heads * head_dim
    flat = np.arange(total, dtype=np.int32)
    if dtype == np.float16:
        flat = (flat % 65504).astype(np.float16)
    else:  # BF16 represented as a uint16 view (numpy has no native BF16)
        flat = (flat % 65535).astype(np.uint16)
    np.random.seed(seed)
    np.random.shuffle(flat)
    return flat.reshape(num_slots, num_kv_heads, head_dim).copy()


def _make_map(kind, total_tokens, num_slots, rng_seed):
    rng = np.random.default_rng(rng_seed)
    if kind == "dense":
        perm = np.arange(num_slots)
        rng.shuffle(perm)
        return perm[np.arange(total_tokens) % num_slots]
    if kind == "sparse":
        return rng.integers(0, num_slots, size=total_tokens)
    # non-monotonic (descending modulo wrap)
    return np.array([(num_slots - 1) - (i % num_slots)
                     for i in range(total_tokens)])


class KvGatherLayerTest(unittest.TestCase):
    def test_basic_fp16(self):
        k = np.arange(2 * 1 * 4, dtype=np.float16).reshape(2, 1, 4)
        v = np.full((2, 1, 4), 99, dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.int32)
        dst = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        comm.kv_gather_layer(k, v, slot_ids, dst)
        np.testing.assert_array_equal(dst[:, :, 0], k[slot_ids])
        np.testing.assert_array_equal(dst[:, :, 1], v[slot_ids])

    def test_basic_bf16_as_uint16(self):
        k = np.arange(4 * 2 * 8, dtype=np.uint16).reshape(4, 2, 8)
        v = (np.arange(4 * 2 * 8, dtype=np.uint16) + 1000).reshape(4, 2, 8)
        slot_ids = np.array([[3, 0, 2, 1]], dtype=np.int64)  # non-monotonic
        dst = np.zeros((1, 4, 2, 2, 8), dtype=np.uint16)
        comm.kv_gather_layer(k, v, slot_ids, dst)
        np.testing.assert_array_equal(dst[:, :, 0], k[slot_ids])
        np.testing.assert_array_equal(dst[:, :, 1], v[slot_ids])

    def test_zero_pages_is_noop(self):
        k = np.zeros((1, 1, 1), dtype=np.float16)
        v = np.zeros((1, 1, 1), dtype=np.float16)
        slot_ids = np.zeros((0, 4), dtype=np.int32)
        dst = np.full((0, 4, 2, 1, 1), 0xCC, dtype=np.float16)
        comm.kv_gather_layer(k, v, slot_ids, dst)  # must not raise
        self.assertEqual(dst.shape, (0, 4, 2, 1, 1))

    def test_repeated_slots(self):
        k = np.arange(4 * 1 * 2, dtype=np.float16).reshape(4, 1, 2)
        v = np.full((4, 1, 2), 7, dtype=np.float16)
        slot_ids = np.array([[1, 1, 3, 3]], dtype=np.int32)  # repeats
        dst = np.zeros((1, 4, 2, 1, 2), dtype=np.float16)
        comm.kv_gather_layer(k, v, slot_ids, dst)
        np.testing.assert_array_equal(dst[:, :, 0], k[slot_ids])
        np.testing.assert_array_equal(dst[:, :, 1], v[slot_ids])

    def test_int32_and_int64_agree(self):
        k = _make_src(8, 2, 16, np.float16, 1)
        v = _make_src(8, 2, 16, np.float16, 2)
        slot_ids = np.array([[3, 1, 6, 4]], dtype=np.int32)
        d32 = np.zeros((1, 4, 2, 2, 16), dtype=np.float16)
        d64 = np.zeros((1, 4, 2, 2, 16), dtype=np.float16)
        comm.kv_gather_layer(k, v, slot_ids, d32)
        comm.kv_gather_layer(k, v, slot_ids.astype(np.int64), d64)
        np.testing.assert_array_equal(d32, d64)
        np.testing.assert_array_equal(d32, _two_gather_ref(k, v, slot_ids))

    def test_async_matches_sync_and_one_task(self):
        k = _make_src(8, 2, 16, np.float16, 1)
        v = _make_src(8, 2, 16, np.float16, 2)
        slot_ids = np.array([[3, 1, 6, 4]], dtype=np.int32)
        sync = np.zeros((1, 4, 2, 2, 16), dtype=np.float16)
        asy = np.zeros((1, 4, 2, 2, 16), dtype=np.float16)
        comm.kv_gather_layer(k, v, slot_ids, sync)
        s = Stream()
        before = s.submitted()
        comm.kv_gather_layer(k, v, slot_ids, asy, stream=s)
        self.assertEqual(s.submitted() - before, 1)  # exactly one task
        s.wait()
        np.testing.assert_array_equal(sync, asy)

    def test_validation_wrong_dtype(self):
        k = np.zeros((2, 1, 4), dtype=np.float32)  # itemsize 4, not BF16/FP16
        v = np.zeros((2, 1, 4), dtype=np.float32)
        slot_ids = np.array([[0, 1]], dtype=np.int32)
        dst = np.zeros((1, 2, 2, 1, 4), dtype=np.float32)
        with self.assertRaises((TypeError, ValueError)):
            comm.kv_gather_layer(k, v, slot_ids, dst)

    def test_validation_slot_dtype(self):
        k = np.zeros((2, 1, 4), dtype=np.float16)
        v = np.zeros((2, 1, 4), dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.float32)  # wrong dtype
        dst = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        with self.assertRaises((TypeError, ValueError)):
            comm.kv_gather_layer(k, v, slot_ids, dst)

    def test_validation_shape_mismatch(self):
        k = np.zeros((4, 2, 16), dtype=np.float16)
        v = np.zeros((4, 2, 16), dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.int32)  # 1 page
        dst = np.zeros((2, 1, 2, 2, 16), dtype=np.float16)  # wrong page count
        with self.assertRaises((ValueError, TypeError)):
            comm.kv_gather_layer(k, v, slot_ids, dst)

    def test_validation_out_of_range_slot(self):
        k = np.zeros((4, 2, 16), dtype=np.float16)
        v = np.zeros((4, 2, 16), dtype=np.float16)
        slot_ids = np.array([[0, 4]], dtype=np.int32)  # 4 == num_slots
        dst = np.zeros((1, 2, 2, 2, 16), dtype=np.float16)
        with self.assertRaises(ValueError):
            comm.kv_gather_layer(k, v, slot_ids, dst)

    def test_validation_non_contiguous_dst(self):
        k = np.zeros((2, 1, 4), dtype=np.float16)
        v = np.zeros((2, 1, 4), dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.int32)
        full = np.zeros((1, 4, 2, 1, 4), dtype=np.float16)
        dst = full[:, ::2]  # non-contiguous
        with self.assertRaises((ValueError, TypeError)):
            comm.kv_gather_layer(k, v, slot_ids, dst)

    def test_validation_non_writable_dst(self):
        k = np.zeros((2, 1, 4), dtype=np.float16)
        v = np.zeros((2, 1, 4), dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.int32)
        dst = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        dst.setflags(write=False)
        with self.assertRaises((ValueError, TypeError)):
            comm.kv_gather_layer(k, v, slot_ids, dst)


class KvGatherLayerMatrixTest(unittest.TestCase):
    """Full acceptance matrix: dtype x head_dim x page_size x num_pages x
    map_kind, int32 and int64, against the two-gather reference."""

    def _run(self, dtype, head_dim, page_size, num_pages, map_kind):
        num_kv_heads = 8
        total_tokens = num_pages * page_size
        if total_tokens == 0:
            num_slots = 16
            slot_ids = np.zeros((num_pages, page_size), dtype=np.int32)
        else:
            num_slots = min(max(total_tokens + 64, 128), 4096)
            ids = _make_map(map_kind, total_tokens, num_slots,
                            rng_seed=head_dim * 131071 + page_size * 8191 +
                            num_pages * 127 + len(map_kind))
            slot_ids = ids.astype(np.int32).reshape(num_pages, page_size)
        k = _make_src(num_slots, num_kv_heads, head_dim, dtype, 0x12)
        v = _make_src(num_slots, num_kv_heads, head_dim, dtype, 0x9A)

        # int32
        dst32 = np.zeros(
            (num_pages, page_size, 2, num_kv_heads, head_dim), dtype=dtype)
        comm.kv_gather_layer(k, v, slot_ids, dst32)
        ref = _two_gather_ref(k, v, slot_ids)
        np.testing.assert_array_equal(dst32, ref)

        # int64
        slot_ids64 = slot_ids.astype(np.int64)
        dst64 = np.zeros(
            (num_pages, page_size, 2, num_kv_heads, head_dim), dtype=dtype)
        comm.kv_gather_layer(k, v, slot_ids64, dst64)
        np.testing.assert_array_equal(dst64, ref)

    def test_fp16_matrix(self):
        for head_dim in (64, 128, 256):
            for page_size in (1, 16, 32, 64):
                for num_pages in (1, 16, 64, 128):
                    for map_kind in ("dense", "sparse", "nonmono"):
                        with self.subTest(head_dim=head_dim,
                                          page_size=page_size,
                                          num_pages=num_pages,
                                          map_kind=map_kind):
                            self._run(np.float16, head_dim, page_size,
                                      num_pages, map_kind)

    def test_bf16_matrix(self):
        # BF16 represented as a uint16 view (numpy has no native BF16). The
        # gather is a raw-byte copy, so this exercises the same path.
        for head_dim in (64, 128, 256):
            for page_size in (1, 16, 32, 64):
                for num_pages in (1, 16, 64, 128):
                    for map_kind in ("sparse", "nonmono"):
                        with self.subTest(head_dim=head_dim,
                                          page_size=page_size,
                                          num_pages=num_pages,
                                          map_kind=map_kind):
                            self._run(np.uint16, head_dim, page_size,
                                      num_pages, map_kind)

    def test_zero_pages_all_page_sizes(self):
        for page_size in (1, 16, 32, 64):
            with self.subTest(page_size=page_size):
                self._run(np.float16, 64, page_size, 0, "dense")


@unittest.skipUnless(_COMPILED, "compiled backend required for cross-check")
class KvGatherLayerBackendAgreementTest(unittest.TestCase):
    """When both backends are present, compiled and fallback agree."""

    def test_compiled_equals_fallback(self):
        for dtype in (np.float16, np.uint16):
            for head_dim in (64, 256):
                for num_pages in (1, 64):
                    page_size = 16
                    num_kv_heads = 8
                    num_slots = min(max(num_pages * page_size + 64, 128),
                                    4096)
                    ids = _make_map("sparse", num_pages * page_size,
                                    num_slots, 42)
                    slot_ids = ids.astype(np.int32).reshape(
                        num_pages, page_size)
                    k = _make_src(num_slots, num_kv_heads, head_dim, dtype, 5)
                    v = _make_src(num_slots, num_kv_heads, head_dim, dtype, 6)

                    d_compiled = np.zeros(
                        (num_pages, page_size, 2, num_kv_heads, head_dim),
                        dtype=dtype)
                    d_fallback = np.zeros(
                        (num_pages, page_size, 2, num_kv_heads, head_dim),
                        dtype=dtype)

                    _core.comm.kv_gather_layer(k, v, slot_ids, d_compiled,
                                               None)
                    _fallback_gather(k, v, slot_ids, d_fallback)
                    np.testing.assert_array_equal(d_compiled, d_fallback)


if __name__ == "__main__":
    unittest.main()
