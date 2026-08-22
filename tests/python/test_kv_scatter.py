"""Contract tests for the fused indexed K/V layer scatter (issue #1).

Covers the full acceptance matrix (BF16/FP16 x head dims {64,128,256} x
page sizes {1,16,32,64} x page counts {0..128+} x {dense,sparse,non-
monotonic} x {int32,int64}) against the two-write PyTorch reference
(``k_dst[slot_ids] = src[:, :, 0]; v_dst[slot_ids] = src[:, :, 1]``), the
async stream contract (exactly one task, result matches sync), and the
validation surface (dtype, contiguity, shape, out-of-range slot, non-unique
slot, non-writable k_dst/v_dst).

Unlike the gather (issue #2), the scatter writes DISJOINT destinations, so
slot_ids must be UNIQUE; the matrix maps are built without replacement and
the non-uniqueness case is asserted to raise.

Runs against the active backend (compiled when the extension is built,
otherwise the pure-Python fallback) and, when both are present, cross-checks
them.
"""

from __future__ import annotations

import unittest

import numpy as np

from vkernels import _backend, comm
from vkernels._fallback import kv_scatter_layer as _fallback_scatter
from vkernels.core import Stream

try:
    _core = _backend.load_extension()
    _COMPILED = _core is not None
except Exception:  # pragma: no cover
    _core = None
    _COMPILED = False


def _two_scatter_ref(src: np.ndarray, slot_ids: np.ndarray,
                     num_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """The PyTorch two-write reference (numpy): K and V from the [2] dim."""
    num_pages, page_size = slot_ids.shape
    num_kv_heads, head_dim = src.shape[3], src.shape[4]
    k = np.zeros((num_slots, num_kv_heads, head_dim), dtype=src.dtype)
    v = np.zeros((num_slots, num_kv_heads, head_dim), dtype=src.dtype)
    k[slot_ids] = src[:, :, 0]
    v[slot_ids] = src[:, :, 1]
    return k, v


def _make_src(num_slots, num_kv_heads, head_dim, dtype, seed):
    """Deterministic per-element destination buffer so mis-routing is obvious."""
    total = num_slots * num_kv_heads * head_dim
    flat = np.arange(total, dtype=np.int32)
    if dtype == np.float16:
        flat = (flat % 65504).astype(np.float16)
    else:  # BF16 represented as a uint16 view (numpy has no native BF16)
        flat = (flat % 65535).astype(np.uint16)
    np.random.seed(seed)
    np.random.shuffle(flat)
    return flat.reshape(num_slots, num_kv_heads, head_dim).copy()


def _make_src_packed(num_pages, page_size, num_kv_heads, head_dim, dtype,
                     seed):
    """Deterministic per-element packed [pages, page_size, 2, heads, dim]."""
    total = num_pages * page_size * 2 * num_kv_heads * head_dim
    flat = np.arange(total, dtype=np.int32)
    if dtype == np.float16:
        flat = (flat % 65504).astype(np.float16)
    else:
        flat = (flat % 65535).astype(np.uint16)
    np.random.seed(seed)
    np.random.shuffle(flat)
    return flat.reshape(num_pages, page_size, 2, num_kv_heads, head_dim).copy()


def _make_unique_map(kind, total_tokens, num_slots, rng_seed):
    """A UNIQUE slot map (scatter writes disjoint destinations).

    `num_slots` must be >= `total_tokens`. `dense` shuffles a contiguous
    range; `nonmono` is a descending contiguous range; `sparse` draws
    distinct slots from the whole [0, num_slots) range (genuinely sparse).
    """
    rng = np.random.default_rng(rng_seed)
    if kind == "dense":
        perm = np.arange(num_slots)
        rng.shuffle(perm)
        return perm[:total_tokens]
    if kind == "nonmono":
        return np.arange(total_tokens, 0, -1) - 1  # [total-1, ..., 0]
    return rng.choice(num_slots, size=total_tokens, replace=False)  # sparse


class KvScatterLayerTest(unittest.TestCase):
    def test_basic_fp16(self):
        num_slots = 4
        k = np.zeros((num_slots, 1, 4), dtype=np.float16)
        v = np.zeros((num_slots, 1, 4), dtype=np.float16)
        slot_ids = np.array([[3, 1]], dtype=np.int32)  # unique, non-monotonic
        src = np.arange(2 * 2 * 4, dtype=np.float16).reshape(1, 2, 2, 1, 4)
        comm.kv_scatter_layer(k, v, slot_ids, src)
        np.testing.assert_array_equal(k[slot_ids], src[:, :, 0])
        np.testing.assert_array_equal(v[slot_ids], src[:, :, 1])

    def test_basic_bf16_as_uint16(self):
        num_slots = 8
        k = np.zeros((num_slots, 2, 8), dtype=np.uint16)
        v = np.zeros((num_slots, 2, 8), dtype=np.uint16)
        slot_ids = np.array([[3, 0, 2, 1]], dtype=np.int64)  # unique, non-monotonic
        src = _make_src_packed(1, 4, 2, 8, np.uint16, 0x42)
        comm.kv_scatter_layer(k, v, slot_ids, src)
        np.testing.assert_array_equal(k[slot_ids], src[:, :, 0])
        np.testing.assert_array_equal(v[slot_ids], src[:, :, 1])

    def test_zero_pages_is_noop(self):
        num_slots = 4
        k = np.full((num_slots, 1, 1), 0xCC, dtype=np.float16)
        v = np.full((num_slots, 1, 1), 0xDD, dtype=np.float16)
        before_k, before_v = k.copy(), v.copy()
        slot_ids = np.zeros((0, 4), dtype=np.int32)
        src = np.zeros((0, 4, 2, 1, 1), dtype=np.float16)
        comm.kv_scatter_layer(k, v, slot_ids, src)  # must not raise/mutate
        np.testing.assert_array_equal(k, before_k)
        np.testing.assert_array_equal(v, before_v)

    def test_int32_and_int64_agree(self):
        num_kv_heads, head_dim = 2, 16
        num_pages, page_size = 1, 4
        total_tokens = num_pages * page_size
        num_slots = total_tokens + 4
        src = _make_src_packed(num_pages, page_size, num_kv_heads, head_dim,
                               np.float16, 1)
        ids = _make_unique_map("sparse", total_tokens, num_slots, 7)
        slot32 = ids.astype(np.int32).reshape(num_pages, page_size)
        slot64 = ids.astype(np.int64).reshape(num_pages, page_size)

        k32 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        v32 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        k64 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        v64 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        comm.kv_scatter_layer(k32, v32, slot32, src)
        comm.kv_scatter_layer(k64, v64, slot64, src)
        np.testing.assert_array_equal(k32, k64)
        np.testing.assert_array_equal(v32, v64)
        rk, rv = _two_scatter_ref(src, slot32, num_slots)
        np.testing.assert_array_equal(k32, rk)
        np.testing.assert_array_equal(v32, rv)

    def test_async_matches_sync_and_one_task(self):
        num_kv_heads, head_dim = 2, 16
        num_pages, page_size = 1, 4
        total_tokens = num_pages * page_size
        num_slots = total_tokens + 4
        src = _make_src_packed(num_pages, page_size, num_kv_heads, head_dim,
                               np.float16, 2)
        ids = _make_unique_map("dense", total_tokens, num_slots, 9)
        slot_ids = ids.astype(np.int32).reshape(num_pages, page_size)

        k_sync = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        v_sync = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        k_asy = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        v_asy = np.zeros((num_slots, num_kv_heads, head_dim), dtype=np.float16)
        comm.kv_scatter_layer(k_sync, v_sync, slot_ids, src)
        s = Stream()
        before = s.submitted()
        comm.kv_scatter_layer(k_asy, v_asy, slot_ids, src, stream=s)
        self.assertEqual(s.submitted() - before, 1)  # exactly one task
        s.wait()
        np.testing.assert_array_equal(k_sync, k_asy)
        np.testing.assert_array_equal(v_sync, v_asy)

    # --- validation -------------------------------------------------------
    def test_validation_wrong_dtype(self):
        k = np.zeros((2, 1, 4), dtype=np.float32)  # itemsize 4, not BF16/FP16
        v = np.zeros((2, 1, 4), dtype=np.float32)
        slot_ids = np.array([[0, 1]], dtype=np.int32)
        src = np.zeros((1, 2, 2, 1, 4), dtype=np.float32)
        with self.assertRaises((TypeError, ValueError)):
            comm.kv_scatter_layer(k, v, slot_ids, src)

    def test_validation_slot_dtype(self):
        k = np.zeros((2, 1, 4), dtype=np.float16)
        v = np.zeros((2, 1, 4), dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.float32)  # wrong dtype
        src = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        with self.assertRaises((TypeError, ValueError)):
            comm.kv_scatter_layer(k, v, slot_ids, src)

    def test_validation_shape_mismatch(self):
        num_slots = 4
        k = np.zeros((num_slots, 2, 16), dtype=np.float16)
        v = np.zeros((num_slots, 2, 16), dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.int32)  # 1 page
        src = np.zeros((2, 1, 2, 2, 16), dtype=np.float16)  # wrong page count
        with self.assertRaises((ValueError, TypeError)):
            comm.kv_scatter_layer(k, v, slot_ids, src)

    def test_validation_out_of_range_slot(self):
        num_slots = 4
        k = np.zeros((num_slots, 1, 4), dtype=np.float16)
        v = np.zeros((num_slots, 1, 4), dtype=np.float16)
        slot_ids = np.array([[0, 4]], dtype=np.int32)  # 4 == num_slots
        src = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        with self.assertRaises(ValueError):
            comm.kv_scatter_layer(k, v, slot_ids, src)

    def test_validation_non_unique_slot(self):
        num_slots = 4
        k = np.zeros((num_slots, 1, 4), dtype=np.float16)
        v = np.zeros((num_slots, 1, 4), dtype=np.float16)
        slot_ids = np.array([[1, 1]], dtype=np.int32)  # duplicate destination
        src = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        with self.assertRaises(ValueError):
            comm.kv_scatter_layer(k, v, slot_ids, src)

    def test_validation_non_contiguous_dst(self):
        num_slots = 4
        full = np.zeros((num_slots, 4, 1, 4), dtype=np.float16)
        k = full[:, ::2]  # non-contiguous
        v = np.zeros((num_slots, 2, 4), dtype=np.float16)
        slot_ids = np.array([[0, 1]], dtype=np.int32)
        src = np.zeros((1, 2, 2, 2, 4), dtype=np.float16)
        with self.assertRaises((ValueError, TypeError)):
            comm.kv_scatter_layer(k, v, slot_ids, src)

    def test_validation_non_writable_dst(self):
        num_slots = 4
        k = np.zeros((num_slots, 1, 4), dtype=np.float16)
        v = np.zeros((num_slots, 1, 4), dtype=np.float16)
        k.setflags(write=False)
        slot_ids = np.array([[0, 1]], dtype=np.int32)
        src = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        with self.assertRaises((ValueError, TypeError)):
            comm.kv_scatter_layer(k, v, slot_ids, src)


class KvScatterLayerMatrixTest(unittest.TestCase):
    """Full acceptance matrix: dtype x head_dim x page_size x num_pages x
    map_kind, int32 and int64, against the two-write reference. Slot maps
    are UNIQUE (scatter writes disjoint destinations)."""

    def _run(self, dtype, head_dim, page_size, num_pages, map_kind):
        num_kv_heads = 8
        total_tokens = num_pages * page_size
        if total_tokens == 0:
            num_slots = 16
            slot_ids = np.zeros((num_pages, page_size), dtype=np.int32)
            src = np.zeros((num_pages, page_size, 2, num_kv_heads, head_dim),
                           dtype=dtype)
        else:
            # num_slots >= total_tokens with headroom so sparse maps are
            # genuinely sparse and dense/nonmono still fit.
            num_slots = max(total_tokens + 64, 128)
            ids = _make_unique_map(map_kind, total_tokens, num_slots,
                                   rng_seed=head_dim * 131071 +
                                   page_size * 8191 + num_pages * 127 +
                                   len(map_kind))
            slot_ids = ids.astype(np.int32).reshape(num_pages, page_size)
            src = _make_src_packed(num_pages, page_size, num_kv_heads,
                                   head_dim, dtype, 0x9A)

        # int32
        k32 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=dtype)
        v32 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=dtype)
        comm.kv_scatter_layer(k32, v32, slot_ids, src)
        rk, rv = _two_scatter_ref(src, slot_ids, num_slots)
        np.testing.assert_array_equal(k32, rk)
        np.testing.assert_array_equal(v32, rv)

        # int64
        slot_ids64 = slot_ids.astype(np.int64)
        k64 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=dtype)
        v64 = np.zeros((num_slots, num_kv_heads, head_dim), dtype=dtype)
        comm.kv_scatter_layer(k64, v64, slot_ids64, src)
        np.testing.assert_array_equal(k64, rk)
        np.testing.assert_array_equal(v64, rv)

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
        # scatter is a raw-byte copy, so this exercises the same path.
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
class KvScatterLayerBackendAgreementTest(unittest.TestCase):
    """When both backends are present, compiled and fallback agree."""

    def test_compiled_equals_fallback(self):
        for dtype in (np.float16, np.uint16):
            for head_dim in (64, 256):
                for num_pages in (1, 64):
                    num_kv_heads = 8
                    page_size = 16
                    total_tokens = num_pages * page_size
                    num_slots = max(total_tokens + 64, 128)
                    ids = _make_unique_map("sparse", total_tokens, num_slots,
                                           42)
                    slot_ids = ids.astype(np.int32).reshape(
                        num_pages, page_size)
                    src = _make_src_packed(num_pages, page_size,
                                           num_kv_heads, head_dim, dtype, 5)

                    kc = np.zeros((num_slots, num_kv_heads, head_dim),
                                  dtype=dtype)
                    vc = np.zeros((num_slots, num_kv_heads, head_dim),
                                  dtype=dtype)
                    kf = np.zeros((num_slots, num_kv_heads, head_dim),
                                  dtype=dtype)
                    vf = np.zeros((num_slots, num_kv_heads, head_dim),
                                  dtype=dtype)

                    _core.comm.kv_scatter_layer(kc, vc, slot_ids, src, None)
                    _fallback_scatter(kf, vf, slot_ids, src)
                    np.testing.assert_array_equal(kc, kf)
                    np.testing.assert_array_equal(vc, vf)


if __name__ == "__main__":
    unittest.main()
