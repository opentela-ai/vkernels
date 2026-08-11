"""Contract tests for vkernels.kernels (elementwise, reduce, gemm).

Every test runs against the active backend; when the compiled extension and
the pure-Python reference are both available they are additionally compared
bit-for-bit on random float32 data.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from vkernels import _backend, kernels
from vkernels.kernels import add, gemm, max as vk_max, relu, scale, sum as vk_sum

_F32 = np.dtype(np.float32) if np is not None else None

_COMPILED = _backend.load_extension() is not None


def _fallback_impl():
    from vkernels import _fallback
    return _fallback


@unittest.skipIf(np is None, "numpy is required for these tests")
class AddTest(unittest.TestCase):
    def test_basic(self):
        out = add([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(out, np.array([11.0, 22.0, 33.0], dtype=_F32))
        self.assertEqual(out.dtype, _F32)

    def test_float64_input_is_converted(self):
        out = add(np.array([1.0, 2.0]), np.array([3.0, 4.0]))  # float64 in
        self.assertEqual(out.dtype, _F32)
        np.testing.assert_array_equal(out, np.array([4.0, 6.0], dtype=_F32))

    def test_out_is_written_in_place_and_returned(self):
        a = np.array([1.0, 2.0], dtype=_F32)
        b = np.array([3.0, 4.0], dtype=_F32)
        out = np.empty(2, dtype=_F32)
        self.assertIs(add(a, b, out=out), out)
        np.testing.assert_array_equal(out, np.array([4.0, 6.0], dtype=_F32))

    def test_raises_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            add([1.0], [1.0, 2.0])

    def test_out_validation(self):
        a = np.array([1.0], dtype=_F32)
        b = a.copy()
        with self.assertRaises(TypeError):
            add(a, b, out=[0.0])  # not an ndarray
        with self.assertRaises(TypeError):
            add(a, b, out=np.zeros(1))  # float64
        with self.assertRaises(ValueError):
            add(a, b, out=np.zeros(2, dtype=_F32))  # wrong length
        with self.assertRaises(ValueError):
            add(a, b, out=np.zeros(4, dtype=_F32)[::2])  # non-contiguous
        ro = np.zeros(1, dtype=_F32)
        ro.setflags(write=False)
        with self.assertRaises(ValueError):
            add(a, b, out=ro)  # read-only

    def test_random_matches_numpy(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal(1024).astype(_F32)
        b = rng.standard_normal(1024).astype(_F32)
        np.testing.assert_array_equal(add(a, b), a + b)


@unittest.skipIf(np is None, "numpy is required for these tests")
class ScaleReluTest(unittest.TestCase):
    def test_scale(self):
        out = scale([1.0, 2.0, 3.0], 2.0)
        np.testing.assert_array_equal(out, np.array([2.0, 4.0, 6.0], dtype=_F32))

    def test_scale_random(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(512).astype(_F32)
        alpha = np.float32(0.25)
        np.testing.assert_array_equal(scale(x, float(alpha)), alpha * x)

    def test_scale_mismatch_raises(self):
        with self.assertRaises(ValueError):
            scale([1.0, 2.0], 1.0, out=np.zeros(3, dtype=_F32))

    def test_relu(self):
        out = relu([-1.0, 0.0, 2.5])
        np.testing.assert_array_equal(out, np.array([0.0, 0.0, 2.5], dtype=_F32))

    def test_relu_random(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal(512).astype(_F32)
        np.testing.assert_array_equal(relu(x), np.maximum(x, np.float32(0.0)))


@unittest.skipIf(np is None, "numpy is required for these tests")
class ReduceTest(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(vk_sum([1.0, 2.0, 3.0]), 6.0)
        self.assertEqual(vk_sum(np.array([0.5, -0.25], dtype=_F32)), 0.25)

    def test_sum_empty_raises(self):
        with self.assertRaises(ValueError):
            vk_sum([])

    def test_max(self):
        self.assertEqual(vk_max([1.0, 5.0, 3.0]), 5.0)
        self.assertEqual(vk_max([-2.0, -1.0]), -1.0)

    def test_max_empty_raises(self):
        with self.assertRaises(ValueError):
            vk_max([])

    def test_sum_random_close_to_numpy(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(4096).astype(_F32)
        expected = float(np.sum(x, dtype=_F32))
        self.assertAlmostEqual(vk_sum(x), expected, delta=1e-3)

    def test_reduce_returns_python_float(self):
        self.assertIsInstance(vk_sum([1.0, 2.0]), float)
        self.assertIsInstance(vk_max([1.0, 2.0]), float)


@unittest.skipIf(np is None, "numpy is required for these tests")
class GemmTest(unittest.TestCase):
    def test_basic(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=_F32)
        B = np.array([[1.0], [1.0]], dtype=_F32)
        np.testing.assert_array_equal(gemm(A, B), np.array([[3.0], [7.0]], dtype=_F32))

    def test_alpha_beta(self):
        A = np.array([[1.0, 2.0]], dtype=_F32)
        B = np.array([[1.0], [1.0]], dtype=_F32)
        C = np.array([[10.0]], dtype=_F32)
        out = gemm(A, B, alpha=2.0, beta=0.5, out=C)
        self.assertIs(out, C)
        np.testing.assert_array_equal(C, np.array([[11.0]], dtype=_F32))  # 2*3 + 0.5*10

    def test_out_shape_inference(self):
        A = np.random.default_rng(11).standard_normal((4, 5)).astype(_F32)
        B = np.random.default_rng(12).standard_normal((5, 6)).astype(_F32)
        out = gemm(A, B)
        self.assertEqual(out.shape, (4, 6))
        self.assertEqual(out.dtype, _F32)

    def test_flat_out(self):
        A = np.eye(2, dtype=_F32)
        B = np.eye(2, dtype=_F32)
        out = gemm(A, B, out=np.zeros(4, dtype=_F32))
        self.assertEqual(out.shape, (2, 2))

    def test_random_matches_numpy(self):
        rng = np.random.default_rng(13)
        A = rng.standard_normal((8, 9)).astype(_F32)
        B = rng.standard_normal((9, 10)).astype(_F32)
        np.testing.assert_allclose(gemm(A, B), A @ B, rtol=1e-5, atol=1e-6)

    def test_inner_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gemm(np.zeros((2, 3), dtype=_F32), np.zeros((4, 5), dtype=_F32))

    def test_requires_2d(self):
        with self.assertRaises(ValueError):
            gemm(np.zeros(4, dtype=_F32), np.zeros((2, 2), dtype=_F32))


@unittest.skipIf(np is None, "numpy is required for these tests")
class BackendConsistencyTest(unittest.TestCase):
    """Bit-for-bit agreement between compiled and fallback backends."""

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_elementwise_identical(self):
        rng = np.random.default_rng(21)
        a = rng.standard_normal(300).astype(_F32)
        b = rng.standard_normal(300).astype(_F32)
        fb = _fallback_impl()
        for name, args in [
            ("add", (a, b)),
            ("scale", (a, 0.5)),
            ("relu", (a,)),
        ]:
            out_c = np.empty_like(a)
            out_f = np.empty_like(a)
            getattr(_backend.load_extension().kernels, name)(*args, out_c)
            getattr(fb, name)(*args, out_f)
            np.testing.assert_array_equal(out_c, out_f, err_msg=name)

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_reduce_identical(self):
        rng = np.random.default_rng(22)
        x = rng.standard_normal(1000).astype(_F32)
        fb = _fallback_impl()
        c = _backend.load_extension().kernels
        self.assertEqual(c.sum(x), fb.sum(x))
        self.assertEqual(c.max(x), fb.max(x))

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_gemm_identical(self):
        rng = np.random.default_rng(23)
        A = rng.standard_normal((7, 8)).astype(_F32)
        B = rng.standard_normal((8, 6)).astype(_F32)
        Cc = np.zeros((7, 6), dtype=_F32)
        Cf = np.zeros((7, 6), dtype=_F32)
        _backend.load_extension().kernels.gemm(7, 6, 8, 1.0, A, B, 0.5, Cc)
        _fallback_impl().gemm(7, 6, 8, 1.0, A, B, 0.5, Cf)
        np.testing.assert_array_equal(Cc, Cf)


if __name__ == "__main__":
    unittest.main()
