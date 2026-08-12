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
from vkernels.kernels import (add, fused_moe_mxfp4, gemm, max as vk_max,
                              moe_align_block_size, relu, scale,
                              sum as vk_sum)

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


# --- MoE helpers (mirror tests/kernels/moe/test_moe_fused.cpp) ------------

def _bf2f(v):
    return float(np.frombuffer((int(v) << 16).to_bytes(4, "little"),
                               dtype=np.float32)[0])


def _f2bf(v):
    bits = int.from_bytes(np.float32(v).tobytes(), "little")
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
    return np.uint16((bits >> 16) & 0xFFFF)


def _e2m1_nibble(f):
    neg = f < 0
    af = abs(f)
    if af == 0.0 or np.isnan(af):
        return 0x8 if neg else 0x0
    if np.isinf(af):
        return 0xE if neg else 0x6
    vals = [0.25, 1.0, 1.5, 2.0, 3.0]
    nibs = [1, 2, 3, 4, 5]
    best_d = abs(af - vals[0])
    best_n = nibs[0]
    for v, n in zip(vals[1:], nibs[1:]):
        d = abs(af - v)
        if d < best_d:
            best_d = d
            best_n = n
    return (best_n | 0x8) if neg else best_n


def _pack_e2m1_pair(v0, v1):
    return _e2m1_nibble(v0) | (_e2m1_nibble(v1) << 4)


def _random_packed(shape, rng):
    """Random packed E2M1 weight array (each byte = two nearest-nibble fp4)."""
    n = int(np.prod(shape))
    vals = rng.uniform(-3.0, 3.0, n * 2)
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = _pack_e2m1_pair(float(vals[2 * i]), float(vals[2 * i + 1]))
    return out.reshape(shape)


def _ones_weight_bytes(*shape):
    return np.full(shape, _pack_e2m1_pair(1.0, 1.0), dtype=np.uint8)


@unittest.skipIf(np is None, "numpy is required for these tests")
class FusedMoeTinySanityTest(unittest.TestCase):
    def test_tiny_sanity(self):
        M, hidden, ispp = 16, 128, 64
        group_size, BLOCK_M = 32, 16
        EM = M  # top_k=1

        A = np.full((M, hidden), _f2bf(1.0), dtype=np.uint16)
        w13 = _ones_weight_bytes(1, 2 * ispp, hidden // 2)
        w13_scale = np.full((1, 2 * ispp, hidden // group_size), 127, dtype=np.uint8)
        w2 = _ones_weight_bytes(1, hidden, ispp // 2)
        w2_scale = np.full((1, hidden, ispp // group_size), 127, dtype=np.uint8)
        sorted_ids = np.arange(EM, dtype=np.int32)
        topk_w = np.ones(EM, dtype=_F32)
        expert_ids = np.zeros(EM // BLOCK_M, dtype=np.int32)
        act = np.empty(EM * ispp, dtype=np.uint16)

        out = fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              sorted_ids, topk_w, expert_ids,
                              act_scratch=act, out=None, top_k=1,
                              group_size=group_size)

        silu128 = 128.0 / (1.0 + np.exp(-128.0))
        expected_act = silu128 * 128.0
        expected_out = expected_act * ispp

        np.testing.assert_allclose(out, expected_out, rtol=1e-2)
        np.testing.assert_allclose([_bf2f(v) for v in act], expected_act, atol=0.5)

    def test_returns_2d_and_zeroes_out(self):
        M, hidden, ispp = 16, 128, 64
        A = np.full((M, hidden), _f2bf(0.0), dtype=np.uint16)
        w13 = np.zeros((1, 2 * ispp, hidden // 2), dtype=np.uint8)
        w13_scale = np.full((1, 2 * ispp, hidden // 32), 127, dtype=np.uint8)
        w2 = np.zeros((1, hidden, ispp // 2), dtype=np.uint8)
        w2_scale = np.full((1, hidden, ispp // 32), 127, dtype=np.uint8)
        out = fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              np.arange(16, dtype=np.int32),
                              np.ones(16, dtype=_F32),
                              np.zeros(1, dtype=np.int32), top_k=1)
        self.assertEqual(out.shape, (M, hidden))
        self.assertEqual(out.dtype, _F32)
        np.testing.assert_array_equal(out, np.zeros((M, hidden), dtype=_F32))


@unittest.skipIf(np is None, "numpy is required for these tests")
class FusedMoeBiasSanityTest(unittest.TestCase):
    def test_bias_sanity(self):
        M, hidden, ispp = 16, 128, 64
        group_size, BLOCK_M = 32, 16
        EM = M

        A = np.full((M, hidden), _f2bf(0.0), dtype=np.uint16)
        w13 = np.zeros((1, 2 * ispp, hidden // 2), dtype=np.uint8)
        w13_scale = np.full((1, 2 * ispp, hidden // group_size), 127, dtype=np.uint8)
        w2 = np.zeros((1, hidden, ispp // 2), dtype=np.uint8)
        w2_scale = np.full((1, hidden, ispp // group_size), 127, dtype=np.uint8)
        b13 = np.empty(2 * ispp, dtype=_F32)
        b13[:ispp] = 1.0
        b13[ispp:] = 2.0
        b2 = np.ones(hidden, dtype=_F32)

        act = np.empty(EM * ispp, dtype=np.uint16)
        out = fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              np.arange(EM, dtype=np.int32),
                              np.ones(EM, dtype=_F32),
                              np.zeros(EM // BLOCK_M, dtype=np.int32),
                              act_scratch=act, top_k=1,
                              group_size=group_size, b13=b13, b2=b2)

        silu1 = 1.0 / (1.0 + np.exp(-1.0))
        expected_act = silu1 * 2.0
        np.testing.assert_allclose([_bf2f(v) for v in act], expected_act, rtol=1e-2)
        np.testing.assert_allclose(out, 1.0, rtol=1e-4)


@unittest.skipIf(np is None, "numpy is required for these tests")
class FusedMoeSwiGLUClampTest(unittest.TestCase):
    def test_swiglu_clamp(self):
        M, hidden, ispp = 16, 128, 64
        group_size, BLOCK_M = 32, 16
        EM = M

        A = np.full((M, hidden), _f2bf(1.0), dtype=np.uint16)
        w13 = _ones_weight_bytes(1, 2 * ispp, hidden // 2)
        w13_scale = np.full((1, 2 * ispp, hidden // group_size), 127, dtype=np.uint8)
        w2 = _ones_weight_bytes(1, hidden, ispp // 2)
        w2_scale = np.full((1, hidden, ispp // group_size), 127, dtype=np.uint8)

        act = np.empty(EM * ispp, dtype=np.uint16)
        out = fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              np.arange(EM, dtype=np.int32),
                              np.ones(EM, dtype=_F32),
                              np.zeros(EM // BLOCK_M, dtype=np.int32),
                              act_scratch=act, top_k=1,
                              group_size=group_size, swiglu_limit=10.0)

        silu10 = 10.0 / (1.0 + np.exp(-10.0))
        expected_act = silu10 * 10.0
        expected_out = expected_act * ispp
        np.testing.assert_allclose([_bf2f(v) for v in act], expected_act, atol=0.2)
        np.testing.assert_allclose(out, expected_out, rtol=1e-2)


@unittest.skipIf(np is None, "numpy is required for these tests")
class FusedMoeSituTest(unittest.TestCase):
    def test_situ_ignores_clamp(self):
        M, hidden, ispp = 16, 128, 64
        group_size, BLOCK_M = 32, 16
        EM = M

        A = np.full((M, hidden), _f2bf(1.0), dtype=np.uint16)
        w13 = _ones_weight_bytes(1, 2 * ispp, hidden // 2)
        w13_scale = np.full((1, 2 * ispp, hidden // group_size), 127,
                            dtype=np.uint8)
        w2 = _ones_weight_bytes(1, hidden, ispp // 2)
        w2_scale = np.full((1, hidden, ispp // group_size), 127, dtype=np.uint8)

        act = np.empty(EM * ispp, dtype=np.uint16)
        # swiglu_limit=1.0 must be IGNORED on the SiTU path.
        out = fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              np.arange(EM, dtype=np.int32),
                              np.ones(EM, dtype=_F32),
                              np.zeros(EM // BLOCK_M, dtype=np.int32),
                              act_scratch=act, top_k=1, group_size=group_size,
                              swiglu_limit=1.0, activation="situ",
                              beta=4.0, linear_beta=25.0)

        # situ_and_mul reference (vLLM): gate = up = 128, unclamped.
        gate = up = 128.0
        sig = 1.0 / (1.0 + np.exp(-gate))
        expected_act = (4.0 * np.tanh(gate / 4.0) * sig) * (
            25.0 * np.tanh(up / 25.0))
        expected_out = expected_act * ispp

        np.testing.assert_allclose([_bf2f(v) for v in act], expected_act,
                                   atol=0.5)
        np.testing.assert_allclose(out, expected_out, rtol=1e-2)

    def test_rejects_unknown_activation(self):
        M, hidden, ispp = 16, 128, 64
        A = np.zeros((M, hidden), dtype=np.uint16)
        w13 = np.zeros((1, 2 * ispp, hidden // 2), dtype=np.uint8)
        w13_scale = np.full((1, 2 * ispp, hidden // 32), 127, dtype=np.uint8)
        w2 = np.zeros((1, hidden, ispp // 2), dtype=np.uint8)
        w2_scale = np.full((1, hidden, ispp // 32), 127, dtype=np.uint8)
        with self.assertRaises(ValueError):
            fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                            np.arange(16, dtype=np.int32),
                            np.ones(16, dtype=_F32),
                            np.zeros(1, dtype=np.int32),
                            top_k=1, activation="gelu")


@unittest.skipIf(np is None, "numpy is required for these tests")
class FusedMoeMultiExpertTest(unittest.TestCase):
    def test_multi_expert(self):
        M, hidden, ispp, E = 8, 128, 64, 4
        group_size, BLOCK_M = 32, 16
        EM = 16  # padded

        A = np.full((M, hidden), _f2bf(1.0), dtype=np.uint16)
        w13 = np.empty((E, 2 * ispp, hidden // 2), dtype=np.uint8)
        w13_scale = np.full((E, 2 * ispp, hidden // group_size), 127, dtype=np.uint8)
        w2 = np.empty((E, hidden, ispp // 2), dtype=np.uint8)
        w2_scale = np.full((E, hidden, ispp // group_size), 127, dtype=np.uint8)
        for e in range(E):
            wv = 1.0 + e  # 1, 2, 3, 4
            w13[e] = _pack_e2m1_pair(wv, wv)
            w2[e] = _pack_e2m1_pair(wv, wv)

        sorted_ids = np.array([i % M for i in range(EM)], dtype=np.int32)
        expert_ids = np.array([i % E for i in range(EM // BLOCK_M)], dtype=np.int32)

        out = fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              sorted_ids, np.ones(EM, dtype=_F32),
                              expert_ids, top_k=1, group_size=group_size)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out > 0.0))


@unittest.skipIf(np is None, "numpy is required for these tests")
class FusedMoeExpertFilterTest(unittest.TestCase):
    def test_expert_filter(self):
        M, hidden, ispp = 16, 128, 64
        group_size, BLOCK_M = 32, 16
        EM = M

        A = np.full((M, hidden), _f2bf(1.0), dtype=np.uint16)
        w13 = _ones_weight_bytes(1, 2 * ispp, hidden // 2)
        w13_scale = np.full((1, 2 * ispp, hidden // group_size), 127, dtype=np.uint8)
        w2 = _ones_weight_bytes(1, hidden, ispp // 2)
        w2_scale = np.full((1, hidden, ispp // group_size), 127, dtype=np.uint8)
        expert_ids = np.full(EM // BLOCK_M, -1, dtype=np.int32)  # all filtered

        out = fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              np.arange(EM, dtype=np.int32),
                              np.ones(EM, dtype=_F32), expert_ids,
                              out=np.full(M * hidden, 99.0, dtype=_F32),
                              top_k=1, group_size=group_size)
        np.testing.assert_allclose(out, 99.0, rtol=0.01)


@unittest.skipIf(np is None, "numpy is required for these tests")
class MoeAlignTest(unittest.TestCase):
    def test_basic_8x4(self):
        M, top_k, E, BS = 8, 4, 4, 16
        N = M * top_k  # 32

        # Expert 0: 5 (token0 ×4, token1 sel0); expert 1: 11; expert 2: 16
        topk_ids = np.zeros((M, top_k), dtype=np.int32)
        topk_ids[0] = 0
        topk_ids[1] = [0, 1, 1, 1]
        topk_ids[2] = 1
        topk_ids[3] = 1
        topk_ids[4] = 2
        topk_ids[5] = 2
        topk_ids[6] = 2
        topk_ids[7] = 2

        sorted_ids, expert_ids, EM = moe_align_block_size(topk_ids, E, BS)

        self.assertEqual(EM, 48)
        self.assertEqual(expert_ids[0], 0)
        self.assertEqual(expert_ids[1], 1)
        self.assertEqual(expert_ids[2], 2)

        # Expert 0 block: 5 real flat indices [0,1,2,3,4], padded with 32
        self.assertEqual(np.count_nonzero(sorted_ids[:BS] == 0), 1)
        self.assertEqual(sorted_ids[4], 4)  # flat 4 = token 1 sel 0

        # Expert 2 block: flat indices 16..31 (tokens 4,5,6,7)
        cnt16 = np.count_nonzero(sorted_ids[32:48] == 16)
        cnt31 = np.count_nonzero(sorted_ids[32:48] == 31)
        self.assertEqual(cnt16, 1)
        self.assertEqual(cnt31, 1)

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            moe_align_block_size(np.zeros(8, dtype=np.int32), 4)


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

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_fused_moe_consistent(self):
        rng = np.random.default_rng(24)
        M, hidden, ispp, E = 32, 128, 64, 2
        group_size, BLOCK_M = 32, 16
        EM = 2 * BLOCK_M  # one 16-row block per expert; all 32 rows real

        A = np.array([_f2bf(float(v))
                      for v in rng.uniform(-1.0, 1.0, M * hidden)]).reshape(
            M, hidden).astype(np.uint16)
        w13 = _random_packed((E, 2 * ispp, hidden // 2), rng)
        w13_scale = np.full((E, 2 * ispp, hidden // group_size), 127,
                            dtype=np.uint8)
        w2 = _random_packed((E, hidden, ispp // 2), rng)
        w2_scale = np.full((E, hidden, ispp // group_size), 127, dtype=np.uint8)
        sorted_ids = np.arange(EM, dtype=np.int32)  # flat, token = i (top_k=1)
        topk_w = rng.uniform(0.5, 1.5, EM).astype(_F32)
        expert_ids = np.array([0, 1], dtype=np.int32)

        fb = _fallback_impl()
        act_c = np.empty(EM * ispp, dtype=np.uint16)
        act_f = np.empty(EM * ispp, dtype=np.uint16)
        out_c = np.zeros(M * hidden, dtype=_F32)
        out_f = np.zeros(M * hidden, dtype=_F32)
        common = (A, w13, w13_scale, w2, w2_scale, sorted_ids, topk_w,
                  expert_ids)
        _backend.load_extension().kernels.fused_moe_mxfp4(
            *common, act_c, out_c, M, hidden, ispp, 1, EM, group_size, 0.0,
            0, 4.0, 25.0, None, None)
        fb.fused_moe_mxfp4(*common, act_f, out_f, M, hidden, ispp, 1, EM,
                           group_size, 0.0, 0, 4.0, 25.0, None, None)

        np.testing.assert_allclose(out_c, out_f, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose([_bf2f(int(v)) for v in act_c],
                                   [_bf2f(int(v)) for v in act_f], atol=0.5)


if __name__ == "__main__":
    unittest.main()
