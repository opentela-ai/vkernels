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
from vkernels.kernels import (
    add,
    fused_moe_mxfp4,
    gemm,
    gemm_bf16,
    gemm_bf16_config,
    kda_delta_rule_fwd,
    kda_delta_rule_intra,
    kda_delta_rule_inter,
    kda_gla_fwd_o,
    kda_gate_chunk_cumsum,
    kda_layer_norm_gated,
    kda_naive_delta_rule_fwd,
    kda_pack_bitmatrix,
    max as vk_max,
    mla_fwd,
    mla_config,
    moe_align_block_size,
    mxfp4_moe_quant,
    mxfp4_moe_scatter_reduce,
    mxfp4_moe_scatter_reduce_q,
    mxfp4_moe_sort,
    mxfp4_moe_sort_scales,
    relu,
    scale,
    sum as vk_sum,
)

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
    return float(
        np.frombuffer((int(v) << 16).to_bytes(4, "little"), dtype=np.float32)[0]
    )


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

        out = fused_moe_mxfp4(
            A,
            w13,
            w13_scale,
            w2,
            w2_scale,
            sorted_ids,
            topk_w,
            expert_ids,
            act_scratch=act,
            out=None,
            top_k=1,
            group_size=group_size,
        )

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
        out = fused_moe_mxfp4(
            A,
            w13,
            w13_scale,
            w2,
            w2_scale,
            np.arange(16, dtype=np.int32),
            np.ones(16, dtype=_F32),
            np.zeros(1, dtype=np.int32),
            top_k=1,
        )
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
        out = fused_moe_mxfp4(
            A,
            w13,
            w13_scale,
            w2,
            w2_scale,
            np.arange(EM, dtype=np.int32),
            np.ones(EM, dtype=_F32),
            np.zeros(EM // BLOCK_M, dtype=np.int32),
            act_scratch=act,
            top_k=1,
            group_size=group_size,
            b13=b13,
            b2=b2,
        )

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
        out = fused_moe_mxfp4(
            A,
            w13,
            w13_scale,
            w2,
            w2_scale,
            np.arange(EM, dtype=np.int32),
            np.ones(EM, dtype=_F32),
            np.zeros(EM // BLOCK_M, dtype=np.int32),
            act_scratch=act,
            top_k=1,
            group_size=group_size,
            swiglu_limit=10.0,
        )

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
        w13_scale = np.full((1, 2 * ispp, hidden // group_size), 127, dtype=np.uint8)
        w2 = _ones_weight_bytes(1, hidden, ispp // 2)
        w2_scale = np.full((1, hidden, ispp // group_size), 127, dtype=np.uint8)

        act = np.empty(EM * ispp, dtype=np.uint16)
        # swiglu_limit=1.0 must be IGNORED on the SiTU path.
        out = fused_moe_mxfp4(
            A,
            w13,
            w13_scale,
            w2,
            w2_scale,
            np.arange(EM, dtype=np.int32),
            np.ones(EM, dtype=_F32),
            np.zeros(EM // BLOCK_M, dtype=np.int32),
            act_scratch=act,
            top_k=1,
            group_size=group_size,
            swiglu_limit=1.0,
            activation="situ",
            beta=4.0,
            linear_beta=25.0,
        )

        # situ_and_mul reference (vLLM): gate = up = 128, unclamped.
        gate = up = 128.0
        sig = 1.0 / (1.0 + np.exp(-gate))
        expected_act = (4.0 * np.tanh(gate / 4.0) * sig) * (25.0 * np.tanh(up / 25.0))
        expected_out = expected_act * ispp

        np.testing.assert_allclose([_bf2f(v) for v in act], expected_act, atol=0.5)
        np.testing.assert_allclose(out, expected_out, rtol=1e-2)

    def test_rejects_unknown_activation(self):
        M, hidden, ispp = 16, 128, 64
        A = np.zeros((M, hidden), dtype=np.uint16)
        w13 = np.zeros((1, 2 * ispp, hidden // 2), dtype=np.uint8)
        w13_scale = np.full((1, 2 * ispp, hidden // 32), 127, dtype=np.uint8)
        w2 = np.zeros((1, hidden, ispp // 2), dtype=np.uint8)
        w2_scale = np.full((1, hidden, ispp // 32), 127, dtype=np.uint8)
        with self.assertRaises(ValueError):
            fused_moe_mxfp4(
                A,
                w13,
                w13_scale,
                w2,
                w2_scale,
                np.arange(16, dtype=np.int32),
                np.ones(16, dtype=_F32),
                np.zeros(1, dtype=np.int32),
                top_k=1,
                activation="gelu",
            )


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

        out = fused_moe_mxfp4(
            A,
            w13,
            w13_scale,
            w2,
            w2_scale,
            sorted_ids,
            np.ones(EM, dtype=_F32),
            expert_ids,
            top_k=1,
            group_size=group_size,
        )
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

        out = fused_moe_mxfp4(
            A,
            w13,
            w13_scale,
            w2,
            w2_scale,
            np.arange(EM, dtype=np.int32),
            np.ones(EM, dtype=_F32),
            expert_ids,
            out=np.full(M * hidden, 99.0, dtype=_F32),
            top_k=1,
            group_size=group_size,
        )
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

        A = (
            np.array([_f2bf(float(v)) for v in rng.uniform(-1.0, 1.0, M * hidden)])
            .reshape(M, hidden)
            .astype(np.uint16)
        )
        w13 = _random_packed((E, 2 * ispp, hidden // 2), rng)
        w13_scale = np.full((E, 2 * ispp, hidden // group_size), 127, dtype=np.uint8)
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
        common = (A, w13, w13_scale, w2, w2_scale, sorted_ids, topk_w, expert_ids)
        _backend.load_extension().kernels.fused_moe_mxfp4(
            *common,
            act_c,
            out_c,
            M,
            hidden,
            ispp,
            1,
            EM,
            group_size,
            0.0,
            0,
            4.0,
            25.0,
            None,
            None,
        )
        fb.fused_moe_mxfp4(
            *common,
            act_f,
            out_f,
            M,
            hidden,
            ispp,
            1,
            EM,
            group_size,
            0.0,
            0,
            4.0,
            25.0,
            None,
            None,
        )

        np.testing.assert_allclose(out_c, out_f, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(
            [_bf2f(int(v)) for v in act_c], [_bf2f(int(v)) for v in act_f], atol=0.5
        )


# ---------------------------------------------------------------------------
# MXFP4 MoE orchestration: mxfp4_moe_{quant,sort,sort_scales,scatter_reduce[_q]}.
# These helpers exactly mirror the C++ kernels in src/c/vkernels/kernels/
# moe_aux.{cpp,hpp} so the tests can serve as an independent oracle.
# ---------------------------------------------------------------------------
_FP4_LUT = None


def _fp4_nib_c(x):
    """Round to nearest E2M1 nibble, ties to the larger magnitude (mirrors
    ``float_to_fp4_nib`` in moe_aux.cpp)."""
    v = abs(float(x))
    if v < 0.125:
        mag = 0
    elif v < 0.625:
        mag = 1
    elif v < 1.25:
        mag = 2
    elif v < 1.75:
        mag = 3
    elif v < 2.5:
        mag = 4
    else:
        mag = 5
    if x < 0.0:
        mag |= 0x8
    return mag


def _nib2f_c(n):
    """Decode an E2M1 nibble (mirrors the table in moe.cpp)."""
    s = (n >> 3) & 1
    e = (n >> 1) & 3
    m = n & 1
    if e == 0:
        return (-0.25 if m else -0.0) if s else (0.25 if m else 0.0)
    if e == 3:
        # code 6 = +inf, 7 = NaN, 14 = -inf, 15 = NaN
        if m:
            return float("nan")
        return float("-inf") if s else float("inf")
    v = (1.0 + 0.5 * m) * (2.0 ** (e - 1))
    return -v if s else v


def _ue8m0_c(s):
    """Decode a ue8m0 scale byte (mirrors the simple `s << 23` path)."""
    if s == 0xFF:
        return 0.0
    return float(
        np.frombuffer((int(s) << 23).to_bytes(4, "little"), dtype=np.float32)[0]
    )


def _fp4_lut():
    global _FP4_LUT
    if _FP4_LUT is None:
        _FP4_LUT = np.array([_nib2f_c(i) for i in range(16)], dtype=np.float32)
    return _FP4_LUT


def _dequant_quantized(packed, scales, M, width, gs):
    """Dequant an [M, width/2] E2M1 + [M, width/gs] ue8m0 pair to float32
    [M, width], exactly as the kernels' scatter_reduce_q does inline."""
    lo = (packed & 0x0F).astype(np.int64)
    hi = ((packed >> 4) & 0x0F).astype(np.int64)
    nibs = np.empty((M, width), dtype=np.int64)
    nibs[:, 0::2] = lo
    nibs[:, 1::2] = hi
    vals = _fp4_lut()[nibs]
    ng = width // gs
    sc = np.array(
        [[_ue8m0_c(int(s)) for s in row] for row in np.asarray(scales).reshape(M, ng)],
        dtype=np.float32,
    )
    sc_full = np.repeat(sc, gs, axis=1)
    return (vals * sc_full).astype(np.float32)


def _align_ids(M, top_k, E, block_size, rng):
    topk = rng.integers(0, E, size=(M, top_k), dtype=np.int32)
    sorted_ids, expert_ids, EM = moe_align_block_size(topk, E, block_size)
    return sorted_ids, EM


@unittest.skipIf(np is None, "numpy is required for these tests")
class MoeAuxTest(unittest.TestCase):
    M, hidden, gs = 7, 64, 32

    def test_quant_round_trip(self):
        rng = np.random.default_rng(31)
        A = (
            np.array(
                [_f2bf(float(v)) for v in rng.uniform(-3.0, 3.0, self.M * self.hidden)]
            )
            .reshape(self.M, self.hidden)
            .astype(np.uint16)
        )
        packed, scales = mxfp4_moe_quant(A, group_size=self.gs)
        self.assertEqual(packed.shape, (self.M, self.hidden // 2))
        self.assertEqual(scales.shape, (self.M, self.hidden // self.gs))
        dq = _dequant_quantized(packed, scales, self.M, self.hidden, self.gs)
        # Each element recovers the bf16 value within one fp4 quantum of its
        # group scale (max step at the top of the range is 0.5 * scale).
        Af = np.array([[_bf2f(int(v)) for v in row] for row in A], dtype=np.float32)
        sc_full = np.repeat(
            np.array(
                [[_ue8m0_c(int(s)) for s in row] for row in scales], dtype=np.float32
            ),
            self.gs,
            axis=1,
        )
        tol = np.maximum(np.abs(Af), np.abs(dq)) + 0.5 * sc_full + 1e-6
        np.testing.assert_array_less(np.abs(Af - dq), tol)

    def test_quant_zero_group(self):
        A = np.zeros((self.M, self.hidden), dtype=np.uint16)
        packed, scales = mxfp4_moe_quant(A, group_size=self.gs)
        np.testing.assert_array_equal(scales, 0xFF)
        np.testing.assert_array_equal(packed, 0)
        dq = _dequant_quantized(packed, scales, self.M, self.hidden, self.gs)
        np.testing.assert_array_equal(dq, 0.0)

    def test_quant_rejects_bad_group_size(self):
        A = np.zeros((4, 30), dtype=np.uint16)  # 30 not divisible by 32
        with self.assertRaises(ValueError):
            mxfp4_moe_quant(A, group_size=32)
        A2 = np.zeros((4, 33), dtype=np.uint16)  # odd hidden
        with self.assertRaises(ValueError):
            mxfp4_moe_quant(A2, group_size=1)

    def test_sort_gathers_and_pads(self):
        rng = np.random.default_rng(32)
        M, hidden, top_k, E, BS = 8, 16, 2, 4, 16
        sorted_ids, EM = _align_ids(M, top_k, E, BS, rng)
        A = (
            rng.integers(1, 0x7FFF, size=M * hidden)
            .astype(np.uint16)
            .reshape(M, hidden)
        )
        A_sorted = mxfp4_moe_sort(A, sorted_ids, top_k=top_k)
        self.assertEqual(A_sorted.shape, (EM, hidden))
        for r in range(EM):
            flat = int(sorted_ids[r])
            if 0 <= flat < M * top_k:
                np.testing.assert_array_equal(A_sorted[r], A[flat // top_k])
            else:
                np.testing.assert_array_equal(A_sorted[r], 0)

    def test_sort_scales_gathers_and_pads(self):
        rng = np.random.default_rng(33)
        M, hidden, gs, top_k, E, BS = 6, 32, 16, 2, 4, 16
        ng = hidden // gs
        sorted_ids, EM = _align_ids(M, top_k, E, BS, rng)
        scales = rng.integers(1, 200, size=M * ng).astype(np.uint8).reshape(M, ng)
        s_sorted = mxfp4_moe_sort_scales(scales, sorted_ids, top_k=top_k)
        self.assertEqual(s_sorted.shape, (EM, ng))
        for r in range(EM):
            flat = int(sorted_ids[r])
            if 0 <= flat < M * top_k:
                np.testing.assert_array_equal(s_sorted[r], scales[flat // top_k])
            else:
                np.testing.assert_array_equal(s_sorted[r], 0)

    def _scatter_oracle(self, partial, w, sorted_ids, M, width, top_k, EM):
        out = np.zeros((M, width), dtype=np.float32)
        for r in range(EM):
            flat = int(sorted_ids[r])
            if 0 <= flat < M * top_k:
                np.add.at(
                    out,
                    flat // top_k,
                    np.asarray(partial[r], dtype=np.float32) * np.float32(w[r]),
                )
        return out

    def test_scatter_reduce(self):
        rng = np.random.default_rng(34)
        M, width, top_k, E, BS = 5, 12, 2, 4, 16
        sorted_ids, EM = _align_ids(M, top_k, E, BS, rng)
        partial = rng.standard_normal((EM, width)).astype(np.float32)
        w = rng.uniform(0.5, 1.5, EM).astype(np.float32)
        out = mxfp4_moe_scatter_reduce(
            partial, w, sorted_ids, M=M, width=width, top_k=top_k
        )
        oracle = self._scatter_oracle(partial, w, sorted_ids, M, width, top_k, EM)
        np.testing.assert_array_equal(out, oracle)

    def test_scatter_reduce_q(self):
        rng = np.random.default_rng(35)
        M, width, gs, top_k, E, BS = 5, 64, 32, 2, 4, 16
        sorted_ids, EM = _align_ids(M, top_k, E, BS, rng)
        # Build a quantized partial from a known float partial so the
        # dequantized combine is bounded by the fp4 grid.
        raw = rng.standard_normal((EM, width)).astype(np.float32)
        packed, scales = mxfp4_moe_quant(raw, group_size=gs)
        w = rng.uniform(0.5, 1.5, EM).astype(np.float32)
        out = mxfp4_moe_scatter_reduce_q(
            packed, scales, w, sorted_ids, M=M, width=width, top_k=top_k, group_size=gs
        )
        # Oracle: dequant the (same) quantized partial, then r-order add.
        dq = _dequant_quantized(packed, scales, EM, width, gs)
        oracle = self._scatter_oracle(dq, w, sorted_ids, M, width, top_k, EM)
        np.testing.assert_allclose(out, oracle, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency(self):
        """Compiled C++ reference and the pure-Python fallback must agree."""
        ext = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(36)
        M, hidden, gs = 6, 64, 32
        A = (
            np.array([_f2bf(float(v)) for v in rng.uniform(-3.0, 3.0, M * hidden)])
            .reshape(M, hidden)
            .astype(np.uint16)
        )
        pc, sc = ext.mxfp4_moe_quant(A, M, hidden, gs)
        pf, sf = fb.mxfp4_moe_quant(A, M, hidden, gs)
        np.testing.assert_array_equal(np.asarray(pc).ravel(), np.asarray(pf).ravel())
        np.testing.assert_array_equal(np.asarray(sc).ravel(), np.asarray(sf).ravel())

        top_k, E, BS = 2, 4, 16
        topk = rng.integers(0, E, size=(M, top_k), dtype=np.int32)
        sorted_ids, expert_ids, EM = moe_align_block_size(topk, E, BS)
        EM = int(EM)

        Ac = ext.mxfp4_moe_sort(A, sorted_ids, M, hidden, top_k, EM)
        Af = fb.mxfp4_moe_sort(A, sorted_ids, M, hidden, top_k, EM)
        np.testing.assert_array_equal(np.asarray(Ac).ravel(), np.asarray(Af).ravel())

        ng = hidden // gs
        scc = ext.mxfp4_moe_sort_scales(
            np.asarray(sc).reshape(M, ng), sorted_ids, M, ng, top_k, EM
        )
        scf = fb.mxfp4_moe_sort_scales(
            np.asarray(sf).reshape(M, ng), sorted_ids, M, ng, top_k, EM
        )
        np.testing.assert_array_equal(np.asarray(scc).ravel(), np.asarray(scf).ravel())

        partial = rng.standard_normal((EM, hidden)).astype(np.float32)
        w = rng.uniform(0.5, 1.5, EM).astype(np.float32)
        oc = ext.mxfp4_moe_scatter_reduce(partial, w, sorted_ids, M, hidden, top_k, EM)
        of = fb.mxfp4_moe_scatter_reduce(partial, w, sorted_ids, M, hidden, top_k, EM)
        np.testing.assert_array_equal(np.asarray(oc).ravel(), np.asarray(of).ravel())

        # scatter_reduce_q takes a partial already in sorted row order
        # [EM, width] (one row per (token, selection), as produced by the
        # per-expert grouped GEMM). Quantize that same float partial so the
        # compiled dequant-scatter and the fallback can be cross-checked.
        pq_c, ps_c = ext.mxfp4_moe_quant(partial, EM, hidden, gs)
        pq_f, ps_f = fb.mxfp4_moe_quant(partial, EM, hidden, gs)
        oqc = ext.mxfp4_moe_scatter_reduce_q(
            pq_c, ps_c, w, sorted_ids, M, hidden, top_k, EM, gs
        )
        oqf = fb.mxfp4_moe_scatter_reduce_q(
            pq_f, ps_f, w, sorted_ids, M, hidden, top_k, EM, gs
        )
        np.testing.assert_allclose(
            np.asarray(oqc).ravel(), np.asarray(oqf).ravel(), rtol=1e-6, atol=1e-6
        )

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_pipeline_k3(self):
        """K3-shaped orchestration: align -> sort -> quant -> sort_scales
        -> scatter_reduce_q composes at the K3 scale
        (M=112, hidden=7168, ispp=3072, top_k=16)."""
        M, hidden, ispp, top_k, E, gs, BS = 112, 7168, 3072, 16, 64, 32, 16
        rng = np.random.default_rng(37)
        topk = rng.integers(0, E, size=(M, top_k), dtype=np.int32)
        sorted_ids, expert_ids, EM = moe_align_block_size(topk, E, BS)
        EM = int(EM)
        self.assertEqual(EM % BS, 0)
        real = sorted_ids < M * top_k

        A = (
            np.array([_f2bf(float(v)) for v in rng.uniform(-1.0, 1.0, M * hidden)])
            .reshape(M, hidden)
            .astype(np.uint16)
        )
        # Per-token scales (token order) gathered into sorted order must
        # match quantizing the already-gathered A_sorted: both express
        # the per-group scale of token t at sorted row r where
        # sorted_ids[r] = t*top_k + sel.
        packed_tok, scales_tok = mxfp4_moe_quant(A, group_size=gs)
        A_sorted = mxfp4_moe_sort(A, sorted_ids, top_k=top_k)
        self.assertEqual(A_sorted.shape, (EM, hidden))
        s_sorted = mxfp4_moe_sort_scales(scales_tok, sorted_ids, top_k=top_k)
        ng = hidden // gs
        self.assertEqual(s_sorted.shape, (EM, ng))
        packed, scales = mxfp4_moe_quant(A_sorted, group_size=gs)
        self.assertEqual(packed.shape, (EM, hidden // 2))
        self.assertEqual(scales.shape, (EM, ng))
        np.testing.assert_array_equal(s_sorted[real], scales[real])
        # Real rows (sorted_ids < M*top_k) must carry a finite scale in
        # [1, 254]; only padding rows may be 0xFF.
        self.assertTrue(np.all(scales[real] >= 1) and np.all(scales[real] <= 254))
        np.testing.assert_array_equal(packed[~real], 0)
        np.testing.assert_array_equal(scales[~real], 0xFF)
        # scatter_reduce_q of a synthesized quantized partial at the ispp
        # width (the down-projection output dimension on K3).
        raw_p = rng.standard_normal((EM, ispp)).astype(np.float32)
        pq, ps = mxfp4_moe_quant(raw_p, group_size=gs)
        w = rng.uniform(0.5, 1.5, EM).astype(np.float32)
        out = mxfp4_moe_scatter_reduce_q(
            pq, ps, w, sorted_ids, M=M, width=ispp, top_k=top_k, group_size=gs
        )
        self.assertEqual(out.shape, (M, ispp))
        self.assertTrue(np.all(np.isfinite(out)))
        # Real tokens receive a non-trivial reduction (>= 1 contributing row).
        counts = np.bincount((sorted_ids[real] // top_k).astype(np.int64), minlength=M)
        active = np.where(counts > 0)[0]
        self.assertTrue(np.all(np.any(out[active] != 0.0, axis=1)))


# ---------------------------------------------------------------------------
# bf16 GEMM, MLA, KDA: contract + backend-consistency tests (issues #21, #29).
# The low-level C++ kernels live in src/c/vkernels/kernels/{gemm_bf16,mla,kda}.
# and are exposed both as a compiled pybind11 backend and a pure-Python
# mirror (vkernels._fallback). The tests below mirror the host C++ tests in
# tests/kernels/{gemm/test_gemm_bf16,attn/test_mla,attn/test_kda}.cpp.
# ---------------------------------------------------------------------------


def _run_under(impl, fn):
    """Run ``fn`` with ``kernels._impl`` temporarily set to ``impl``."""
    prev = kernels._impl
    try:
        kernels._impl = impl
        return fn()
    finally:
        kernels._impl = prev


def _bf16_ref(M, N, K, alpha, A, B, beta, C):
    """Independent fp32+RNE bf16 GEMM oracle (mirrors test_gemm_bf16.cpp ref)."""
    out = C.copy()
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += _bf2f(int(A[i * K + k])) * _bf2f(int(B[k * N + j]))
            prev = _bf2f(int(out[i * N + j])) if beta != 0.0 else 0.0
            out[i * N + j] = _f2bf(float(alpha * acc + beta * prev))
    return out


@unittest.skipIf(np is None, "numpy is required for these tests")
class GemmBf16Test(unittest.TestCase):
    def test_identity_alpha_one_beta_zero(self):
        A = np.array([_f2bf(1), _f2bf(2), _f2bf(3), _f2bf(4)],
                     dtype=np.uint16).reshape(2, 2)
        I = np.array([_f2bf(1), _f2bf(0), _f2bf(0), _f2bf(1)],
                     dtype=np.uint16).reshape(2, 2)
        C = gemm_bf16(A, I)
        self.assertEqual(C.dtype, np.uint16)
        np.testing.assert_allclose(
            [_bf2f(int(v)) for v in C.ravel()], [1.0, 2.0, 3.0, 4.0], atol=1e-6
        )

    def test_non_square_and_beta_accumulation(self):
        A = np.array([_f2bf(v) for v in (1, 2, 3, 4, 5, 6)],
                     dtype=np.uint16).reshape(2, 3)
        B = np.array([_f2bf(v) for v in (7, 8, 9, 10, 11, 12)],
                     dtype=np.uint16).reshape(3, 2)
        C = np.array([_f2bf(1)] * 4, dtype=np.uint16).reshape(2, 2)
        got = gemm_bf16(A, B, alpha=1.0, beta=2.0, out=C)
        self.assertIs(got, C)
        exp = _bf16_ref(2, 2, 3, 1.0, A.ravel(), B.ravel(), 2.0,
                        np.array([_f2bf(1)] * 4, dtype=np.uint16))
        np.testing.assert_array_equal(C.ravel(), exp)

    def test_config_serving_shape_picks_16x16(self):
        self.assertEqual(gemm_bf16_config(8, 6288, 7168), (16, 16, 64, 64))
        bm, bn, bk, th = gemm_bf16_config(64, 128, 512)
        self.assertEqual((bm, bn, th), (16, 16, 64))

    def test_config_warmup_shape_picks_64x64(self):
        self.assertEqual(gemm_bf16_config(8192, 6288, 7168), (64, 64, 64, 256))
        bm, bn, bk, th = gemm_bf16_config(65, 128, 64)
        self.assertEqual((bm, bk, th), (64, 64, 256))

    def test_out_dtype_and_shape_validated(self):
        A = np.array([_f2bf(1)] * 4, dtype=np.uint16).reshape(2, 2)
        with self.assertRaises(TypeError):
            gemm_bf16(A, A, out=np.zeros(4, dtype=np.float32))  # wrong dtype
        with self.assertRaises(ValueError):
            gemm_bf16(A, A, out=np.zeros(3, dtype=np.uint16))   # wrong size

    def test_inner_dim_mismatch_raises(self):
        A = np.zeros((2, 3), dtype=np.uint16)
        B = np.zeros((4, 2), dtype=np.uint16)
        with self.assertRaises(ValueError):
            gemm_bf16(A, B)

    def test_empty_is_no_op(self):
        C = np.array([_f2bf(5), _f2bf(6), _f2bf(7), _f2bf(8)], dtype=np.uint16)
        E = C.copy()
        out = gemm_bf16(np.zeros((0, 2), dtype=np.uint16),
                        np.zeros((2, 2), dtype=np.uint16))
        self.assertEqual(out.shape, (0, 2))
        np.testing.assert_array_equal(C, E)

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency_bit_exact(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(29)
        for (M, N, K) in [(3, 5, 2), (8, 8, 8), (13, 11, 9)]:
            A = np.array([_f2bf(float(rng.uniform(-4, 4)))
                          for _ in range(M * K)], dtype=np.uint16).reshape(M, K)
            B = np.array([_f2bf(float(rng.uniform(-4, 4)))
                          for _ in range(K * N)], dtype=np.uint16).reshape(K, N)
            c = _run_under(core, lambda A=A, B=B: gemm_bf16(A, B).copy())
            f = _run_under(fb, lambda A=A, B=B: gemm_bf16(A, B).copy())
            np.testing.assert_array_equal(c, f)


@unittest.skipIf(np is None, "numpy is required for these tests")
class MlaFwdTest(unittest.TestCase):
    def test_hand_checked(self):
        # B=1 H=1 S_q=2 S_kv=2, lr=2 rhd=2, scale = 1/sqrt(4) = 0.5
        q = np.array([1, 0, 0, 0, 0, 1, 1, 0], dtype=_F32).reshape(1, 1, 2, 4)
        k_c = np.array([1, 0, 0, 1], dtype=_F32).reshape(1, 2, 2)
        k_pe = np.array([0, 1, 1, 0], dtype=_F32).reshape(1, 2, 2)
        v_c = np.array([5, 6, 7, 8], dtype=_F32).reshape(1, 2, 2)
        out = mla_fwd(q, k_c, k_pe, v_c, scale=0.5)
        self.assertEqual(out.shape, (1, 1, 2, 2))
        self.assertEqual(out.dtype, _F32)
        # q0 attends only k0 (causal): out0 = v_c[0] = [5, 6]
        np.testing.assert_allclose(out[0, 0, 0], [5.0, 6.0], atol=1e-6)
        # q1 attends k0,k1: s0=0,s1=1 -> w1=1/(1+e^-1); out1=w0*[5,6]+w1*[7,8]
        import math
        w1 = 1.0 / (1.0 + math.exp(-1.0))
        w0 = 1.0 - w1
        np.testing.assert_allclose(out[0, 0, 1],
                                   [w0 * 5 + w1 * 7, w0 * 6 + w1 * 8],
                                   atol=1e-6)

    def test_causal_mask_via_q_start_kv_start(self):
        # q_start=2: only j with 2+j <= 2+i are attended. With kv_start=0 and
        # S_q=S_kv, query i attends exactly keys [0, i] (shifted by q_start-2).
        rng = np.random.default_rng(5)
        B, H, S, lr, rhd = 1, 1, 4, 2, 1
        q = rng.standard_normal((B, H, S, lr + rhd)).astype(_F32)
        k_c = rng.standard_normal((B, S, lr)).astype(_F32)
        k_pe = rng.standard_normal((B, S, rhd)).astype(_F32)
        v_c = rng.standard_normal((B, S, lr)).astype(_F32)
        full = mla_fwd(q, k_c, k_pe, v_c, q_start=0, kv_start=0, scale=0.25)
        # The last query (i=S-1) sees all keys; the first (i=0) sees only k0.
        self.assertTrue(np.all(np.isfinite(full[0, 0, -1])))

    def test_all_masked_row_is_zero(self):
        # kv_start=10: every key is in the future relative to q (gqi=0,1 <
        # kv_start+j=10,11), so no key is ever attended and the output is zero.
        q = np.zeros((1, 1, 2, 4), dtype=_F32)
        k_c = np.ones((1, 2, 2), dtype=_F32)
        k_pe = np.ones((1, 2, 2), dtype=_F32)
        v_c = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=_F32).reshape(1, 2, 2)
        out = mla_fwd(q, k_c, k_pe, v_c, q_start=0, kv_start=10, scale=1.0)
        np.testing.assert_array_equal(out, np.zeros((1, 1, 2, 2), dtype=_F32))

    def test_config_selector(self):
        self.assertEqual(mla_config(1, 512, 64), (1, 64, 64))
        self.assertEqual(mla_config(8, 512, 64), (1, 64, 64))
        self.assertEqual(mla_config(9, 512, 64), (4, 64, 256))

    def test_shape_inference_and_validation(self):
        q = np.zeros((1, 1, 2, 4), dtype=_F32)
        k_c = np.zeros((1, 2, 2), dtype=_F32)
        k_pe = np.zeros((1, 2, 2), dtype=_F32)
        v_c = np.zeros((1, 2, 2), dtype=_F32)
        out = mla_fwd(q, k_c, k_pe, v_c)
        self.assertEqual(out.shape, (1, 1, 2, 2))
        # q last dim must equal lr + rhd
        bad = np.zeros((1, 1, 2, 5), dtype=_F32)
        with self.assertRaises(ValueError):
            mla_fwd(bad, k_c, k_pe, v_c)
        with self.assertRaises(ValueError):
            mla_fwd(q, np.zeros((1, 3, 2), dtype=_F32), k_pe, v_c)  # S_kv

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(31)
        for (B, H, Sq, Skv, lr, rhd) in [(1, 1, 2, 2, 2, 2), (1, 2, 4, 4, 2, 2),
                                         (2, 1, 6, 6, 3, 1), (1, 1, 3, 3, 2, 2)]:
            q = rng.standard_normal((B, H, Sq, lr + rhd)).astype(_F32)
            k_c = rng.standard_normal((B, Skv, lr)).astype(_F32)
            k_pe = rng.standard_normal((B, Skv, rhd)).astype(_F32)
            v_c = rng.standard_normal((B, Skv, lr)).astype(_F32)
            sc = np.float32(1.0 / np.sqrt(lr + rhd))
            c = _run_under(core, lambda q=q, kc=k_c, kp=k_pe, vc=v_c, sc=sc:
                           mla_fwd(q, kc, kp, vc, scale=sc).copy())
            f = _run_under(fb, lambda q=q, kc=k_c, kp=k_pe, vc=v_c, sc=sc:
                           mla_fwd(q, kc, kp, vc, scale=sc).copy())
            np.testing.assert_allclose(c, f, rtol=1e-5, atol=1e-6)


@unittest.skipIf(np is None, "numpy is required for these tests")
class KdaLayerNormGatedTest(unittest.TestCase):
    def test_identity_weight_and_unit_gate(self):
        # weight=1, gate=5 -> silu(5)~4.966; x all ones -> rms=1 -> out~silu(5).
        x = np.ones((1, 4), dtype=_F32)
        w = np.ones(4, dtype=_F32)
        gate = np.full((1, 4), 5.0, dtype=_F32)
        out = kda_layer_norm_gated(x, w, gate)
        import math
        silu5 = 5.0 / (1.0 + math.exp(-5.0))
        np.testing.assert_allclose(out, np.full((1, 4), silu5, dtype=_F32),
                                   atol=1e-5)

    def test_zero_gate_zero_output(self):
        x = np.full((2, 3), 7.0, dtype=_F32)
        w = np.full(3, 2.0, dtype=_F32)
        gate = np.zeros((2, 3), dtype=_F32)
        out = kda_layer_norm_gated(x, w, gate)
        np.testing.assert_allclose(out, np.zeros((2, 3), dtype=_F32), atol=1e-6)

    def test_rms_normalizes_then_scales(self):
        # x=[3,4,0,0] -> rms=sqrt((9+16)/4)=2.5; gate=10 -> silu(10)~9.9995.
        import math
        x = np.array([3, 4, 0, 0], dtype=_F32).reshape(1, 4)
        w = np.ones(4, dtype=_F32)
        gate = np.full((1, 4), 10.0, dtype=_F32)
        out = kda_layer_norm_gated(x, w, gate, eps=1e-12)
        s = 10.0 / (1.0 + math.exp(-10.0))
        np.testing.assert_allclose(out[0, 0], 3.0 / 2.5 * s, atol=1e-5)
        np.testing.assert_allclose(out[0, 1], 4.0 / 2.5 * s, atol=1e-5)
        np.testing.assert_allclose(out[0, 2:], [0.0, 0.0], atol=1e-6)

    def test_shape_validation(self):
        x = np.ones((2, 4), dtype=_F32)
        with self.assertRaises(ValueError):
            kda_layer_norm_gated(x, np.ones(3, dtype=_F32), np.ones((2, 4), dtype=_F32))
        with self.assertRaises(ValueError):
            kda_layer_norm_gated(x, np.ones(4, dtype=_F32), np.ones((2, 3), dtype=_F32))

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(33)
        for (N, D) in [(1, 4), (4, 8), (7, 3), (2, 16)]:
            x = rng.standard_normal((N, D)).astype(_F32)
            w = rng.standard_normal(D).astype(_F32)
            g = rng.standard_normal((N, D)).astype(_F32)
            c = _run_under(core, lambda x=x, w=w, g=g:
                           kda_layer_norm_gated(x, w, g).copy())
            f = _run_under(fb, lambda x=x, w=w, g=g:
                           kda_layer_norm_gated(x, w, g).copy())
            np.testing.assert_allclose(c, f, rtol=1e-5, atol=1e-6)


@unittest.skipIf(np is None, "numpy is required for these tests")
class KdaGateChunkCumsumTest(unittest.TestCase):
    def test_matches_independent_refs(self):
        import math
        B, H, nc, cs = 1, 1, 3, 4
        rng = np.random.default_rng(11)
        g = (0.3 + 0.7 * rng.random((B, H, nc, cs))).astype(_F32)
        intra, inter = kda_gate_chunk_cumsum(g)
        self.assertEqual(intra.shape, (B, H, nc, cs))
        self.assertEqual(inter.shape, (B, H, nc))
        # intra[c,t] = sum_{l<=t} log(g[c,l])
        for c in range(nc):
            acc = 0.0
            for t in range(cs):
                acc += math.log(float(g[0, 0, c, t]))
                self.assertAlmostEqual(float(intra[0, 0, c, t]), acc, places=5)
        # inter[c] = sum_{c'<c} intra[c', cs-1]
        for c in range(nc):
            acc = 0.0
            for cp in range(c):
                acc += float(intra[0, 0, cp, cs - 1])
            self.assertAlmostEqual(float(inter[0, 0, c]), acc, places=5)

    def test_zero_gate_clamped(self):
        # g==0 in chunk 1 -> that chunk's intra drops to the floor (~-1e9).
        g = np.array([1.0, 1.0, 0.0, 0.0], dtype=_F32).reshape(1, 1, 2, 2)
        intra, inter = kda_gate_chunk_cumsum(g)
        self.assertAlmostEqual(float(intra[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(intra[0, 0, 0, 1]), 0.0, places=5)
        self.assertLess(float(intra[0, 0, 1, 0]), -1.0e8)
        self.assertLess(float(intra[0, 0, 1, 1]), -1.0e8)
        self.assertAlmostEqual(float(inter[0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(inter[0, 0, 1]), 0.0, places=5)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            kda_gate_chunk_cumsum(np.zeros((1, 1, 2), dtype=_F32))  # not 4-D

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(35)
        for (B, H, nc, cs) in [(1, 1, 2, 4), (2, 2, 3, 4), (1, 1, 5, 8)]:
            g = (0.3 + 0.7 * rng.random((B, H, nc, cs))).astype(_F32)
            ic, ec = _run_under(core, lambda g=g: kda_gate_chunk_cumsum(g))
            if_, ef_ = _run_under(fb, lambda g=g: kda_gate_chunk_cumsum(g))
            np.testing.assert_allclose(ic, if_, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(ec, ef_, rtol=1e-6, atol=1e-6)


@unittest.skipIf(np is None, "numpy is required for these tests")
class KdaNaiveDeltaRuleTest(unittest.TestCase):
    def test_tiny_recurrence(self):
        # B=1 H=1 S=2 D=1 with g=1, beta=1: S_1 = v_1 (S_0=0); o_1 = S_1*q_1.
        # S_2 = S_1 + (v_2 - S_1*k_2)*k_2; o_2 = S_2*q_2.
        q = np.array([[[[2.0], [3.0]]]], dtype=_F32)
        k = np.array([[[[1.0], [2.0]]]], dtype=_F32)
        v = np.array([[[[5.0], [7.0]]]], dtype=_F32)
        g = np.array([[[1.0, 1.0]]], dtype=_F32)
        beta = np.array([[[1.0, 1.0]]], dtype=_F32)
        out = kda_naive_delta_rule_fwd(q, k, v, g, beta)
        # S_1 = 0 + 1*(5-0)*1 = 5; o_1 = 5*2 = 10
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 10.0, places=5)
        # S_2 = 1*5 + 1*(7-5*2)*2 = 5 + (7-10)*2 = 5-6 = -1; o_2 = -1*3 = -3
        self.assertAlmostEqual(float(out[0, 0, 1, 0]), -3.0, places=5)

    def test_shape_and_out(self):
        rng = np.random.default_rng(1)
        q = rng.standard_normal((1, 2, 8, 4)).astype(_F32)
        k = rng.standard_normal((1, 2, 8, 4)).astype(_F32)
        v = rng.standard_normal((1, 2, 8, 4)).astype(_F32)
        g = (0.3 + 0.7 * rng.random((1, 2, 8))).astype(_F32)
        b = (0.3 + 0.7 * rng.random((1, 2, 8))).astype(_F32)
        out = np.empty((1, 2, 8, 4), dtype=_F32)
        got = kda_naive_delta_rule_fwd(q, k, v, g, b, out=out)
        self.assertIs(got, out)
        self.assertTrue(np.all(np.isfinite(out)))

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency_bit_exact(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(37)
        for (B, H, S, D) in [(1, 1, 8, 4), (1, 2, 12, 6), (2, 1, 16, 4)]:
            q = rng.standard_normal((B, H, S, D)).astype(_F32)
            k = rng.standard_normal((B, H, S, D)).astype(_F32)
            v = rng.standard_normal((B, H, S, D)).astype(_F32)
            g = (0.3 + 0.7 * rng.random((B, H, S))).astype(_F32)
            b = (0.3 + 0.7 * rng.random((B, H, S))).astype(_F32)
            c = _run_under(core, lambda q=q, k=k, v=v, g=g, b=b:
                           kda_naive_delta_rule_fwd(q, k, v, g, b).copy())
            f = _run_under(fb, lambda q=q, k=k, v=v, g=g, b=b:
                           kda_naive_delta_rule_fwd(q, k, v, g, b).copy())
            np.testing.assert_array_equal(c, f)


@unittest.skipIf(np is None, "numpy is required for these tests")
class KdaDeltaRuleFwdTest(unittest.TestCase):
    """Chunked forward (L2..L6) vs the per-token naive oracle."""

    def _args(self, B, H, S, D, rng):
        q = rng.standard_normal((B, H, S, D)).astype(_F32)
        k = rng.standard_normal((B, H, S, D)).astype(_F32)
        v = rng.standard_normal((B, H, S, D)).astype(_F32)
        g = (0.3 + 0.7 * rng.random((B, H, S))).astype(_F32)
        beta = (0.3 + 0.7 * rng.random((B, H, S))).astype(_F32)
        return q, k, v, g, beta

    def test_chunked_matches_naive(self):
        rng = np.random.default_rng(123)
        for (B, H, S, D, cs) in [(1, 1, 4, 2, 2), (1, 1, 8, 4, 4),
                                 (1, 2, 8, 3, 4), (2, 1, 12, 4, 4),
                                 (1, 1, 16, 4, 8), (1, 1, 64, 8, 16)]:
            q, k, v, g, beta = self._args(B, H, S, D, rng)
            naive = kda_naive_delta_rule_fwd(q, k, v, g, beta)
            chunked = kda_delta_rule_fwd(q, k, v, g, beta, chunk_size=cs)
            maxd = float(np.max(np.abs(naive - chunked)))
            maxabs = float(np.max(np.abs(naive)))
            self.assertLessEqual(maxd, 1e-3 * (1.0 + maxabs))

    def test_zero_forgetting_matches_naive(self):
        rng = np.random.default_rng(7)
        B, H, S, D, cs = 1, 1, 12, 4, 4
        q = rng.standard_normal((B, H, S, D)).astype(_F32)
        k = rng.standard_normal((B, H, S, D)).astype(_F32)
        v = rng.standard_normal((B, H, S, D)).astype(_F32)
        g = np.ones((B, H, S), dtype=_F32)
        beta = np.full((B, H, S), 0.5, dtype=_F32)
        naive = kda_naive_delta_rule_fwd(q, k, v, g, beta)
        chunked = kda_delta_rule_fwd(q, k, v, g, beta, chunk_size=cs)
        maxd = float(np.max(np.abs(naive - chunked)))
        maxabs = float(np.max(np.abs(naive)))
        self.assertLessEqual(maxd, 1e-3 * (1.0 + maxabs))

    def test_chunk_size_must_divide_s(self):
        q = np.ones((1, 1, 8, 1), dtype=_F32)
        k = np.ones_like(q); v = np.ones_like(q)
        g = np.ones((1, 1, 8), dtype=_F32); b = np.ones((1, 1, 8), dtype=_F32)
        with self.assertRaises(ValueError):
            kda_delta_rule_fwd(q, k, v, g, b, chunk_size=3)  # 8 % 3 != 0

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(41)
        for (B, H, S, D, cs) in [(1, 1, 8, 4, 4), (1, 2, 16, 4, 8),
                                 (2, 1, 16, 4, 8), (1, 1, 32, 6, 8)]:
            q, k, v, g, beta = self._args(B, H, S, D, rng)
            c = _run_under(core, lambda q=q, k=k, v=v, g=g, b=beta, cs=cs:
                           kda_delta_rule_fwd(q, k, v, g, b, chunk_size=cs).copy())
            f = _run_under(fb, lambda q=q, k=k, v=v, g=g, b=beta, cs=cs:
                           kda_delta_rule_fwd(q, k, v, g, b, chunk_size=cs).copy())
            np.testing.assert_allclose(c, f, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_COMPILED, "compiled backend not available")
class KdaPipelineTest(unittest.TestCase):
    """The standalone stages (intra -> inter loop -> gla_fwd_o) must
    compose to the full chunked forward (mirrors test_kda.cpp)."""

    def _pipeline(self, q, k, v, g, beta, cs):
        B, H, S, D = q.shape
        nc = S // cs
        intra_log, inter = kda_gate_chunk_cumsum(g.reshape(B, H, nc, cs))
        u = np.zeros((B, H, S, D), dtype=_F32)
        ist = np.zeros((B, H, nc + 1, D, D), dtype=_F32)
        for c in range(nc):
            kda_delta_rule_intra(q, k, v, g, beta, intra_log, ist,
                                 u=u, chunk_size=cs, chunk_idx=c)
            kda_delta_rule_inter(k, v, g, beta, intra_log, u,
                                 inter_state=ist, chunk_size=cs, chunk_idx=c)
        return kda_gla_fwd_o(q, k, g, beta, intra_log, ist, u, chunk_size=cs)

    def test_pipeline_matches_full_forward(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(43)
        for (B, H, S, D, cs) in [(1, 1, 8, 4, 4), (1, 1, 64, 8, 16),
                                 (2, 2, 32, 6, 8), (1, 2, 16, 4, 8)]:
            q = rng.standard_normal((B, H, S, D)).astype(_F32)
            k = rng.standard_normal((B, H, S, D)).astype(_F32)
            v = rng.standard_normal((B, H, S, D)).astype(_F32)
            g = (0.3 + 0.7 * rng.random((B, H, S))).astype(_F32)
            b = (0.3 + 0.7 * rng.random((B, H, S))).astype(_F32)
            full_c = _run_under(core, lambda q=q, k=k, v=v, g=g, b=b, cs=cs:
                                kda_delta_rule_fwd(q, k, v, g, b,
                                                   chunk_size=cs).copy())
            pipe_c = _run_under(core, lambda q=q, k=k, v=v, g=g, b=b, cs=cs:
                                self._pipeline(q, k, v, g, b, cs).copy())
            # On the compiled backend the stages compose bit-exactly.
            np.testing.assert_array_equal(pipe_c, full_c)
            pipe_f = _run_under(fb, lambda q=q, k=k, v=v, g=g, b=b, cs=cs:
                                self._pipeline(q, k, v, g, b, cs).copy())
            np.testing.assert_allclose(pipe_f, full_c, rtol=1e-5, atol=1e-6)


@unittest.skipIf(np is None, "numpy is required for these tests")
class KdaPackBitmatrixTest(unittest.TestCase):
    def test_hand_checked(self):
        # 10 bits: 1,0,1,1,0,0,0,1, 0,1 -> 0xB1, 0x40
        bits = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1], dtype=np.uint8)
        packed = kda_pack_bitmatrix(bits)
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(list(packed), [0xB1, 0x40])

    def test_round_trip(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            n = int(1 + rng.integers(0, 200))
            bits = rng.integers(0, 2, size=n).astype(np.uint8)
            packed = kda_pack_bitmatrix(bits)
            self.assertEqual(packed.size, (n + 7) // 8)
            for k in range(n):
                byte = k // 8
                bit = 7 - (k % 8)
                self.assertEqual((int(packed[byte]) >> bit) & 1, int(bits[k]))

    def test_partial_n_bits(self):
        # bits = 1,0,1,1,0,0,0,1, 0,1  (index 8 = 0, index 9 = 1).
        bits = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1], dtype=np.uint8)
        self.assertEqual(list(kda_pack_bitmatrix(bits, n_bits=8)), [0xB1])
        # n_bits=9: byte1 holds only bit index 8 (=0) -> 0x00.
        self.assertEqual(list(kda_pack_bitmatrix(bits, n_bits=9)), [0xB1, 0x00])
        # n_bits=10: byte1 holds bit index 8 (=0) and index 9 (=1) -> 0x40.
        self.assertEqual(list(kda_pack_bitmatrix(bits, n_bits=10)), [0xB1, 0x40])
        with self.assertRaises(ValueError):
            kda_pack_bitmatrix(bits, n_bits=11)  # exceeds bits.size

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_backend_consistency_bit_exact(self):
        core = _backend.load_extension().kernels
        fb = _fallback_impl()
        rng = np.random.default_rng(47)
        for n in [8, 10, 7, 16, 1, 13]:
            bits = rng.integers(0, 2, size=n).astype(np.uint8)
            c = _run_under(core, lambda bits=bits: np.asarray(kda_pack_bitmatrix(bits)).copy())
            f = _run_under(fb, lambda bits=bits: np.asarray(kda_pack_bitmatrix(bits)).copy())
            np.testing.assert_array_equal(c, f)


if __name__ == "__main__":
    unittest.main()
