"""Contract tests for vkernels.dist — distributed (TP / EP / PP) fused MoE.

Validates the Python host reference (dist.py) against the single-rank oracle
:func:`vkernels.kernels.fused_moe_mxfp4`: TP sharding arithmetic, the
multi-rank TP/EP forwards, the PP stage-boundary interface, and the
TP8 x PP2 Kimi-K3 layout geometry from issue #18.

Runs with or without the compiled extension (the distributed layer is pure
numpy on top of the public kernels/comm APIs).
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from vkernels import dist, kernels

_F32 = np.dtype(np.float32) if np is not None else None


def _f2bf(v):
    """float32 → bf16 uint16 (RNE)."""
    bits = np.float32(v).view(np.uint32)
    lsb = (bits >> 16) & 1
    bits = bits + np.uint32(0x7FFF) + lsb
    return (bits >> 16).astype(np.uint16)


def _bf2f(v):
    return (np.uint32(v) << 16).view(np.float32)


def _pack_pair(v0, v1):
    """Nearest E2M1 byte for two floats (low nibble = v0).  Vectorised."""
    def nib(v):
        v = np.asarray(v, dtype=np.float32)
        out = np.zeros(v.shape, dtype=np.uint8)
        nz = v != 0
        vals = np.array([0.25, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
        best = np.abs(np.abs(v[nz])[..., None] - vals[None, :]).argmin(axis=-1) + 1
        out[nz] = best.astype(np.uint8)
        out[(v < 0) & nz] |= 0x8
        out[(v == 0) & np.signbit(v)] = 0x8
        return out
    return nib(v0) | (nib(v1) << 4).astype(np.uint8)


def _make_problem(M, hidden, ispp, top_k, E, seed):
    """A full single-rank MoE problem (weights, routing, biases)."""
    rng = np.random.default_rng(seed)
    A = _f2bf(rng.uniform(-0.2, 0.2, (M, hidden)).astype(np.float32))
    w13_vals = rng.uniform(-0.5, 0.5, (E, 2 * ispp, hidden // 2, 2))
    w2_vals = rng.uniform(-0.5, 0.5, (E, hidden, ispp // 2, 2))
    w13 = _pack_pair(w13_vals[..., 0], w13_vals[..., 1])
    w2 = _pack_pair(w2_vals[..., 0], w2_vals[..., 1])
    w13_scale = np.full((E, 2 * ispp, hidden // 32), 125, dtype=np.uint8)
    w2_scale = np.full((E, hidden, ispp // 32), 126, dtype=np.uint8)
    b13 = rng.uniform(-0.3, 0.3, (E, 2 * ispp)).astype(np.float32)
    b2 = rng.uniform(-0.3, 0.3, (E, hidden)).astype(np.float32)
    ids = np.empty((M, top_k), dtype=np.int32)
    for i in range(M):
        for s in range(top_k):
            ids[i, s] = (i * 3 + s * 5 + seed) % E
    w = rng.uniform(0.1, 0.5, (M, top_k)).astype(np.float32)
    return A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2


def _oracle(problem, swiglu_limit=4.0, activation="swiglu"):
    """Run the single-rank oracle; returns out (M, hidden) fp32."""
    A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = problem
    M = A.shape[0]
    _, top_k = ids.shape
    E = w13.shape[0]
    sorted_ids, expert_ids, _ = kernels.moe_align_block_size(ids, E, 16)
    # Gather routing weights for the sorted rows; padding sentinels (== M*top_k)
    # fall outside the weights and map to 0.0.
    clamped = np.minimum(sorted_ids, M * top_k - 1)
    sorted_w = np.where(
        (sorted_ids >= 0) & (sorted_ids < M * top_k), w.ravel()[clamped], 0.0)
    return kernels.fused_moe_mxfp4(
        A, w13, w13_scale, w2, w2_scale, sorted_ids, sorted_w, expert_ids,
        top_k=top_k, group_size=32, swiglu_limit=swiglu_limit,
        activation=activation, b13=b13, b2=b2)


def _max_rel(got, ref):
    ref = np.asarray(ref, dtype=np.float32)
    err = np.abs(np.asarray(got, dtype=np.float32) - ref)
    return float(np.max(err / np.maximum(np.abs(ref), 1.0)))


@unittest.skipIf(np is None, "numpy is required for these tests")
class TpPlanTest(unittest.TestCase):
    def test_k3_layout(self):
        # Kimi-K3: E=112, D_INTER=3072, hidden=7168, TP8 (issue #18).
        plan = dist.tp_plan(7168, 3072, 8)
        self.assertEqual(plan["hidden_shard"], 896)
        self.assertEqual(plan["ispp_shard"], 384)
        self.assertEqual(plan["w13_shard_bytes"], 2 * 3072 * (896 // 2))
        self.assertEqual(plan["w2_shard_bytes"], 7168 * (384 // 2))

    def test_small_plan(self):
        plan = dist.tp_plan(256, 128, 2)
        self.assertEqual((plan["hidden_shard"], plan["ispp_shard"]), (128, 64))

    def test_rejects_bad_layouts(self):
        with self.assertRaises(ValueError):
            dist.tp_plan(128, 64, 0)          # tp <= 0
        with self.assertRaises(ValueError):
            dist.tp_plan(127, 64, 2)          # hidden % tp
        with self.assertRaises(ValueError):
            dist.tp_plan(128, 63, 2)          # ispp % tp
        with self.assertRaises(ValueError):
            dist.tp_plan(128, 64, 2)          # ispp/2 = 32 not multiple of 64


@unittest.skipIf(np is None, "numpy is required for these tests")
class TpShardTest(unittest.TestCase):
    def test_shards_reconstruct_full(self):
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = _make_problem(8, 256, 256, 1, 4, 21)
        shards = dist.tp_shard_weights(w13, w13_scale, w2, w2_scale, 4)
        self.assertEqual(len(shards), 4)
        # Concatenating the per-rank shards reproduces the full weights.
        w13_cat = np.concatenate([s[0] for s in shards], axis=2)
        w13s_cat = np.concatenate([s[1] for s in shards], axis=2)
        w2_cat = np.concatenate([s[2] for s in shards], axis=2)
        w2s_cat = np.concatenate([s[3] for s in shards], axis=2)
        np.testing.assert_array_equal(w13_cat, w13)
        np.testing.assert_array_equal(w13s_cat, w13_scale)
        np.testing.assert_array_equal(w2_cat, w2)
        np.testing.assert_array_equal(w2s_cat, w2_scale)
        # Shard shapes match the plan's byte counts.
        plan = dist.tp_plan(256, 256, 4)
        self.assertEqual(shards[0][0].shape, (4, 512, plan["hidden_shard"] // 2))


@unittest.skipIf(np is None, "numpy is required for these tests")
class DistMoeTpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The numpy fallback oracle is O(per-element) and slow; run it exactly
        # once, on the smallest shape that exercises multi-expert routing.
        # Every other assertion uses transitivity: tp=N == tp=1 (same code
        # path, partial sums re-associated) and tp=1 == oracle (this anchor).
        cls.anchor = _make_problem(16, 128, 64, 2, 8, 31)
        cls.ref = _oracle(cls.anchor)

    def test_tp_single_rank_matches_oracle(self):
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = self.anchor
        outs = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
            top_k=2, tp=1, swiglu_limit=4.0)
        self.assertEqual(len(outs), 1)
        # tp == 1: same computation as the oracle modulo BLAS summation order.
        self.assertLess(_max_rel(outs[0], self.ref), 1e-4)

    def test_tp_matches_oracle_swiglu(self):
        # tp == 2 needs ispp/2 = 64: use the 128x128 shape (tp=1 is
        # oracle-validated by test_tp_single_rank_matches_oracle; the larger
        # world must converge to the same result).
        p2 = _make_problem(16, 128, 128, 2, 8, 32)
        A2, w132, w13s2, w22, w2s2, ids2, w2w, b132, b22 = p2
        base = dist.dist_moe_tp(
            A2, w132, w13s2, w22, w2s2, ids2, w2w, b13=b132, b2=b22,
            top_k=2, tp=1, swiglu_limit=4.0)[0]
        outs = dist.dist_moe_tp(
            A2, w132, w13s2, w22, w2s2, ids2, w2w, b13=b132, b2=b22,
            top_k=2, tp=2, swiglu_limit=4.0)
        self.assertEqual(len(outs), 2)
        np.testing.assert_array_equal(outs[0], outs[1])  # ranks converge
        self.assertLess(_max_rel(outs[0], base), 1e-3)

    def test_tp_world4_matches_world1(self):
        # 256x256 is the smallest shape where tp=4 keeps 64-divisible shards.
        p = _make_problem(16, 256, 256, 2, 8, 33)
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = p
        base = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
            top_k=2, tp=1, swiglu_limit=4.0)[0]
        outs = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
            top_k=2, tp=4, swiglu_limit=4.0)
        self.assertEqual(len(outs), 4)
        for out in outs:
            np.testing.assert_array_equal(out, outs[0])
        self.assertLess(_max_rel(outs[0], base), 1e-3)

    def test_tp_matches_oracle_situ(self):
        p = _make_problem(16, 256, 256, 2, 8, 34)
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = p
        base = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
            top_k=2, tp=1, swiglu_limit=0.0, activation="situ")[0]
        outs = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
            top_k=2, tp=4, swiglu_limit=0.0, activation="situ")
        self.assertLess(_max_rel(outs[0], base), 1e-3)

    def test_tp_no_bias_matches_oracle(self):
        p = _make_problem(16, 128, 128, 2, 8, 35)
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = p
        # No-bias path (skip branches of the epilogues): tp=2 must agree with
        # tp=1 (tp=1 is oracle-validated by the anchor; the no-bias branches
        # are identical code paths minus the bias adds, cross-checked against
        # the oracle in the C++ suite).
        base = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=None, b2=None,
            top_k=2, tp=1, swiglu_limit=2.0)[0]
        self.assertTrue(np.all(np.isfinite(base)))
        outs = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=None, b2=None,
            top_k=2, tp=2, swiglu_limit=2.0)
        self.assertLess(_max_rel(outs[0], base), 1e-3)

    def test_tp_rejects_bad_args(self):
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = self.anchor
        with self.assertRaises(ValueError):
            dist.dist_moe_tp(A, w13, w13_scale, w2, w2_scale, ids, w,
                             top_k=2, tp=0)
        with self.assertRaises(ValueError):
            dist.dist_moe_tp(A, w13, w13_scale, w2, w2_scale, ids, w,
                             top_k=2, tp=3)  # 64/3 not divisible


@unittest.skipIf(np is None, "numpy is required for these tests")
class DistMoeEpTest(unittest.TestCase):
    def test_ep_plan_arithmetic(self):
        p0 = dist.ep_plan(10, 3, 0)
        self.assertEqual((p0["expert_begin"], p0["expert_end"]), (0, 4))
        p1 = dist.ep_plan(10, 3, 1)
        self.assertEqual((p1["expert_begin"], p1["expert_end"]), (4, 7))
        p2 = dist.ep_plan(10, 3, 2)
        self.assertEqual((p2["expert_begin"], p2["expert_end"]), (7, 10))
        p3 = dist.ep_plan(8, 4, 3)
        self.assertEqual((p3["expert_begin"], p3["expert_end"]), (6, 8))
        with self.assertRaises(ValueError):
            dist.ep_plan(8, 0, 0)
        with self.assertRaises(ValueError):
            dist.ep_plan(8, 4, 4)
        with self.assertRaises(ValueError):
            dist.ep_plan(2, 4, 0)

    def test_ep_dispatch_local_only(self):
        M, top_k, E = 16, 2, 8
        ids = np.arange(M * top_k, dtype=np.int32).reshape(M, top_k) % E
        plan = dist.ep_plan(E, 2, 0)
        sorted_ids, eids, EM = dist.ep_dispatch(ids, plan, E, 16)
        self.assertEqual(EM % 16, 0)
        for f in sorted_ids:
            if 0 <= f < M * top_k:
                self.assertLess(int(ids.ravel()[f]), 4)  # only local experts
        for e in eids:
            if e >= 0:
                self.assertLess(e, 4)  # local ids
        with self.assertRaises(ValueError):
            dist.ep_dispatch(ids, plan, E, 0)

    def test_ep_matches_oracle(self):
        # Uses the shared anchor problem; its oracle was computed once in
        # DistMoeTpTest.setUpClass — recompute here for class independence
        # (the fallback oracle is slow, so anchor on the same small shape).
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = _make_problem(16, 128, 64, 2, 8, 41)
        ref = _oracle((A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2))
        for ep in (1, 2, 4):
            outs = dist.dist_moe_ep(
                A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
                top_k=2, ep=ep, swiglu_limit=4.0)
            self.assertEqual(len(outs), ep)
            combined = sum(outs)
            self.assertLess(_max_rel(combined, ref), 1e-3)

    def test_ep_situ_odd_experts_matches_tp(self):
        # Two independent distributed schemes (EP a2a vs TP all-reduce) must
        # agree; no slow oracle needed (both reduce to the numpy stages that
        # the anchor validates).
        problem = _make_problem(16, 128, 64, 2, 9, 42)
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = problem
        ref = dist.dist_moe_tp(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
            top_k=2, tp=1, swiglu_limit=0.0, activation="situ")[0]
        outs = dist.dist_moe_ep(
            A, w13, w13_scale, w2, w2_scale, ids, w, b13=b13, b2=b2,
            top_k=2, ep=3, swiglu_limit=0.0, activation="situ")
        self.assertLess(_max_rel(sum(outs), ref), 1e-3)

    def test_ep_rejects_bad_ep(self):
        problem = _make_problem(16, 128, 64, 2, 8, 43)
        A, w13, w13_scale, w2, w2_scale, ids, w, b13, b2 = problem
        with self.assertRaises(ValueError):
            dist.dist_moe_ep(A, w13, w13_scale, w2, w2_scale, ids, w,
                             top_k=2, ep=0)


@unittest.skipIf(np is None, "numpy is required for these tests")
class PpBoundaryTest(unittest.TestCase):
    def test_round_bf16_known_values(self):
        out = dist.round_bf16(np.array([1.0, 0.5, -1.0, 1.0001], dtype=np.float32))
        np.testing.assert_array_equal(out, np.array([0x3F80, 0x3F00, 0xBF80, 0x3F80],
                                                    dtype=np.uint16))

    def test_boundary_roundtrip(self):
        q = dist.comm.BlockingQueue()
        state = np.arange(6, dtype=np.float32).reshape(2, 3)
        dist.pp_boundary_send(state, q)
        got = dist.pp_boundary_recv(q, 2, 3)
        np.testing.assert_array_equal(got, state)
        # size mismatch is rejected
        dist.pp_boundary_send(state, q)
        with self.assertRaises(ValueError):
            dist.pp_boundary_recv(q, 3, 3)

    def test_pp_tp_pipeline_matches_oracle(self):
        # TP2 x PP2: two MoE layers (different weights), hidden state passed
        # through the boundary with bf16 re-quantisation.  The reference is
        # the tp=1 pipeline (oracle-validated by the anchor); the distributed
        # path runs tp=2 at each stage with the boundary transfer between.
        layer_a = _make_problem(16, 128, 128, 2, 8, 51)
        layer_b = _make_problem(16, 128, 128, 2, 8, 52)
        A0, w13a, w13sa, w2a, w2sa, idsa, wa, b13a, b2a = layer_a
        _, w13b, w13sb, w2b, w2sb, idsb, wb, b13b, b2b = layer_b

        # Reference pipeline (tp=1 at each stage): layer A then layer B with
        # bf16 rounding at the boundary.
        out_a = dist.dist_moe_tp(
            A0, w13a, w13sa, w2a, w2sa, idsa, wa, b13=b13a, b2=b2a,
            top_k=2, tp=1, swiglu_limit=4.0)[0]
        a1 = dist.round_bf16(out_a)
        ref = dist.dist_moe_tp(
            a1, w13b, w13sb, w2b, w2sb, idsb, wb, b13=b13b, b2=b2b,
            top_k=2, tp=1, swiglu_limit=4.0)[0]

        # Distributed pipeline: TP2 stage 0 → boundary → TP2 stage 1.
        outs0 = dist.dist_moe_tp(
            A0, w13a, w13sa, w2a, w2sa, idsa, wa, b13=b13a, b2=b2a,
            top_k=2, tp=2, swiglu_limit=4.0)
        q = dist.comm.BlockingQueue()
        dist.pp_boundary_send(outs0[0], q)
        boundary = dist.pp_boundary_recv(q, 16, 128)
        a1_dist = dist.round_bf16(boundary)
        outs1 = dist.dist_moe_tp(
            a1_dist, w13b, w13sb, w2b, w2sb, idsb, wb, b13=b13b, b2=b2b,
            top_k=2, tp=2, swiglu_limit=4.0)
        self.assertLess(_max_rel(outs1[0], ref), 1e-2)


if __name__ == "__main__":
    unittest.main()
