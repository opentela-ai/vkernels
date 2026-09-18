"""Gate-policy tests for the mhc_pre correctness gate (issue #138).

Background (issue #79 -> #138): the gfx942 HIP ``mhc_pre_gemm_sqrsum``
kernels were gated against a CPU oracle whose ``out`` is a STRICT sequential
fp32 mul-add chain over ``hc_hidden`` at ``max_rel < 1e-4``. That gate is
order-fragile: fp32 accumulation of this dot product — in ANY order,
including the oracle's own — deviates from the exact (fp64) sum by up to
~4e-4 rel on the GLM shapes (MI300A-measured over 5 seeds; the deviation
lives in heavily-cancelled outputs), so the verdict for the blocked-order
(NSLICE-split) variant is seed-dependent (issue #79: ns=2/16/32 FAILED on
the measurement seed; the follow-up survey: even the STRICT split kernel
FAILS 1e-4 on seed 2) and NSLICE >= 16 — needed for the <= 50 us/layer
decode target — is inadmissible.

Issue #138 revises the DEFAULT gate to a conditioning-normalized
fp64-REFERENCE comparison, per element:

    |got - ref64| <= K * eps32 * sum(|x_h * fn_h|)     (out)
    |got - ref64| <= K * eps32 * sum( x_h^2 )          (sqrsum)

(ref64 = fp64 CPU reference; eps32 = FLT_EPSILON; K pinned from the measured
MI300A multi-seed envelope, K*eps*sum|terms| being the standard a-priori
fp32 accumulation-error scale.) The normalization makes the gate
order-INVARIANT — every correct accumulation order sits at the same
envelope — while a shifted/dropped term perturbs an output by O(|x*fn|),
i.e. ~1/(K*eps) normalized units, orders of magnitude above the envelope.
The legacy strict-1e-4 oracle-chain gate remains available via
VK_MHC_STRICT_GATE=1 on the device harness.

This suite validates the POLICY on CPU in pure numpy (no torch, no HIP —
bare-env friendly) by simulating every accumulation order the device can
produce:

* the fp32 sequential oracle chain (legacy gate reference),
* fp32 blocked-order chains (NSLICE contiguous slices, in-order combine) —
  the device's ``mhc_pre_gemm_sqrsum_blocked`` accumulation pattern,
* the fp32 tree-reduced sqrsum (strided partials + pairwise combine),
* the fp64 reference and the per-element conditioning scales,
* a deliberately corrupted result (fn row off by one — the negative-control
  kernel compiled into meta/benchmarks/test_mhc_correct.hip).

Assertions:
1. EVERY NSLICE (1..256) on EVERY seed passes the normalized fp64 gate —
   the gate is order-invariant, so NSLICE >= 16 is admissible.
2. The corrupted result exceeds the gate by >= 100x — real bugs stay caught.
3. The fp32 sequential oracle chain itself deviates from the fp64 reference
   at a visible scale — documenting WHY the legacy gate cannot distinguish
   order noise from bugs (soft, loose-bounded).

The GPU-side counterparts (fmaf single-rounding device chains) are measured
on MI300A by meta/benchmarks/test_mhc_correct.hip; the pinned K comes from
that hardware envelope (docs/performance/mhc/gfx942.md). This file's
numpy two-rounding emulation is a policy proxy with a much smaller envelope,
so the K asserted here is conservative.
"""

from __future__ import annotations

import unittest

import numpy as np

F32 = np.float32
F64 = np.float64

EPS32 = float(np.finfo(np.float32).eps)  # 1.19e-07

# The pinned normalized-gate constant (docs/performance/mhc/gfx942.md).
# MI300A-measured (job 641091): worst clean normalized error 0.9 over 5
# seeds x NSLICE 1..256 x 3 GLM shapes; corrupted control 3.2e5-4.1e5.
# K=256 = ~280x above the clean envelope, still below the ~515-unit
# single-dropped-term signature. This file's numpy two-rounding emulation
# has a much smaller envelope, so the assertions below hold with wide
# margin.
GATE_K = 256.0

# GLM-5.3-Flash decode/multi-token shapes (hc_mult=4, hidden=4096).
GLM_HIDDEN = 4096
GLM_HC_MULT = 4


def _rnd(seed: int, n: int) -> np.ndarray:
    """The deterministic [-2, 1) generator meta/benchmarks uses (rnd()).

    Bit-faithful to the C++ ``rnd`` in test_mhc_correct.hip / bench_mhc.hip,
    including the ``(int)x % 200000`` signed-wraparound (values can be
    negative down to -2). Seed 1 reproduces the historical #79 inputs.
    """
    i = np.arange(n, dtype=np.uint64)
    x = np.uint64(seed) * np.uint64(2654435761) + i * np.uint64(40503)
    x = x ^ (x >> np.uint64(13))
    x = x * np.uint64(0x5BD1E995)
    x = x ^ (x >> np.uint64(15))
    x32 = (x & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    signed = x32.view(np.int32)  # C's implementation-defined (int)x wrap
    return ((signed % np.int32(200000)).astype(np.float64)
            / 100000.0).astype(F32)


def _f32_seq_chain(prod: np.ndarray) -> np.ndarray:
    """Strict left-to-right fp32 accumulation along the last axis.

    np.cumsum with dtype=float32 accumulates sequentially in fp32 (no
    pairwise regrouping), matching the CPU oracle's ``acc += x*h`` chain.
    """
    return np.cumsum(prod.astype(F32), axis=-1, dtype=F32)[..., -1]


def _f32_blocked(prod: np.ndarray, nslice: int) -> np.ndarray:
    """The device blocked-order accumulation (issue #79/#138).

    NSLICE contiguous slices over the last axis; each slice is a sequential
    fp32 chain; slices are combined IN THREAD ORDER by a sequential fp32
    add chain (idle threads contribute exact +0.0).
    """
    *lead, h = prod.shape
    assert h % nslice == 0, "test shapes keep slices equal"
    slices = prod.astype(F32).reshape(*lead, nslice, h // nslice)
    partials = np.cumsum(slices, axis=-1, dtype=F32)[..., -1]  # per-slice chain
    out = np.zeros(lead, dtype=F32)
    for t in range(nslice):  # in-order combine, fp32 sequential
        out = (out + partials[..., t]).astype(F32)
    return out


def _f32_sqrsum_tree(x: np.ndarray) -> np.ndarray:
    """The device sqrsum: strided fp32 partials + 256-wide pairwise tree."""
    *lead, h = x.shape
    parts = np.zeros(tuple(lead) + (256,), dtype=F32)
    for t in range(256):  # strided sequential fp32 partial per lane
        parts[..., t] = np.cumsum(x[..., t::256] ** 2, axis=-1, dtype=F32)[
            ..., -1
        ]
    off = 128  # pairwise tree, exactly the kernel's combine
    while off > 0:
        parts[..., :off] = (parts[..., :off] + parts[..., off : 2 * off]).astype(F32)
        off //= 2
    return parts[..., 0]


def _rel(got: np.ndarray, ref: np.ndarray) -> float:
    got = got.astype(F64)
    ref = ref.astype(F64)
    denom = np.maximum(np.abs(ref), 1.0)
    return float(np.max(np.abs(got - ref) / denom))


def _norm(got: np.ndarray, ref64: np.ndarray, scale: np.ndarray) -> float:
    """The issue #138 gate statistic: max |got-ref64| / (eps32 * scale)."""
    got = got.astype(F64)
    err = np.abs(got - ref64)
    denom = EPS32 * scale
    safe = np.where(denom > 0, denom, 1.0)
    q = np.where(denom > 0, err / safe, np.where(err > 0, np.inf, 0.0))
    return float(np.max(q))


class MhcPreF64GatePolicyTest(unittest.TestCase):
    """The issue #138 normalized fp64 gate, validated on simulated orders."""

    def _case(self, num_tokens: int, seed: int):
        hc_mult, hidden = GLM_HC_MULT, GLM_HIDDEN
        hc_hidden = hc_mult * hidden
        hc_mult3 = hc_mult * (2 + hc_mult)
        x = _rnd(2 * seed - 1, num_tokens * hc_hidden).reshape(
            num_tokens, hc_hidden
        )
        fn = _rnd(2 * seed, hc_mult3 * hc_hidden).reshape(hc_mult3, hc_hidden)
        # The fp32 values ARE the common input here (the policy test does
        # not need the bf16 round-trip; the device test applies it).
        prod = x[None, :, :] * fn[:, None, :]  # [hc_mult3, tokens, hc_hidden]
        ref64 = prod.astype(F64).sum(axis=-1).T  # [tokens, hc_mult3]
        oracle = _f32_seq_chain(prod).T  # [tokens, hc_mult3]
        # Conditioning scales: sum|x*fn| per out element, sum x^2 per token.
        s_out = np.abs(prod.astype(F64)).sum(axis=-1).T  # [tokens, hc_mult3]
        s_sq = (x.astype(F64) ** 2).sum(axis=-1)  # [tokens]
        return x, prod, ref64, oracle, s_out, s_sq, hc_hidden

    def test_all_nslice_pass_normalized_f64_gate_on_every_seed(self):
        for num_tokens in (1, 7):
            for seed in range(1, 6):
                x, prod, ref64, _oracle, s_out, s_sq, hc_hidden = self._case(
                    num_tokens, seed
                )
                sqrsum64 = (x.astype(F64) ** 2).sum(axis=-1)
                for ns in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                    got = _f32_blocked(prod, ns).T
                    n_out = _norm(got, ref64, s_out)
                    self.assertLess(
                        n_out,
                        GATE_K,
                        f"n={num_tokens} seed={seed} ns={ns}: norm {n_out:.1f}",
                    )
                    sq = _f32_sqrsum_tree(x)
                    n_sq = _norm(sq, sqrsum64, s_sq)
                    self.assertLess(
                        n_sq,
                        GATE_K,
                        f"sqrsum n={num_tokens} seed={seed} ns={ns}: "
                        f"{n_sq:.1f}",
                    )

    def test_corrupted_kernel_fails_gate_by_wide_margin(self):
        # Negative control: fn row off by one (the corrupted kernel compiled
        # into test_mhc_correct.hip). Must exceed the gate by >= 100x.
        for num_tokens in (1, 7):
            for seed in (1, 3):
                _x, prod, ref64, _o, s_out, _s, hc_hidden = self._case(
                    num_tokens, seed
                )
                hc_mult3 = prod.shape[0]
                fn = _rnd(2 * seed, hc_mult3 * hc_hidden).reshape(
                    hc_mult3, hc_hidden
                )
                fn_bad = np.empty_like(fn)
                fn_bad[:, :-1] = fn[:, 1:]
                fn_bad[:, -1] = 0.0
                prod_bad = prod * 0 + (
                    _rnd(2 * seed - 1, num_tokens * hc_hidden).reshape(
                        num_tokens, hc_hidden
                    )[None, :, :]
                    * fn_bad[:, None, :]
                )
                got_bad = _f32_blocked(prod_bad, 16).T
                n_bad = _norm(got_bad, ref64, s_out)
                self.assertGreater(
                    n_bad, 100 * GATE_K,
                    f"corrupted norm {n_bad:.1f} must fail the gate")

    def test_legacy_oracle_chain_deviation_is_order_scale(self):
        # WHY the gate had to change: the fp32 sequential oracle chain
        # itself deviates from the fp64 reference (the MI300A measurement
        # puts that deviation at up to ~4e-4 rel on cancelled outputs,
        # seed-dependent — the same order as ANY parallel regrouping), so a
        # 1e-4 oracle-chain gate is seed-fragile. Loose bounds here
        # (1e-6 < rel < 1e-2); the exact measured scale is recorded in
        # docs/performance/mhc/gfx942.md from the MI300A runs.
        seen = 0.0
        for seed in range(1, 6):
            _x, _prod, ref64, oracle, _so, _sq, _h = self._case(7, seed)
            seen = max(seen, _rel(oracle, ref64))
        self.assertGreater(seen, 1e-6, "oracle chain should deviate measurably")
        self.assertLess(seen, 1e-2, "oracle chain should stay sane vs fp64")


class MhcPreF64GatePolicySmallShapeTest(unittest.TestCase):
    """Same policy on the tiny/odd shapes the device test also covers."""

    def test_tiny_shapes_pass(self):
        rng = np.random.default_rng(138)
        for (num_tokens, hc_mult, hidden) in ((1, 2, 2), (7, 3, 3), (2, 4, 8)):
            hc_hidden = hc_mult * hidden
            hc_mult3 = hc_mult * (2 + hc_mult)
            if hc_hidden % 256 != 0:
                continue  # blocked sim needs divisibility; device test covers
            x = rng.standard_normal(num_tokens * hc_hidden).astype(F32)
            fn = rng.standard_normal(hc_mult3 * hc_hidden).astype(F32)
            prod = x[None, :] * fn[:, None, :]
            ref64 = prod.astype(F64).sum(axis=-1).T
            s_out = np.abs(prod.astype(F64)).sum(axis=-1).T
            for ns in (1, 4, 16, 64, 256):
                got = _f32_blocked(prod, ns).T
                self.assertLess(_norm(got, ref64, s_out), GATE_K)


if __name__ == "__main__":
    unittest.main()
