"""Tests for the numerics layer (tolerances, comparison, verify, bisect, CLI).

Everything here runs on host CI without a GPU: verification compares the
active backend against the pure-Python fallback oracle, which is exactly
what host CI does for the kernel contract tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels import numerics
from vkernels.numerics import bisect as bisect_mod
from vkernels.numerics import cli as cli_mod
from vkernels.numerics.tolerance import tolerance_for

_F32 = np.dtype(np.float32)


# --- tolerance policy -------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("float32", (0.0, 0.0)),  # documented bit-identical oracle pairs
        ("float64", (1e-12, 1e-12)),
        ("float16", (1e-3, 1e-3)),
        ("bfloat16", (2e-2, 2e-2)),
        ("int32", (0.0, 0.0)),
        ("bool", (0.0, 0.0)),
    ],
)
def test_tolerance_for(dtype, expected):
    assert tolerance_for(dtype) == expected


def test_tolerance_for_numpy_dtypes_and_exact_override():
    assert tolerance_for(np.float32) == (0.0, 0.0)
    assert tolerance_for("float32", exact=True) == (0.0, 0.0)
    assert tolerance_for("bfloat16", exact=True) == (0.0, 0.0)
    assert tolerance_for("unknown_dtype") == (1e-5, 1e-5)  # conservative default


# --- comparison utilities (DeepGEMM ports) -----------------------------------


def test_calc_diff_identical_and_orthogonal_and_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256).astype(_F32)
    assert numerics.calc_diff(x, x) == 0.0
    # orthogonal inputs: similarity ~ 0 -> diff ~ 1
    assert numerics.calc_diff(np.ones(4), np.array([1, -1, 1, -1], dtype=_F32)) > 0.9
    assert numerics.calc_diff(np.zeros(4), np.zeros(4)) == 0.0


def test_count_bytes_nested():
    a = np.zeros(8, dtype=_F32)
    b = np.zeros((4, 4), dtype=np.float64)
    assert numerics.count_bytes(a, (b, [a, None])) == 8 * 4 + 4 * 4 * 8 + 8 * 4


def test_assert_bitwise_equal_reports_first_mismatch():
    x = np.array([1.0, 2.0, 3.0], dtype=_F32)
    y = x.copy()
    numerics.assert_bitwise_equal(x, y)
    y[2] = 4.0
    with pytest.raises(AssertionError, match=r"coord=\(2,\).*x_val=3\.0.*y_val=4\.0"):
        numerics.assert_bitwise_equal(x, y, label="gemm")


def test_assert_close_nan_aware():
    numerics.assert_close(np.array([np.nan]), np.array([np.nan]))
    with pytest.raises(AssertionError, match="nan|actual"):
        numerics.assert_close(np.array([np.nan]), np.array([1.0]))


def test_assert_close_names_worst_element():
    actual = np.zeros((3, 3), dtype=_F32)
    expected = np.zeros((3, 3), dtype=_F32)
    expected[1, 2] = 5.0
    with pytest.raises(AssertionError, match=r"\(1, 2\)"):
        numerics.assert_close(actual, expected, label="scale")


def test_mismatch_stats_shape_mismatch():
    stats = numerics.mismatch_stats(np.zeros(3), np.zeros(4))
    assert stats["ok"] is False
    assert "shape mismatch" in stats["reason"]


# --- verify: kernel vs oracle ------------------------------------------------


@pytest.mark.parametrize("op", numerics.available_ops())
def test_verify_all_ops_pass(op):
    result = numerics.verify(op, 64, seed=3)
    assert result.ok, result.message
    assert result.stats["calc_diff"] == 0.0


def test_verify_detects_a_broken_kernel(monkeypatch):
    from vkernels import kernels

    original = kernels.add

    def broken_add(a, b, out=None):
        result = original(a, b, out=out)
        if result.size > 8:
            result = result.astype(np.float64).copy()
            result[3] += 1.0
            result = result.astype(_F32)
        return result

    monkeypatch.setattr(kernels, "add", broken_add)
    result = numerics.verify("add", 64)
    assert not result.ok
    assert "at (3,)" in result.message or "(3,)" in result.message
    assert result.stats["max_abs"] == pytest.approx(1.0)
    assert result.stats["first_mismatch"]["coords"] == (3,)


def test_verify_respects_seed():
    args1, _ = numerics.make_case("add", 32, seed=1)
    args2, _ = numerics.make_case("add", 32, seed=1)
    args3, _ = numerics.make_case("add", 32, seed=2)
    assert np.array_equal(args1[0], args2[0])
    assert not np.array_equal(args1[0], args3[0])


def test_make_case_unknown_op_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        numerics.make_case("nope", 8)


# --- bisect -------------------------------------------------------------------


def test_bisect_finds_smallest_failing_size(monkeypatch):
    def fake_verify(op, size, *, seed=0, **extra):
        return numerics.VerificationResult(
            op, size, "fake", ok=size <= 100, exact=True
        )

    monkeypatch.setattr(bisect_mod, "verify", fake_verify)
    report = numerics.bisect_size("add", 1, 1000)
    assert report.smallest_failing == 101
    assert all(t.ok for t in report.trials if t.size <= 100)
    assert any(not t.ok for t in report.trials if t.size > 100)
    assert "smallest failing size=101" in str(report)


def test_bisect_no_failure_short_circuits(monkeypatch):
    def fake_verify(op, size, *, seed=0, **extra):
        return numerics.VerificationResult(op, size, "fake", ok=True, exact=True)

    monkeypatch.setattr(bisect_mod, "verify", fake_verify)
    report = numerics.bisect_size("add", 1, 64)
    assert report.smallest_failing is None
    assert len(report.trials) == 1  # apex only
    assert "no failure" in str(report)


@pytest.mark.parametrize("lo,hi", [(0, 10), (10, 10), (5, 2)])
def test_bisect_rejects_bad_bounds(lo, hi):
    with pytest.raises(ValueError):
        numerics.bisect_size("add", lo, hi)


# --- CLI ----------------------------------------------------------------------


def test_cli_ops(capsys):
    assert cli_mod.main(["ops"]) == 0
    out = capsys.readouterr().out.split()
    assert set(out) == set(numerics.available_ops())


def test_cli_verify_pass_and_fail(capsys, monkeypatch):
    assert cli_mod.main(["verify", "add", "--size", "32"]) == 0
    assert "ok" in capsys.readouterr().out

    def fake_verify(op, size, *, seed=0, exact=None, **extra):
        return numerics.VerificationResult(op, size, "fake", ok=False, exact=bool(exact))

    monkeypatch.setattr(cli_mod, "verify", fake_verify)
    assert cli_mod.main(["verify", "add", "--size", "32"]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_cli_bisect(capsys, monkeypatch):
    def fake_verify(op, size, *, seed=0, exact=None, **extra):
        return numerics.VerificationResult(
            op, size, "fake", ok=size <= 4, exact=bool(exact)
        )

    # cli calls bisect_size, which resolves verify in its own module.
    monkeypatch.setattr(cli_mod, "verify", fake_verify)
    monkeypatch.setattr(bisect_mod, "verify", fake_verify)
    assert cli_mod.main(["bisect", "add", "--lo", "1", "--hi", "64"]) == 1
    out = capsys.readouterr().out
    assert "smallest failing size=5" in out
    assert "size=5: FAILED" in out
