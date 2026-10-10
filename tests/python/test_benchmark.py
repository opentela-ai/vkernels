"""Tests for the benchmark harness (timers, throughput, runner, CI compare).

All host-runnable: the graph/event timers are exercised through a torch
guard and skip without CUDA.
"""

from __future__ import annotations

import json

import pytest

from vkernels import benchmark as bm
from vkernels.benchmark import cli as cli_mod
from vkernels.benchmark import runner as runner_mod
from vkernels.benchmark.timing import HostTimer, TimingResult


# --- throughput models --------------------------------------------------------


@pytest.mark.parametrize(
    "op,params,expected",
    [
        ("add", {"size": 100}, (100, 3 * 100 * 4)),
        ("scale", {"size": 100}, (100, 2 * 100 * 4)),
        ("relu", {"size": 100}, (100, 2 * 100 * 4)),
        ("sum", {"size": 100}, (100, 100 * 4 + 4)),
        ("max", {"size": 100}, (100, 100 * 4 + 4)),
        ("gemm", {"size": 8, "n": 4, "k": 2}, (2 * 8 * 4 * 2, (8 * 2 + 2 * 4 + 2 * 8 * 4) * 4)),
    ],
)
def test_throughput_models(op, params, expected):
    assert bm.model_for(op, params) == expected


def test_throughput_model_defaults_and_errors():
    flops, bytes_ = bm.model_for("gemm", {"size": 4})
    assert flops == 2 * 4 * 16 * 16  # default n=k=16 mirrors the generator
    with pytest.raises(KeyError, match="available"):
        bm.model_for("nope", {"size": 1})


def test_register_throughput_model():
    bm.register_throughput_model("custom_op", lambda p: (p["size"], p["size"] * 8))
    assert bm.model_for("custom_op", {"size": 3}) == (3, 24)


# --- timing -------------------------------------------------------------------


def test_host_timer_records_per_call_samples():
    calls = []

    def fn():
        calls.append(1)

    result = HostTimer().measure(fn, warmup=2, iters=5)
    assert result.method == "host"
    assert len(result.samples) == 5
    assert len(calls) == 7  # warmup counted
    assert result.seconds > 0


def test_timing_result_median_and_spread():
    result = TimingResult("host", (1.0, 2.0, 3.0))
    assert result.seconds == 2.0
    assert result.spread == pytest.approx(1.0)
    assert TimingResult("host", (1.0,)).spread == 0.0


def test_cuda_graph_timer_runs_a_torch_op():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    result = bm.GraphTimer(batches=2).measure(lambda: a @ b, warmup=2, iters=3)
    assert result.method in ("cuda-graph", "cuda-events")
    assert result.seconds > 0


# --- runner -------------------------------------------------------------------


def test_run_benchmark_add():
    request = bm.BenchmarkRequest("add", size=256, warmup=2, iters=5)
    result = bm.run_benchmark(request)
    assert result.error is None
    assert result.method == "host"
    assert len(result.samples) == 5
    assert result.bytes == 3 * 256 * 4
    assert result.verified is None  # verify defaults off
    assert result.gflops == pytest.approx(256 / result.seconds / 1e9)


def test_run_benchmark_gemm_with_verification():
    request = bm.BenchmarkRequest("gemm", size=16, n=4, k=4, warmup=1, iters=2, verify=True)
    result = bm.run_benchmark(request)
    assert result.verified is True
    assert result.flops == 2 * 16 * 4 * 4


def test_run_benchmark_reports_verification_failure_without_raising(monkeypatch):
    # Patch the verify symbol the runner resolves at call time.
    monkeypatch.setattr(
        "vkernels.numerics.verify", lambda *a, **k: _failing_verification()
    )
    request = bm.BenchmarkRequest("add", size=64, warmup=1, iters=2, verify=True)
    result = bm.run_benchmark(request)
    assert result.verified is False
    assert result.error == "not close: broken"
    assert result.method == "none"  # no timing after a failed gate
    assert result.samples == ()


def _failing_verification():
    from vkernels.numerics import VerificationResult

    return VerificationResult("add", 64, "fake", ok=False, exact=True,
                              stats={}, message="not close: broken")


def test_register_benchmark_case_custom_op():
    bm.register_benchmark_case(
        "custom_counter",
        bm.BenchmarkCase(
            factory=lambda params: (lambda: None),
            model=lambda params: (params.get("flops", 0), params.get("bytes", 0)),
        ),
    )
    request = bm.BenchmarkRequest("custom_counter", size=1, warmup=1, iters=2)
    result = bm.run_benchmark(request)
    assert result.flops == 0 and result.bytes == 0


# --- report + compare -----------------------------------------------------------


def _result_document(seconds: float, key: str = "add|size=64") -> dict:
    return {
        "schema": 1,
        "results": [
            {
                "request": {"op": key.split("|")[0], "size": int(key.split("=")[1])},
                "backend": "fallback",
                "method": "host",
                "samples": [seconds],
                "seconds": seconds,
                "flops": 64,
                "bytes": 768,
                "gflops": 0.0,
                "gbytes": 0.0,
                "verified": None,
                "error": None,
                "key": key,
            }
        ],
    }


def test_json_roundtrip(tmp_path):
    request = bm.BenchmarkRequest("add", size=64, warmup=1, iters=2)
    result = bm.run_benchmark(request)
    bm.write_json(tmp_path / "r.json", [result])
    loaded = bm.read_json(tmp_path / "r.json")
    assert loaded[0].request.key == result.request.key
    assert loaded[0].seconds == result.seconds


def test_json_schema_mismatch_rejected():
    with pytest.raises(ValueError, match="schema"):
        bm.from_json({"schema": 99, "results": []})


def test_compare_verdicts():
    base = _result_document(1.0)
    # 1.2x slower at threshold 0.1 -> regression
    slow = _result_document(1.2)
    report = bm.compare(base, slow, threshold=0.10)
    assert not report.ok
    assert report.regressions[0].verdict == "regression"
    assert report.regressions[0].ratio == pytest.approx(1.2)
    # 1.05x at threshold 0.1 -> ok
    assert bm.compare(base, _result_document(1.05), threshold=0.10).ok
    # 0.5x -> improved
    improved = bm.compare(base, _result_document(0.5), threshold=0.10)
    assert improved.rows[0].verdict == "improved" and improved.ok


def test_compare_new_missing_and_error_rows():
    base = _result_document(1.0)
    cur = _result_document(1.0, key="gemm|size=8")
    report = bm.compare(base, cur)
    verdicts = {r.key: r.verdict for r in report.rows}
    assert verdicts["add|size=64"] == "missing"
    assert verdicts["gemm|size=8"] == "new"
    assert report.ok  # neither fails the check

    errored = _result_document(1.0)
    errored["results"][0]["error"] = "boom"
    report = bm.compare(base, errored)
    assert report.rows[0].verdict == "error"


def test_format_table_includes_op_and_throughput():
    request = bm.BenchmarkRequest("add", size=64, warmup=1, iters=2)
    table = bm.format_table([bm.run_benchmark(request)])
    assert "add" in table and "GB/s" in table


# --- CLI ------------------------------------------------------------------------


def test_cli_ops(capsys):
    assert cli_mod.main(["ops"]) == 0
    assert "gemm" in capsys.readouterr().out


def test_cli_run_and_compare(tmp_path, capsys, monkeypatch):
    # Freeze timing so the comparison is deterministic: patch HostTimer.
    class FrozenTimer:
        ticks = 0

        def measure(self, fn, *, warmup, iters):
            FrozenTimer.ticks += 1
            return TimingResult("host", (1e-4,))

    monkeypatch.setattr(runner_mod, "HostTimer", FrozenTimer)
    monkeypatch.setattr(runner_mod, "auto_timer", lambda: FrozenTimer())

    out = tmp_path / "out.json"
    assert cli_mod.main(["run", "--ops", "add", "--sizes", "64", "--json", str(out)]) == 0
    assert "add" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["schema"] == 1

    # Same suite again: identical timings -> ok, exit 0.
    out2 = tmp_path / "out2.json"
    cli_mod.main(["run", "--ops", "add", "--sizes", "64", "--json", str(out2)])
    assert cli_mod.main(["compare", str(out), str(out2)]) == 0

    # And a doctored regression: 2x slower -> exit 1.
    slower = json.loads(out2.read_text())
    slower["results"][0]["seconds"] *= 2
    slower["results"][0]["samples"] = [slower["results"][0]["seconds"]]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(slower))
    assert cli_mod.main(["compare", str(out), str(bad)]) == 1
    assert "regression" in capsys.readouterr().out


def test_cli_run_suite_file(tmp_path, capsys, monkeypatch):
    class FrozenTimer:
        def measure(self, fn, *, warmup, iters):
            return TimingResult("host", (2e-4,))

    monkeypatch.setattr(runner_mod, "HostTimer", FrozenTimer)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"requests": [{"op": "relu", "size": 32}]}))
    out = tmp_path / "suite-out.json"
    assert cli_mod.main(["run", "--suite", str(suite), "--json", str(out)]) == 0
    assert "relu" in capsys.readouterr().out
    assert "relu" in out.read_text()
