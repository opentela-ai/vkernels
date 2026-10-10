"""Tests for the JIT compile monitor (no GPU or real Triton required).

The bookkeeping (:class:`CompileMonitor`) is exercised directly, and the
hook plumbing is exercised against a stub ``triton`` module that mimics
``triton.knobs.runtime`` — the only Triton surface the monitor touches.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from vkernels import compile_monitor as cm
from vkernels.compile_monitor import CompileMonitor, CompileStats


def _spec(**constants: int) -> tuple[dict, dict]:
    """A rendered specialization + constexpr pair like the hook produces."""
    rendered = {name: repr(value) for name, value in constants.items()}
    return rendered, dict(constants)


def test_record_splits_startup_and_serving_stats():
    monitor = CompileMonitor("warn", cm.UNBOUNDED_VALUE_LIMIT)
    monitor.record("k.x", *_spec(BLOCK=64), seconds=1.0, site="a.py:1 (f)")
    monitor.serving = True
    monitor.record("k.x", *_spec(BLOCK=64), seconds=0.5, site="a.py:2 (f)")
    stats = monitor.stats()
    assert stats == CompileStats(1, 1.0, 1, 0.5)


def test_record_describes_change_against_nearest():
    monitor = CompileMonitor("warn", cm.UNBOUNDED_VALUE_LIMIT)
    monitor.record("k", *_spec(BLOCK=64, W=4), seconds=0.0, site="a.py:1 (f)")
    monitor.serving = True
    # The info log names what changed.
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("vkernels.compile_monitor")
    logger.addHandler(handler)
    try:
        logger.setLevel(logging.INFO)
        monitor.record("k", *_spec(BLOCK=128, W=4), seconds=0.02, site="a.py:2 (f)")
    finally:
        logger.removeHandler(handler)
    assert "BLOCK: 64 -> 128" in stream.getvalue()


def test_unbounded_parameter_warns_after_limit(caplog):
    monitor = CompileMonitor("warn", value_limit=4)
    monitor.serving = True
    with caplog.at_level(logging.WARNING, logger="vkernels.compile_monitor"):
        for value in (11, 12, 13, 14):  # below the limit: no report yet
            monitor.record("k", *_spec(T=value), seconds=0.0, site="a.py:1 (f)")
        # value_limit=4: the 5th new serving value crosses it
        monitor.record("k", *_spec(T=15), seconds=0.0, site="a.py:1 (f)")
    assert any("compile-time parameter T" in r.message for r in caplog.records)


def test_unbounded_parameter_error_mode_raises():
    monitor = CompileMonitor("error", value_limit=2)
    monitor.serving = True
    for value in (11, 12):
        monitor.record("k", *_spec(T=value), seconds=0.0, site="a.py:1 (f)")
    with pytest.raises(cm.UnboundedSpecializationError, match="parameter T"):
        monitor.record("k", *_spec(T=13), seconds=0.0, site="a.py:1 (f)")


def test_power_of_two_values_do_not_count_as_unbounded():
    monitor = CompileMonitor("error", value_limit=2)
    monitor.serving = True
    for value in (2, 4, 8, 16, 32, 64, 128):  # bucketed bounds are fine
        monitor.record("k", *_spec(BOUND=value), seconds=0.0, site="a.py:1 (f)")
    # only the reporting is suppressed; the compile itself was recorded
    assert monitor.stats().serving_compiles == 7


def test_startup_values_never_count_as_unbounded():
    monitor = CompileMonitor("error", value_limit=1)
    for value in range(20):  # all before the serving mark
        monitor.record("k", *_spec(T=value), seconds=0.0, site="a.py:1 (f)")
    monitor.serving = True
    monitor.record("k", *_spec(T=100), seconds=0.0, site="a.py:1 (f)")  # first serving value


def test_call_sites_are_distinguished(caplog):
    monitor = CompileMonitor("warn", value_limit=2)
    monitor.serving = True
    with caplog.at_level(logging.WARNING, logger="vkernels.compile_monitor"):
        for value in (11, 12, 13):
            monitor.record("k", *_spec(T=value), seconds=0.0, site="layer1.py:1 (f)")
        # same parameter, different call site: independent budget
        for value in (21, 22, 23):
            monitor.record("k", *_spec(T=value), seconds=0.0, site="layer2.py:9 (g)")
    warned = [r.message for r in caplog.records if "layer1.py:1" in r.message]
    assert warned and "layer2" not in warned[0]


def test_check_mode_env(monkeypatch):
    monkeypatch.setenv("VKERNELS_JIT_COMPILE_CHECK", "error")
    assert cm.check_mode() == "error"
    monkeypatch.setenv("VKERNELS_JIT_COMPILE_CHECK", "bogus")
    with pytest.raises(ValueError, match="warn"):
        cm.check_mode()


def test_serving_switch_and_stats_without_install():
    assert not cm.is_serving()
    assert cm.compile_stats() is None
    cm.mark_serving()
    assert cm.is_serving()
    assert cm.compile_stats() is None
    # reset for other tests (module state is process-global)
    cm._serving = False


# --- hook plumbing against a stub triton -------------------------------------


class _Param:
    def __init__(self, name: str, is_constexpr: bool):
        self.name = name
        self.is_constexpr = is_constexpr


class _JitFunction:
    def __init__(self, *params: _Param):
        self.params = list(params)


class _Fn:
    """The ``fn`` object Triton passes to its compile hooks."""

    def __init__(self, name: str, *params: _Param):
        self.name = name
        self.module = "some.kernels"
        self.jit_function = _JitFunction(*params)


@pytest.fixture()
def stub_triton(monkeypatch):
    triton = types.ModuleType("triton")
    knobs = types.ModuleType("triton.knobs")
    runtime = types.SimpleNamespace(jit_cache_hook=None, jit_post_compile_hook=None)
    knobs.runtime = runtime
    triton.knobs = knobs
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.knobs", knobs)
    return runtime


def _drive_compile(runtime, fn, *, key="k1", constants=None, signature=None):
    if runtime.jit_cache_hook is not None:
        runtime.jit_cache_hook(fn=fn, key=key)
    compile_info = {
        "constants": constants or {},
        "configs": [{}],
        "signature": signature or {},
    }
    runtime.jit_post_compile_hook(fn=fn, key=key, compile=compile_info)


def test_install_records_compiles_through_stub_hooks(stub_triton):
    assert cm.install_compile_monitor() is True
    try:
        fn = _Fn(
            "topk_kernel",
            _Param("x", False),
            _Param("BLOCK", True),
            _Param("num_tokens", True),
        )
        cm._hooks.monitor.serving = True
        _drive_compile(
            stub_triton,
            fn,
            constants={(0,): None, (1,): 64, (2,): 123},
            signature={"x": "fp32", "BLOCK": "constexpr", "num_tokens": "constexpr"},
        )
        stats = cm.compile_stats()
        assert stats is not None and stats.serving_compiles == 1
    finally:
        cm.uninstall_compile_monitor()
    assert stub_triton.jit_cache_hook is None
    assert stub_triton.jit_post_compile_hook is None


def test_install_returns_false_without_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)  # import → ImportError
    assert cm.install_compile_monitor() is False
    assert cm.compile_stats() is None


def test_hooks_chain_onto_previous_hooks(stub_triton):
    calls = []
    stub_triton.jit_post_compile_hook = lambda **kw: calls.append("previous")
    assert cm.install_compile_monitor() is True
    try:
        fn = _Fn("k", _Param("B", True))
        _drive_compile(stub_triton, fn, constants={(0,): 8}, signature={"B": "constexpr"})
        assert calls == ["previous"]
        assert cm.compile_stats().startup_compiles == 1
    finally:
        cm.uninstall_compile_monitor()
    # restored, not replaced
    assert stub_triton.jit_post_compile_hook is not None


def test_bad_hook_payload_does_not_break_the_launch(stub_triton):
    assert cm.install_compile_monitor() is True
    try:
        stub_triton.jit_cache_hook(fn=object(), key="x")  # no .jit_function
        stub_triton.jit_post_compile_hook(fn=object(), key="x", compile=None)
        assert cm.compile_stats().startup_compiles == 1  # still recorded
    finally:
        cm.uninstall_compile_monitor()


def test_assert_no_triton_compile_guard(stub_triton):
    fn = _Fn("warm_kernel", _Param("B", True))

    with cm.assert_no_triton_compile():
        pass  # no compilation: passes

    with pytest.raises(AssertionError, match="warm_kernel"):
        with cm.assert_no_triton_compile():
            _drive_compile(stub_triton, fn, constants={(0,): 8}, signature={"B": "constexpr"})

    with cm.assert_no_triton_compile(allow=("warm_kernel",)):
        _drive_compile(stub_triton, fn, constants={(0,): 16}, signature={"B": "constexpr"})

    # guard restores the hook it saw
    assert stub_triton.jit_post_compile_hook is None


def test_assert_no_triton_compile_noop_without_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    with cm.assert_no_triton_compile():
        pass
