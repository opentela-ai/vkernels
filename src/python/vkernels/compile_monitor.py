"""Report JIT compilations that happen after startup ends.

Every compile-time kernel parameter (``tl.constexpr`` in Triton, template
arguments in DeepGEMM-style libraries) is part of the compile-cache key.
A per-batch value passed as one — a token or row count, a table width, a
sequence length — compiles a new binary for every new batch shape, on the
forward thread, stalling serving for 100+ ms each time.

This module hooks Triton's JIT and records every compilation.
Compilations during startup are expected. After :func:`mark_serving`,
each one is logged with its duration and with what changed against the
kernel's earlier specializations, and a compile-time parameter that keeps
taking new values from one call site is reported by name once more than
:data:`UNBOUNDED_VALUE_LIMIT` distinct non-power-of-two values were first
compiled there while serving. The call site is the first Python frame
outside Triton, torch and this package, so layers that legitimately launch
a kernel with different fixed dimensions count separately. Powers of two
are not counted: rounding up to one is how a kernel buckets a
compile-time bound, and it is log-bounded.

The serving mark is also the compile switch, kept whether or not the
monitor is installed: :func:`is_serving` turns true, and code whose
library compiles once per batch shape (outside Triton, where no key can
be bucketed for it) may query it to pick an implementation that never
compiles.

The check mode comes from ``VKERNELS_JIT_COMPILE_CHECK`` (``warn``, the
default, or ``error``); serving CI runs with ``error`` so a per-batch
constexpr fails the job instead of stalling it.

Unlike torch or triton-dependent modules, importing this one is always
safe: Triton is touched only by :func:`install_compile_monitor`.

Design adapted from tokenspeed-kernel's ``compile_monitor`` (MIT,
Copyright (c) 2026 LightSeek Foundation).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

__all__ = [
    "UNBOUNDED_VALUE_LIMIT",
    "CompileMonitor",
    "CompileStats",
    "UnboundedSpecializationError",
    "assert_no_triton_compile",
    "check_mode",
    "compile_stats",
    "install_compile_monitor",
    "is_serving",
    "mark_serving",
    "uninstall_compile_monitor",
]

logger = logging.getLogger(__name__)

# A compile-time parameter may take this many new non-power-of-two values
# from one call site while serving before it is reported. A call site passes
# one model dimension; a per-batch value crosses the limit within the first
# few distinct batch shapes.
UNBOUNDED_VALUE_LIMIT = 4

OnUnbounded = Literal["warn", "error"]

_ENV_CHECK = "VKERNELS_JIT_COMPILE_CHECK"


class UnboundedSpecializationError(RuntimeError):
    """A compile-time kernel parameter keeps taking new values while serving."""


def check_mode() -> OnUnbounded:
    """The unbounded-parameter reaction: ``VKERNELS_JIT_COMPILE_CHECK``."""
    mode = os.environ.get(_ENV_CHECK, "warn")
    if mode not in ("warn", "error"):
        raise ValueError(f"{_ENV_CHECK} must be 'warn' or 'error', got {mode!r}")
    return mode  # type: ignore[return-value]


@dataclass(frozen=True)
class CompileStats:
    """Cumulative JIT compilations, split at :func:`mark_serving`."""

    startup_compiles: int
    startup_seconds: float
    serving_compiles: int
    serving_seconds: float


@dataclass
class _KernelHistory:
    # Rendered specialization of every compiled variant: parameter -> text.
    specializations: list[dict[str, str]] = field(default_factory=list)
    # Every value each numeric compile-time parameter was compiled with.
    values: dict[str, set[int | float]] = field(default_factory=dict)
    # Per (parameter, call site): values first compiled while serving,
    # powers of two excluded.
    serving_values: dict[tuple[str, str], list[int | float]] = field(
        default_factory=dict
    )
    # serving_values count at the last report, so reports repeat on doubling.
    reported: dict[tuple[str, str], int] = field(default_factory=dict)


def _is_power_of_two(value: int | float) -> bool:
    return isinstance(value, int) and value > 0 and value & (value - 1) == 0


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _param_name(names: list[str], path: tuple[int, ...]) -> str:
    return names[path[0]] + "".join(f"[{index}]" for index in path[1:])


def _describe_change(
    previous: list[dict[str, str]], specialization: dict[str, str]
) -> str:
    """Name what distinguishes a compilation from its nearest earlier one."""
    if not previous:
        return "first specialization of this kernel"

    def differing(other: dict[str, str]) -> list[str]:
        return [
            name
            for name in specialization.keys() | other.keys()
            if specialization.get(name) != other.get(name)
        ]

    nearest = min(previous, key=lambda other: len(differing(other)))
    changes = [
        f"{name}: {nearest.get(name, '-')} -> {specialization.get(name, '-')}"
        for name in sorted(differing(nearest))
    ]
    return "; ".join(changes) if changes else "same key recompiled"


class CompileMonitor:
    """Bookkeeping behind the JIT hooks; usable without Triton for tests.

    Args:
        on_unbounded: ``"warn"`` logs a parameter that keeps taking new
            values while serving; ``"error"`` raises
            :class:`UnboundedSpecializationError` from the compiling launch.
        value_limit: How many new non-power-of-two values one compile-time
            parameter may take from one call site while serving before it
            is reported.
    """

    def __init__(self, on_unbounded: OnUnbounded, value_limit: int) -> None:
        if on_unbounded not in ("warn", "error"):
            raise ValueError(f"on_unbounded must be 'warn' or 'error', got {on_unbounded!r}")
        self.on_unbounded: OnUnbounded = on_unbounded
        self.value_limit = value_limit
        self.serving = False
        self._lock = threading.Lock()
        self._kernels: dict[str, _KernelHistory] = {}
        self._startup_compiles = 0
        self._startup_seconds = 0.0
        self._serving_compiles = 0
        self._serving_seconds = 0.0

    def stats(self) -> CompileStats:
        with self._lock:
            return CompileStats(
                startup_compiles=self._startup_compiles,
                startup_seconds=self._startup_seconds,
                serving_compiles=self._serving_compiles,
                serving_seconds=self._serving_seconds,
            )

    def record(
        self,
        kernel: str,
        specialization: dict[str, str],
        constexprs: dict[str, Any],
        seconds: float,
        site: str,
    ) -> None:
        """Record one compilation and report it if it happened while serving.

        Args:
            kernel: Qualified kernel name.
            specialization: Rendered compile-cache key, parameter -> text,
                for describing what a new compilation changed.
            constexprs: Values of the parameters the kernel declares
                compile-time, flattened for tuple parameters.
            seconds: Wall time of the compilation.
            site: The launching call site, ``file:line (function)``.
        """
        unbounded: list[tuple[str, list[int | float]]] = []
        with self._lock:
            history = self._kernels.setdefault(kernel, _KernelHistory())
            change = _describe_change(history.specializations, specialization)
            history.specializations.append(specialization)
            serving = self.serving
            if serving:
                self._serving_compiles += 1
                self._serving_seconds += seconds
            else:
                self._startup_compiles += 1
                self._startup_seconds += seconds
            for name, value in constexprs.items():
                if not _numeric(value):
                    continue
                seen = history.values.setdefault(name, set())
                if value in seen:
                    continue
                seen.add(value)
                if not serving or _is_power_of_two(value):
                    continue
                new_values = history.serving_values.setdefault((name, site), [])
                new_values.append(value)
                reported = history.reported.get((name, site), 0)
                if len(new_values) > max(self.value_limit, 2 * reported):
                    history.reported[(name, site)] = len(new_values)
                    unbounded.append((name, list(new_values)))
        if not serving:
            return
        logger.info(
            "Triton JIT compiled %s while serving in %.0f ms (%s) from %s",
            kernel, seconds * 1e3, change, site,
        )
        for name, values in unbounded:
            shown = ", ".join(str(value) for value in values[:12])
            more = ", ..." if len(values) > 12 else ""
            message = (
                f"{kernel}: compile-time parameter {name} has compiled "
                f"{len(values)} new values while serving from {site} "
                f"({shown}{more}). Each new value is a JIT compilation on "
                "the forward thread; a value that varies per batch must be "
                "a runtime argument, or be bucketed to a power of two "
                "where the kernel needs a compile-time bound."
            )
            if self.on_unbounded == "error":
                raise UnboundedSpecializationError(message)
            logger.warning(message)


def _render(kind: str, value: Any, attrs: list[list[Any]]) -> str:
    """Render one parameter's part of the compile-cache key."""
    if kind == "constexpr":
        return repr(value)
    flags = "".join(f" {name.removeprefix('tt.')}={setting}" for name, setting in attrs)
    return f"{kind}{flags}"


def _specialization(fn: Any, compile_info: dict[str, Any]) -> tuple[
    dict[str, str],
    dict[str, Any],
]:
    """Split a hook payload into its rendered key and declared constexprs.

    Defensive by design: Triton's hook payload is an internal contract that
    moves between versions, and a monitoring failure must never break the
    launch it observes.
    """
    params = fn.jit_function.params
    names = [param.name for param in params]
    constants = compile_info.get("constants") or {}
    configs = compile_info.get("configs") or [{}]
    attrs = configs[0] if configs else {}
    rendered: dict[str, str] = {}
    for name, kind in compile_info.get("signature", {}).items():
        if name not in names:
            continue
        index = names.index(name)
        rendered[name] = _render(kind, constants.get((index,)), attrs.get((index,), []))
    constexprs = {
        _param_name(names, path): value
        for path, value in constants.items()
        if len(path) and path[0] < len(params) and params[path[0]].is_constexpr
    }
    # A tuple-valued compile-time parameter arrives as one value; count each
    # element as its own parameter so a varying element is named.
    for name, value in list(constexprs.items()):
        if isinstance(value, tuple):
            del constexprs[name]
            for index, element in enumerate(value):
                constexprs[f"{name}[{index}]"] = element
    return rendered, constexprs


def _internal_dirs() -> tuple[str, ...]:
    """Directories whose frames are the JIT and its wrappers, not call sites."""
    here = os.path.dirname(os.path.dirname(__file__ or ".")) + os.sep
    dirs = [here]  # this package
    for module_name in ("triton", "torch"):
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None) if module is not None else None
        if module_file:
            dirs.append(os.path.dirname(str(module_file)) + os.sep)
    return tuple(dirs)


def _call_site(internal_dirs: tuple[str, ...]) -> str:
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(internal_dirs):
        frame = frame.f_back
    if frame is None:
        return "<unknown>"
    parts = frame.f_code.co_filename.split(os.sep)
    return f"{os.sep.join(parts[-3:])}:{frame.f_lineno} ({frame.f_code.co_name})"


def _runtime_knobs():
    """``triton.knobs.runtime`` if present, else ``None`` (defensive)."""
    try:
        import triton.knobs as _knobs

        return _knobs.runtime
    except Exception:  # noqa: BLE001 — no triton, or knobs moved
        return None


class _Hooks:
    """Triton's pre- and post-compile hooks, chained onto earlier ones."""

    def __init__(self, monitor: CompileMonitor) -> None:
        knobs = _runtime_knobs()
        if knobs is None:
            raise RuntimeError("triton.knobs.runtime is unavailable")
        self.monitor = monitor
        self.previous_cache_hook = knobs.jit_cache_hook
        self.previous_post_compile_hook = knobs.jit_post_compile_hook
        self._knobs = knobs
        self._started: dict[tuple[int, int, str], float] = {}
        self._internal_dirs = _internal_dirs()

    def _timer_key(self, fn: Any, key: str) -> tuple[int, int, str]:
        return threading.get_ident(), id(fn.jit_function), key

    def cache_hook(self, **kwargs: Any) -> "bool | None":
        if self.previous_cache_hook is not None:
            skip = self.previous_cache_hook(**kwargs)
            if skip:
                return skip
        try:
            self._started[self._timer_key(kwargs["fn"], kwargs["key"])] = (
                time.perf_counter()
            )
        except Exception:  # noqa: BLE001 — monitoring must not break launches
            pass
        return None

    def post_compile_hook(self, **kwargs: Any) -> "bool | None":
        result = None
        if self.previous_post_compile_hook is not None:
            result = self.previous_post_compile_hook(**kwargs)
        fn = kwargs.get("fn")
        seconds = 0.0
        try:
            started = self._started.pop(self._timer_key(fn, kwargs.get("key", "")), None)
            seconds = 0.0 if started is None else time.perf_counter() - started
        except Exception:  # noqa: BLE001 — monitoring must not break launches
            pass
        try:
            specialization, constexprs = _specialization(fn, kwargs.get("compile") or {})
            name = f"{getattr(fn, 'module', '<unknown>')}.{getattr(fn, 'name', '<unknown>')}"
        except Exception:  # noqa: BLE001 — Triton internals moved again
            specialization, constexprs, name = {}, {}, str(getattr(fn, "name", "<unknown>"))
        self.monitor.record(
            name, specialization, constexprs, seconds, _call_site(self._internal_dirs)
        )
        return result


_hooks: _Hooks | None = None
_serving = False


def install_compile_monitor(on_unbounded: OnUnbounded | None = None) -> bool:
    """Start recording every Triton compilation in this process.

    Args:
        on_unbounded: ``"warn"`` or ``"error"``; defaults to
            :func:`check_mode` (the ``VKERNELS_JIT_COMPILE_CHECK`` env).

    Chains onto hooks installed earlier. Installing again replaces the
    monitor, which resets its history. Returns ``True`` when the hooks were
    installed, ``False`` when Triton (or its JIT hook knobs) is unavailable
    — importing this module stays safe either way.
    """
    global _hooks
    if _runtime_knobs() is None:
        logger.info("Triton compile monitor unavailable; no JIT hooks installed")
        return False
    uninstall_compile_monitor()
    _hooks = _Hooks(CompileMonitor(on_unbounded or check_mode(), UNBOUNDED_VALUE_LIMIT))
    _hooks._knobs.jit_cache_hook = _hooks.cache_hook
    _hooks._knobs.jit_post_compile_hook = _hooks.post_compile_hook
    return True


def uninstall_compile_monitor() -> None:
    """Remove the monitor and restore the hooks it chained onto."""
    global _hooks
    if _hooks is None:
        return
    _hooks._knobs.jit_cache_hook = _hooks.previous_cache_hook
    _hooks._knobs.jit_post_compile_hook = _hooks.previous_post_compile_hook
    _hooks = None


def mark_serving() -> None:
    """Mark the end of startup: close the compile switch, report later compiles."""
    global _serving
    _serving = True
    if _hooks is not None:
        _hooks.monitor.serving = True


def is_serving() -> bool:
    """Whether startup has ended, after which kernels must not compile per batch shape."""
    return _serving


def compile_stats() -> CompileStats | None:
    """Cumulative compilations, or ``None`` when the monitor is not installed."""
    return None if _hooks is None else _hooks.monitor.stats()


@contextmanager
def assert_no_triton_compile(*, allow: "tuple[str, ...] | set[str]" = ()) -> Iterator[None]:
    """Fail if any Triton kernel compiles inside this block (test guard).

    Guards the devflow rule that batch-varying values must be runtime
    arguments: a kernel whose test launches many distinct shapes inside
    this context manager must not trigger one compilation per shape.
    ``allow`` names kernels (bare function name, e.g. ``"moe_topk_kernel"``)
    permitted to compile — for one-time warmups the test deliberately
    performs here. Without Triton the block is a no-op, so the guard runs
    on host CI too.
    """
    compiled: list[str] = []
    knobs = _runtime_knobs()
    if knobs is None:
        yield
        return

    previous = knobs.jit_post_compile_hook

    def _record(**kwargs: Any) -> "bool | None":
        result = previous(**kwargs) if previous is not None else None
        fn = kwargs.get("fn")
        compiled.append(str(getattr(fn, "name", "<unknown>")))
        return result

    knobs.jit_post_compile_hook = _record
    try:
        yield
    finally:
        knobs.jit_post_compile_hook = previous
    unexpected = [
        name for name in compiled if not any(name.endswith(a) or a in name for a in allow)
    ]
    if unexpected:
        raise AssertionError(
            f"Triton compiled {len(unexpected)} kernel(s) inside the guard: "
            f"{', '.join(sorted(set(unexpected)))}. A value that varies per "
            "batch must be a runtime argument, not a compile-time constant."
        )
