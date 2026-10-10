"""Timing methods for benchmarking kernels: host wall clock, CUDA events,
and warmed CUDA-graph replay.

The harness picks the strongest method the callable supports: graph replay
for graph-capturable GPU callables (torch/triton ops), CUDA events for GPU
callables that are not capturable, and host wall clock for everything else
(including the numpy-facing public kernels, which are synchronous pybind
or pure-Python calls). Each timer returns per-batch means as samples so
callers can see variance, never a single opaque number.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

__all__ = [
    "TimingResult",
    "Timer",
    "HostTimer",
    "CudaEventTimer",
    "GraphTimer",
    "auto_timer",
]


@dataclass(frozen=True)
class TimingResult:
    """Per-batch mean runtimes, in seconds, plus the method that produced them."""

    method: str  # "host" | "cuda-events" | "cuda-graph"
    samples: tuple[float, ...]

    @property
    def seconds(self) -> float:
        """Median sample — the reported runtime of the callable."""
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def spread(self) -> float:
        """(max - min) / median; 0.0 for a single sample."""
        if len(self.samples) < 2:
            return 0.0
        med = self.seconds
        return (max(self.samples) - min(self.samples)) / med if med else 0.0


class Timer(Protocol):
    def measure(self, fn: Callable[[], object], *, warmup: int, iters: int) -> TimingResult:
        ...


class HostTimer:
    """Wall-clock timing for synchronous callables (host CI default)."""

    def measure(self, fn: Callable[[], object], *, warmup: int, iters: int) -> TimingResult:
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        return TimingResult("host", tuple(samples))


def _torch_cuda() -> Any:
    """The torch module when CUDA is usable, else ``None``."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch
    except Exception:  # noqa: BLE001 — no torch, or CUDA init failed
        pass
    return None


class CudaEventTimer:
    """CUDA-event timing; one sample per batch of ``iters`` launches."""

    def __init__(self, batches: int = 3) -> None:
        self.batches = batches

    def measure(self, fn: Callable[[], object], *, warmup: int, iters: int) -> TimingResult:
        torch = _torch_cuda()
        if torch is None:
            raise RuntimeError("CUDA events unavailable (no torch.cuda)")
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(self.batches):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 1e3 / iters)
        return TimingResult("cuda-events", tuple(samples))


class GraphTimer:
    """Warm the callable, capture ``iters`` launches in one CUDA graph, and
    time graph replays.

    Graph replay removes per-launch CPU overhead, which is the honest
    device-time measurement for kernels served under graphs. Falls back to
    :class:`CudaEventTimer` when the callable cannot be captured (raises
    during capture, allocates unbacked memory, synchronizes, ...).
    """

    def __init__(self, batches: int = 3) -> None:
        self.batches = batches

    def measure(self, fn: Callable[[], object], *, warmup: int, iters: int) -> TimingResult:
        torch = _torch_cuda()
        if torch is None:
            raise RuntimeError("CUDA graphs unavailable (no torch.cuda)")
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(iters):
                    fn()
        except Exception:  # noqa: BLE001 — not capturable; time it eagerly
            return CudaEventTimer(self.batches).measure(fn, warmup=0, iters=iters)
        samples = []
        for _ in range(self.batches):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 1e3 / iters)
        return TimingResult("cuda-graph", tuple(samples))


def auto_timer() -> Timer:
    """The strongest timer this process can run: graph > events > host.

    Only capability is probed here — the actual method still degrades per
    callable (``GraphTimer`` falls back to events when capture fails).
    """
    return GraphTimer() if _torch_cuda() is not None else HostTimer()
