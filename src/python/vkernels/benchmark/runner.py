"""The benchmark runner: turn a request into timed, verified throughput.

A :class:`BenchmarkRequest` names an op (from the numerics generators) and
its shape parameters; :func:`run_benchmark` builds the standard seeded
inputs, optionally verifies the kernel against its oracle first, times it
with the strongest available :class:`~vkernels.benchmark.timing.Timer`, and
attaches the throughput model's FLOPs/bytes so the report can state
FLOP/s and GB/s against the roofline.

Ops outside the numerics generators (e.g. torch_ops serving kernels)
register a :class:`BenchmarkCase` with :func:`register_benchmark_case` —
the harness then times and reports them through the same machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from vkernels.benchmark.throughput import model_for
from vkernels.benchmark.timing import HostTimer, Timer, TimingResult, auto_timer

__all__ = [
    "BenchmarkCase",
    "BenchmarkRequest",
    "BenchmarkResult",
    "register_benchmark_case",
    "run_benchmark",
]


@dataclass(frozen=True)
class BenchmarkRequest:
    """One measurable case of one op."""

    op: str
    size: int
    n: int | None = None
    k: int | None = None
    seed: int = 0
    warmup: int = 5
    iters: int = 20
    verify: bool = False  # oracle comparison first (slow for big gemms)

    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"size": self.size}
        if self.n is not None:
            params["n"] = self.n
        if self.k is not None:
            params["k"] = self.k
        return params

    @property
    def key(self) -> str:
        parts = [self.op, f"size={self.size}"]
        if self.n is not None:
            parts.append(f"n={self.n}")
        if self.k is not None:
            parts.append(f"k={self.k}")
        return "|".join(parts)


@dataclass
class BenchmarkCase:
    """Extension point for ops outside the numerics generators.

    ``factory(params)`` returns the zero-argument callable to time;
    ``model(params)`` returns ``(flops, bytes)``.
    """

    factory: Callable[[dict], Callable[[], Any]]
    model: Callable[[dict], tuple[int, int]]


_CUSTOM_CASES: dict[str, BenchmarkCase] = {}


def register_benchmark_case(op: str, case: BenchmarkCase) -> None:
    """Teach the harness a new op (e.g. a torch_ops serving kernel)."""
    _CUSTOM_CASES[op] = case


def _kernel_case(request: BenchmarkRequest) -> tuple[Callable[[], Any], dict]:
    from vkernels.numerics.generators import make_case

    args, kwargs = make_case(
        request.op,
        request.size,
        seed=request.seed,
        **{k: v for k, v in request.params().items() if k != "size"},
    )

    def fn() -> Any:
        from vkernels import kernels

        return getattr(kernels, request.op)(*args, **kwargs)

    return fn, request.params()


def run_benchmark(
    request: BenchmarkRequest, *, timer: Timer | None = None
) -> "BenchmarkResult":
    """Time one request; never raises on verification failure (reports it).

    Timer policy: the built-in kernel cases are numpy-facing synchronous
    calls — CUDA-graph replay would capture nothing and time an empty
    graph — so they default to :class:`HostTimer`. Custom registered cases
    (torch/triton ops) default to :func:`auto_timer`, the strongest method
    the callable supports.
    """
    from vkernels import _backend

    custom = _CUSTOM_CASES.get(request.op) is not None
    timer = timer or (auto_timer() if custom else HostTimer())
    params = request.params()
    verified: bool | None = None
    error: str | None = None

    if _CUSTOM_CASES.get(request.op) is not None:
        fn = _CUSTOM_CASES[request.op].factory(params)
        flops, bytes_ = _CUSTOM_CASES[request.op].model(params)
    else:
        fn, params = _kernel_case(request)
        flops, bytes_ = model_for(request.op, params)

    if request.verify:
        from vkernels.numerics import verify

        result = verify(request.op, request.size, seed=request.seed, **{
            k: v for k, v in params.items() if k != "size"
        })
        verified = result.ok
        error = None if result.ok else result.message

    timing: TimingResult | None = None
    if error is None:
        timing = timer.measure(fn, warmup=request.warmup, iters=request.iters)

    seconds = timing.seconds if timing else float("nan")
    return BenchmarkResult(
        request=request,
        backend=_backend.backend_name(),
        method=timing.method if timing else "none",
        samples=timing.samples if timing else (),
        seconds=seconds,
        flops=flops,
        bytes=bytes_,
        gflops=flops / seconds / 1e9 if seconds and seconds > 0 else float("nan"),
        gbytes=bytes_ / seconds / 1e9 if seconds and seconds > 0 else float("nan"),
        verified=verified,
        error=error,
    )


@dataclass
class BenchmarkResult:
    request: BenchmarkRequest
    backend: str
    method: str
    samples: tuple[float, ...]
    seconds: float
    flops: int
    bytes: int
    gflops: float
    gbytes: float
    verified: bool | None
    error: str | None
