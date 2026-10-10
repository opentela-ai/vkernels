"""Kernel benchmark harness: timing, throughput models, reporting, CI compare.

Layered like tokenspeed-kernel's benchmark package, scaled to this repo:

* :mod:`vkernels.benchmark.timing` — pluggable timers: host wall clock,
  CUDA events, and warmed CUDA-graph replay (the strongest method a
  callable supports; graph capture failure degrades to events).
* :mod:`vkernels.benchmark.throughput` — FLOPs/bytes per op (the roofline
  denominators), extensible via ``@ThroughputModel``.
* :mod:`vkernels.benchmark.runner` — :func:`run_benchmark` turns a
  :class:`BenchmarkRequest` into a timed, optionally oracle-verified
  :class:`BenchmarkResult`; ops outside the numerics generators plug in
  with :func:`register_benchmark_case` (the torch_ops extension point).
* :mod:`vkernels.benchmark.report` — fixed-width tables and versioned JSON.
* :mod:`vkernels.benchmark.ci` — ``compare``: the PR check that flags a
  case whose median runtime regressed beyond a threshold against the
  merge-base baseline.

Command line::

    python -m vkernels.benchmark ops
    python -m vkernels.benchmark run --ops add gemm --sizes 1024 4096 --json out.json
    python -m vkernels.benchmark run --suite my-arch.json --json out.json
    python -m vkernels.benchmark compare baseline.json current.json --threshold 0.1
"""

from __future__ import annotations

from vkernels.benchmark.ci import ComparisonReport, ComparisonRow, compare
from vkernels.benchmark.report import format_table, from_json, read_json, to_json, write_json
from vkernels.benchmark.runner import (
    BenchmarkCase,
    BenchmarkRequest,
    BenchmarkResult,
    register_benchmark_case,
    run_benchmark,
)
from vkernels.benchmark.throughput import model_for, register_throughput_model
from vkernels.benchmark.timing import (
    CudaEventTimer,
    GraphTimer,
    HostTimer,
    TimingResult,
    auto_timer,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkRequest",
    "BenchmarkResult",
    "ComparisonReport",
    "ComparisonRow",
    "CudaEventTimer",
    "GraphTimer",
    "HostTimer",
    "TimingResult",
    "auto_timer",
    "compare",
    "format_table",
    "from_json",
    "model_for",
    "read_json",
    "register_benchmark_case",
    "register_throughput_model",
    "run_benchmark",
    "to_json",
    "write_json",
]
