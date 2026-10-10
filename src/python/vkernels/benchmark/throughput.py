"""Throughput models: FLOPs and bytes moved per op, as a function of the
case parameters.

These are the roofline denominators (see the project wiki's roofline
method): FLOP/s against the compute roof, bytes/s against the HBM roof.
Models assume float32 (4 bytes/element) and beta=0 GEMMs (C written once).
"""

from __future__ import annotations

from typing import Callable

__all__ = ["model_for", "register_throughput_model", "ThroughputModel"]

#: op -> callable(params dict) -> (flops, bytes)
_MODELS: dict[str, Callable[[dict], tuple[int, int]]] = {}


def ThroughputModel(op: str):  # noqa: N802 — decorator, named like the concept
    """Register a throughput model: ``@ThroughputModel("gemm")``."""

    def decorate(fn: Callable[[dict], tuple[int, int]]) -> Callable[[dict], tuple[int, int]]:
        _MODELS[op] = fn
        return fn

    return decorate


def register_throughput_model(op: str, fn: Callable[[dict], tuple[int, int]]) -> None:
    """Imperative form of :func:`ThroughputModel`."""
    _MODELS[op] = fn


def _elementwise(rw_arrays: int) -> Callable[[dict], tuple[int, int]]:
    def model(params: dict) -> tuple[int, int]:
        n = params["size"]
        return n, rw_arrays * n * 4

    return model


@ThroughputModel("add")
def _add(params: dict) -> tuple[int, int]:
    return _elementwise(3)(params)


@ThroughputModel("scale")
def _scale(params: dict) -> tuple[int, int]:
    return _elementwise(2)(params)


@ThroughputModel("relu")
def _relu(params: dict) -> tuple[int, int]:
    return _elementwise(2)(params)


@ThroughputModel("sum")
def _sum(params: dict) -> tuple[int, int]:
    n = params["size"]
    return n, n * 4 + 4  # read n, write one scalar


@ThroughputModel("max")
def _max(params: dict) -> tuple[int, int]:
    n = params["size"]
    return n, n * 4 + 4


@ThroughputModel("gemm")
def _gemm(params: dict) -> tuple[int, int]:
    m = params["size"]
    n = params.get("n", 16)
    k = params.get("k", 16)
    flops = 2 * m * n * k
    bytes_ = (m * k + k * n + 2 * m * n) * 4  # A read, B read, C write
    return flops, bytes_


def model_for(op: str, params: dict) -> tuple[int, int]:
    """``(flops, bytes)`` for one case of ``op`` at these parameters."""
    try:
        model = _MODELS[op]
    except KeyError:
        raise KeyError(
            f"no throughput model for {op!r}; available: {', '.join(sorted(_MODELS))}"
        ) from None
    return model(dict(params))
