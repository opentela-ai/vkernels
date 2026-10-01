"""Symbolic capture values and diagnostics."""
from __future__ import annotations


from .operator_ir import (
    DType,
    Region,
    TensorValue,
)
class CaptureError(Exception):
    """Capture failed: the model body violated the capture contract."""


class UnsupportedOperator(CaptureError):
    """An operation outside the supported subset was encountered (§3.3)."""

    def __init__(self, message: str, node: str = "<unknown>", missing_contract: str = ""):
        super().__init__(message)
        self.node = node
        self.missing_contract = missing_contract


class SymbolicTensor:
    """Handle passed to model bodies; wraps a :class:`TensorValue`."""

    __slots__ = ("value",)

    def __init__(self, value: TensorValue):
        self.value = value

    # -- metadata passthrough (useful for guards in model bodies) ----------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> DType:
        return self.value.dtype

    @property
    def name(self) -> str:
        return self.value.name

    def item(self):  # pragma: no cover - guard path
        raise CaptureError(".item() on a symbolic tensor would freeze a data-dependent value into the graph; host control flow depending on tensor values is outside the capture contract (§4.2)")

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SymbolicTensor({self.value.name}{self.value.shape})"


def _regional_reads(view: TensorValue) -> Region:
    return Region.whole(view)

_RMS_GATED_ACTIVATIONS = ("sigmoid",)
