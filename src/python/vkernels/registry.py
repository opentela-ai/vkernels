"""Metadata-only kernel selection, safe to use before importing GPU runtimes.

Registries are immutable snapshots. Selection is deterministic (descending
priority, then name), cached by the full request, and never executes kernels or
imports their modules. Call ``selection.load()`` only when ready to execute.
Reference implementations require explicit opt-in and are never a silent
fallback. Capability predicates must be pure functions of request metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str
    contiguous: bool = True


@dataclass(frozen=True)
class KernelRequest:
    operation: str
    tensors: tuple[TensorMetadata, ...]
    architecture: str = "unknown"
    backends: frozenset[str] = frozenset()
    graph_capture: bool = False
    workspace_bytes: int = 0
    parameters: tuple[tuple[str, int | float | str | bool], ...] = ()


@dataclass(frozen=True)
class KernelImplementation:
    operation: str
    name: str
    entrypoint: str
    check: Callable[[KernelRequest], tuple[str, ...]]
    backend: str
    priority: int = 0
    architectures: tuple[str, ...] = ()
    graph_capture: bool = False
    workspace_bytes: int = 0
    reference: bool = False

    @lru_cache(maxsize=256)
    def load(self) -> Callable:
        """Import the selected launcher; selection itself never calls this."""
        module, name = self.entrypoint.split(":")
        return getattr(import_module(module), name)


@dataclass(frozen=True)
class CandidateExplanation:
    implementation: KernelImplementation
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectionExplanation:
    request: KernelRequest
    override: str | None
    selected: KernelImplementation | None
    candidates: tuple[CandidateExplanation, ...]

    def __str__(self) -> str:
        chosen = self.selected.name if self.selected else "none"
        lines = [f"{self.request.operation}: selected={chosen}, override={self.override!r}"]
        lines.extend(
            f"  {candidate.implementation.name}: " + ("; ".join(candidate.reasons) or "eligible")
            for candidate in self.candidates
        )
        if self.override and not any(c.implementation.name == self.override for c in self.candidates):
            lines.append(f"  unknown implementation: {self.override}")
        return "\n".join(lines)


class KernelSelectionError(ValueError):
    """No implementation satisfies the request or explicit override."""


@dataclass(frozen=True)
class KernelRegistry:
    implementations: tuple[KernelImplementation, ...]

    def __post_init__(self) -> None:
        names = [(impl.operation, impl.name) for impl in self.implementations]
        if len(set(names)) != len(names):
            raise ValueError("duplicate kernel operation/name")
        for impl in self.implementations:
            if len(impl.entrypoint.split(":")) != 2 or not all(impl.entrypoint.split(":")):
                raise ValueError(f"invalid entrypoint: {impl.entrypoint!r}")
            if impl.workspace_bytes < 0:
                raise ValueError("workspace_bytes must be nonnegative")

    @lru_cache(maxsize=1024)
    def explain(
        self, request: KernelRequest, *, override: str | None = None, allow_reference: bool = False
    ) -> SelectionExplanation:
        """Explain all candidates without loading torch, Triton or a launcher.

        Backend availability and architecture are provided by the caller, so
        offline planning and startup can use the same API. Capture support
        describes the warmed launcher; callers must compile/warm it eagerly.
        """
        if request.workspace_bytes < 0:
            raise ValueError("workspace_bytes must be nonnegative")
        candidates = []
        selected = None
        for impl in sorted(self.implementations, key=lambda i: (-i.priority, i.name)):
            if impl.operation != request.operation:
                continue
            reasons = []
            if impl.reference and not allow_reference:
                reasons.append("reference implementation requires allow_reference=True")
            if impl.backend not in request.backends:
                reasons.append(f"backend {impl.backend!r} unavailable")
            if impl.architectures and request.architecture not in impl.architectures:
                reasons.append(f"architecture {request.architecture!r} unsupported")
            if request.graph_capture and not impl.graph_capture:
                reasons.append("graph capture unsupported")
            if request.workspace_bytes < impl.workspace_bytes:
                reasons.append(f"requires {impl.workspace_bytes} workspace bytes")
            reasons.extend(impl.check(request))
            candidates.append(CandidateExplanation(impl, tuple(reasons)))
            if not reasons and selected is None and (override is None or override == impl.name):
                selected = impl
        return SelectionExplanation(request, override, selected, tuple(candidates))

    def select(
        self, request: KernelRequest, *, override: str | None = None, allow_reference: bool = False
    ) -> KernelImplementation:
        explanation = self.explain(request, override=override, allow_reference=allow_reference)
        if explanation.selected is None:
            raise KernelSelectionError(str(explanation))
        return explanation.selected
