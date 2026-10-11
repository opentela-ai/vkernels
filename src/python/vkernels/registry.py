"""Metadata-only kernel selection, safe to use before importing GPU runtimes.

Registries are immutable snapshots. Selection is deterministic (descending
priority, then name), cached by the full request, and never executes kernels or
imports their modules. Call ``selection.load()`` only when ready to execute.
Reference implementations require explicit opt-in and are never a silent
fallback. Capability predicates must be pure functions of request metadata.

Priorities are drawn from the :class:`Priority` band layout (borrowed from
tokenspeed-kernel): REFERENCE < PORTABLE < PERFORMANT < SPECIALIZED <
PLUGIN, each band four wide in ``[0, 20)``. Bands encode a
portability/performance contract rather than a bare ranking, and the PLUGIN
band is reserved for out-of-tree registrations so they always have headroom
over in-tree kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from importlib import import_module
from typing import Callable

from vkernels.signature import FormatSignature


# Hard upper bound on priority values; see :class:`Priority`.
_PRIORITY_MAX = 20


class Priority(IntEnum):
    """Selection-priority bands for registered kernels.

    Priority is the tiebreaker among kernels that already match the request's
    capability gates (``check``, backends, architecture, graph capture,
    workspace). A kernel that fails gating is filtered out before priority is
    consulted.

    Bands group kernels by their portability/performance contract so an
    out-of-tree registration can predict what offset it needs to win without
    auditing every in-tree registration. Use band members directly, or add a
    small intra-band offset for relative preference within the band::

        priority=Priority.PERFORMANT       # band start (8)
        priority=Priority.PERFORMANT + 2   # +2 within the band (10)

    Band layout (each occupies a contiguous range of ints in ``[0, 20)``):

    +--------------+--------+--------------------------------------------------+
    | Band         | Range  | When to use                                      |
    +==============+========+==================================================+
    | REFERENCE    |    0   | Correctness reference / oracle. Never            |
    |              |        | auto-selected when a real implementation is      |
    |              |        | available (pair with ``reference=True``).        |
    +--------------+--------+--------------------------------------------------+
    | PORTABLE     |  4..7  | Generic implementation with no arch or shape     |
    |              |        | gating beyond the family contract.               |
    +--------------+--------+--------------------------------------------------+
    | PERFORMANT   |  8..11 | Generally optimized kernel covering a broad      |
    |              |        | arch range. The default winner on supported      |
    |              |        | platforms.                                       |
    +--------------+--------+--------------------------------------------------+
    | SPECIALIZED  | 12..15 | Highly optimized kernel, narrowly gated on arch   |
    |              |        | and/or shape (e.g. a native path for one         |
    |              |        | storage format).                                |
    +--------------+--------+--------------------------------------------------+
    | PLUGIN       | 16..19 | Reserved for out-of-tree registrations to        |
    |              |        | override the in-tree default; in-tree kernels    |
    |              |        | must not use this band.                          |
    +--------------+--------+--------------------------------------------------+
    """

    REFERENCE = 0
    PORTABLE = 4
    PERFORMANT = 8
    SPECIALIZED = 12
    PLUGIN = 16

    @classmethod
    def band_of(cls, value: int) -> "Priority":
        """The band containing ``value`` (largest band start <= value)."""
        return max((band for band in cls if int(band) <= value), key=int, default=cls.REFERENCE)


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
    # Declarative per-role dtype contract; ``explain`` filters on it
    # alongside ``check``, so the format gate is data, not predicate
    # prose (see vkernels.signature).
    signature: FormatSignature | None = None
    # Entrypoint ("module:function") of a one-time weight transform this
    # implementation requires at LOAD time — e.g. the e4m3fn->e4m3fnuz
    # in-place rewrite the gfx942 storage paths need. Kept as a string so
    # selection stays import-free; callers resolve it via
    # :meth:`load_weight_preprocessor` after startup planning, never under
    # graph capture (borrowed from tokenspeed-kernel's registration hook).
    weight_preprocessor: str | None = None

    @lru_cache(maxsize=256)
    def load(self) -> Callable:
        """Import the selected launcher; selection itself never calls this."""
        module, name = self.entrypoint.split(":")
        return getattr(import_module(module), name)

    @lru_cache(maxsize=256)
    def load_weight_preprocessor(self) -> Callable | None:
        """Import the declared weight preprocessor, or ``None``.

        Like :meth:`load`, this only happens when the caller is ready to
        execute — importing it may pull GPU runtimes. The preprocessor
        contract is the loading side's: run once per weight version, keyed
        by ``(data_ptr, _version, device)``, eagerly outside graph capture.
        """
        if self.weight_preprocessor is None:
            return None
        module, name = self.weight_preprocessor.split(":")
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
            if impl.weight_preprocessor is not None and (
                len(impl.weight_preprocessor.split(":")) != 2
                or not all(impl.weight_preprocessor.split(":"))
            ):
                raise ValueError(
                    f"invalid weight_preprocessor entrypoint: {impl.weight_preprocessor!r}"
                )
            if impl.workspace_bytes < 0:
                raise ValueError("workspace_bytes must be nonnegative")
            if not 0 <= impl.priority < _PRIORITY_MAX:
                raise ValueError(
                    f"priority must be in [0, {_PRIORITY_MAX}), got {impl.priority}; "
                    "use a Priority band (optionally with a small intra-band offset)"
                )

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
            if impl.signature is not None:
                reasons.extend(impl.signature.matches(request.tensors))
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
