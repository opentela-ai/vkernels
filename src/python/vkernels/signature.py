"""Typed tensor-format signatures for registry gating.

A :class:`FormatSignature` declares, per argument *role*, which storage
dtypes (and, later, layouts like paged KV) a kernel accepts. Registrations
that carry one get signature-based filtering in addition to their
``check`` predicates, so the dtype/format contract is data — explainable,
testable, and reusable — instead of prose buried in predicate code.

Borrowed from tokenspeed-kernel's ``signature.py`` vocabulary (roles +
dtype sets), scaled down to this registry's metadata-only design: no torch
imports, dtype names are the registry's canonical strings
(``"bfloat16"``, ``"float8_e4m3fn"``, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkernels.registry import TensorMetadata

__all__ = ["FormatSignature"]


@dataclass(frozen=True)
class FormatSignature:
    """Per-role dtype contract for one registration.

    Args:
        roles: Argument names in positional order (``("x", "weights")``).
        dtypes: Accepted dtype-name sets per role, aligned with ``roles``.
        layout: Reserved format discriminator (``"dense"`` today; ``"paged"``
            and friends join as kernels that care appear).

    ``matches(tensors)`` returns rejection *reasons* (empty tuple = match)
    in the registry's vocabulary, so :meth:`KernelRegistry.explain` can
    surface them exactly like ``check`` findings.
    """

    roles: tuple[str, ...]
    dtypes: tuple[frozenset[str], ...]
    layout: str = "dense"

    def __post_init__(self) -> None:
        if len(self.roles) != len(self.dtypes):
            raise ValueError(
                f"roles and dtypes must align: {len(self.roles)} roles, "
                f"{len(self.dtypes)} dtype sets"
            )
        for role, accepted in zip(self.roles, self.dtypes):
            if not accepted:
                raise ValueError(f"role {role!r} accepts no dtype")

    def matches(self, tensors: "tuple[TensorMetadata, ...]") -> tuple[str, ...]:
        """Rejection reasons for ``tensors``; empty when the signature fits."""
        if len(tensors) != len(self.roles):
            return (
                f"expected {len(self.roles)} tensors "
                f"({', '.join(self.roles)}), got {len(tensors)}",
            )
        reasons = []
        for role, accepted, tensor in zip(self.roles, self.dtypes, tensors):
            if tensor.dtype not in accepted:
                reasons.append(f"{role} requires {'/'.join(sorted(accepted))}")
        return tuple(reasons)

    def describe(self) -> str:
        """One-line rendering for ``vkl info``-style tooling."""
        parts = [f"{role}<{'/'.join(sorted(accepted))}>" for role, accepted in zip(self.roles, self.dtypes)]
        return f"{self.layout}: " + " ".join(parts)
