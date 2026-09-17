"""Legality: supported-subset and uniformity checks (§3.3, §12).

Strict single-kernel mode must *reject* an input when no legal device
lowering exists, with a diagnostic that identifies the unsupported node and
the missing contract — never silently call an external kernel. This module
re-checks the recorded graph after capture so the obligation does not depend
solely on the recorder's inline guards.
"""

from __future__ import annotations

from dataclasses import dataclass

from .operator_ir import ARITHMETIC_OP_KINDS, OperatorGraph

__all__ = ["Diagnostic", "LegalityError", "check_graph"]

# Operators allowed to consume the symbolic decode position (§4.2). Each has
# a documented runtime-bound contract in its lowering.
POSITION_CONSUMERS = frozenset(
    {
        "embedding",
        "cache_append",
        "compressor_append",
        "attention_scores",
        "softmax",
        "attention_values",
        "rope",
        # MLA decode (issue #95): fused latent-attention scores/values and
        # the conjugate output-side rope all bind the runtime decode position
        # (window bound t in (p-W, p], per-row rope position).
        "cache_append_paged",
        "attention_scores_paged",
        "attention_values_paged",
        "mla_scores",
        "mla_values",
        "conjugate_rope",
    }
)


@dataclass(frozen=True)
class Diagnostic:
    severity: str  # "error" | "warning" | "note"
    code: str
    message: str
    location: str = ""


class LegalityError(Exception):
    """Strict mode: the graph is outside the supported subset."""

    def __init__(self, diagnostics: list[Diagnostic]):
        lines = [f"[{d.severity}] {d.code}: {d.message} ({d.location})" for d in diagnostics]
        super().__init__("legality check failed:\n  " + "\n  ".join(lines))
        self.diagnostics = diagnostics


def check_graph(graph: OperatorGraph) -> list[Diagnostic]:
    """Return all diagnostics; callers raise on any ``error`` severity."""
    diags: list[Diagnostic] = []

    # 1. Every operator kind must be in the supported subset.
    for op in graph.ops:
        if op.kind not in ARITHMETIC_OP_KINDS:
            diags.append(
                Diagnostic(
                    "error",
                    "unsupported-operator",
                    f"operator kind {op.kind!r} has no device-callable task lowering",
                    op.source_location,
                )
            )

    # 2. Shapes must be fully static (dimensions specialize compilation, §1.3).
    for name, tv in graph.tensors.items():
        if not all(isinstance(d, int) and d > 0 for d in tv.shape):
            diags.append(Diagnostic("error", "non-static-shape", f"tensor {name!r} has non-static shape {tv.shape}", name))
        if tv.storage_id not in graph.storage_sizes:
            diags.append(Diagnostic("error", "unregistered-storage", f"tensor {name!r} references unknown storage", name))

    # 3. Symbolic scalars may be consumed only by position consumers.
    scalar_names = set(graph.scalars)
    # Per-row position tensors (issue #93) are scoped exactly like the
    # scalar: only attention/append/rope/embedding consumers may read them.
    row_position_names = getattr(graph, "row_position_tensors", set())
    for op in graph.ops:
        consumed = [v for v in op.attributes.values() if isinstance(v, str) and v in scalar_names]
        if consumed and op.kind not in POSITION_CONSUMERS:
            diags.append(
                Diagnostic(
                    "error",
                    "symbolic-scalar-consumer",
                    f"{op.kind} consumes runtime scalar(s) {consumed} without a supported bound contract",
                    op.source_location,
                )
            )
        consumed_row = [v for v in op.attributes.values() if isinstance(v, str) and v in row_position_names]
        if consumed_row and op.kind not in POSITION_CONSUMERS:
            diags.append(
                Diagnostic(
                    "error",
                    "row-position-consumer",
                    f"{op.kind} consumes per-row position tensor(s) {consumed_row} without a supported per-row bound contract",
                    op.source_location,
                )
            )

    # 4. Every operator must produce at least one effect or output.
    for op in graph.ops:
        if not op.outputs and not op.write_regions:
            diags.append(
                Diagnostic(
                    "error",
                    "effect-free-op",
                    f"{op.kind} records neither outputs nor write effects; nothing orders it",
                    op.source_location,
                )
            )

    # Notes (never errors) — documented numerical transformations (§12).
    diags.append(
        Diagnostic(
            "note",
            "numerical-contract",
            "f32 accumulation, k/t ascending reductions; reduction-order changes vs. the original standalone kernels are expected and tracked separately from schedule legality",
        )
    )
    return diags


def enforce(graph: OperatorGraph) -> None:
    diags = check_graph(graph)
    errors = [d for d in diags if d.severity == "error"]
    if errors:
        raise LegalityError(errors)
