"""Recording backend: capture the model body without executing it (§4).

Executing the model body with a :class:`RecordingBackend` constructs the
operator graph instead of computing numbers. The recorded interface mirrors
the model-level pseudocode of §4.1:

    x = ops.embedding(ids, weights.token, weights.position, position)
    n = ops.layer_norm(x, gamma, beta, eps)
    u = ops.linear(n, w, b)        # y = x @ w + b, w stored [Cin, Cout]
    u = ops.gelu(u)
    z = ops.add(x, z)
    (k', v') = ops.cache_append(k, v, qkv, position)
    s  = ops.attention_scores(q, k', position)
    p  = ops.softmax(s, position)
    a  = ops.attention_values(p, v', position)

Fidelity obligations implemented here (§4, §12 "capture fidelity"):

* The recorder returns symbolic tensors with identity, shape, dtype,
  strides, storage identity and byte offset (views keep their alias
  relationship — see :meth:`RecordingBackend.transpose_view`).
* Runtime scalars stay symbolic: the decode position is guarded, never
  frozen to the example value used during capture (§4.2).
* Mutation is captured through storage effects: ``cache_append`` records a
  write effect on the K/V storages and bumps their versions; the *real*
  cache arrays passed by the caller are never touched, and no cache length
  bookkeeping advances (§4.3). Consumers read the post-append version.
* A Python branch on a symbolic value fails loudly rather than silently
  recording one execution path (§4.2).
* Unknown operations are rejected with a diagnostic naming the node and
  the missing contract (§3.3).
"""

from __future__ import annotations

from typing import Optional, Sequence

from .operator_ir import (
    ARITHMETIC_OP_KINDS,
    DType,
    F32,
    I32,
    OperatorGraph,
    Region,
    SymbolicScalar,
    ValidLength,
    _row_major_strides,
)

__all__ = [
    "CaptureError",
    "UnsupportedOperator",
    "SymbolicTensor",
    "RecordingBackend",
    "capture_model",
]






# Gated-RMSNorm activations in the supported subset (issue #100). GLM's
# o_norm uses sigmoid (floe Glm53RMSNormGated); silu is deliberately NOT
# here — vkernels' kda_layer_norm_gated hardcodes silu and does not cover
# this variant, which is exactly why the op exists.
_RMS_GATED_ACTIVATIONS = ("sigmoid",)






from .capture_types import CaptureError as CaptureError, UnsupportedOperator as UnsupportedOperator, SymbolicTensor as SymbolicTensor
from .capture_dense import DenseRecording
from .capture_attention import AttentionRecording
from .capture_recurrent import RecurrentRecording

class RecordingBackend(DenseRecording, AttentionRecording, RecurrentRecording):
    """The ``ops`` interface of §4.1, implemented as a graph recorder."""

    def __init__(self, graph: Optional[OperatorGraph] = None):
        self.graph = graph if graph is not None else OperatorGraph()
        self._op_counter = 0
        # Dedupe registry: registering the same external tensor twice returns
        # the same symbolic handle (model bodies re-request parameters).
        self._externals: dict[str, SymbolicTensor] = {}

    def __getattr__(self, name):
        """§3.3: an unknown op is an explicit unsupported-operator diagnostic,
        not a silent AttributeError buried in capture."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise UnsupportedOperator(
            f"backend has no operation {name!r}; it is outside the supported subset",
            node=name,
            missing_contract="a device-callable task lowering with modeled memory effects (§3.3)",
        )

    # ------------------------------------------------------------------
    # Scalar + external tensor plumbing
    # ------------------------------------------------------------------

    def define_position(self, capacity: int) -> SymbolicScalar:
        """The decode position p with the host-side guard 0 <= p < S."""
        return self.graph.add_scalar(SymbolicScalar("p", 0, capacity))

    def define_row_positions(
        self,
        name: str,
        batch: int,
        capacity: int,
        *,
        storage_id: int,
    ) -> SymbolicTensor:
        """Per-row decode positions for ragged batches (issue #93).

        Registers an external **i32 [B]** tensor of per-row decode
        positions (host tensors may be i64; they narrow on upload). Each
        row ``b`` then carries its own valid length ``pos[b] + 1``: ops
        that accept this tensor as their ``position`` record per-row
        :class:`ValidLength` regions instead of the shared scalar form
        ``p+1``, and legality scopes it exactly like the scalar ``p`` —
        only attention/append/rope/embedding consumers may read it.
        """
        if not (isinstance(batch, int) and batch > 0):
            raise CaptureError(f"row-position tensor needs a static batch > 0; got {batch!r}")
        if capacity <= 0:
            raise CaptureError(f"row-position tensor needs capacity > 0; got {capacity!r}")
        sym = self.external_tensor(name, (batch,), I32, storage_id=storage_id)
        self.graph.row_position_tensors.add(name)
        self.graph.row_position_capacity[name] = capacity
        return sym

    def external_tensor(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DType = F32,
        *,
        storage_id: int,
        strides: Optional[Sequence[int]] = None,
        offset: int = 0,
        register_storage: bool = True,
    ) -> SymbolicTensor:
        """Register caller-owned storage (weights, ids, K/V cache)."""
        if name in self._externals:
            existing = self._externals[name]
            if existing.value.shape != tuple(shape) or existing.value.storage_id != storage_id:
                raise ValueError(f"external tensor {name!r} re-registered with different identity")
            return existing
        strides = strides if strides is not None else _row_major_strides(shape)
        if register_storage:
            numel = 1
            for d in shape:
                numel *= d
            self.graph.add_external_storage(storage_id, name, numel)
        tv = self.graph.add_tensor(name, shape, dtype, storage_id=storage_id, strides=strides, offset=offset)
        sym = SymbolicTensor(tv)
        self._externals[name] = sym
        return sym

    def fresh_buffer(self, name: str, shape: Sequence[int], dtype: DType = F32) -> SymbolicTensor:
        """A compiler-planned intermediate (workspace candidate)."""
        numel = 1
        for d in shape:
            numel *= d
        sid = self.graph.new_storage(name, numel)
        tv = self.graph.add_tensor(name, shape, dtype, storage_id=sid, strides=_row_major_strides(shape))
        return SymbolicTensor(tv)

    def view_of(self, base: SymbolicTensor, name: str, shape: Sequence[int]) -> SymbolicTensor:
        """Row-major reshape aliasing the base storage."""
        tv = base.value.view(name, shape)
        tv = self.graph.add_tensor(name, tv.shape, tv.dtype, storage_id=tv.storage_id, strides=tv.strides, offset=tv.offset)
        return SymbolicTensor(tv)

    def narrow(self, base: SymbolicTensor, name: str, *, axis: int, start: int, length: int) -> SymbolicTensor:
        """Slice ``length`` entries along ``axis`` starting at ``start``.

        Used to take per-layer K/V views out of the packed [L, B, H, S, D]
        cache storages; aliases the storage like any other view.
        """
        bv = base.value
        if not (0 <= start and start + length <= bv.shape[axis]):
            raise ValueError(f"narrow out of bounds: axis {axis} [{start}, {start + length}) of {bv.shape}")
        shape = bv.shape[:axis] + (length,) + bv.shape[axis + 1 :]
        offset = bv.offset + start * bv.strides[axis]
        tv = self.graph.add_tensor(name, shape, bv.dtype, storage_id=bv.storage_id, strides=bv.strides, offset=offset)
        return SymbolicTensor(tv)

    def transpose_view(self, w: SymbolicTensor, name: Optional[str] = None) -> SymbolicTensor:
        """Transposed alias used by the tied language-model head (§5.2)."""
        name = name or f"{w.value.name}.T"
        tv = w.value.transposed(name)
        tv = self.graph.add_tensor(name, tv.shape, tv.dtype, storage_id=tv.storage_id, strides=tv.strides, offset=tv.offset)
        return SymbolicTensor(tv)

    # ------------------------------------------------------------------
    # Recorded operations
    # ------------------------------------------------------------------

    def _record(
        self,
        kind: str,
        *,
        inputs: Sequence[SymbolicTensor | str],
        outputs: Sequence[SymbolicTensor],
        attributes: dict,
        reads: Sequence[Region],
        writes: Sequence[Region],
        source_location: str = "",
        numerical_contract: Optional[dict] = None,
    ):
        if kind not in ARITHMETIC_OP_KINDS:
            raise UnsupportedOperator(
                f"operator kind {kind!r} is outside the supported subset",
                node=source_location or kind,
                missing_contract="a device-callable task lowering with modeled memory effects",
            )
        self._op_counter += 1
        loc = source_location or f"op#{self._op_counter} {kind}"
        input_names = tuple(x.value.name if isinstance(x, SymbolicTensor) else str(x) for x in inputs)
        output_names = tuple(o.value.name for o in outputs)
        return self.graph.record(
            kind,
            inputs=input_names,
            outputs=output_names,
            attributes=attributes,
            reads=reads,
            writes=writes,
            source_location=loc,
            numerical_contract=numerical_contract,
        )






















    # ------------------------------------------------------------------
    # Paged decode (#94): pools addressed through external slot tables
    # ------------------------------------------------------------------







    # ------------------------------------------------------------------
    # MoE decode (issue #98)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _require_position(self, position, op: str) -> None:
        """Accepts the symbolic scalar ``p`` or a per-row positions tensor
        registered via :meth:`define_row_positions` (issue #93)."""
        if isinstance(position, SymbolicTensor):
            if position.name not in self.graph.row_position_tensors:
                raise CaptureError(
                    f"{op} accepts a per-row position tensor only when registered via define_row_positions; got {position!r}"
                )
            if position.value.dtype != I32 or len(position.value.shape) != 1:
                raise CaptureError(f"per-row positions must be an i32 [B] tensor; got {position.value.shape} {position.value.dtype.name}")
            return
        if not isinstance(position, SymbolicScalar):
            raise CaptureError(f"{op} requires the symbolic decode position; got {position!r}. A concrete position would freeze the cache length into the graph (§4.2).")
        if position.name not in self.graph.scalars:
            raise CaptureError(f"unknown symbolic scalar {position.name!r}")

    @staticmethod
    def _position_name(position) -> str:
        assert isinstance(position, (SymbolicScalar, SymbolicTensor))
        return position.name

    def _valid_plus_one(self, position, op: str) -> ValidLength:
        """The valid cache length implied by ``position``: scalar ``p+1`` or
        the per-row form ``pos[b]+1`` (issue #93)."""
        self._require_position(position, op)
        p = self._position_name(position)
        if isinstance(position, SymbolicTensor):
            return ValidLength.from_positions(p)
        return ValidLength(f"{p}+1")

    def _suffix(self) -> str:
        return f"_t{self._op_counter:02d}"


def capture_model(
    build_forward,
    *args,
    backend: Optional[RecordingBackend] = None,
    **kwargs,
) -> tuple[OperatorGraph, RecordingBackend]:
    """Run ``build_forward(ops, ...)`` under the recording backend (§4.1).

    Returns the recorded :class:`OperatorGraph` and the backend (useful for
    querying symbolic scalars). The real cache passed by the caller is never
    mutated: the model body works against recorded views only.
    """
    recorder = backend or RecordingBackend()
    try:
        build_forward(recorder, *args, **kwargs)
    except CaptureError:
        raise
    except Exception as exc:  # surface capture failures with context
        raise CaptureError(f"model body raised {type(exc).__name__}: {exc}") from exc
    return recorder.graph, recorder
