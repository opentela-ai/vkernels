"""Operator IR: tensor values, views, memory regions, and effects.

This module implements the operator-IR level described in Section 5 of
``cutedsl_megakernel_compiler_design.md``: what the model *computes*, before
any decision about how work is partitioned across persistent workers.

Design points realized here:

* A :class:`TensorValue` carries identity, shape, dtype, strides, storage
  identity and element offset — enough to express aliasing views (the tied
  language-model head reads a transposed view of the token embedding) rather
  than unrelated tensor names (§5.2).
* A :class:`Region` describes a storage object plus an index set and its
  address mapping ``addr = offset + sum_i index_i * stride_i`` (§5.2).
  Overlap testing uses the mapped bounding interval in storage space, which
  is exact for the dense/strided layouts this compiler supports and
  conservative otherwise.
* An :class:`Operator` records read/write effects over regions, so that
  read-after-write, write-after-read and write-after-write conflicts between
  any two recorded operations can be recovered (§5.2). Only read-after-write
  edges are visible in a purely functional graph; the in-place KV-cache
  append introduces the other two kinds.
* Cache validity is separate from allocation (§5.3): attention-shaped
  regions carry a *valid length* bound (a runtime expression in the decode
  position ``p``) so lowerings must emit masked loads rather than reading
  uninitialized cache tails.

The module is deliberately stdlib-only: importing the compiler package never
pulls torch, cutlass, or numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Dtypes (element-level, no torch dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DType:
    """Element type descriptor. ``size`` is the element size in bytes."""

    name: str
    size: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


F32 = DType("f32", 4)
F16 = DType("f16", 2)
BF16 = DType("bf16", 2)
F8_E4M3 = DType("f8e4m3", 1)
I32 = DType("i32", 4)
I64 = DType("i64", 8)


# ---------------------------------------------------------------------------
# Symbolic runtime scalars
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolicScalar:
    """A runtime scalar that stays dynamic through compilation (§4.2).

    The decode position ``p`` is the canonical example: it must not be
    frozen to the example value used during capture. ``lo``/``hi`` record
    the guard the host must check before every invocation.
    """

    name: str
    lo: int
    hi: int

    def guard(self, value: int) -> None:
        if not (self.lo <= value < self.hi):
            raise ValueError(f"runtime scalar {self.name!r} out of range: {value} not in [{self.lo}, {self.hi})")


@dataclass(frozen=True)
class ValidLength:
    """Number of leading valid entries along a cache axis (§5.3).

    Two forms:

    * **Scalar form** (``row_tensor is None``): ``expr`` is either a
      concrete int or the string name of a symbolic scalar (e.g.
      ``"p+1"``). One valid length for the whole batch.
    * **Per-row form** (ragged decode, floe ``forward_batch``): the valid
      length is a runtime **tensor** — an external i32 ``[B]`` array named
      by ``row_tensor``. With ``mode="positions"`` row ``b``'s valid
      length is ``pos[b] + 1``; with ``mode="lengths"`` it is the value
      ``lengths[b]`` itself. Resolution happens per task at execution —
      :meth:`resolve` deliberately does *not* collapse it to one int.

    Entries at or beyond the valid length are *uninitialized storage*:
    implementations must use masked loads or control flow that never
    reads them, because multiplying an uninitialized NaN by zero still
    produces NaN. In per-row form the mask is per row: row ``b`` never
    reads beyond its own valid length (§5.3 NaN-tail contract, per row).
    """

    expr: int | str | None = None
    row_tensor: Optional[str] = None
    mode: str = "positions"  # "positions" (len = pos[b] + 1) | "lengths"

    def __post_init__(self):
        if self.row_tensor is None:
            if self.expr is None:
                raise ValueError("ValidLength needs an int/str expr or a row_tensor")
        else:
            if self.expr is not None:
                raise ValueError("ValidLength row form takes row_tensor, not expr")
            if self.mode not in ("positions", "lengths"):
                raise ValueError(f"ValidLength row mode {self.mode!r} not supported")

    @property
    def is_row(self) -> bool:
        return self.row_tensor is not None

    @classmethod
    def from_positions(cls, tensor_name: str) -> "ValidLength":
        """Per-row lengths from an external i32 [B] decode-position tensor."""
        return cls(row_tensor=tensor_name, mode="positions")

    @classmethod
    def from_lengths(cls, tensor_name: str) -> "ValidLength":
        """Per-row lengths from an external i32 [B] cache-seqlen tensor."""
        return cls(row_tensor=tensor_name, mode="lengths")

    def resolve(self, scalars: dict[str, int]) -> int:
        if self.is_row:
            raise ValueError(
                f"ValidLength row tensor {self.row_tensor!r} resolves per row at execution, not to a single int"
            )
        if isinstance(self.expr, int):
            return self.expr
        # Support the single supported symbolic form "<name>+1".
        expr = self.expr
        add = 0
        if expr.endswith("+1"):
            expr, add = expr[:-2], 1
        return scalars[expr] + add

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.is_row:
            tail = "pos[b]+1" if self.mode == "positions" else "len[b]"
            return f"row:{self.row_tensor}[{tail}]"
        return str(self.expr)


# ---------------------------------------------------------------------------
# Tensor values and views
# ---------------------------------------------------------------------------


class TensorValue:
    """A symbolic tensor: identity plus layout (§5.1).

    Two views that alias share a ``storage_id``; their strides/offset
    describe how view indices map into the shared storage. Strides are in
    *elements*, matching device pointer arithmetic.
    """

    __slots__ = (
        "vid",
        "name",
        "shape",
        "dtype",
        "strides",
        "storage_id",
        "offset",
        "valid_length",
    )

    def __init__(
        self,
        vid: int,
        name: str,
        shape: Sequence[int],
        dtype: DType,
        strides: Sequence[int],
        storage_id: int,
        offset: int = 0,
        valid_length: Optional[ValidLength] = None,
    ):
        self.vid = vid
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.strides = tuple(strides)
        self.storage_id = storage_id
        self.offset = offset
        self.valid_length = valid_length

    # -- basic queries -----------------------------------------------------

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.size

    def is_contiguous(self) -> bool:
        """Row-major contiguity check (used to legalize reshapes)."""
        expected = 1
        for d, s in zip(reversed(self.shape), reversed(self.strides)):
            if d != 1 and s != expected:
                return False
            expected *= d
        return True

    # -- derived views -----------------------------------------------------

    def view(self, name: str, shape: Sequence[int]) -> "TensorValue":
        """Layout-exact reshape: row-major of a contiguous view, or a pure
        squeeze/keep of unit dimensions (dropping a size-1 axis is exact for
        any strides). Aliases the storage."""
        new_shape = tuple(shape)
        if _numel(self.shape) != _numel(new_shape):
            raise ValueError(f"reshape {self.shape} -> {new_shape} changes numel")
        # Squeeze path: identical non-unit extents in the same order.
        old_kept = [(d, s) for d, s in zip(self.shape, self.strides) if d != 1]
        if [d for d, _ in old_kept] == [d for d in new_shape if d != 1]:
            it = iter(old_kept)
            strides = []
            for d in new_shape:
                strides.append(next(it)[1] if d != 1 else 0)
            for i in range(len(strides) - 1, -1, -1):
                if strides[i] == 0:
                    strides[i] = strides[i + 1] if i + 1 < len(strides) else 1
            return TensorValue(
                vid=None,
                name=name,
                shape=new_shape,
                dtype=self.dtype,
                strides=tuple(strides),
                storage_id=self.storage_id,
                offset=self.offset,
                valid_length=None,
            )
        # Contiguous path.
        if not self.is_contiguous():
            raise ValueError(f"reshape of non-contiguous view {self.name!r} is unsupported")
        return TensorValue(
            vid=None,
            name=name,
            shape=new_shape,
            dtype=self.dtype,
            strides=_row_major_strides(new_shape),
            storage_id=self.storage_id,
            offset=self.offset,
            valid_length=None,
        )

    def transposed(self, name: str) -> "TensorValue":
        """Swap the last two dimensions (the tied-head pattern, §5.2)."""
        if len(self.shape) < 2:
            raise ValueError("transpose requires >= 2 dims")
        shape = self.shape[:-2] + (self.shape[-1], self.shape[-2])
        strides = self.strides[:-2] + (self.strides[-1], self.strides[-2])
        return TensorValue(
            vid=None,
            name=name,
            shape=shape,
            dtype=self.dtype,
            strides=strides,
            storage_id=self.storage_id,
            offset=self.offset,
        )

    def element_slice(self, name: str, axis: int, valid: ValidLength) -> "TensorValue":
        """A view of the valid prefix along ``axis`` (used for cache reads)."""
        return TensorValue(
            vid=None,
            name=name,
            shape=self.shape,
            dtype=self.dtype,
            strides=self.strides,
            storage_id=self.storage_id,
            offset=self.offset,
            valid_length=(valid if axis == len(self.shape) - 2 else None),
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TensorValue({self.name}, shape={self.shape}, strides={self.strides}, storage=s{self.storage_id}, offset={self.offset}, vid={self.vid})"


def _row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Memory regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A storage object plus an index set and address mapping (§5.2).

    The index set is described per *view dimension* as an interval
    ``[lo, hi)`` where ``hi`` is either an int or a symbolic bound string
    (``"p+1"``). Together with the owning view's strides/offset this gives
    the element address ``offset + sum_i idx_i * stride_i``.

    Symbolic bounds (cache valid lengths) are treated as the *full*
    allocated extent for overlap analysis — conservative, never unsound,
    and irrelevant while phase order is preserved.
    """

    storage_id: int
    view: TensorValue
    # (lo, hi) per dimension; hi may be a str symbolic expression or a
    # per-row ValidLength (row-tensor form, stored as the object itself).
    boxes: tuple[tuple[int, int | str | ValidLength], ...]
    # Paged indirection (#94): when set, the position axis is addressed
    # through an external i32 [B, S] slot table rather than a static box.
    indirect_table: Optional[TensorValue] = None
    indirect_axis: int = -1

    @staticmethod
    def whole(view: TensorValue) -> "Region":
        boxes = tuple((0, d) for d in view.shape)
        return Region(view.storage_id, view, boxes)

    @staticmethod
    def prefix(view: TensorValue, axis: int, valid: ValidLength) -> "Region":
        """Whole view, but the extent on ``axis`` is the valid prefix.

        Per-row (row-tensor) valid lengths are stored as the
        :class:`ValidLength` object itself; :meth:`resolved_boxes` treats
        them conservatively as the full extent.
        """
        boxes = []
        for i, d in enumerate(view.shape):
            if i == axis:
                boxes.append((0, valid if valid.is_row else valid.expr))
            else:
                boxes.append((0, d))
        return Region(view.storage_id, view, tuple(boxes))

    @staticmethod
    def indirect(
        view: TensorValue,
        table: TensorValue,
        axis: int,
        valid: Optional[ValidLength] = None,
    ) -> "Region":
        """Paged region (#94): ``axis`` of ``view`` is indexed through
        ``table`` — an external i32 ``[B, S]`` slot table. Element
        addresses become ``slot = table[b, t]``,
        ``addr = pool_base + slot * row_stride``; writes land at
        ``table[b, p_row]`` and slot 0 is the reserved null/sink page
        (never written by a live row).

        The static boxes keep the view extent (optionally bounded by
        ``valid`` on ``axis``), but storage-span analysis overapproximates
        to the *whole pool*: indirected regions on one pool conflict with
        everything on that pool. Sound under phase order — same reasoning
        as today's prefix regions, whose symbolic bounds are likewise
        treated as full extent (§5.2).
        """
        assert 0 <= axis < len(view.shape), f"indirect axis {axis} out of range for {view.shape}"
        boxes = []
        for i, d in enumerate(view.shape):
            boxes.append((0, valid.expr) if (i == axis and valid is not None) else (0, d))
        return Region(view.storage_id, view, tuple(boxes), indirect_table=table, indirect_axis=axis)

    @staticmethod
    def tile(view: TensorValue, tile: Sequence[tuple[int, int]]) -> "Region":
        """A sub-box given as per-dimension [lo, hi) element ranges."""
        assert len(tile) == len(view.shape)
        return Region(view.storage_id, view, tuple((lo, hi) for lo, hi in tile))

    # -- analysis ----------------------------------------------------------

    def resolved_boxes(self, scalars: Optional[dict[str, int]] = None) -> tuple[tuple[int, int], ...]:
        """Boxes with symbolic bounds resolved (defaults: full extent)."""
        out = []
        for (lo, hi), extent in zip(self.boxes, self.view.shape):
            if isinstance(hi, ValidLength):
                # Per-row form: exact extent depends on runtime row data.
                # Conservative full-extent, exactly like the symbolic case
                # — never unsound for overlap analysis.
                hi = extent
            elif isinstance(hi, str):
                hi = ValidLength(hi).resolve(scalars or {})
                hi = min(hi, extent)
            out.append((lo, hi))
        return tuple(out)

    def storage_span(self) -> tuple[int, int]:
        """Bounding [min, max] element interval in storage space.

        Exact for dense and transposed layouts; conservative in general.
        Strides must be non-negative (asserted at view creation sites we
        control).
        """
        if self.indirect_table is not None:
            # Paged (#94): addressed slots are runtime data — overapproximate
            # to the whole pool extent (conservative, never unsound).
            total = 1
            for d in self.view.shape:
                total *= d
            return (self.view.offset, self.view.offset + max(total - 1, 0))
        lo_addr = self.view.offset
        hi_addr = self.view.offset
        for (lo, hi), stride, extent in zip(self.boxes, self.view.strides, self.view.shape):
            hi_eff = min(hi, extent) if isinstance(hi, int) else extent
            candidates = (lo * stride, (hi_eff - 1) * stride) if hi_eff > lo else (0, 0)
            lo_addr += min(candidates)
            hi_addr += max(candidates)
        return (lo_addr, max(lo_addr, hi_addr))

    def overlaps(self, other: "Region") -> bool:
        if self.storage_id != other.storage_id:
            return False
        a0, a1 = self.storage_span()
        b0, b1 = other.storage_span()
        return a0 <= b1 and b0 <= a1

    def __repr__(self) -> str:  # pragma: no cover - trivial
        ind = f" via table[{self.indirect_table.name}]" if self.indirect_table is not None else ""
        return f"Region(s{self.storage_id}, {self.boxes}){ind}"


# ---------------------------------------------------------------------------
# Operators and effects
# ---------------------------------------------------------------------------

# Operator kinds in the supported subset (§3.1, §4.3). Anything outside this
# set is rejected in strict mode with a diagnostic (§3.3).
OP_EMBEDDING = "embedding"
OP_LAYER_NORM = "layer_norm"
OP_RMS_NORM = "rms_norm"
OP_ROPE = "rope"
OP_LINEAR = "linear"
# fp8-blockwise decode projection (issue #91): y = x @ dequant(w_fp8, scales)^T
# with DeepSeek-style 128x128 block scales; same task shape as ``linear``, the
# scale tensor is the second weight external.
OP_LINEAR_FP8 = "linear_fp8"
OP_GELU = "gelu"
OP_SWIGLU = "swiglu"
OP_ADD = "add"
OP_CACHE_APPEND = "cache_append"
OP_CACHE_APPEND_PAGED = "cache_append_paged"
OP_ATTENTION_SCORES = "attention_scores"
OP_ATTENTION_SCORES_PAGED = "attention_scores_paged"
OP_SOFTMAX = "softmax"
OP_ATTENTION_VALUES = "attention_values"
OP_ATTENTION_VALUES_PAGED = "attention_values_paged"

ARITHMETIC_OP_KINDS = (
    OP_EMBEDDING,
    OP_LAYER_NORM,
    OP_RMS_NORM,
    OP_ROPE,
    OP_LINEAR,
    OP_LINEAR_FP8,
    OP_GELU,
    OP_SWIGLU,
    OP_ADD,
    OP_CACHE_APPEND,
    OP_CACHE_APPEND_PAGED,
    OP_ATTENTION_SCORES,
    OP_ATTENTION_SCORES_PAGED,
    OP_SOFTMAX,
    OP_ATTENTION_VALUES,
    OP_ATTENTION_VALUES_PAGED,
)


@dataclass(frozen=True)
class Operator:
    """One recorded model operation (§5.1)."""

    opid: int
    kind: str
    inputs: tuple[str, ...]  # tensor value names
    outputs: tuple[str, ...]
    attributes: dict
    read_regions: tuple[Region, ...]
    write_regions: tuple[Region, ...]
    source_location: str  # e.g. "layer 1 mlp down projection"
    numerical_contract: dict = field(default_factory=dict)

    def reads(self) -> tuple[Region, ...]:
        return self.read_regions

    def writes(self) -> tuple[Region, ...]:
        return self.write_regions

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"#{self.opid} {self.kind}({', '.join(self.inputs)}) -> {self.outputs} @ {self.source_location}"


# Hazard kinds between two operators u (earlier) and v (later) (§5.2).
HAZARD_RAW = "RAW"
HAZARD_WAR = "WAR"
HAZARD_WAW = "WAW"


@dataclass(frozen=True)
class Hazard:
    kind: str
    producer: int  # opid of the earlier operator u
    consumer: int  # opid of the later operator v
    storage_id: int

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind}: op{self.producer} -> op{self.consumer} on storage s{self.storage_id}"


def compute_hazards(ops: Sequence[Operator]) -> list[Hazard]:
    """All pairwise RAW/WAR/WAW conflicts, in program order (§5.2).

    Whole-operator regions are used, which is the documented granularity
    for the phase-synchronous target; precise tile regions become
    necessary only when relaxing phase order (§5.2, §10).
    """
    hazards: list[Hazard] = []
    for i, u in enumerate(ops):
        for v in ops[i + 1 :]:
            # RAW: writes of u read by v
            for wu in u.write_regions:
                for rv in v.read_regions:
                    if wu.overlaps(rv):
                        hazards.append(Hazard(HAZARD_RAW, u.opid, v.opid, wu.storage_id))
            # WAR: reads of u written by v
            for ru in u.read_regions:
                for wv in v.write_regions:
                    if ru.overlaps(wv):
                        hazards.append(Hazard(HAZARD_WAR, u.opid, v.opid, ru.storage_id))
            # WAW: writes of u written by v
            for wu in u.write_regions:
                for wv in v.write_regions:
                    if wu.overlaps(wv):
                        hazards.append(Hazard(HAZARD_WAW, u.opid, v.opid, wu.storage_id))
    return _dedupe_hazards(hazards)


def _dedupe_hazards(hazards: Iterable[Hazard]) -> list[Hazard]:
    seen: set[tuple[str, int, int, int]] = set()
    out: list[Hazard] = []
    for h in hazards:
        key = (h.kind, h.producer, h.consumer, h.storage_id)
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


class OperatorGraph:
    """A captured model body: ordered operators plus tensor/scalar tables."""

    def __init__(self) -> None:
        self.tensors: dict[str, TensorValue] = {}
        self.storage_sizes: dict[int, int] = {}  # storage_id -> element count
        self.storage_names: dict[int, str] = {}
        self.scalars: dict[str, SymbolicScalar] = {}
        self.ops: list[Operator] = []
        # Storages created by the recorder (compiler-planned intermediates);
        # everything else is caller-owned (weights, ids, K/V cache).
        self.fresh_storages: set[int] = set()
        # storage_id -> version, bumped by each effectful write; expresses
        # ordering without copying the cache (§4.3).
        self.storage_versions: dict[int, int] = {}
        # Per-row decode-position tensors (issue #93): name -> declared
        # capacity. Legality scopes consumers exactly like the scalar p.
        self.row_position_tensors: set[str] = set()
        self.row_position_capacity: dict[str, int] = {}
        self._next_vid = 0
        self._next_opid = 0
        self._next_storage = 0

    # -- construction ------------------------------------------------------

    def new_storage(self, name: str, numel: int) -> int:
        sid = self._next_storage
        self._next_storage += 1
        self.storage_sizes[sid] = numel
        self.storage_names[sid] = name
        self.storage_versions[sid] = 0
        self.fresh_storages.add(sid)
        return sid

    def add_external_storage(self, sid: int, name: str, numel: int) -> None:
        """Register caller-owned storage (weights, cache, inputs)."""
        self.storage_sizes.setdefault(sid, numel)
        self.storage_names.setdefault(sid, name)
        self.storage_versions.setdefault(sid, 0)

    def add_tensor(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DType,
        *,
        storage_id: int,
        strides: Sequence[int],
        offset: int = 0,
        valid_length: Optional[ValidLength] = None,
    ) -> TensorValue:
        if name in self.tensors:
            raise ValueError(f"duplicate tensor name {name!r}")
        if any(s < 0 for s in strides):
            raise ValueError(f"negative strides are unsupported (tensor {name!r})")
        vid = self._next_vid
        self._next_vid += 1
        tv = TensorValue(vid, name, shape, dtype, strides, storage_id, offset, valid_length)
        self.tensors[name] = tv
        return tv

    def record(
        self,
        kind: str,
        *,
        inputs: Sequence[str],
        outputs: Sequence[str],
        attributes: dict,
        reads: Sequence[Region],
        writes: Sequence[Region],
        source_location: str,
        numerical_contract: Optional[dict] = None,
    ) -> Operator:
        op = Operator(
            opid=self._next_opid,
            kind=kind,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            attributes=dict(attributes),
            read_regions=tuple(reads),
            write_regions=tuple(writes),
            source_location=source_location,
            numerical_contract=dict(numerical_contract or {}),
        )
        self._next_opid += 1
        self.ops.append(op)
        for w in writes:
            self.storage_versions[w.storage_id] = self.storage_versions.get(w.storage_id, 0) + 1
        return op

    def add_scalar(self, scalar: SymbolicScalar) -> SymbolicScalar:
        self.scalars[scalar.name] = scalar
        return scalar

    # -- queries ------------------------------------------------------------

    def arithmetic_ops(self) -> list[Operator]:
        """Ops that become phases (views and no-ops excluded)."""
        return list(self.ops)

    def tensor(self, name: str) -> TensorValue:
        return self.tensors[name]

    def op_phase_count(self) -> int:
        return len(self.ops)

    def summary(self) -> str:
        lines = [f"OperatorGraph: {len(self.ops)} operators, {len(self.tensors)} tensors"]
        for op in self.ops:
            lines.append(f"  {op!r}")
        return "\n".join(lines)
