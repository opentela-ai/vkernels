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
    TensorValue,
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


class RecordingBackend:
    """The ``ops`` interface of §4.1, implemented as a graph recorder."""

    def __init__(self, graph: Optional[OperatorGraph] = None):
        self.graph = graph if graph is not None else OperatorGraph()
        # Live-version tracking per storage, so cache appends produce
        # ordered read/write versions (§4.3).
        self._storage_views: dict[int, str] = {}
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

    def embedding(
        self,
        ids: SymbolicTensor,
        token: SymbolicTensor,
        position_emb: Optional[SymbolicTensor],
        position,
        *,
        out: Optional[SymbolicTensor] = None,
    ) -> SymbolicTensor:
        """Token-row lookup, plus the learned position row for GPT-2-style
        embeddings. ``position_emb=None`` for Qwen3 (position enters via RoPE
        only); the symbolic position is still required to keep the op's
        runtime-bound contract explicit."""
        self._require_position(position, "embedding")
        out = out or self.fresh_buffer(f"hidden{self._suffix()}", (ids.value.shape[0], token.value.shape[1]))
        p = self._position_name(position)
        position_form = "row" if isinstance(position, SymbolicTensor) else "scalar"
        reads = [_regional_reads(token.value), _regional_reads(ids.value)]
        inputs = [ids, token]
        if position_emb is not None:
            # Position row p is symbolic: whole table read, conservative.
            reads.append(_regional_reads(position_emb.value))
            inputs.append(position_emb)
        self._record(
            "embedding",
            inputs=tuple(inputs),
            outputs=(out,),
            attributes={"position": p, "position_form": position_form, "pos_table": position_emb is not None},
            reads=tuple(reads),
            writes=(_regional_reads(out.value),),
            source_location="embedding lookup",
            numerical_contract={"sum": "token_row(ids[b]) + position_row(pos[b])" if position_emb is not None else "token_row(ids[b])", "dtype": "f32"},
        )
        return out

    def layer_norm(
        self,
        x: SymbolicTensor,
        gamma: SymbolicTensor,
        beta: SymbolicTensor,
        eps: float,
        *,
        out: Optional[SymbolicTensor] = None,
        name: str = "ln",
    ) -> SymbolicTensor:
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", x.value.shape)
        self._record(
            "layer_norm",
            inputs=(x, gamma, beta),
            outputs=(out,),
            attributes={"eps": eps},
            reads=(_regional_reads(x.value), _regional_reads(gamma.value), _regional_reads(beta.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "mean/var": "row-wise, var = mean((x-mean)^2)",
                "eps": "inside sqrt",
                "reduction": "row width C, single task per row",
            },
        )
        return out

    def linear(
        self,
        x: SymbolicTensor,
        w: SymbolicTensor,
        b: Optional[SymbolicTensor] = None,
        *,
        out: Optional[SymbolicTensor] = None,
        name: str = "linear",
        grouped_heads: Optional[int] = None,
    ) -> SymbolicTensor:
        """y = x @ w (+ b); ``w`` is stored [Cin, Cout] (§3.1).

        ``grouped_heads=H`` (issue #95, DeepSeek-V4 ``GroupedLinear``):
        block-diagonal per-head projection — ``x`` is [B, H*K_g], ``w``
        stays [H*K_g, H*N_g] = [Cin, Cout] with only the per-head
        diagonal blocks ``w[h*K_g:(h+1)*K_g, h*N_g:(h+1)*N_g]`` ever
        read (off-block storage is never touched), and ``y[b, h*N_g +
        n] = <w[h*K_g:(h+1)*K_g, h*N_g + n], x[b, h*K_g:(h+1)*K_g]>``.
        Requires H | K and H | N.
        """
        m = x.value.shape[0]
        n = w.value.shape[1]
        if len(x.value.shape) > 2:
            # 3D activations [B, H, K_g] (e.g. a conjugate-rope context) are
            # flattened row-major for the block-diagonal projection; the
            # flattened layout is exactly [B, H*K_g].
            k = 1
            for dim in x.value.shape[1:]:
                k *= dim
            x = self.view_of(x, f"{name}_flat{self._suffix()}", (m, k))
        else:
            k = x.value.shape[1]
        if grouped_heads is not None:
            if grouped_heads < 1 or k % grouped_heads or n % grouped_heads:
                raise ValueError(
                    f"grouped linear requires grouped_heads to divide K={k} and N={n}; got H={grouped_heads}"
                )
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", (m, n))
        reads = [_regional_reads(x.value), _regional_reads(w.value)]
        inputs = [x, w]
        if b is not None:
            reads.append(_regional_reads(b.value))
            inputs.append(b)
        attributes = {"bias": b is not None}
        if grouped_heads is not None:
            attributes["grouped_heads"] = grouped_heads
        self._record(
            "linear",
            inputs=tuple(inputs),
            outputs=(out,),
            attributes=attributes,
            reads=tuple(reads),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "accumulation": "f32, full-K reduction per output tile, k ascending",
                "bias": "added after the K reduction" if b is not None else "none",
            },
        )
        return out

    def linear_fp8(
        self,
        x: SymbolicTensor,
        w: SymbolicTensor,
        scale: SymbolicTensor,
        *,
        out: Optional[SymbolicTensor] = None,
        name: str = "linear_fp8",
        quant_block: int = 128,
    ) -> SymbolicTensor:
        """fp8-blockwise decode projection (issue #91):

            y = x @ dequant(w_fp8, scale)^T

        ``w`` is stored [N, K] row-major (the checkpoint's ``nn.Linear``
        layout) in fp8 e4m3; ``scale`` is the second weight external, fp32
        [ceil(N/quant_block), K/quant_block] in block-major order
        (DeepSeek-style 128x128 block-FP8; ragged trailing N block
        allowed). Numerics: fp32 accumulation over K, dequant
        in-register per block — the scale for one block multiplies the
        products of that block only.
        """
        m, k = x.value.shape
        n, k_w = w.value.shape
        if k_w != k:
            raise ValueError(f"linear_fp8 weight contraction mismatch: x K={k}, w K={k_w}")
        if k % quant_block:
            raise ValueError(
                f"linear_fp8 requires K divisible by the {quant_block}-wide quant block; got K={k}"
            )
        if n % 16:
            raise ValueError(
                f"linear_fp8 requires N divisible by the 16-wide output tile (ragged trailing scale block allowed); got N={n}"
            )
        expected_scale = (-(-n // quant_block), k // quant_block)  # ceil rows for ragged N
        if tuple(scale.value.shape) != expected_scale:
            raise ValueError(
                f"linear_fp8 scale shape {tuple(scale.value.shape)} != {expected_scale} (block-major [N/{quant_block}, K/{quant_block}])"
            )
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", (m, n))
        self._record(
            "linear_fp8",
            inputs=(x, w, scale),
            outputs=(out,),
            attributes={"weight_layout": "fp8_block", "quant_block": quant_block, "bias": False},
            reads=(_regional_reads(x.value), _regional_reads(w.value), _regional_reads(scale.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "dequant": f"per {quant_block}x{quant_block} block: w_fp8 * scale (fp32 scale, e4m3 weights)",
                "accumulation": "f32, full-K reduction per output tile, k ascending within each block",
                "bias": "none (checkpoint fp8 projections are bias-free)",
            },
        )
        return out

    def indexer_scores(
        self,
        q: SymbolicTensor,
        entries: SymbolicTensor,
        mix_w: SymbolicTensor,
        *,
        out: Optional[SymbolicTensor] = None,
        name: str = "indexer_scores",
    ) -> SymbolicTensor:
        """Lightning-indexer scoring over the compressed entries (issue #97).

        Per floe ``IndexerScorer``/``LightningIndexer`` (deepseek_v4) and
        ``Glm53Indexer`` (glm53flash), for a single decode token per batch
        row:

            scores[b, h, j] = relu(<q[b, h, :], c[b, j, :]>) * head_dim**-0.5
            s[b, j]         = sum_h scores[b, h, j] * mix_w[b, h]

        ``q`` is the indexer query [B, H, D], ``entries`` the compressed
        token pool [B, M, D] (M = capacity, an upper bound — the per-row
        valid candidate count masks it at selection time), and ``mix_w``
        the per-head mixing weights [B, H] (``weights_proj(x) * n_heads**-0.5``
        computed upstream). Activation is ReLU, not softmax. Scoring and
        mixing accumulate in f32 (the q/entries may be stored bf16, ``.cg``
        streamed on device).
        """
        b, h, d = q.value.shape
        b_c, m, d_c = entries.value.shape
        if (b_c, d_c) != (b, d):
            raise ValueError(
                f"indexer_scores shape mismatch: q {q.value.shape} vs entries {entries.value.shape} (shared B and head_dim required)"
            )
        if tuple(mix_w.value.shape) != (b, h):
            raise ValueError(
                f"indexer_scores mix weights must be [B, H] = {(b, h)}, got {tuple(mix_w.value.shape)}"
            )
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", (b, m))
        self._record(
            "indexer_scores",
            inputs=(q, entries, mix_w),
            outputs=(out,),
            attributes={"heads": h, "head_dim": d, "capacity": m, "scale": d ** -0.5, "activation": "relu"},
            reads=(_regional_reads(q.value), _regional_reads(entries.value), _regional_reads(mix_w.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "scale": f"1/sqrt(head_dim) = {d ** -0.5}",
                "activation": "relu (elementwise, after the per-head dot, before mixing)",
                "accumulation": "f32 dot over head_dim, then f32 weighted sum over heads ascending",
                "validity": "the full capacity M is scored; row masking happens in index_topk via the per-row valid candidate count",
            },
        )
        return out

    def index_topk(
        self,
        scores: SymbolicTensor,
        valid_counts: SymbolicTensor,
        *,
        k: int,
        idx: Optional[SymbolicTensor] = None,
        bias: Optional[SymbolicTensor] = None,
        name: str = "index_topk",
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Fixed-count top-k selection over the fused indexer scores (issue
        #97). Produces the static-count indirection table the attention
        scores/values tasks consume via #94: the output count is ``k`` at
        decode even though selection is data-dependent.

        ``scores`` is the fused score [B, M] (f32); ``valid_counts`` is the
        per-row number of valid candidates [B] (i32) — candidates at or
        beyond a row's valid count are uninitialized storage and are never
        read (masked loads, §5.3; multiplying an uninitialized NaN by zero
        still produces NaN, so the mask is a hard exclude).

        Semantics (deterministic):

        * candidates are ordered by descending score, ties resolved to the
          *lowest* candidate index j (``tie_break="lowest_index"``);
        * a candidate whose score is NaN inside the valid prefix is
          excluded from selection (canary semantics — corrupt scores must
          not win);
        * the first ``min(k, valid_counts[b])`` slots of ``idx`` receive
          the selected candidate indices (i32); slots beyond a row's valid
          count are filled with ``-1`` (no candidate);
        * ``bias`` carries the selected rows' scores normalized by the
          L2 norm of the row's valid scores (the block_bias consumed by
          #95/#96), with 0.0 in the ``-1`` slots.

        k must satisfy 1 <= k <= M; rows may have any valid count in
        [0, M] (ragged candidate counts, including k > valid_count).
        """
        b, m = scores.value.shape
        if scores.value.dtype != F32:
            raise ValueError(f"index_topk scores must be f32, got {scores.value.dtype}")
        if tuple(valid_counts.value.shape) != (b,) or valid_counts.value.dtype != I32:
            raise ValueError(
                f"index_topk valid_counts must be i32 [B] = {(b,)}, got {valid_counts.value.shape}/{valid_counts.value.dtype}"
            )
        if not isinstance(k, int) or k < 1 or k > m:
            raise ValueError(f"index_topk requires 1 <= k <= capacity M={m}; got k={k!r}")
        idx = idx or self.fresh_buffer(f"{name}_idx{self._suffix()}", (b, k), I32)
        bias = bias or self.fresh_buffer(f"{name}_bias{self._suffix()}", (b, k), F32)
        self._record(
            "index_topk",
            inputs=(scores, valid_counts),
            outputs=(idx, bias),
            attributes={
                "k": k,
                "capacity": m,
                "tie_break": "lowest_index",
                "score_dtype": "f32",
                "index_dtype": "i32",
                "nan_policy": "exclude",
                "pad": "idx=-1, bias=0.0 beyond a row's valid count",
            },
            reads=(_regional_reads(scores.value), _regional_reads(valid_counts.value)),
            writes=(_regional_reads(idx.value), _regional_reads(bias.value)),
            source_location=name,
            numerical_contract={
                "selection": f"descending score, ties to the lowest candidate index; top-{k}",
                "normalization": "bias = s_j / ||s_valid||_2 per row (fp32)",
                "masking": "candidates j >= valid_counts[b] are never observed (uninitialized storage)",
                "ordering": "one task per batch row; the consumer (#95/#96) is a later phase (RAW on the indirection table)",
            },
        )
        return idx, bias

    def gelu(self, x: SymbolicTensor, *, out: Optional[SymbolicTensor] = None, name: str = "gelu") -> SymbolicTensor:
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", x.value.shape)
        self._record(
            "gelu",
            inputs=(x,),
            outputs=(out,),
            attributes={"approximation": "tanh"},
            reads=(_regional_reads(x.value),),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={"formula": "0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))"},
        )
        return out

    def add(self, a: SymbolicTensor, b: SymbolicTensor, *, name: str = "add") -> SymbolicTensor:
        out = self.fresh_buffer(f"{name}{self._suffix()}", a.value.shape)
        self._record(
            "add",
            inputs=(a, b),
            outputs=(out,),
            attributes={},
            reads=(_regional_reads(a.value), _regional_reads(b.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={"formula": "a + b elementwise"},
        )
        return out

    def rms_norm(
        self,
        x: SymbolicTensor,
        gamma: SymbolicTensor,
        eps: float,
        *,
        out: Optional[SymbolicTensor] = None,
        name: str = "rms",
    ) -> SymbolicTensor:
        """RMSNorm over the last dim (no mean subtraction, no bias).

        Works on [B, C] hidden states (one task per row) and on [B, H, D]
        head views (one task per head row — Qwen3's q_norm/k_norm).
        """
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", x.value.shape)
        self._record(
            "rms_norm",
            inputs=(x, gamma),
            outputs=(out,),
            attributes={"eps": eps},
            reads=(_regional_reads(x.value), _regional_reads(gamma.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "formula": "x * rsqrt(mean(x^2) + eps) * gamma",
                "upcast": "statistics accumulated in fp32",
            },
        )
        return out

    def rope(
        self,
        x: SymbolicTensor,
        cos_table: SymbolicTensor,
        sin_table: SymbolicTensor,
        position,
        *,
        layer: int,
        which: str = "q",
        rotary_dim: Optional[int] = None,
        convention: str = "rotate_half",
    ) -> SymbolicTensor:
        """RoPE at the runtime position p.

        ``convention="rotate_half"`` (default — Qwen3): full-width
        rotate-half form ``x' = x*cos[p] + rotate_half(x)*sin[p]`` where
        rotate_half(x) = cat(-x[D/2:], x[:D/2]). Tables are precomputed
        [max_positions, D] externals with cos = cat([f, f]).

        ``convention="interleaved"`` (DeepSeek-V4 / GPT-J style, issue #95):
        adjacent-pair rotation over the first ``rotary_dim`` dims — pairs
        ``(2i, 2i+1)``::

            x[2i]'   = x[2i]*c_i - x[2i+1]*s_i
            x[2i+1]' = x[2i+1]*c_i + x[2i]*s_i      (c_i, s_i = tables[p, i])

        and dims ``[rotary_dim, head_dim)`` pass through unchanged. Tables
        are fp32 externals of shape [max_positions, rotary_dim // 2].

        ``convention="neox_partial"`` (Qwen3.5 ``PartialRotaryEmbedding``):
        NeoX split-half over the first ``rotary_dim`` dims only — with
        ``half = rotary_dim // 2``, ``x1 = x[:half]``, ``x2 = x[half:rotary_dim]``::

            x1' = x1*c - x2*s ;  x2' = x2*c + x1*s      (c,s = tables[p])

        and dims ``[rotary_dim, head_dim)`` pass through unchanged. Tables
        are fp32 externals of shape [max_positions, rotary_dim // 2].
        """
        if convention not in ("rotate_half", "neox_partial", "interleaved"):
            raise CaptureError(f"rope convention {convention!r} not supported (expected 'rotate_half', 'neox_partial' or 'interleaved')")
        if convention in ("neox_partial", "interleaved"):
            head_dim = x.value.shape[-1]
            if rotary_dim is None or rotary_dim % 2 != 0 or not 0 < rotary_dim <= head_dim:
                raise CaptureError(f"neox_partial rope requires an even 0 < rotary_dim <= head_dim ({head_dim}); got {rotary_dim!r}")
            for t in (cos_table, sin_table):
                if t.value.shape[-1] != rotary_dim // 2:
                    raise CaptureError(
                        f"neox_partial rope tables must be [max_positions, rotary_dim//2] = [.., {rotary_dim // 2}]; got shape {t.value.shape}"
                    )
        self._require_position(position, "rope")
        out = self.fresh_buffer(f"rope_{which}_l{layer}{self._suffix()}", x.value.shape)
        p = self._position_name(position)
        attributes = {"layer": layer, "which": which, "position": p, "convention": convention,
                      "position_form": "row" if isinstance(position, SymbolicTensor) else "scalar"}
        if rotary_dim is not None:
            attributes["rotary_dim"] = rotary_dim
        if convention == "rotate_half":
            numerical_contract = {
                "rotate_half": "cat(-x[D/2:], x[:D/2])",
                "tables": "full-width cos/sin (HF rotate_half convention)",
            }
        elif convention == "interleaved":
            numerical_contract = {
                "interleaved": "x[2i]' = x[2i]*c_i - x[2i+1]*s_i ; x[2i+1]' = x[2i+1]*c_i + x[2i]*s_i over pairs i < rotary_dim//2",
                "pass_through": f"dims [{rotary_dim}, {x.value.shape[-1]}) unchanged",
                "tables": "cos/sin [max_positions, rotary_dim//2], fp32, indexed at the runtime position (pair stride 2)",
                "upcast": "rotation math in fp32 (bf16 activations in/out)",
            }
        else:
            numerical_contract = {
                "neox_partial": "x1' = x1*c - x2*s ; x2' = x2*c + x1*s over dims [0, rotary_dim), half = rotary_dim//2",
                "pass_through": f"dims [{rotary_dim}, {x.value.shape[-1]}) unchanged",
                "tables": "cos/sin [max_positions, rotary_dim//2], fp32, indexed at the runtime position",
                "upcast": "rotation math in fp32 (bf16 activations in/out)",
            }
        self._record(
            "rope",
            inputs=(x, cos_table, sin_table),
            outputs=(out,),
            attributes=attributes,
            reads=(_regional_reads(x.value), _regional_reads(cos_table.value), _regional_reads(sin_table.value)),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} rope {which}",
            numerical_contract=numerical_contract,
        )
        return out

    def swiglu(
        self,
        gate: SymbolicTensor,
        up: SymbolicTensor,
        *,
        out: Optional[SymbolicTensor] = None,
        name: str = "swiglu",
    ) -> SymbolicTensor:
        """out = silu(gate) * up over a [B, F] activation shard (Qwen3 MLP)."""
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", gate.value.shape)
        self._record(
            "swiglu",
            inputs=(gate, up),
            outputs=(out,),
            attributes={},
            reads=(_regional_reads(gate.value), _regional_reads(up.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={"formula": "gate * sigmoid(gate) * up"},
        )
        return out

    def cache_append(
        self,
        k_cache: SymbolicTensor,
        v_cache: SymbolicTensor,
        k_new: SymbolicTensor,
        v_new: SymbolicTensor,
        position,
        *,
        layer: int,
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Append the new K/V rows at position p (§4.3).

        Records write effects on the cache storages and returns *post-append
        views* — same storage, bumped version — without touching real cache
        storage. Consumers must use the returned views.
        """
        valid = self._valid_plus_one(position, "cache_append")
        p = self._position_name(position)
        writes = [
            Region.prefix(k_cache.value, axis=2, valid=valid),
            Region.prefix(v_cache.value, axis=2, valid=valid),
        ]
        position_form = "row" if valid.is_row else "scalar"
        self._record(
            "cache_append",
            inputs=(k_cache, v_cache, k_new, v_new),
            outputs=(),  # mutation through storage effects; views returned below
            attributes={"layer": layer, "position": p, "position_form": position_form},
            reads=(_regional_reads(k_new.value), _regional_reads(v_new.value)),
            writes=tuple(writes),
            source_location=f"layer {layer} kv cache append",
            numerical_contract={
                "write": "K[l,b,h,pos[b],:] = k_new[b,h,:]; same for V",
                "valid_length": str(valid),
            },
        )
        # Post-append versions: same storages, prefix now valid to pos[b]+1.
        k_post = self.graph.add_tensor(
            f"k_cache_l{layer}_v{self.graph.storage_versions[k_cache.value.storage_id]}",
            k_cache.value.shape,
            k_cache.value.dtype,
            storage_id=k_cache.value.storage_id,
            strides=k_cache.value.strides,
            offset=k_cache.value.offset,
            valid_length=valid,
        )
        v_post = self.graph.add_tensor(
            f"v_cache_l{layer}_v{self.graph.storage_versions[v_cache.value.storage_id]}",
            v_cache.value.shape,
            v_cache.value.dtype,
            storage_id=v_cache.value.storage_id,
            strides=v_cache.value.strides,
            offset=v_cache.value.offset,
            valid_length=valid,
        )
        return SymbolicTensor(k_post), SymbolicTensor(v_post)

    def gdn_conv(
        self,
        conv_state: SymbolicTensor,
        x: SymbolicTensor,
        w: SymbolicTensor,
        *,
        layer: int,
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """GDN short-conv decode step (Qwen3.5 ``GatedDeltaNet``, seq==1 path).

        Causal depthwise conv1d over the packed qkv row, folding the
        persistent conv state::

            full = cat(conv_state[b], x[b])      # [K, C]
            out[b] = silu(sum_k full[:, k] * w[:, k])   # depthwise FIR
            conv_state'[b] = full[1:]            # time-major shift

        ``conv_state`` is an external caller-owned pool [B, K-1, C], fp32,
        time-major (floe's eager init uses the embed dtype — the compiled
        pool is fp32; oracle tolerance ~1e-3). ``x`` is the mixed qkv row
        [B, C] and ``w`` the FIR weights [C, K] (grouped conv, one tap
        vector per channel).

        The op is position-independent: it consumes no decode-position
        scalar, and its ordering obligation is the read-modify-write on
        the state pool (RAW/WAR/WAW hazards vs any other op touching that
        storage). Records write effects on the pool and returns
        ``(out, conv_state_post)`` — the post-step state view (same
        storage, bumped version) that later layers must read.
        """
        sv, xv, wv = conv_state.value, x.value, w.value
        if len(sv.shape) != 3:
            raise CaptureError(f"gdn_conv state pool must be [B, K-1, C]; got shape {sv.shape}")
        if len(xv.shape) != 2:
            raise CaptureError(f"gdn_conv input row must be [B, C]; got shape {xv.shape}")
        B, Km1, C = sv.shape
        K = Km1 + 1
        if xv.shape != (B, C):
            raise CaptureError(f"gdn_conv input row shape {xv.shape} does not match state pool [B, C] = {(B, C)}")
        if tuple(wv.shape) != (C, K):
            raise CaptureError(f"gdn_conv FIR weights must be [C, K] = [{C}, {K}]; got shape {wv.shape}")
        out = self.fresh_buffer(f"gdn_conv_l{layer}{self._suffix()}", (B, C), dtype=xv.dtype)
        self._record(
            "gdn_conv",
            inputs=(conv_state, x, w),
            outputs=(out,),
            attributes={"layer": layer, "conv_kernel": K},
            reads=(_regional_reads(sv), _regional_reads(xv), _regional_reads(wv)),
            writes=(_regional_reads(sv), _regional_reads(out.value)),
            source_location=f"layer {layer} gdn conv",
            numerical_contract={
                "fir": "out[b] = silu(sum_{j<K-1} w[:, j] * state[b, j, :] + w[:, K-1] * x[b, :])",
                "state_shift": "state'[b, j, :] = state[b, j+1, :] for j < K-2; state'[b, K-2, :] = x[b, :]",
                "silu": "x / (1 + exp(-x))",
                "accumulate": "fp32 accumulation",
                "dtypes": "x/w bf16 (.cg loads), state pool fp32 [B, K-1, C] time-major (floe eager init uses the embed dtype — compiled pool is fp32; oracle tolerance ~1e-3)",
                "pool": "external caller-owned state pool; read-modify-write per decode step",
            },
        )
        # Post-step state view: same pool, bumped version (cache_append's
        # §4.3 pattern). Later layers must consume this view.
        sid = sv.storage_id
        post = self.graph.add_tensor(
            f"conv_state_l{layer}_v{self.graph.storage_versions[sid]}",
            sv.shape,
            sv.dtype,
            storage_id=sid,
            strides=sv.strides,
            offset=sv.offset,
        )
        return out, SymbolicTensor(post)

    def attention_scores(
        self,
        q: SymbolicTensor,
        k_cache: SymbolicTensor,
        position,
        *,
        scale: float,
        layer: int,
        kv_heads: Optional[int] = None,
    ) -> SymbolicTensor:
        """scores[b,h,t] = scale * <q[b,h,:], K[l,b,kv(h),t,:]>, t in [0, p].

        GQA (Qwen3): q head h reads kv head ``h // (H/KVH)`` when
        ``kv_heads`` is given; recorded regions reflect the grouped mapping.
        """
        self._require_position(position, "attention_scores")
        b, h, s = q.value.shape[0], q.value.shape[1], k_cache.value.shape[2]
        out = self.fresh_buffer(f"scores_l{layer}{self._suffix()}", (b, h, s))
        p = self._position_name(position)
        valid = self._valid_plus_one(position, "attention_scores")
        kvh = kv_heads if kv_heads is not None else h

        def k_regions() -> tuple[Region, ...]:
            if kvh == h:
                return (Region.prefix(k_cache.value, axis=2, valid=valid),)
            group = h // kvh
            return tuple(
                Region(
                    k_cache.value.storage_id,
                    k_cache.value,
                    ((0, b), (g * group, (g + 1) * group), (0, valid if valid.is_row else f"{p}+1"), (0, k_cache.value.shape[3])),
                )
                for g in range(kvh)
            )

        self._record(
            "attention_scores",
            inputs=(q, k_cache),
            outputs=(out,),
            attributes={"scale": float(scale), "layer": layer, "position": p, "position_form": "row" if valid.is_row else "scalar", "kv_heads": kvh},
            reads=(_regional_reads(q.value),) + k_regions(),
            writes=(Region.tile(out.value, ((0, b), (0, h), (0, s))),),
            source_location=f"layer {layer} attention scores",
            numerical_contract={
                "scale": f"1/sqrt(D) = {scale}",
                "gqa": f"q head h reads kv head h // {h // kvh}" if kvh != h else "none (MHA)",
                "valid": "only each row's positions [0, pos[b]] are read; scores beyond a row's valid length are not observed by consumers",
                "reduction": "f32 dot over D, k ascending",
            },
        )
        return out

    def softmax(self, scores: SymbolicTensor, position, *, layer: int) -> SymbolicTensor:
        """Row softmax over each row's valid prefix [0, pos[b]]; invalid tail forced to 0."""
        self._require_position(position, "softmax")
        out = self.fresh_buffer(f"probs_l{layer}{self._suffix()}", scores.value.shape)
        p = self._position_name(position)
        valid = self._valid_plus_one(position, "softmax")
        self._record(
            "softmax",
            inputs=(scores,),
            outputs=(out,),
            attributes={"layer": layer, "position": p, "position_form": "row" if valid.is_row else "scalar"},
            reads=(Region.prefix(scores.value, axis=2, valid=valid),),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} softmax",
            numerical_contract={
                "stability": "max-subtraction",
                "invalid_tail": "probabilities beyond each row's valid length are exactly 0 (written, not read from cache)",
            },
        )
        return out

    def attention_values(
        self,
        probs: SymbolicTensor,
        v_cache: SymbolicTensor,
        position,
        *,
        layer: int,
        kv_heads: Optional[int] = None,
    ) -> SymbolicTensor:
        """ctx[b,h,:] = sum_{t<=p} probs[b,h,t] * V[l,b,kv(h),t,:] (GQA-aware)."""
        self._require_position(position, "attention_values")
        b, h, d = probs.value.shape[0], probs.value.shape[1], v_cache.value.shape[3]
        out = self.fresh_buffer(f"ctx_l{layer}{self._suffix()}", (b, h, d))
        p = self._position_name(position)
        valid = self._valid_plus_one(position, "attention_values")
        kvh = kv_heads if kv_heads is not None else h

        def v_regions() -> tuple[Region, ...]:
            if kvh == h:
                return (Region.prefix(v_cache.value, axis=2, valid=valid),)
            group = h // kvh
            return tuple(
                Region(
                    v_cache.value.storage_id,
                    v_cache.value,
                    ((0, b), (g * group, (g + 1) * group), (0, valid if valid.is_row else f"{p}+1"), (0, v_cache.value.shape[3])),
                )
                for g in range(kvh)
            )

        self._record(
            "attention_values",
            inputs=(probs, v_cache),
            outputs=(out,),
            attributes={"layer": layer, "position": p, "position_form": "row" if valid.is_row else "scalar", "kv_heads": kvh},
            reads=(Region.prefix(probs.value, axis=2, valid=valid),) + v_regions(),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} attention values",
            numerical_contract={
                "gqa": f"q head h reads kv head h // {h // kvh}" if kvh != h else "none (MHA)",
                "valid": "only each row's positions [0, pos[b]] contribute; V beyond a row's valid length is never loaded (NaN tails must not leak)",
                "reduction": "f32, t ascending",
            },
        )
        return out

    # ------------------------------------------------------------------
    # Paged decode (#94): pools addressed through external slot tables
    # ------------------------------------------------------------------

    def cache_append_paged(
        self,
        k_pool: SymbolicTensor,
        v_pool: SymbolicTensor,
        slot_table: SymbolicTensor,
        k_new: SymbolicTensor,
        v_new: SymbolicTensor,
        position,
        *,
        layer: int,
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Append new K/V rows into token-slot-major pools at
        ``slot_table[b, p]`` (paged decode, §4.3 + #94).

        Pool layout ``[slots, KVH, D]`` — element address
        ``slot * KVH * D + kvh * D``, exactly the addressing the device
        kernels already use. The slot table is an external i32 ``[B, S]``
        input; slot 0 is the reserved null/sink page, never written by a
        live row. Returns post-append pool views (bumped versions).
        """
        self._require_position(position, "cache_append_paged")
        p = self._position_name(position)
        b_new, kvh_new, d_new = k_new.value.shape
        if k_pool.value.shape != (k_pool.value.shape[0], kvh_new, d_new):
            raise CaptureError(
                f"paged pool {k_pool.value.shape} does not match k_new {(b_new, kvh_new, d_new)} "
                "on (KVH, D); pools are token-slot-major [slots, KVH, D]"
            )
        if slot_table.value.shape[0] != b_new:
            raise CaptureError(
                f"slot table rows {slot_table.value.shape[0]} != batch {b_new}"
            )
        self._record(
            "cache_append_paged",
            inputs=(k_pool, v_pool, slot_table, k_new, v_new),
            outputs=(),  # mutation through storage effects; pool views returned below
            attributes={"layer": layer, "position": p},
            reads=(_regional_reads(k_new.value), _regional_reads(v_new.value)),
            writes=(
                Region.indirect(k_pool.value, slot_table.value, axis=0),
                Region.indirect(v_pool.value, slot_table.value, axis=0),
            ),
            source_location=f"layer {layer} paged kv cache append",
            numerical_contract={
                "write": "K_pool[slot_table[b, p], h, :] = k_new[b, h, :]; same for V",
                "slot0": "reserved null/sink page — never written by a live row",
                "table": "external i32 [B, S]; row b maps positions through table[b, :]",
            },
        )
        # Post-append versions: same storages, bumped (§4.3).
        k_post = self.graph.add_tensor(
            f"k_pool_l{layer}_v{self.graph.storage_versions[k_pool.value.storage_id]}",
            k_pool.value.shape,
            k_pool.value.dtype,
            storage_id=k_pool.value.storage_id,
            strides=k_pool.value.strides,
            offset=k_pool.value.offset,
        )
        v_post = self.graph.add_tensor(
            f"v_pool_l{layer}_v{self.graph.storage_versions[v_pool.value.storage_id]}",
            v_pool.value.shape,
            v_pool.value.dtype,
            storage_id=v_pool.value.storage_id,
            strides=v_pool.value.strides,
            offset=v_pool.value.offset,
        )
        return SymbolicTensor(k_post), SymbolicTensor(v_post)

    def attention_scores_paged(
        self,
        q: SymbolicTensor,
        k_pool: SymbolicTensor,
        slot_table: SymbolicTensor,
        position,
        *,
        scale: float,
        layer: int,
        kv_heads: Optional[int] = None,
    ) -> SymbolicTensor:
        """scores[b, h, t] = scale * <q[b, h, :], K_pool[table[b, t], kv(h), :]>
        for t in [0, p] (GQA-aware; #94).
        """
        self._require_position(position, "attention_scores_paged")
        b, h = q.value.shape[0], q.value.shape[1]
        s = slot_table.value.shape[1]
        d = k_pool.value.shape[2]
        out = self.fresh_buffer(f"scores_l{layer}{self._suffix()}", (b, h, s))
        p = self._position_name(position)
        kvh = kv_heads if kv_heads is not None else h
        self._record(
            "attention_scores_paged",
            inputs=(q, k_pool, slot_table),
            outputs=(out,),
            attributes={"scale": float(scale), "layer": layer, "position": p, "kv_heads": kvh},
            reads=(
                _regional_reads(q.value),
                Region.indirect(k_pool.value, slot_table.value, axis=0),
            ),
            writes=(Region.tile(out.value, ((0, b), (0, h), (0, s))),),
            source_location=f"layer {layer} paged attention scores",
            numerical_contract={
                "scale": f"1/sqrt(D) = {scale}",
                "gqa": f"q head h reads kv head h // {h // kvh}" if kvh != h else "none (MHA)",
                "gather": f"K rows gathered from slots table[b, t], t in [0, {p}]",
                "reduction": "f32 dot over D, t ascending",
            },
        )
        return out

    def attention_values_paged(
        self,
        probs: SymbolicTensor,
        v_pool: SymbolicTensor,
        slot_table: SymbolicTensor,
        position,
        *,
        layer: int,
        kv_heads: Optional[int] = None,
    ) -> SymbolicTensor:
        """ctx[b, h, :] = sum_{t<=p} probs[b, h, t] * V_pool[table[b, t], kv(h), :]
        (GQA-aware; #94). V rows beyond p are never gathered (NaN slots
        must not leak).
        """
        self._require_position(position, "attention_values_paged")
        b, h = probs.value.shape[0], probs.value.shape[1]
        d = v_pool.value.shape[2]
        out = self.fresh_buffer(f"ctx_l{layer}{self._suffix()}", (b, h, d))
        p = self._position_name(position)
        kvh = kv_heads if kv_heads is not None else h
        self._record(
            "attention_values_paged",
            inputs=(probs, v_pool, slot_table),
            outputs=(out,),
            attributes={"layer": layer, "position": p, "kv_heads": kvh},
            reads=(
                Region.prefix(probs.value, axis=2, valid=ValidLength(f"{p}+1")),
                Region.indirect(v_pool.value, slot_table.value, axis=0),
            ),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} paged attention values",
            numerical_contract={
                "gqa": f"q head h reads kv head h // {h // kvh}" if kvh != h else "none (MHA)",
                "gather": f"V rows gathered from slots table[b, t], t in [0, {p}]",
                "reduction": "f32, t ascending",
            },
        )
        return out

    def conjugate_rope(
        self,
        x: SymbolicTensor,
        cos_table: SymbolicTensor,
        sin_table: SymbolicTensor,
        position,
        *,
        layer: int,
        which: str = "o",
        convention: str = "interleaved",
        rotary_dim: Optional[int] = None,
        out: Optional[SymbolicTensor] = None,
    ) -> SymbolicTensor:
        """Conjugate (output-side) rope — issue #95, DeepSeek-V4 attention.

        Rotates by the NEGATIVE angle: same tables, sin negated. For the
        interleaved convention (full width by default)::

            y[2i]   = x[2i]*c_i + x[2i+1]*s_i
            y[2i+1] = x[2i+1]*c_i - x[2i]*s_i       (c_i, s_i = tables[p, i])

        This is the exact inverse of the q/k rotation at the same position,
        so ``rope(...) -> conjugate_rope(...)`` round-trips to identity
        (validated to 1e-12 in fp64). Decode rotates ONLY the query — the
        cached latent rows stay unrotated (the entry-side rotation happened
        once at emission).
        """
        if convention not in ("interleaved", "rotate_half"):
            raise CaptureError(f"conjugate_rope convention {convention!r} not supported (expected 'interleaved' or 'rotate_half')")
        head_dim = x.value.shape[-1]
        # rotary_dim defaults to the table-defined rotated width (tables are
        # [max_positions, rotary_dim//2]) — full width when the tables span
        # the whole head.
        table_half = cos_table.value.shape[-1]
        if sin_table.value.shape != cos_table.value.shape:
            raise CaptureError(f"conjugate_rope cos/sin tables must match; got {cos_table.value.shape} vs {sin_table.value.shape}")
        rot = rotary_dim if rotary_dim is not None else table_half * 2
        if rot % 2 != 0 or not 0 < rot <= head_dim:
            raise CaptureError(f"conjugate_rope requires an even 0 < rotary_dim <= head_dim ({head_dim}); got {rot!r}")
        if table_half != rot // 2:
            raise CaptureError(
                f"conjugate_rope tables must be [max_positions, rotary_dim//2] = [.., {rot // 2}]; got shape {cos_table.value.shape}"
            )
        self._require_position(position, "conjugate_rope")
        out = out or self.fresh_buffer(f"crope_{which}_l{layer}{self._suffix()}", x.value.shape)
        if out.value.shape != x.value.shape:
            raise ValueError(f"conjugate_rope out must match x {x.value.shape}; got {out.value.shape}")
        p = self._position_name(position)
        self._record(
            "conjugate_rope",
            inputs=(x, cos_table, sin_table),
            outputs=(out,),
            attributes={"layer": layer, "which": which, "position": p, "convention": convention,
                        "rotary_dim": rot,
                        "position_form": "row" if isinstance(position, SymbolicTensor) else "scalar"},
            reads=(_regional_reads(x.value), _regional_reads(cos_table.value), _regional_reads(sin_table.value)),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} conjugate rope {which}",
            numerical_contract={
                "interleaved": "y[2i] = x[2i]*c_i + x[2i+1]*s_i ; y[2i+1] = x[2i+1]*c_i - x[2i]*s_i (sin NEGATED)",
                "inverse": "exact inverse of the rope rotation at the same position (round-trip = identity)",
                "tables": "cos/sin [max_positions, rotary_dim//2], fp32, indexed at the runtime position",
                "upcast": "rotation math in fp32 (bf16 activations in/out)",
            },
        )
        return out

    def mla_scores(
        self,
        q: SymbolicTensor,
        latent_pool: SymbolicTensor,
        window_table: SymbolicTensor,
        comp_pool: SymbolicTensor,
        comp_idx: SymbolicTensor,
        sink: SymbolicTensor,
        position,
        *,
        scale: float,
        layer: int,
        window: int,
        bias: Optional[SymbolicTensor] = None,
    ) -> SymbolicTensor:
        """MLA decode scores — fused scores + softmax + sink (issue #95).

        Shared-KV MQA (``num_key_value_heads == 1``) over a latent cache.
        For row ``b``, head ``h`` at decode position ``p`` the candidate set
        is {window keys} ∪ {selected compressed entries} ∪ {sink}::

            window slot i < W:  t = p - W + 1 + i   (valid iff 0 <= t <= p,
                                i.e. the sliding-window bound |q - t| < W,
                                t <= q; slots with t < 0 are masked)
            compressed slot j < K: e = comp_idx[b, j]  (valid iff e >= 0;
                                the #97 table is -1-padded beyond a row's
                                valid candidate count)
            sink slot (last):   always valid

            logit(window i)    = scale * <q[b, h, :], latent_pool[b, window_table[b, t], :]>
            logit(compressed j) = scale * <q[b, h, :], comp_pool[b, e, :]>
                                  + bias[b, j]        (when ``bias`` is given;
                                  the #97 block_bias, normalized over valid
                                  scores with non-finite entries excluded —
                                  that landed contract is the producer of record)
            logit(sink)        = sink[b, h]   (per-head learnable scalar)

        The output is POST-softmax ``probs[b, h, W + K + 1]`` (the raw
        logits have no second consumer; the sink column is LAST, index
        ``W + K``). Invalid candidates receive exact 0.0 (§4.3); the sink
        absorbs softmax mass but contributes NO value (see
        :meth:`mla_values`). Accumulation fp32 over D, candidates in
        window order then #97 rank order then sink.
        """
        self._require_position(position, "mla_scores")
        b, h, d = q.value.shape
        if latent_pool.value.shape[0] != b or latent_pool.value.shape[2] != d:
            raise ValueError(f"mla_scores latent_pool {latent_pool.value.shape} must be [B, S, D] with B={b}, D={d}")
        if window_table.value.shape[0] != b or window_table.value.dtype != I32:
            raise ValueError(f"mla_scores window_table must be i32 [B, S_w], got {window_table.value.shape} {window_table.value.dtype.name}")
        if latent_pool.value.shape[1] != window_table.value.shape[1]:
            raise ValueError(
                f"mla_scores latent_pool S={latent_pool.value.shape[1]} must match window_table S_w={window_table.value.shape[1]}"
            )
        if comp_pool.value.shape[0] != b or comp_pool.value.shape[2] != d:
            raise ValueError(f"mla_scores comp_pool {comp_pool.value.shape} must be [B, M, D] with B={b}, D={d}")
        k = comp_idx.value.shape[1]
        if comp_idx.value.shape[0] != b or comp_idx.value.dtype != I32:
            raise ValueError(f"mla_scores comp_idx must be i32 [B, K], got {comp_idx.value.shape} {comp_idx.value.dtype.name}")
        if tuple(sink.value.shape) not in ((h,), (b, h)):
            raise ValueError(f"mla_scores sink must be [H] = ({h},) or [B, H] = ({b}, {h}); got {sink.value.shape}")
        if bias is not None and (bias.value.shape != (b, k) or bias.value.dtype != F32):
            raise ValueError(f"mla_scores bias must be f32 [B, K] = ({b}, {k}); got {bias.value.shape}")
        if window < 1:
            raise ValueError(f"mla_scores requires window >= 1; got {window}")
        width = window + k + 1
        out = self.fresh_buffer(f"mla_probs_l{layer}{self._suffix()}", (b, h, width), F32)
        p = self._position_name(position)
        inputs = [q, latent_pool, window_table, comp_pool, comp_idx, sink]
        reads = [
            _regional_reads(q.value),
            Region.indirect(latent_pool.value, window_table.value, axis=1),
            _regional_reads(window_table.value),
            _regional_reads(comp_pool.value),
            _regional_reads(comp_idx.value),
            _regional_reads(sink.value),
        ]
        if bias is not None:
            inputs.append(bias)
            reads.append(_regional_reads(bias.value))
        self._record(
            "mla_scores",
            inputs=tuple(inputs),
            outputs=(out,),
            attributes={"scale": float(scale), "layer": layer, "position": p, "window": window,
                        "comp_slots": k, "width": width, "sink": "last",
                        "position_form": "row" if isinstance(position, SymbolicTensor) else "scalar"},
            reads=tuple(reads),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} mla scores",
            numerical_contract={
                "scale": f"1/sqrt(D) = {scale}",
                "window": f"logical positions (p-{window}, p]; slots with t < 0 masked (row start)",
                "compressed": "logit = scale*<q, comp_pool[b, comp_idx[b,j]]> (+ bias[b,j] when given); comp_idx < 0 masked",
                "sink": "per-head learnable logit, no scale; absorbs softmax mass, contributes no value",
                "softmax": "f32, over valid candidates ∪ sink; invalid candidates exact 0.0",
                "output": f"post-softmax probs [B, H, {width}]; sink column last",
                "accumulation": "f32 dot over D per candidate; online-free two-pass max-subtract softmax in f32",
            },
        )
        return out

    def mla_values(
        self,
        probs: SymbolicTensor,
        latent_pool: SymbolicTensor,
        window_table: SymbolicTensor,
        comp_pool: SymbolicTensor,
        comp_idx: SymbolicTensor,
        position,
        *,
        layer: int,
        window: int,
        out: Optional[SymbolicTensor] = None,
    ) -> SymbolicTensor:
        """MLA decode context gather (issue #95).

        ``ctx[b, h, :] = sum_{i < W, t_i valid} probs[b, h, i] *
        latent_pool[b, window_table[b, t_i], :] + sum_{j < K, e_j >= 0}
        probs[b, h, W + j] * comp_pool[b, comp_idx[b, j], :]`` — the sink
        column (``W + K``) contributes NOTHING (its probability mass is
        absorbed). Latent rows double as keys and values (K == V, one
        shared head); invalid/zero-prob candidates contribute exactly zero.
        fp32 accumulation, single bf16/fp32 store.
        """
        self._require_position(position, "mla_values")
        b, h, width = probs.value.shape
        d = latent_pool.value.shape[2]
        if comp_pool.value.shape[2] != d:
            raise ValueError(f"mla_values comp_pool {comp_pool.value.shape} must share D={d} with the latent pool")
        if comp_idx.value.shape[1] != comp_pool.value.shape[1] and comp_idx.value.dtype != I32:
            raise ValueError(f"mla_values comp_idx must be i32 [B, K]; got {comp_idx.value.shape} {comp_idx.value.dtype.name}")
        k = comp_idx.value.shape[1]
        if width != window + k + 1:
            raise ValueError(f"mla_values probs width {width} != window {window} + K {k} + 1 (sink last)")
        # Epilogue: fp32 accumulation, ONE store — bf16 activations round to
        # bf16 here (default fp32 for fp32 chains).
        out = out or self.fresh_buffer(f"mla_ctx_l{layer}{self._suffix()}", (b, h, d))
        if out.value.shape != (b, h, d):
            raise ValueError(f"mla_values out must be [B, H, D] = ({b}, {h}, {d}); got {out.value.shape}")
        p = self._position_name(position)
        self._record(
            "mla_values",
            inputs=(probs, latent_pool, window_table, comp_pool, comp_idx),
            outputs=(out,),
            attributes={"layer": layer, "position": p, "window": window, "comp_slots": k, "width": width,
                        "position_form": "row" if isinstance(position, SymbolicTensor) else "scalar"},
            reads=(
                _regional_reads(probs.value),
                Region.indirect(latent_pool.value, window_table.value, axis=1),
                _regional_reads(window_table.value),
                _regional_reads(comp_pool.value),
                _regional_reads(comp_idx.value),
            ),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} mla values",
            numerical_contract={
                "gather": "window rows via window_table (logical t = p-W+1+i), compressed rows via comp_idx",
                "sink": "column W+K contributes no value (mass absorbed)",
                "reduction": "f32, window candidates then compressed candidates ascending",
                "dtypes": "pools bf16 (`.cg` streamed), accumulate fp32, single store",
            },
        )
        return out

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
