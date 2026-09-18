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


# Gated-RMSNorm activations in the supported subset (issue #100). GLM's
# o_norm uses sigmoid (floe Glm53RMSNormGated); silu is deliberately NOT
# here — vkernels' kda_layer_norm_gated hardcodes silu and does not cover
# this variant, which is exactly why the op exists.
_RMS_GATED_ACTIVATIONS = ("sigmoid",)


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

    def rms_norm_gated(
        self,
        x: SymbolicTensor,
        gate: SymbolicTensor,
        gamma: SymbolicTensor,
        eps: float,
        *,
        activation: str = "sigmoid",
        out: Optional[SymbolicTensor] = None,
        name: str = "rms_gated",
    ) -> SymbolicTensor:
        """Sigmoid-gated RMSNorm over the last dim (GLM ``o_norm``).

        ``o = rmsnorm(x) * gamma * activation(gate)`` with the gate applied
        per element after the weight multiply. ``activation`` is an IR
        attribute — only ``"sigmoid"`` is in the supported subset (floe
        ``Glm53RMSNormGated``); vkernels' ``kda_layer_norm_gated`` hardcodes
        silu and does *not* cover this op.

        Works on [B, C] hidden states (one task per row) and on [B, H, D]
        head views (one task per head row — the linear-attention output
        norm sits per-head before ``o_proj``); ``gate`` must match ``x``
        shape for elementwise multiplication.
        """
        if activation not in _RMS_GATED_ACTIVATIONS:
            raise UnsupportedOperator(
                f"rms_norm_gated activation {activation!r} is outside the supported subset {_RMS_GATED_ACTIVATIONS}",
                node=name,
                missing_contract="a gated-norm activation the device task implements (sigmoid for GLM o_norm)",
            )
        xv, gv = x.value, gate.value
        if gv.shape != xv.shape:
            raise CaptureError(
                f"rms_norm_gated gate shape {gv.shape} must match the normalized tensor {xv.shape} (elementwise gate)"
            )
        if len(xv.shape) not in (2, 3):
            raise CaptureError(
                f"rms_norm_gated input must be [B, C] or [B, H, D] (row-wise norm); got shape {xv.shape}"
            )
        if gv.shape[-1] != gamma.value.shape[-1] or len(gamma.value.shape) != 1:
            raise CaptureError(
                f"rms_norm_gated weight must be [width]={xv.shape[-1]}; got shape {gamma.value.shape}"
            )
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", xv.shape)
        self._record(
            "rms_norm_gated",
            inputs=(x, gate, gamma),
            outputs=(out,),
            attributes={"eps": eps, "activation": activation},
            reads=(_regional_reads(xv), _regional_reads(gv), _regional_reads(gamma.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "formula": "x * rsqrt(mean(x^2) + eps) * gamma * activation(gate)",
                "activation": "sigmoid(g) = 1 / (1 + exp(-g))" if activation == "sigmoid" else activation,
                "upcast": "strict fp32 statistics and math (floe Glm53RMSNormGated): x/gate upcast before the row reduction, gate sigmoid in fp32, output cast back to the storage dtype",
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

    def compressor_append(
        self,
        entry_pool: SymbolicTensor,
        series_state: SymbolicTensor,
        window: SymbolicTensor,
        gates: SymbolicTensor,
        rms_weight: SymbolicTensor,
        cos: SymbolicTensor,
        sin: SymbolicTensor,
        position,
        *,
        m: int,
        r: int,
        eps: float,
        name: str = "compressor_append",
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Emit one compressed entry at the m-token boundary (issue #96).

        DeepSeek Sparse Attention's compressor (floe
        ``deepseek_v4/forward.py::_BaseCompressor``): every ``m`` tokens
        compress into one cached entry, kept in a **two-series** per-layer
        pool — the prior completed series ``Ca`` and the in-flight series
        ``Cb`` (width ``2r`` stride ``r`` tokens: at each ``r``-boundary the
        just-completed ``Cb`` becomes the new ``Ca`` and ``Cb`` restarts, so
        the attention window [Ca ∪ Cb] slides by ``r`` per rotation). Entries
        are rope-rotated **once at emission** (at the emitting row's own
        position), so decode only rotates the query.

        Emission math for a boundary row ``b`` (fp32 accumulated, per floe
        ``Compressor.emit``)::

            w = softmax(gates[b, :])                    # over the m window
            e = Σ_t w_t · window[b, t, :]               # weighted latent fold
            e = e / sqrt(mean(e²) + eps) · rms_weight   # rms_norm
            e = rotate_half(e, cos[b], sin[b])          # rotated ONCE, at emission
            entry_pool[b, l, slot, cb_len, :] = bf16(e)

        with ``slot``/``cb_len`` from the persistent per-(row, layer) series
        state, then the rotation bookkeeping (``cb_len += 1``; at ``cb_len ==
        r // m`` the completed ``Cb`` becomes ``Ca`` — slot roles ping-pong —
        and ``Cb`` restarts). Rows whose position is not at a boundary
        (``p[b] % m != m-1``) are exact no-ops — the per-row mask comes from
        the issue #93 positions tensor, so ragged rows emit at their own
        cadence.

        Tensor layout:

        * ``window`` [B, m, D] — the row's in-flight window of shared-latent
          kv vectors (compressed-entry source), from the compressor state
          pool; bf16 stored, fp32-accumulated;
        * ``gates`` [B, m] — fp32 compression gates over the window;
        * ``entry_pool`` [B, L, 2, R, D] bf16 — persistent, two series slots
          of ``R = r // m`` entries each; slot roles ping-pong;
        * ``series_state`` [B, L, 2] i32 — persistent ``[active_slot,
          cb_len]`` per (row, layer);
        * ``cos``/``sin`` [B, D//2] fp32 — gathered by the caller at each
          row's emission position (per-row ragged cadence).

        Reads the series state (RAW), writes the entry pool slab and the
        series state (RMW, ``cache_append`` §4.3 pattern); returns
        *post-append views* — same storages, bumped versions.

        block_bias contract (issue #97, producer of record): entries are
        appended in global emission order, so flat entry index ``j`` is a
        stable per-row identity across the pool; the indexer's
        ``block_bias[b, j]`` aligns 1:1 with that index, and the Ca/Cb
        attention window selects contiguous ``j`` ranges via the #94 slot
        indirection at scores time (#95's integration).
        """
        if m <= 0 or r <= 0 or r % m:
            raise CaptureError(
                f"{name}: series rotation needs r % m == 0; got m={m}, r={r}"
            )
        b, mw, d = window.value.shape
        if mw != m:
            raise CaptureError(f"{name}: window token axis {mw} != m={m}")
        bg, mg = gates.value.shape
        if (bg, mg) != (b, m):
            raise CaptureError(f"{name}: gates {gates.value.shape} != [{b}, {m}]")
        R = r // m
        be, le, slots, re, de = entry_pool.value.shape
        if (be, slots, re, de) != (b, 2, R, d):
            raise CaptureError(
                f"{name}: entry_pool {entry_pool.value.shape} != [B={b}, L, 2, R={R}, D={d}]"
            )
        if series_state.value.shape != (b, le, 2):
            raise CaptureError(
                f"{name}: series_state {series_state.value.shape} != [{b}, {le}, 2]"
            )
        if rms_weight.value.shape != (d,):
            raise CaptureError(f"{name}: rms_weight {rms_weight.value.shape} != [{d}]")
        dh = cos.value.shape[-1]
        if cos.value.shape != (b, dh) or sin.value.shape != (b, dh) or 2 * dh != d:
            raise CaptureError(
                f"{name}: cos/sin must be [B={b}, D/2={d // 2}]; got {cos.value.shape} / {sin.value.shape}"
            )
        valid = self._valid_plus_one(position, name)
        p = self._position_name(position)
        writes = (
            Region.tile(
                entry_pool.value,
                ((0, b), (0, le), (0, 2), (0, R), (0, d)),
            ),
            Region.tile(series_state.value, ((0, b), (0, le), (0, 2))),
        )
        position_form = "row" if valid.is_row else "scalar"
        self._record(
            "compressor_append",
            inputs=(entry_pool, series_state, window, gates, rms_weight, cos, sin),
            outputs=(),  # mutation through storage effects; views returned below
            attributes={
                "m": m,
                "r": r,
                "eps": eps,
                "position": p,
                "position_form": position_form,
            },
            reads=(
                _regional_reads(window.value),
                _regional_reads(gates.value),
                _regional_reads(rms_weight.value),
                _regional_reads(cos.value),
                _regional_reads(sin.value),
                _regional_reads(series_state.value),
            ),
            writes=writes,
            source_location=name,
            numerical_contract={
                "emission": (
                    "boundary rows (p[b] % m == m-1): w = softmax(gates[b]) fp32; "
                    "e = sum_t w_t*window[b,t] fp32; e = e/sqrt(mean(e^2)+eps)*rms_weight; "
                    "e = rotate_half(e, cos[b], sin[b]) — rotated once at emission; "
                    "entry_pool[b,l,slot,cb_len,:] = bf16(e)"
                ),
                "boundary_mask": "rows with p[b] % m != m-1 are exact no-ops (per-row, issue #93)",
                "series_rotation": (
                    "cb_len += 1 after append; at cb_len == r//m the completed Cb "
                    "becomes Ca (slot roles swap) and Cb restarts empty"
                ),
                "block_bias": (
                    "entry flat index j = global emission order (stable per row); "
                    "block_bias[b, j] aligns 1:1 with that index (issue #97 producer "
                    "contract, non-finite-valid normalization as landed); Ca/Cb "
                    "windows select contiguous j ranges via #94 indirection at "
                    "scores time (#95 integration)"
                ),
            },
        )
        pool_post = self.graph.add_tensor(
            f"entry_pool_s{entry_pool.value.storage_id}_v{self.graph.storage_versions[entry_pool.value.storage_id]}",
            entry_pool.value.shape,
            entry_pool.value.dtype,
            storage_id=entry_pool.value.storage_id,
            strides=entry_pool.value.strides,
            offset=entry_pool.value.offset,
            valid_length=valid,
        )
        state_post = self.graph.add_tensor(
            f"series_state_s{series_state.value.storage_id}_v{self.graph.storage_versions[series_state.value.storage_id]}",
            series_state.value.shape,
            series_state.value.dtype,
            storage_id=series_state.value.storage_id,
            strides=series_state.value.strides,
            offset=series_state.value.offset,
            valid_length=valid,
        )
        return SymbolicTensor(pool_post), SymbolicTensor(state_post)

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
    def mhc_pre(
        self,
        streams: SymbolicTensor,
        fn: SymbolicTensor,
        base: SymbolicTensor,
        scale: SymbolicTensor,
        *,
        layer: int,
        iters: int,
        eps: float,
        rms_eps: float,
    ) -> tuple[SymbolicTensor, SymbolicTensor, SymbolicTensor]:
        """mHC hyper-connection pre-mix (issue #99; floe
        ``DeepseekV4HyperConnection.forward`` / ``Glm53HyperConnection.forward``
        — one op family, family attributes).

        Per decode token (row ``b`` of the ``[B, hc, C]`` stream stack), with
        ``mix = (2 + hc)·hc``:

            flat    = unweighted_rms_norm(flatten(streams[b]))    # [hc·C], fp32
            logits  = fn @ flat                                   # [mix] GEMV — NO bias:
                                                                  # base enters only in the gates
            pre_w, post_w, comb_w = split(logits, [hc, hc, hc²])
            pre     = sigmoid(pre_w·pre_s + pre_b) + eps          # [hc]
            post    = 2·sigmoid(post_w·post_s + post_b)           # [hc], (0, 2)
            comb    = softmax((comb_w·comb_s + comb_b).view(hc, hc), -1) + eps
            comb    = comb / (colsum(comb) + eps)                 # Sinkhorn-Knopp:
            for _ in range(iters − 1):                            # alternate row/col
                comb = comb / (rowsum(comb) + eps)                # normalization to
                comb = comb / (colsum(comb) + eps)                # doubly-stochastic
            h_in   = Σ_h pre_h · streams[b, h, :]                 # stream collapse

        ``pre``/``post``/``comb`` are DATA-DEPENDENT — computed from the stream
        contents every step, so the Sinkhorn projection runs per token, not at
        load time. ``base`` splits into (pre_b, post_b, comb_b) with comb_b
        viewed ``[hc, hc]``; ``scale`` = (pre_s, post_s, comb_s) scalars. All
        mixing math is fp32 (Sinkhorn numerics demand it); ``h_in`` follows the
        streams dtype.

        Stream state is intermediate workspace: ``streams`` is read-only here
        (the composing ``mhc_post`` writes the updated stack to a fresh buffer),
        and ``h_in``/``post``/``comb`` are fresh compiler-planned buffers — no
        persistent state pool, no read-modify-write effects.

        Returns ``(h_in, post, comb)``; the block body consumes ``h_in`` and
        ``mhc_post`` consumes ``(streams, body_out, post, comb)``.
        """
        sv, fnv, bv, scv = (
            streams.value, fn.value, base.value, scale.value,
        )
        if len(sv.shape) != 3:
            raise CaptureError(f"mhc_pre streams must be [B, hc, C]; got shape {sv.shape}")
        B, hc, C = sv.shape
        mix = (2 + hc) * hc
        if fnv.shape != (mix, hc * C):
            raise CaptureError(
                f"mhc_pre fn must be [{mix}, {hc * C}] = [(2+hc)·hc, hc·C]; got shape {fnv.shape}"
            )
        if bv.shape != (mix,):
            raise CaptureError(f"mhc_pre base must be [{mix}]; got shape {bv.shape}")
        if scv.shape != (3,):
            raise CaptureError(f"mhc_pre scale must be [3] = (pre_s, post_s, comb_s); got shape {scv.shape}")
        if iters < 1:
            raise CaptureError(f"mhc_pre Sinkhorn iters must be >= 1; got {iters}")
        h_in = self.fresh_buffer(f"mhc_h_in_l{layer}{self._suffix()}", (B, C), dtype=sv.dtype)
        post = self.fresh_buffer(f"mhc_post_w_l{layer}{self._suffix()}", (B, hc), dtype=F32)
        comb = self.fresh_buffer(f"mhc_comb_l{layer}{self._suffix()}", (B, hc, hc), dtype=F32)
        self._record(
            "mhc_pre",
            inputs=(streams, fn, base, scale),
            outputs=(h_in, post, comb),
            attributes={"layer": layer, "hc": hc, "iters": iters, "eps": eps, "rms_eps": rms_eps},
            reads=(
                _regional_reads(sv), _regional_reads(fnv),
                _regional_reads(bv), _regional_reads(scv),
            ),
            writes=(
                _regional_reads(h_in.value), _regional_reads(post.value),
                _regional_reads(comb.value),
            ),
            source_location=f"layer {layer} mhc pre-mix",
            numerical_contract={
                "input_norm": "flat = flatten(streams[b]) * rsqrt(mean(flat²) + rms_eps), fp32 (unweighted RMSNorm over the full hc·C vector)",
                "projection": "logits = fn @ flat (F.linear, NO projection bias), one [mix=(2+hc)·hc, hc·C] GEMV per token (data-dependent); base enters only inside the gates",
                "pre": "pre = sigmoid(pre_w·pre_s + pre_b) + eps, [hc]",
                "post": "post = 2·sigmoid(post_w·post_s + post_b), [hc], range (0, 2)",
                "comb": "comb = softmax((comb_w·comb_s + comb_b.view(hc, hc)), dim=-1) + eps",
                "sinkhorn": "comb /= (colsum(comb) + eps); then (iters−1)× [comb /= (rowsum+eps); comb /= (colsum+eps)] — alternate normalization to doubly-stochastic, eps inside every denominator",
                "collapse": "h_in = Σ_h pre_h · streams[b, h, :] (raw streams, not the normalized flat)",
                "dtypes": "streams bf16 workspace loads, all mixing math fp32 (Sinkhorn numerics demand it), h_in follows the streams dtype",
                "state": "streams read-only workspace; h_in/post/comb fresh buffers — intermediate, NOT a persistent state pool",
            },
        )
        return h_in, post, comb

    def mhc_post(
        self,
        streams: SymbolicTensor,
        body_out: SymbolicTensor,
        post: SymbolicTensor,
        comb: SymbolicTensor,
        *,
        layer: int,
    ) -> SymbolicTensor:
        """mHC hyper-connection post-compose (issue #99; floe
        ``_mhc_compose``): place the sublayer output back onto the hc parallel
        residual streams with the Sinkhorn-projected mixer —

            streams'[j, :] = post[j]·body_out[:] + Σ_k comb[k, j]·streams[k, :]

        (floe torch reference: ``post.unsqueeze(-1)·sublayer_out.unsqueeze(-2)
        + combᵀ @ residual``). ``post``/``comb`` are the data-dependent weights
        this token's ``mhc_pre`` produced. The updated stream stack is written
        to a FRESH ``[B, hc, C]`` workspace buffer (intermediate, NOT a
        persistent state pool — the next layer's ``mhc_pre`` consumes the
        returned view through an ordinary RAW dependency).
        """
        sv, ov, pv, cv = (
            streams.value, body_out.value, post.value, comb.value,
        )
        if len(sv.shape) != 3:
            raise CaptureError(f"mhc_post streams must be [B, hc, C]; got shape {sv.shape}")
        B, hc, C = sv.shape
        if ov.shape != (B, C):
            raise CaptureError(f"mhc_post body_out must be [B, {C}]; got shape {ov.shape}")
        if pv.shape != (B, hc):
            raise CaptureError(f"mhc_post post weights must be [B, {hc}]; got shape {pv.shape}")
        if cv.shape != (B, hc, hc):
            raise CaptureError(f"mhc_post comb must be [B, {hc}, {hc}]; got shape {cv.shape}")
        streams_post = self.fresh_buffer(
            f"mhc_streams_l{layer}{self._suffix()}", (B, hc, C), dtype=sv.dtype
        )
        self._record(
            "mhc_post",
            inputs=(streams, body_out, post, comb),
            outputs=(streams_post,),
            attributes={"layer": layer, "hc": hc},
            reads=(
                _regional_reads(sv), _regional_reads(ov),
                _regional_reads(pv), _regional_reads(cv),
            ),
            writes=(_regional_reads(streams_post.value),),
            source_location=f"layer {layer} mhc post-compose",
            numerical_contract={
                "compose": "streams'[j] = post[j]·body_out + Σ_k comb[k, j]·streams[k] (combᵀ @ streams + post ⊙ body_out)",
                "dtypes": "fp32 compose arithmetic, streams' follows the streams dtype",
                "state": "fresh [B, hc, C] workspace buffer per layer — intermediate, NOT a persistent state pool",
            },
        )
        return streams_post

    def gdn_delta(
        self,
        ssm_state: SymbolicTensor,
        q: SymbolicTensor,
        k: SymbolicTensor,
        v: SymbolicTensor,
        z: SymbolicTensor,
        a: SymbolicTensor,
        b: SymbolicTensor,
        a_log: SymbolicTensor,
        dt_bias: SymbolicTensor,
        norm_w: SymbolicTensor,
        *,
        layer: int,
        scale: float,
        eps: float,
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Gated delta rule decode step (Qwen3.5 ``GatedDeltaNet``, seq==1).

        Per value head ``h`` over its ``[HV, HK]`` fp32 SSM state slice,
        with ``GROUP = NV // NK`` and ``kh = h // GROUP``::

            g      = -exp(A_log[h]) * softplus(a[b,h] + dt_bias[h])
            beta   = sigmoid(b[b,h])
            q_n    = q[b,kh] / sqrt(|q[b,kh]|^2 + 1e-6) * scale
            k_n    = k[b,kh] / sqrt(|k[b,kh]|^2 + 1e-6)
            s     *= exp(g)                            # per-head scalar decay
            sk     = s @ k_n                           # state read
            s     += beta * (v[b,h] - sk) outer k_n     # delta-rule update
            o      = s @ q_n                           # state readout
            out    = RMSNorm_HV(o) * norm_w * (z * sigmoid(z))

        ``ssm_state`` is an external caller-owned pool ``[B, NV, HV, HK]``,
        fp32, read-modify-write per (batch, head) row. ``q``/``k`` are the
        post-conv rows ``[B, NK, HK]`` (normalized *inside* the op, exactly
        as floe ``qwen35_gdn.GatedDeltaNet.forward`` conditions them);
        ``v``/``z`` are ``[B, NV, HV]``; ``a``/``b`` are ``[B, NV]``;
        ``A_log``/``dt_bias`` are ``[NV]`` and ``norm_w`` ``[HV]`` per-layer
        parameters broadcast over the batch. All fp32 arithmetic between
        the gates and the state (floe keeps the recurrence fp32; inputs may
        arrive bf16 from the projections).

        The op is position-independent: its ordering obligation is the
        read-modify-write on the state pool (RAW/WAR/WAW hazards vs any
        other op touching that storage). Records write effects on the pool
        and returns ``(out, ssm_state_post)`` -- the post-step state view
        (same storage, bumped version) that later layers must read.
        """
        sv, qv, kv, vv, zv, av, bv = (
            ssm_state.value, q.value, k.value, v.value, z.value, a.value, b.value,
        )
        if len(sv.shape) != 4:
            raise CaptureError(f"gdn_delta state pool must be [B, NV, HV, HK]; got shape {sv.shape}")
        B, NV, HV, HK = sv.shape
        if len(qv.shape) != 3 or qv.shape[0] != B or qv.shape[2] != HK:
            raise CaptureError(f"gdn_delta q rows must be [B, NK, {HK}]; got shape {qv.shape}")
        NK = qv.shape[1]
        if NK < 1 or NV % NK:
            raise CaptureError(f"gdn_delta head group must divide: NV={NV} not divisible by NK={NK}")
        if kv.shape != (B, NK, HK):
            raise CaptureError(f"gdn_delta k rows must be [B, {NK}, {HK}]; got shape {kv.shape}")
        if vv.shape != (B, NV, HV):
            raise CaptureError(f"gdn_delta v rows must be [B, {NV}, {HV}]; got shape {vv.shape}")
        if zv.shape != (B, NV, HV):
            raise CaptureError(f"gdn_delta z gate rows must be [B, {NV}, {HV}]; got shape {zv.shape}")
        if av.shape != (B, NV) or bv.shape != (B, NV):
            raise CaptureError(f"gdn_delta a/b rows must be [B, {NV}]; got {av.shape} / {bv.shape}")
        if a_log.value.shape != (NV,) or dt_bias.value.shape != (NV,):
            raise CaptureError(f"gdn_delta A_log/dt_bias must be [{NV}]; got {a_log.value.shape} / {dt_bias.value.shape}")
        if norm_w.value.shape != (HV,):
            raise CaptureError(f"gdn_delta norm_w must be [{HV}]; got shape {norm_w.value.shape}")
        out = self.fresh_buffer(f"gdn_delta_l{layer}{self._suffix()}", (B, NV, HV), dtype=vv.dtype)
        self._record(
            "gdn_delta",
            inputs=(ssm_state, q, k, v, z, a, b, a_log, dt_bias, norm_w),
            outputs=(out,),
            attributes={"layer": layer, "scale": scale, "eps": eps},
            reads=(
                _regional_reads(sv), _regional_reads(qv), _regional_reads(kv),
                _regional_reads(vv), _regional_reads(zv), _regional_reads(av),
                _regional_reads(bv), _regional_reads(a_log.value),
                _regional_reads(dt_bias.value), _regional_reads(norm_w.value),
            ),
            writes=(_regional_reads(sv), _regional_reads(out.value)),
            source_location=f"layer {layer} gdn delta rule",
            numerical_contract={
                "decay": "g = -exp(A_log) * softplus(a + dt_bias); s *= exp(g) (softplus guarded: x>20 -> x)",
                "qk_norm": "q_n = L2(q)*scale, k_n = L2(k) per key head (group-expanded), eps 1e-6 inside the sqrt",
                "delta_rule": "sk = s @ k_n; s += beta*(v - sk) outer k_n with beta = sigmoid(b)",
                "readout": "o = s @ q_n",
                "out_gate": "out = RMSNorm_HV(o) * norm_w * (z * sigmoid(z)), eps inside the rsqrt",
                "accumulate": "fp32 arithmetic between gates and state",
                "dtypes": "q/k/v/z/a/b workspace loads (.cg), A_log/dt_bias/norm_w params, SSM state pool fp32 [B, NV, HV, HK] read-modify-write, out follows v's dtype",
                "pool": "external caller-owned state pool; read-modify-write per decode step",
            },
        )
        # Post-step state view: same pool, bumped version (cache_append's
        # §4.3 pattern). Later layers must consume this view.
        sid = sv.storage_id
        post = self.graph.add_tensor(
            f"ssm_state_l{layer}_v{self.graph.storage_versions[sid]}",
            sv.shape,
            sv.dtype,
            storage_id=sid,
            strides=sv.strides,
            offset=sv.offset,
        )
        return out, SymbolicTensor(post)

    def kda_delta(
        self,
        ssm_state: SymbolicTensor,
        q: SymbolicTensor,
        k: SymbolicTensor,
        v: SymbolicTensor,
        f: SymbolicTensor,
        b: SymbolicTensor,
        dt_bias: SymbolicTensor,
        A_log: SymbolicTensor,
        *,
        layer: int,
        scale: float,
        lower_bound: Optional[float],
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """KDA gated delta rule decode step with element-wise decay
        (GLM-5.3 ``Glm53LinearAttention``, seq==1 — Kimi delta attention).

        Per head ``h`` over its ``[K, V]`` fp32 state slice (KDA heads are
        square: K = V = head_dim; one q/k/v head each, no group expansion)::

            g      = lower_bound * sigmoid(exp(A_log[h]) * (f[b,h] + dt_bias[h]))
                     # log-space [K]; lower_bound None -> -exp(A_log)*softplus
            beta   = sigmoid(b[b,h])
            q_n    = q[b,h] / sqrt(|q|^2 + 1e-6) * scale
            k_n    = k[b,h] / sqrt(|k|^2 + 1e-6)
            s     *= exp(g)[None over V per k-row]      # element-wise decay
            kv     = sum_k s * k_n                      # [V]
            s     += k_n outer (beta * (v - kv))
            o      = sum_k s * q_n                      # [V]

        ``ssm_state`` is an external caller-owned pool ``[B, H, K, V]`` fp32,
        read-modify-write per (batch, head) row — same §4.3 pattern as
        gdn_conv/gdn_delta. ``q``/``k`` are the post-conv rows ``[B, H, K]``
        (normalized *inside* the op, exactly as floe ``_l2norm`` conditions
        them before ``_kda_recurrent``); ``v`` is ``[B, H, V]``; ``f`` is
        the ``f_b(f_a(x))`` projection row ``[B, H, K]`` with ``dt_bias``
        ``[H, K]`` and ``A_log`` ``[H]`` folded inside (the gate arithmetic
        is per-(head, k-dim), so it rides with the task — unlike gdn_delta's
        per-head scalars it cannot stay outside without a materialized
        ``[B, H, K]`` intermediate). ``lower_bound`` is the config scalar
        (``Glm53Config.linear_lower_bound``; ``None`` selects the guarded
        softplus branch). All gate/state arithmetic fp32; output follows
        ``v``'s dtype.

        The op is position-independent: its ordering obligation is the
        read-modify-write on the state pool (RAW/WAR/WAW hazards vs any
        other op touching that storage). Records write effects on the pool
        and returns ``(out, ssm_state_post)`` — the post-step state view
        (same storage, bumped version) that later layers must read.
        """
        sv, qv, kv, vv, fv, bv = (
            ssm_state.value, q.value, k.value, v.value, f.value, b.value,
        )
        if len(sv.shape) != 4:
            raise CaptureError(f"kda_delta state pool must be [B, H, K, V]; got shape {sv.shape}")
        B, H, K, V = sv.shape
        if K != V:
            raise CaptureError(f"kda_delta heads must be square (K = V = head_dim); got K={K}, V={V}")
        if qv.shape != (B, H, K):
            raise CaptureError(f"kda_delta q rows must be [B, {H}, {K}]; got shape {qv.shape}")
        if kv.shape != (B, H, K):
            raise CaptureError(f"kda_delta k rows must be [B, {H}, {K}]; got shape {kv.shape}")
        if vv.shape != (B, H, V):
            raise CaptureError(f"kda_delta v rows must be [B, {H}, {V}]; got shape {vv.shape}")
        if fv.shape != (B, H, K):
            raise CaptureError(f"kda_delta f rows (f_b projection) must be [B, {H}, {K}]; got shape {fv.shape}")
        if bv.shape != (B, H):
            raise CaptureError(f"kda_delta b logits must be [B, {H}]; got shape {bv.shape}")
        if dt_bias.value.shape != (H, K):
            raise CaptureError(f"kda_delta dt_bias must be [{H}, {K}]; got {dt_bias.value.shape}")
        if A_log.value.shape != (H,):
            raise CaptureError(f"kda_delta A_log must be [{H}]; got {A_log.value.shape}")
        out = self.fresh_buffer(f"kda_delta_l{layer}{self._suffix()}", (B, H, V), dtype=vv.dtype)
        self._record(
            "kda_delta",
            inputs=(ssm_state, q, k, v, f, b, dt_bias, A_log),
            outputs=(out,),
            attributes={"layer": layer, "scale": scale, "lower_bound": lower_bound},
            reads=(
                _regional_reads(sv), _regional_reads(qv), _regional_reads(kv),
                _regional_reads(vv), _regional_reads(fv), _regional_reads(bv),
                _regional_reads(dt_bias.value), _regional_reads(A_log.value),
            ),
            writes=(_regional_reads(sv), _regional_reads(out.value)),
            source_location=f"layer {layer} kda delta rule",
            numerical_contract={
                "gate": "g = lower_bound * sigmoid(exp(A_log[h]) * (f[b,h] + dt_bias[h])) per (head, k-dim); lower_bound None -> g = -exp(A_log[h]) * softplus(f + dt_bias) with the x>20 guard",
                "decay": "s *= exp(g) broadcast over the value axis (element-wise per k-row; richer than gdn_delta's scalar-per-head decay)",
                "qk_norm": "q_n = L2(q)*scale, k_n = L2(k) per head, eps 1e-6 inside the sqrt (floe _l2norm)",
                "beta": "beta = sigmoid(b[b,h])",
                "delta_rule": "kv = sum_k s * k_n; s += k_n outer (beta * (v - kv))",
                "readout": "o = sum_k s * q_n",
                "accumulate": "fp32 arithmetic between gates and state (floe keeps the KDA recurrence fp32 unconditionally)",
                "dtypes": "q/k/v/f/b workspace loads (.cg), dt_bias/A_log params, state pool fp32 [B, H, K, V] read-modify-write, out follows v's dtype",
                "pool": "external caller-owned state pool; read-modify-write per decode step",
            },
        )
        # Post-step state view: same pool, bumped version (cache_append's
        # §4.3 pattern). Later layers must consume this view.
        sid = sv.storage_id
        post = self.graph.add_tensor(
            f"kda_state_l{layer}_v{self.graph.storage_versions[sid]}",
            sv.shape,
            sv.dtype,
            storage_id=sid,
            strides=sv.strides,
            offset=sv.offset,
        )
        return out, SymbolicTensor(post)

    def gdn_delta(
        self,
        ssm_state: SymbolicTensor,
        q: SymbolicTensor,
        k: SymbolicTensor,
        v: SymbolicTensor,
        z: SymbolicTensor,
        a: SymbolicTensor,
        b: SymbolicTensor,
        a_log: SymbolicTensor,
        dt_bias: SymbolicTensor,
        norm_w: SymbolicTensor,
        *,
        layer: int,
        scale: float,
        eps: float,
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Gated delta rule decode step (Qwen3.5 ``GatedDeltaNet``, seq==1).

        Per value head ``h`` over its ``[HV, HK]`` fp32 SSM state slice,
        with ``GROUP = NV // NK`` and ``kh = h // GROUP``::

            g      = -exp(A_log[h]) * softplus(a[b,h] + dt_bias[h])
            beta   = sigmoid(b[b,h])
            q_n    = q[b,kh] / sqrt(|q[b,kh]|^2 + 1e-6) * scale
            k_n    = k[b,kh] / sqrt(|k[b,kh]|^2 + 1e-6)
            s     *= exp(g)                            # per-head scalar decay
            sk     = s @ k_n                           # state read
            s     += beta * (v[b,h] - sk) outer k_n     # delta-rule update
            o      = s @ q_n                           # state readout
            out    = RMSNorm_HV(o) * norm_w * (z * sigmoid(z))

        ``ssm_state`` is an external caller-owned pool ``[B, NV, HV, HK]``,
        fp32, read-modify-write per (batch, head) row. ``q``/``k`` are the
        post-conv rows ``[B, NK, HK]`` (normalized *inside* the op, exactly
        as floe ``qwen35_gdn.GatedDeltaNet.forward`` conditions them);
        ``v``/``z`` are ``[B, NV, HV]``; ``a``/``b`` are ``[B, NV]``;
        ``A_log``/``dt_bias`` are ``[NV]`` and ``norm_w`` ``[HV]`` per-layer
        parameters broadcast over the batch. All fp32 arithmetic between
        the gates and the state (floe keeps the recurrence fp32; inputs may
        arrive bf16 from the projections).

        The op is position-independent: its ordering obligation is the
        read-modify-write on the state pool (RAW/WAR/WAW hazards vs any
        other op touching that storage). Records write effects on the pool
        and returns ``(out, ssm_state_post)`` -- the post-step state view
        (same storage, bumped version) that later layers must read.
        """
        sv, qv, kv, vv, zv, av, bv = (
            ssm_state.value, q.value, k.value, v.value, z.value, a.value, b.value,
        )
        if len(sv.shape) != 4:
            raise CaptureError(f"gdn_delta state pool must be [B, NV, HV, HK]; got shape {sv.shape}")
        B, NV, HV, HK = sv.shape
        if len(qv.shape) != 3 or qv.shape[0] != B or qv.shape[2] != HK:
            raise CaptureError(f"gdn_delta q rows must be [B, NK, {HK}]; got shape {qv.shape}")
        NK = qv.shape[1]
        if NK < 1 or NV % NK:
            raise CaptureError(f"gdn_delta head group must divide: NV={NV} not divisible by NK={NK}")
        if kv.shape != (B, NK, HK):
            raise CaptureError(f"gdn_delta k rows must be [B, {NK}, {HK}]; got shape {kv.shape}")
        if vv.shape != (B, NV, HV):
            raise CaptureError(f"gdn_delta v rows must be [B, {NV}, {HV}]; got shape {vv.shape}")
        if zv.shape != (B, NV, HV):
            raise CaptureError(f"gdn_delta z gate rows must be [B, {NV}, {HV}]; got shape {zv.shape}")
        if av.shape != (B, NV) or bv.shape != (B, NV):
            raise CaptureError(f"gdn_delta a/b rows must be [B, {NV}]; got {av.shape} / {bv.shape}")
        if a_log.value.shape != (NV,) or dt_bias.value.shape != (NV,):
            raise CaptureError(f"gdn_delta A_log/dt_bias must be [{NV}]; got {a_log.value.shape} / {dt_bias.value.shape}")
        if norm_w.value.shape != (HV,):
            raise CaptureError(f"gdn_delta norm_w must be [{HV}]; got shape {norm_w.value.shape}")
        out = self.fresh_buffer(f"gdn_delta_l{layer}{self._suffix()}", (B, NV, HV), dtype=vv.dtype)
        self._record(
            "gdn_delta",
            inputs=(ssm_state, q, k, v, z, a, b, a_log, dt_bias, norm_w),
            outputs=(out,),
            attributes={"layer": layer, "scale": scale, "eps": eps},
            reads=(
                _regional_reads(sv), _regional_reads(qv), _regional_reads(kv),
                _regional_reads(vv), _regional_reads(zv), _regional_reads(av),
                _regional_reads(bv), _regional_reads(a_log.value),
                _regional_reads(dt_bias.value), _regional_reads(norm_w.value),
            ),
            writes=(_regional_reads(sv), _regional_reads(out.value)),
            source_location=f"layer {layer} gdn delta rule",
            numerical_contract={
                "decay": "g = -exp(A_log) * softplus(a + dt_bias); s *= exp(g) (softplus guarded: x>20 -> x)",
                "qk_norm": "q_n = L2(q)*scale, k_n = L2(k) per key head (group-expanded), eps 1e-6 inside the sqrt",
                "delta_rule": "sk = s @ k_n; s += beta*(v - sk) outer k_n with beta = sigmoid(b)",
                "readout": "o = s @ q_n",
                "out_gate": "out = RMSNorm_HV(o) * norm_w * (z * sigmoid(z)), eps inside the rsqrt",
                "accumulate": "fp32 arithmetic between gates and state",
                "dtypes": "q/k/v/z/a/b workspace loads (.cg), A_log/dt_bias/norm_w params, SSM state pool fp32 [B, NV, HV, HK] read-modify-write, out follows v's dtype",
                "pool": "external caller-owned state pool; read-modify-write per decode step",
            },
        )
        # Post-step state view: same pool, bumped version (cache_append's
        # §4.3 pattern). Later layers must consume this view.
        sid = sv.storage_id
        post = self.graph.add_tensor(
            f"ssm_state_l{layer}_v{self.graph.storage_versions[sid]}",
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
        gate: Optional[SymbolicTensor] = None,
    ) -> SymbolicTensor:
        """ctx[b,h,:] = sum_{t<=p} probs[b,h,t] * V[l,b,kv(h),t,:] (GQA-aware).

        With ``gate`` (issue #92, Qwen3.5 attn_output_gate): the per-head
        sigmoid output gate is fused into the values task itself —
        ``ctx[b,h,:] *= sigmoid(gate[b,h,:])`` — so the gated multiply costs
        no extra grid barrier per FA layer. ``gate`` is [B, H, D] in the
        same tile domain, emitted by the chunked [q|gate|k|v] projection.
        """
        self._require_position(position, "attention_values")
        b, h, d = probs.value.shape[0], probs.value.shape[1], v_cache.value.shape[3]
        if gate is not None and tuple(gate.value.shape) != (b, h, d):
            raise ValueError(
                f"attention_values gate must be [B, H, D] = {(b, h, d)}, "
                f"got {tuple(gate.value.shape)}"
            )
        inputs = (probs, v_cache) if gate is None else (probs, v_cache, gate)
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

        attributes = {
            "layer": layer,
            "position": p,
            "position_form": "row" if valid.is_row else "scalar",
            "kv_heads": kvh,
        }
        if gate is not None:
            attributes["gated"] = True
        reads: tuple[Region, ...] = (Region.prefix(probs.value, axis=2, valid=valid),) + v_regions()
        if gate is not None:
            reads = reads + (Region.whole(gate.value),)
        contract = {
            "gqa": f"q head h reads kv head h // {h // kvh}" if kvh != h else "none (MHA)",
            "valid": "only each row's positions [0, pos[b]] contribute; V beyond a row's valid length is never loaded (NaN tails must not leak)",
            "reduction": "f32, t ascending",
        }
        if gate is not None:
            contract["gate"] = "y = acc * sigmoid(gate[b,h,:]); sigmoid and multiply in f32 before the single bf16 store (#92); no extra barrier"
        self._record(
            "attention_values",
            inputs=inputs,
            outputs=(out,),
            attributes=attributes,
            reads=reads,
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} attention values",
            numerical_contract=contract,
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
            attributes={"layer": layer, "position": p, "position_form": "row" if isinstance(position, SymbolicTensor) else "scalar"},
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
            attributes={"scale": float(scale), "layer": layer, "position": p, "kv_heads": kvh, "position_form": "row" if isinstance(position, SymbolicTensor) else "scalar"},
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
            attributes={"layer": layer, "position": p, "kv_heads": kvh, "position_form": "row" if isinstance(position, SymbolicTensor) else "scalar"},
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
    # MoE decode (issue #98)
    # ------------------------------------------------------------------

    def moe_route(
        self,
        x: SymbolicTensor,
        router_w: SymbolicTensor,
        ids: SymbolicTensor,
        weights: SymbolicTensor,
        *,
        mode: str = "learned",
        score_fn: str = "sqrtsoftplus",
        top_k: Optional[int] = None,
        routed_scaling_factor: float = 1.0,
        bias: Optional[SymbolicTensor] = None,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        tid2eid: Optional[SymbolicTensor] = None,
        token_ids: Optional[SymbolicTensor] = None,
        name: str = "moe_route",
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Routed-MoE router decode step (issue #98).

        Computes per-row expert scores and writes the **routing table** —
        external scratch ``ids`` i32 [B, k] and ``weights`` f32 [B, k] — that
        the expert and combine phases consume (RAW-ordered). Two families,
        both floe-exact:

        * ``mode="learned", score_fn="sqrtsoftplus"`` (DeepSeek-V4
          ``DeepseekV4TopKRouter``): global top-k over sqrt(softplus(logits)),
          unconditional renorm ``w / (Σw + 1e-20)``, × ``routed_scaling_factor``.
        * ``mode="learned", score_fn="sigmoid_noaux_tc"`` (GLM-5.3
          ``Glm53TopkRouter``): fp32 sigmoid scores; biased choice scores
          ``s + bias`` restricted to the top-2-sum ``topk_group`` expert groups;
          top-k over the masked choice scores (``sorted=False``); weights gather
          the *unbiased* scores, renorm iff ``norm_topk_prob``, × scaling factor.
        * ``mode="hash"`` (``DeepseekV4HashRouter``): expert selection is the
          frozen ``tid2eid[token_id]`` gather; weights are the gathered scores
          renormed (unconditionally), × scaling factor.

        Selection ties resolve to the **lower expert index** (stable descending
        order) — the documented determinism contract of this lowering.
        """
        b, h = x.value.shape
        e, h_w = router_w.value.shape
        if h_w != h:
            raise ValueError(f"{name}: router weight contraction mismatch ({h_w} != {h})")
        k = top_k if top_k is not None else int(weights.value.shape[1])
        if ids.value.dtype != I32 or weights.value.dtype != F32:
            raise ValueError(f"{name}: routing table must be i32 ids + f32 weights, got {ids.value.dtype}/{weights.value.dtype}")
        if tuple(ids.value.shape) != (b, k) or tuple(weights.value.shape) != (b, k):
            raise ValueError(f"{name}: routing table must be [B={b}, k={k}], got {tuple(ids.value.shape)}/{tuple(weights.value.shape)}")
        if k < 1 or k > e:
            raise ValueError(f"{name}: top_k={k} outside [1, E={e}]")
        attrs = {
            "mode": mode,
            "score_fn": score_fn,
            "top_k": k,
            "num_experts": e,
            "routed_scaling_factor": float(routed_scaling_factor),
        }
        inputs: list[SymbolicTensor] = [x, router_w, ids, weights]
        if mode == "hash":
            if tid2eid is None or token_ids is None:
                raise ValueError(f"{name}: hash routing requires tid2eid [V, k] and token_ids [B]")
            if tid2eid.value.dtype != I32 or tuple(tid2eid.value.shape)[1] != k:
                raise ValueError(f"{name}: tid2eid must be i32 [V, k={k}], got {tid2eid.value.shape}")
            if token_ids.value.dtype != I32 or tuple(token_ids.value.shape) != (b,):
                raise ValueError(f"{name}: token_ids must be i32 [B={b}], got {token_ids.value.shape}")
            inputs += [tid2eid, token_ids]
            if score_fn != "sqrtsoftplus":
                raise ValueError(f"{name}: hash routing (DeepSeek-V4) uses the sqrtsoftplus score; got {score_fn!r}")
        elif mode == "learned" and score_fn == "sigmoid_noaux_tc":
            if bias is None:
                raise ValueError(f"{name}: sigmoid_noaux_tc requires the e_score_correction_bias [E]")
            if tuple(bias.value.shape) != (e,):
                raise ValueError(f"{name}: bias must be [E={e}], got {tuple(bias.value.shape)}")
            if n_group < 1 or e % n_group:
                raise ValueError(f"{name}: n_group={n_group} must divide E={e}")
            if not (1 <= topk_group <= n_group):
                raise ValueError(f"{name}: topk_group={topk_group} outside [1, n_group={n_group}]")
            if k > 2 * topk_group * (e // n_group):
                raise ValueError(f"{name}: top_k={k} exceeds the group-restricted selection pool ({2 * topk_group * (e // n_group)})")
            if bias.value.dtype != F32:
                raise ValueError(f"{name}: bias must be f32, got {bias.value.dtype}")
            inputs.append(bias)
            attrs.update({"n_group": n_group, "topk_group": topk_group, "norm_topk_prob": bool(norm_topk_prob), "routed_bias": True})
        elif mode == "learned" and score_fn == "sqrtsoftplus":
            attrs.update({"norm_topk_prob": True, "routed_bias": False})
        else:
            raise ValueError(f"{name}: unsupported router family mode={mode!r} score_fn={score_fn!r}")
        route_reads = [_regional_reads(x.value), _regional_reads(router_w.value)]
        if mode == "hash":
            route_reads += [_regional_reads(tid2eid.value), _regional_reads(token_ids.value)]
        elif mode == "learned" and score_fn == "sigmoid_noaux_tc":
            route_reads.append(_regional_reads(bias.value))
        self._record(
            "moe_route",
            inputs=inputs,
            outputs=(ids, weights),
            attributes=attrs,
            reads=tuple(route_reads),
            writes=(_regional_reads(ids.value), _regional_reads(weights.value)),
            source_location=name,
            numerical_contract={
                "scores": "fp32 (f64 in the reference body): sigmoid(logits) + bias for noaux_tc, sqrt(softplus(logits)) for sqrtsoftplus",
                "selection": f"top-{k} by choice scores, stable descending, ties to the lower expert index; sorted=False semantics",
                "weights": "unbiased score gather" + (", renorm w/(Σw+1e-20)" if attrs.get("norm_topk_prob") else "") + f", × {routed_scaling_factor}",
            },
        )
        return ids, weights

    def moe_expert(
        self,
        x: SymbolicTensor,
        gate_up: SymbolicTensor,
        down: SymbolicTensor,
        ids: SymbolicTensor,
        *,
        out: Optional[SymbolicTensor] = None,
        swiglu_limit: Optional[float] = None,
        name: str = "moe_expert",
    ) -> SymbolicTensor:
        """Routed-expert FFN decode tasks (issue #98): the static k·B grid.

        One task per (row, slot) computes the full expert FFN for that row's
        routed expert — the weight base ``e = ids[b, slot]`` is **runtime
        indirection** through the routing table (the #94 slot-table pattern
        applied to a read-only weight pool):

            g, u = gate_up[e] @ x ;  act = silu(clamp(g, ≤L)) · clamp(u, ±L)
            partials[b, slot] = down[e] @ act

        ``swiglu_limit=L`` (GLM-5.3 ``Glm53Experts`` / DeepSeek-V4 experts:
        gate clamped to ≤L, up clamped to ±L) folds into the activation.
        Weights are fp32 in this slice (bf16/fp8-block variants are dtype
        follow-ups on the same task shape).
        """
        b, h = x.value.shape
        e, two_i, h_w = gate_up.value.shape
        e_d, h_d, i_d = down.value.shape
        k = int(ids.value.shape[1])
        if h_w != h or e_d != e or h_d != h or two_i != 2 * i_d:
            raise ValueError(f"{name}: expert stack shape mismatch gate_up={gate_up.value.shape} down={down.value.shape} x H={h}")
        if ids.value.dtype != I32:
            raise ValueError(f"{name}: routing ids must be i32, got {ids.value.dtype}")
        if gate_up.value.dtype != F32 or down.value.dtype != F32 or x.value.dtype != F32:
            raise ValueError(f"{name}: this slice runs fp32 expert weights; got {x.value.dtype}/{gate_up.value.dtype}/{down.value.dtype}")
        out = out or self.fresh_buffer(f"{name}{self._suffix()}_partials", (b, k, h))
        if tuple(out.value.shape) != (b, k, h):
            raise ValueError(f"{name}: partials must be [B={b}, k={k}, H={h}], got {tuple(out.value.shape)}")
        attrs = {"num_experts": e, "intermediate": i_d, "top_k": k}
        if swiglu_limit is not None:
            attrs["swiglu_limit"] = float(swiglu_limit)
        self._record(
            "moe_expert",
            inputs=(x, gate_up, down, ids),
            outputs=(out,),
            attributes=attrs,
            reads=(_regional_reads(x.value), _regional_reads(gate_up.value), _regional_reads(down.value), _regional_reads(ids.value)),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "indirection": f"weight base = gate_up[ids[b, slot]] / down[ids[b, slot]] — runtime slot gather (#94 pattern)",
                "activation": "g ≤ L, u ∈ [−L, L], silu(g)·u, fp32" + (f" (L={swiglu_limit})" if swiglu_limit is not None else " (unclamped)"),
                "reduction": "f32, h ascending over the full K/H sweep per task",
            },
        )
        return out

    def moe_combine(
        self,
        partials: SymbolicTensor,
        weights: SymbolicTensor,
        *,
        shared: Optional[SymbolicTensor] = None,
        out: Optional[SymbolicTensor] = None,
        name: str = "moe_combine",
    ) -> SymbolicTensor:
        """Weighted scatter-add per row (issue #98): ``y = Σ_k w_k·h_k`` (+ the
        dense shared-expert path when given). One task per row; fp32
        accumulation in slot order. RAW on the routing ``weights`` (after the
        router) and on the expert ``partials`` (after the expert phase).
        """
        b, k, h = partials.value.shape
        if weights.value.dtype != F32 or tuple(weights.value.shape) != (b, k):
            raise ValueError(f"{name}: weights must be f32 [B={b}, k={k}], got {weights.value.dtype}/{tuple(weights.value.shape)}")
        if shared is not None and tuple(shared.value.shape) != (b, h):
            raise ValueError(f"{name}: shared path must be [B={b}, H={h}], got {tuple(shared.value.shape)}")
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", (b, h))
        if tuple(out.value.shape) != (b, h):
            raise ValueError(f"{name}: output must be [B={b}, H={h}], got {tuple(out.value.shape)}")
        inputs = (partials, weights, shared) if shared is not None else (partials, weights)
        self._record(
            "moe_combine",
            inputs=inputs,
            outputs=(out,),
            attributes={"top_k": k, "shared": shared is not None},
            reads=(_regional_reads(partials.value), _regional_reads(weights.value)) + ((_regional_reads(shared.value),) if shared is not None else ()),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "combine": f"Σ over k={k} slots in slot order, fp32 accumulate" + ("; + shared expert row" if shared is not None else ""),
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
