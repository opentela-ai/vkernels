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
            attributes={"position": self._position_name(position), "pos_table": position_emb is not None},
            reads=tuple(reads),
            writes=(_regional_reads(out.value),),
            source_location="embedding lookup",
            numerical_contract={"sum": "token_row(ids[b]) + position_row(p)" if position_emb is not None else "token_row(ids[b])", "dtype": "f32"},
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
    ) -> SymbolicTensor:
        """y = x @ w (+ b); ``w`` is stored [Cin, Cout] (§3.1)."""
        m = x.value.shape[0]
        n = w.value.shape[1]
        out = out or self.fresh_buffer(f"{name}{self._suffix()}", (m, n))
        reads = [_regional_reads(x.value), _regional_reads(w.value)]
        inputs = [x, w]
        if b is not None:
            reads.append(_regional_reads(b.value))
            inputs.append(b)
        self._record(
            "linear",
            inputs=tuple(inputs),
            outputs=(out,),
            attributes={"bias": b is not None},
            reads=tuple(reads),
            writes=(_regional_reads(out.value),),
            source_location=name,
            numerical_contract={
                "accumulation": "f32, full-K reduction per output tile, k ascending",
                "bias": "added after the K reduction" if b is not None else "none",
            },
        )
        return out

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
    ) -> SymbolicTensor:
        """Rotate-half RoPE at the runtime position p (Qwen3 convention).

        Tables are precomputed [max_positions, D] externals with
        cos = cat([f, f]) so x' = x*cos[p] + rotate_half(x)*sin[p].
        """
        self._require_position(position, "rope")
        out = self.fresh_buffer(f"rope_{which}_l{layer}{self._suffix()}", x.value.shape)
        p = self._position_name(position)
        self._record(
            "rope",
            inputs=(x, cos_table, sin_table),
            outputs=(out,),
            attributes={"layer": layer, "which": which, "position": p},
            reads=(_regional_reads(x.value), _regional_reads(cos_table.value), _regional_reads(sin_table.value)),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} rope {which}",
            numerical_contract={
                "rotate_half": "cat(-x[D/2:], x[:D/2])",
                "tables": "full-width cos/sin (HF rotate_half convention)",
            },
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
        self._require_position(position, "cache_append")
        p = self._position_name(position)
        writes = [
            Region.prefix(k_cache.value, axis=2, valid=ValidLength(f"{p}+1")),
            Region.prefix(v_cache.value, axis=2, valid=ValidLength(f"{p}+1")),
        ]
        self._record(
            "cache_append",
            inputs=(k_cache, v_cache, k_new, v_new),
            outputs=(),  # mutation through storage effects; views returned below
            attributes={"layer": layer, "position": p},
            reads=(_regional_reads(k_new.value), _regional_reads(v_new.value)),
            writes=tuple(writes),
            source_location=f"layer {layer} kv cache append",
            numerical_contract={
                "write": "K[l,b,h,p,:] = k_new[b,h,:]; same for V",
                "valid_length": "p+1",
            },
        )
        # Post-append versions: same storages, prefix now valid to p+1.
        k_post = self.graph.add_tensor(
            f"k_cache_l{layer}_v{self.graph.storage_versions[k_cache.value.storage_id]}",
            k_cache.value.shape,
            k_cache.value.dtype,
            storage_id=k_cache.value.storage_id,
            strides=k_cache.value.strides,
            offset=k_cache.value.offset,
            valid_length=ValidLength(f"{p}+1"),
        )
        v_post = self.graph.add_tensor(
            f"v_cache_l{layer}_v{self.graph.storage_versions[v_cache.value.storage_id]}",
            v_cache.value.shape,
            v_cache.value.dtype,
            storage_id=v_cache.value.storage_id,
            strides=v_cache.value.strides,
            offset=v_cache.value.offset,
            valid_length=ValidLength(f"{p}+1"),
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
        kvh = kv_heads if kv_heads is not None else h

        def k_regions() -> tuple[Region, ...]:
            if kvh == h:
                return (Region.prefix(k_cache.value, axis=2, valid=ValidLength(f"{p}+1")),)
            group = h // kvh
            return tuple(
                Region(
                    k_cache.value.storage_id,
                    k_cache.value,
                    ((0, b), (g * group, (g + 1) * group), (0, f"{p}+1"), (0, k_cache.value.shape[3])),
                )
                for g in range(kvh)
            )

        self._record(
            "attention_scores",
            inputs=(q, k_cache),
            outputs=(out,),
            attributes={"scale": float(scale), "layer": layer, "position": p, "kv_heads": kvh},
            reads=(_regional_reads(q.value),) + k_regions(),
            writes=(Region.tile(out.value, ((0, b), (0, h), (0, s))),),
            source_location=f"layer {layer} attention scores",
            numerical_contract={
                "scale": f"1/sqrt(D) = {scale}",
                "gqa": f"q head h reads kv head h // {h // kvh}" if kvh != h else "none (MHA)",
                "valid": "only positions [0, p] are read; scores beyond p are not observed by consumers",
                "reduction": "f32 dot over D, k ascending",
            },
        )
        return out

    def softmax(self, scores: SymbolicTensor, position, *, layer: int) -> SymbolicTensor:
        """Row softmax over the valid prefix [0, p]; invalid tail forced to 0."""
        self._require_position(position, "softmax")
        out = self.fresh_buffer(f"probs_l{layer}{self._suffix()}", scores.value.shape)
        p = self._position_name(position)
        self._record(
            "softmax",
            inputs=(scores,),
            outputs=(out,),
            attributes={"layer": layer, "position": p},
            reads=(Region.prefix(scores.value, axis=2, valid=ValidLength(f"{p}+1")),),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} softmax",
            numerical_contract={
                "stability": "max-subtraction",
                "invalid_tail": "probabilities beyond p are exactly 0 (written, not read from cache)",
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
        kvh = kv_heads if kv_heads is not None else h

        def v_regions() -> tuple[Region, ...]:
            if kvh == h:
                return (Region.prefix(v_cache.value, axis=2, valid=ValidLength(f"{p}+1")),)
            group = h // kvh
            return tuple(
                Region(
                    v_cache.value.storage_id,
                    v_cache.value,
                    ((0, b), (g * group, (g + 1) * group), (0, f"{p}+1"), (0, v_cache.value.shape[3])),
                )
                for g in range(kvh)
            )

        self._record(
            "attention_values",
            inputs=(probs, v_cache),
            outputs=(out,),
            attributes={"layer": layer, "position": p, "kv_heads": kvh},
            reads=(Region.prefix(probs.value, axis=2, valid=ValidLength(f"{p}+1")),) + v_regions(),
            writes=(_regional_reads(out.value),),
            source_location=f"layer {layer} attention values",
            numerical_contract={
                "gqa": f"q head h reads kv head h // {h // kvh}" if kvh != h else "none (MHA)",
                "valid": "only positions [0, p] contribute; V beyond p is never loaded (NaN tails must not leak)",
                "reduction": "f32, t ascending",
            },
        )
        return out

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _require_position(self, position, op: str) -> None:
        if not isinstance(position, SymbolicScalar):
            raise CaptureError(f"{op} requires the symbolic decode position; got {position!r}. A concrete position would freeze the cache length into the graph (§4.2).")
        if position.name not in self.graph.scalars:
            raise CaptureError(f"unknown symbolic scalar {position.name!r}")

    @staticmethod
    def _position_name(position) -> str:
        assert isinstance(position, SymbolicScalar)
        return position.name

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
