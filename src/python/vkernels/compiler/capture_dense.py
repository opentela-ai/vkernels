"""Dense operator capture rules."""
from __future__ import annotations

from typing import Optional

from .operator_ir import (
    F32,
    I32,
)

from .capture_types import SymbolicTensor, UnsupportedOperator, CaptureError, _regional_reads, _RMS_GATED_ACTIVATIONS

class DenseRecording:
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
                "indirection": "weight base = gate_up[ids[b, slot]] / down[ids[b, slot]] — runtime slot gather (#94 pattern)",
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

