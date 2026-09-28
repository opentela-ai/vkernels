"""Attention operator capture rules."""
from __future__ import annotations

from typing import Optional

from .operator_ir import (
    F32,
    I32,
    Region,
    ValidLength,
)

from .capture_types import SymbolicTensor, CaptureError, _regional_reads

class AttentionRecording:
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

