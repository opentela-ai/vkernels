"""Attention operator tile and memory-region rules."""
from __future__ import annotations
from ..operator_ir import Operator, OperatorGraph
from ..task_ir import TaskFamily, TileDomain
from .common import THREADS_PER_WORKER, _tile_region, INDEXER_TILE_M

def lower_indexer_scores(op: Operator, graph: OperatorGraph) -> TaskFamily:
    q, c, w = (graph.tensor(op.inputs[i]) for i in range(3))
    s = graph.tensor(op.outputs[0])
    b, h, d = q.shape
    m = c.shape[1]
    if op.attributes.get("activation") != "relu":
        raise ValueError(f"indexer_scores {op.source_location!r}: unsupported activation {op.attributes.get('activation')!r}")
    if not c.is_contiguous():
        raise ValueError(f"indexer_scores {op.source_location!r}: entries must be row-major contiguous")
    domain = TileDomain(((b, 1), (m, INDEXER_TILE_M)))

    def reads(coords):
        bi, t = coords
        m0 = t * INDEXER_TILE_M
        m1 = min(m0 + INDEXER_TILE_M, m)
        return (
            _tile_region(q, ((bi, bi + 1), (0, h), (0, d))),
            _tile_region(c, ((bi, bi + 1), (m0, m1), (0, d))),
            _tile_region(w, ((bi, bi + 1), (0, h))),
        )

    def writes(coords):
        bi, t = coords
        m0 = t * INDEXER_TILE_M
        m1 = min(m0 + INDEXER_TILE_M, m)
        return (_tile_region(s, ((bi, bi + 1), (m0, m1))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_indexer_scores",
        kind="indexer_scores",
        op=op,
        domain=domain,
        inputs=(q.name, c.name, w.name),
        outputs=(s.name,),
        params={"heads": h, "head_dim": d, "capacity": m, "scale": op.attributes["scale"], "tile_m": INDEXER_TILE_M},
        threads=THREADS_PER_WORKER,
        # In-register staging: one entry tile [TILE_M, D] + one query row [D] (f32).
        scratch_bytes=(INDEXER_TILE_M + 1) * d * 4,
        read_regions=reads,
        write_regions=writes,
    )

def lower_index_topk(op: Operator, graph: OperatorGraph) -> TaskFamily:
    s, valid = graph.tensor(op.inputs[0]), graph.tensor(op.inputs[1])
    idx, bias = graph.tensor(op.outputs[0]), graph.tensor(op.outputs[1])
    b, m = s.shape
    k = op.attributes.get("k")
    if not isinstance(k, int) or not 1 <= k <= m:
        raise ValueError(f"index_topk {op.source_location!r}: k must be an int in [1, M={m}], got {k!r}")
    if op.attributes.get("tie_break") != "lowest_index":
        raise ValueError(f"index_topk {op.source_location!r}: unsupported tie_break {op.attributes.get('tie_break')!r}")
    domain = TileDomain(((b, 1),))  # one task per batch row, full-candidate sweep

    def reads(coords):
        (bi,) = coords
        return (
            _tile_region(s, ((bi, bi + 1), (0, m))),
            _tile_region(valid, ((bi, bi + 1),)),
        )

    def writes(coords):
        (bi,) = coords
        return (
            _tile_region(idx, ((bi, bi + 1), (0, k))),
            _tile_region(bias, ((bi, bi + 1), (0, k))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_index_topk",
        kind="index_topk",
        op=op,
        domain=domain,
        inputs=(s.name, valid.name),
        outputs=(idx.name, bias.name),
        params={"k": k, "capacity": m, "tie_break": "lowest_index"},
        threads=THREADS_PER_WORKER,
        scratch_bytes=0,  # rank counting works in registers
        read_regions=reads,
        write_regions=writes,
    )

def lower_cache_append(op: Operator, graph: OperatorGraph) -> TaskFamily:
    k_cache, v_cache = graph.tensor(op.inputs[0]), graph.tensor(op.inputs[1])
    k_new, v_new = graph.tensor(op.inputs[2]), graph.tensor(op.inputs[3])
    B, H, S, D = k_cache.shape
    domain = TileDomain(((B, 1), (H, 1)))
    p = op.attributes.get("position", "p")

    def reads(coords):
        b, h = coords
        row = ((b, b + 1), (h, h + 1), (0, D))
        return (_tile_region(k_new, row), _tile_region(v_new, row))

    def writes(coords):
        b, h = coords
        # Only row p is written; the declared region is the valid prefix,
        # conservative in storage extent (§5.2).
        prefix = ((b, b + 1), (h, h + 1), (0, f"{p}+1"), (0, D))
        return (_tile_region(k_cache, prefix), _tile_region(v_cache, prefix))

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_cache_append",
        kind="cache_append",
        op=op,
        domain=domain,
        inputs=(k_cache.name, v_cache.name, k_new.name, v_new.name),
        outputs=(),
        params={"layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar")},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )

def lower_cache_append_paged(op: Operator, graph: OperatorGraph) -> TaskFamily:
    k_pool, v_pool = graph.tensor(op.inputs[0]), graph.tensor(op.inputs[1])
    slot_table = graph.tensor(op.inputs[2])
    k_new, v_new = graph.tensor(op.inputs[3]), graph.tensor(op.inputs[4])
    B, KVH, D = k_new.shape
    domain = TileDomain(((B, 1), (KVH, 1)))
    p = op.attributes.get("position", "p")

    def reads(coords):
        b, h = coords
        row = ((b, b + 1), (h, h + 1), (0, D))
        return (_tile_region(k_new, row), _tile_region(v_new, row))

    def writes(coords):
        # Slot is table[b, p] — runtime data. Whole-pool declaration is the
        # conservative sound choice (#94); slot 0 (sink) is never written.
        return (
            _tile_region(k_pool, ((0, k_pool.shape[0]), (0, k_pool.shape[1]), (0, k_pool.shape[2]))),
            _tile_region(v_pool, ((0, v_pool.shape[0]), (0, v_pool.shape[1]), (0, v_pool.shape[2]))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_cache_append_paged",
        kind="cache_append_paged",
        op=op,
        domain=domain,
        inputs=(k_pool.name, v_pool.name, slot_table.name, k_new.name, v_new.name),
        outputs=(),
        params={"layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar")},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )

def lower_attention_scores(op: Operator, graph: OperatorGraph) -> TaskFamily:
    q = graph.tensor(op.inputs[0])
    k_cache = graph.tensor(op.inputs[1])
    y = graph.tensor(op.outputs[0])
    B = q.shape[0]
    Hq = q.shape[1]  # query heads drive the task domain
    S, D = k_cache.shape[2], k_cache.shape[3]
    kvh = op.attributes.get("kv_heads", Hq) or Hq
    group = Hq // kvh
    domain = TileDomain(((B, 1), (Hq, 1)))
    p = op.attributes.get("position", "p")

    def reads(coords):
        b, h = coords
        return (
            _tile_region(q, ((b, b + 1), (h, h + 1), (0, D))),
            _tile_region(k_cache, ((b, b + 1), (h // group, h // group + 1), (0, f"{p}+1"), (0, D))),
        )

    def writes(coords):
        b, h = coords
        return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, f"{p}+1"))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_attn_scores",
        kind="attention_scores",
        op=op,
        domain=domain,
        inputs=(q.name, k_cache.name),
        outputs=(y.name,),
        params={"scale": float(op.attributes.get("scale") or 0.0), "layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar"), "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=S * 4,  # per-task score row
        read_regions=reads,
        write_regions=writes,
    )

def lower_attention_scores_paged(op: Operator, graph: OperatorGraph) -> TaskFamily:
    q = graph.tensor(op.inputs[0])
    k_pool = graph.tensor(op.inputs[1])
    slot_table = graph.tensor(op.inputs[2])
    y = graph.tensor(op.outputs[0])
    B = q.shape[0]
    Hq = q.shape[1]
    S = slot_table.shape[1]
    kvh = op.attributes.get("kv_heads", Hq) or Hq
    group = Hq // kvh
    domain = TileDomain(((B, 1), (Hq, 1)))
    p = op.attributes.get("position", "p")

    def reads(coords):
        b, h = coords
        return (
            _tile_region(q, ((b, b + 1), (h, h + 1), (0, q.shape[2]))),
            _tile_region(k_pool, ((0, k_pool.shape[0]), (0, k_pool.shape[1]), (0, k_pool.shape[2]))),
        )

    def writes(coords):
        b, h = coords
        return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, f"{p}+1"))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_attn_scores_paged",
        kind="attention_scores_paged",
        op=op,
        domain=domain,
        inputs=(q.name, k_pool.name, slot_table.name),
        outputs=(y.name,),
        params={"scale": float(op.attributes.get("scale") or 0.0), "layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar"), "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=S * 4,
        read_regions=reads,
        write_regions=writes,
    )

def lower_mla_scores(op: Operator, graph: OperatorGraph) -> TaskFamily:
    q, latent, window_table, comp_pool, comp_idx, sink = (graph.tensor(op.inputs[i]) for i in range(6))
    bias = graph.tensor(op.inputs[6]) if len(op.inputs) > 6 else None
    probs = graph.tensor(op.outputs[0])
    b, h, _ = q.shape
    W = op.attributes["window"]
    K = op.attributes["comp_slots"]
    domain = TileDomain(((b, 1), (h, 1)))
    names = [t.name for t in (q, latent, window_table, comp_pool, comp_idx, sink)] + ([bias.name] if bias else [])

    def reads(coords):
        bb, hh = coords
        regs = [
            _tile_region(q, ((bb, bb + 1), (hh, hh + 1), (0, q.shape[2]))),
            _tile_region(latent, ((bb, bb + 1), (0, latent.shape[1]), (0, latent.shape[2]))),
            _tile_region(window_table, ((bb, bb + 1), (0, window_table.shape[1]))),
            _tile_region(comp_pool, ((bb, bb + 1), (0, comp_pool.shape[1]), (0, comp_pool.shape[2]))),
            _tile_region(comp_idx, ((bb, bb + 1), (0, K))),
            _tile_region(sink, ((bb, bb + 1), (hh, hh + 1))) if sink.shape.__len__() == 2 else _tile_region(sink, ((hh, hh + 1),)),
        ]
        if bias is not None:
            regs.append(_tile_region(bias, ((bb, bb + 1), (0, K))))
        return tuple(regs)

    def writes(coords):
        bb, hh = coords
        return (_tile_region(probs, ((bb, bb + 1), (hh, hh + 1), (0, W + K + 1))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_mla_scores",
        kind="mla_scores",
        op=op,
        domain=domain,
        inputs=tuple(names),
        outputs=(probs.name,),
        params={"scale": float(op.attributes["scale"]), "layer": op.attributes["layer"],
                "position": op.attributes["position"], "window": W, "comp_slots": K,
                "position_form": op.attributes.get("position_form", "scalar")},
        threads=THREADS_PER_WORKER,
        # Per-(b,h) candidate logits [W + K + 1] staged in registers; no scratch.
        scratch_bytes=0,
        read_regions=reads,
        write_regions=writes,
    )

def lower_mla_values(op: Operator, graph: OperatorGraph) -> TaskFamily:
    probs, latent, window_table, comp_pool, comp_idx = (graph.tensor(op.inputs[i]) for i in range(5))
    ctx = graph.tensor(op.outputs[0])
    b, h, _ = probs.shape
    W = op.attributes["window"]
    K = op.attributes["comp_slots"]
    domain = TileDomain(((b, 1), (h, 1)))
    d = latent.shape[2]

    def reads(coords):
        bb, hh = coords
        return (
            _tile_region(probs, ((bb, bb + 1), (hh, hh + 1), (0, W + K + 1))),
            _tile_region(latent, ((bb, bb + 1), (0, latent.shape[1]), (0, d))),
            _tile_region(window_table, ((bb, bb + 1), (0, window_table.shape[1]))),
            _tile_region(comp_pool, ((bb, bb + 1), (0, comp_pool.shape[1]), (0, d))),
            _tile_region(comp_idx, ((bb, bb + 1), (0, K))),
        )

    def writes(coords):
        bb, hh = coords
        return (_tile_region(ctx, ((bb, bb + 1), (hh, hh + 1), (0, d))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_mla_values",
        kind="mla_values",
        op=op,
        domain=domain,
        inputs=(probs.name, latent.name, window_table.name, comp_pool.name, comp_idx.name),
        outputs=(ctx.name,),
        params={"layer": op.attributes["layer"], "position": op.attributes["position"],
                "window": W, "comp_slots": K,
                "position_form": op.attributes.get("position_form", "scalar")},
        threads=THREADS_PER_WORKER,
        # One [D] accumulator per (b, h) in registers; no scratch.
        scratch_bytes=0,
        read_regions=reads,
        write_regions=writes,
    )

def lower_conjugate_rope(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x, cos_t, sin_t = (graph.tensor(op.inputs[i]) for i in range(3))
    y = graph.tensor(op.outputs[0])
    b, h = x.shape[0], x.shape[1]

    def reads(coords):
        bb, hh = coords
        return (
            _tile_region(x, ((bb, bb + 1), (hh, hh + 1), (0, x.shape[2]))),
            _tile_region(cos_t, ((0, cos_t.shape[0]), (0, cos_t.shape[1]))),
            _tile_region(sin_t, ((0, sin_t.shape[0]), (0, sin_t.shape[1]))),
        )

    def writes(coords):
        bb, hh = coords
        return (_tile_region(y, ((bb, bb + 1), (hh, hh + 1), (0, x.shape[2]))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_conjugate_rope",
        kind="conjugate_rope",
        op=op,
        domain=TileDomain(((b, 1), (h, 1))),
        inputs=(x.name, cos_t.name, sin_t.name),
        outputs=(y.name,),
        params={"layer": op.attributes["layer"], "which": op.attributes["which"],
                "position": op.attributes["position"], "convention": op.attributes["convention"],
                "rotary_dim": op.attributes["rotary_dim"],
                "position_form": op.attributes.get("position_form", "scalar")},
        threads=THREADS_PER_WORKER,
        # Rotation is in-register per (b, h) row; tables stream from L2 (.cg).
        scratch_bytes=0,
        read_regions=reads,
        write_regions=writes,
    )

def lower_softmax(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x = graph.tensor(op.inputs[0])
    y = graph.tensor(op.outputs[0])
    B, H, S = x.shape
    domain = TileDomain(((B, 1), (H, 1)))
    p = op.attributes.get("position", "p")

    def reads(coords):
        b, h = coords
        return (_tile_region(x, ((b, b + 1), (h, h + 1), (0, f"{p}+1"))),)

    def writes(coords):
        # The whole row is written: the invalid tail becomes exact 0.
        b, h = coords
        return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, S))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_softmax",
        kind="softmax",
        op=op,
        domain=domain,
        inputs=(x.name,),
        outputs=(y.name,),
        params={"layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar")},
        threads=THREADS_PER_WORKER,
        scratch_bytes=2 * S * 4,  # row + running max/sum
        read_regions=reads,
        write_regions=writes,
    )

def lower_attention_values(op: Operator, graph: OperatorGraph) -> TaskFamily:
    gated = op.attributes.get("gated", False)
    probs = graph.tensor(op.inputs[0])
    v_cache = graph.tensor(op.inputs[1])
    gate = graph.tensor(op.inputs[2]) if gated else None
    y = graph.tensor(op.outputs[0])
    B = probs.shape[0]
    Hq = probs.shape[1]  # query heads drive the task domain
    D = v_cache.shape[3]
    kvh = op.attributes.get("kv_heads", Hq) or Hq
    group = Hq // kvh
    domain = TileDomain(((B, 1), (Hq, 1)))
    p = op.attributes.get("position", "p")

    def reads(coords):
        b, h = coords
        regions = (
            _tile_region(probs, ((b, b + 1), (h, h + 1), (0, f"{p}+1"))),
            _tile_region(v_cache, ((b, b + 1), (h // group, h // group + 1), (0, f"{p}+1"), (0, D))),
        )
        if gated:
            regions = regions + (_tile_region(gate, ((b, b + 1), (h, h + 1), (0, D))),)
        return regions

    def writes(coords):
        b, h = coords
        return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, D))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_attn_values",
        kind="attention_values",
        op=op,
        domain=domain,
        inputs=(probs.name, v_cache.name) + ((gate.name,) if gated else ()),
        outputs=(y.name,),
        params={
            "layer": op.attributes.get("layer", 0),
            "position": p,
            "position_form": op.attributes.get("position_form", "scalar"),
            "kv_heads": kvh,
            "group": group,
            "gated": gated,
        },
        threads=THREADS_PER_WORKER,
        scratch_bytes=D * 4,  # per-task context accumulator
        read_regions=reads,
        write_regions=writes,
    )

def lower_attention_values_paged(op: Operator, graph: OperatorGraph) -> TaskFamily:
    probs = graph.tensor(op.inputs[0])
    v_pool = graph.tensor(op.inputs[1])
    slot_table = graph.tensor(op.inputs[2])
    y = graph.tensor(op.outputs[0])
    B = probs.shape[0]
    Hq = probs.shape[1]
    D = v_pool.shape[2]
    kvh = op.attributes.get("kv_heads", Hq) or Hq
    group = Hq // kvh
    domain = TileDomain(((B, 1), (Hq, 1)))
    p = op.attributes.get("position", "p")

    def reads(coords):
        b, h = coords
        return (
            _tile_region(probs, ((b, b + 1), (h, h + 1), (0, f"{p}+1"))),
            _tile_region(v_pool, ((0, v_pool.shape[0]), (0, v_pool.shape[1]), (0, v_pool.shape[2]))),
        )

    def writes(coords):
        b, h = coords
        return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, D))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_attn_values_paged",
        kind="attention_values_paged",
        op=op,
        domain=domain,
        inputs=(probs.name, v_pool.name, slot_table.name),
        outputs=(y.name,),
        params={"layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar"), "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=D * 4,
        read_regions=reads,
        write_regions=writes,
    )
