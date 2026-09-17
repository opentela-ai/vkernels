"""Task lowerings: the reusable task-template registry (§3.1, §6, M2).

Each supported operator kind maps to a factory that instantiates a
:class:`~vkernels.compiler.task_ir.TaskFamily` from the recorded
operator, exactly as §3.1 describes ("a lowering registry maps each
supported operation to one or more CuTe task implementations; the compiler
selects implementations, instantiates tile domains, assigns buffers,
constructs a schedule, and emits the final kernel").

Tile constants follow the worked example of §6.4: the GEMM keeps 16x16
output tiles with a full-K reduction per task. Task counts follow from the
output dimensions, not from any desired SM count.

Every family declares the common thread-group contract T=256 (§7.1) and a
block-local finite completion contract (§6.3).
"""

from __future__ import annotations

from typing import Callable

from ..operator_ir import Operator, OperatorGraph, Region, TensorValue
from ..task_ir import TaskFamily, TileDomain

__all__ = [
    "GEMM_TILE_M",
    "GEMM_TILE_N",
    "ELEM_TILE",
    "EMBED_TILE_C",
    "THREADS_PER_WORKER",
    "LOWERINGS",
    "lower_graph",
    "lower_op",
]

# -- tile constants (§6.4) ---------------------------------------------------

GEMM_TILE_M = 16
GEMM_TILE_N = 16
ELEM_TILE = 1024  # elements per elementwise task
EMBED_TILE_C = 256  # hidden-width slice per embedding task
THREADS_PER_WORKER = 256  # common thread-group contract (§7.1)


def _whole(view: TensorValue) -> Region:
    return Region.whole(view)


def _tile_region(view: TensorValue, box) -> Region:
    return Region.tile(view, box)


# ---------------------------------------------------------------------------
# GEMM: y = x @ w (+ b), w stored [K, N] (§6.2)
# ---------------------------------------------------------------------------


def _gemm_domain(x: TensorValue, w: TensorValue) -> TileDomain:
    m = x.shape[0]
    n = w.shape[1]
    return TileDomain(((m, GEMM_TILE_M), (n, GEMM_TILE_N)))


def lower_linear(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x, w = graph.tensor(op.inputs[0]), graph.tensor(op.inputs[1])
    bias = graph.tensor(op.inputs[2]) if len(op.inputs) > 2 else None
    y = graph.tensor(op.outputs[0])
    k = x.shape[1]
    domain = _gemm_domain(x, w)

    def reads(coords):
        (m0, m1), (n0, n1) = _pair_box(domain, coords)
        regs = [_tile_region(x, ((m0, m1), (0, k))), _tile_region(w, ((0, k), (n0, n1)))]
        if bias is not None:
            regs.append(_tile_region(bias, ((n0, n1),)))
        return tuple(regs)

    def writes(coords):
        (m0, m1), (n0, n1) = _pair_box(domain, coords)
        return (_tile_region(y, ((m0, m1), (n0, n1))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_linear",
        kind="gemm",
        op=op,
        domain=domain,
        inputs=(x.name, w.name) + ((bias.name,) if bias else ()),
        outputs=(y.name,),
        params={"bias": bias is not None, "tile_m": GEMM_TILE_M, "tile_n": GEMM_TILE_N},
        threads=THREADS_PER_WORKER,
        # Shared-memory staging of one 16x16 A tile and one 16x16 W tile (f32).
        scratch_bytes=2 * GEMM_TILE_M * 16 * 4,
        read_regions=reads,
        write_regions=writes,
    )


def _pair_box(domain: TileDomain, coords):
    m_idx, n_idx = coords
    (m_extent, m_tile), (n_extent, n_tile) = domain.dims
    m0, n0 = m_idx * m_tile, n_idx * n_tile
    return (m0, min(m0 + m_tile, m_extent)), (n0, min(n0 + n_tile, n_extent))


# ---------------------------------------------------------------------------
# LayerNorm: one task per row (row-wide reduction, §10.3)
# ---------------------------------------------------------------------------


def lower_layer_norm(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x = graph.tensor(op.inputs[0])
    gamma, beta = graph.tensor(op.inputs[1]), graph.tensor(op.inputs[2])
    y = graph.tensor(op.outputs[0])
    rows, width = x.shape
    domain = TileDomain(((rows, 1), (width, width)))

    def reads(coords):
        (r0, r1) = (coords[0], coords[0] + 1)
        return (
            _tile_region(x, ((r0, r1), (0, width))),
            _whole(gamma),
            _whole(beta),
        )

    def writes(coords):
        return (_tile_region(y, ((coords[0], coords[0] + 1), (0, width))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_layernorm",
        kind="layernorm",
        op=op,
        domain=domain,
        inputs=(x.name, gamma.name, beta.name),
        outputs=(y.name,),
        params={"eps": op.attributes.get("eps", 1e-5)},
        threads=THREADS_PER_WORKER,
        scratch_bytes=2 * width * 4,  # row copy + running statistics
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Elementwise: gelu, add — strided over elements with the common block size
# ---------------------------------------------------------------------------


def _flat_box(view: TensorValue, lo: int, hi: int):
    """[lo, hi) flat-element box on a contiguous row-major view."""
    if not view.is_contiguous():
        raise ValueError(f"elementwise tiling requires contiguous views ({view.name})")
    shape = view.shape
    if len(shape) == 1:
        return ((lo, hi),)
    cols = shape[-1]
    r0, c0 = divmod(lo, cols)
    r1, c1 = divmod(hi - 1, cols)
    return ((r0, r1 + 1), (c0 if r0 == r1 else 0, (c1 + 1) if r0 == r1 else cols))


def lower_gelu(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x = graph.tensor(op.inputs[0])
    y = graph.tensor(op.outputs[0])
    domain = TileDomain(((x.numel, ELEM_TILE),))

    def reads(coords):
        lo, hi = coords[0] * ELEM_TILE, min((coords[0] + 1) * ELEM_TILE, x.numel)
        return (_tile_region(x, _flat_box(x, lo, hi)),)

    def writes(coords):
        lo, hi = coords[0] * ELEM_TILE, min((coords[0] + 1) * ELEM_TILE, y.numel)
        return (_tile_region(y, _flat_box(y, lo, hi)),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_gelu",
        kind="elementwise",
        op=op,
        domain=domain,
        inputs=(x.name,),
        outputs=(y.name,),
        params={"op": "gelu"},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )


def lower_add(op: Operator, graph: OperatorGraph) -> TaskFamily:
    a = graph.tensor(op.inputs[0])
    b = graph.tensor(op.inputs[1])
    y = graph.tensor(op.outputs[0])
    domain = TileDomain(((a.numel, ELEM_TILE),))

    def reads(coords):
        box = _flat_box(a, coords[0] * ELEM_TILE, min((coords[0] + 1) * ELEM_TILE, a.numel))
        return (_tile_region(a, box), _tile_region(b, box))

    def writes(coords):
        box = _flat_box(y, coords[0] * ELEM_TILE, min((coords[0] + 1) * ELEM_TILE, y.numel))
        return (_tile_region(y, box),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_add",
        kind="elementwise",
        op=op,
        domain=domain,
        inputs=(a.name, b.name),
        outputs=(y.name,),
        params={"op": "add"},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Embedding: one task per token row, tile over the hidden width
# ---------------------------------------------------------------------------


def lower_embedding(op: Operator, graph: OperatorGraph) -> TaskFamily:
    ids = graph.tensor(op.inputs[0])
    token = graph.tensor(op.inputs[1])
    pos = graph.tensor(op.inputs[2]) if len(op.inputs) > 2 else None
    y = graph.tensor(op.outputs[0])
    rows, width = y.shape
    domain = TileDomain(((rows, 1), (width, min(EMBED_TILE_C, width))))

    def reads(coords):
        b = coords[0]
        (c0, c1) = (0, width) if domain.task_grid[1] == 1 else (coords[1] * EMBED_TILE_C, min((coords[1] + 1) * EMBED_TILE_C, width))
        regs = [_tile_region(ids, ((b, b + 1),)), _whole(token)]
        if pos is not None:
            # Position row p is symbolic: whole table (conservative).
            regs.append(_whole(pos))
        return tuple(regs)

    def writes(coords):
        b = coords[0]
        (c0, c1) = (0, width) if domain.task_grid[1] == 1 else (coords[1] * EMBED_TILE_C, min((coords[1] + 1) * EMBED_TILE_C, width))
        return (_tile_region(y, ((b, b + 1), (c0, c1))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_embedding",
        kind="embedding",
        op=op,
        domain=domain,
        inputs=(ids.name, token.name) + ((pos.name,) if pos else ()),
        outputs=(y.name,),
        params={"position": op.attributes.get("position", "p"), "pos_table": pos is not None},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# RMSNorm: one task per last-dim row (hidden states or per-head QK-norm)
# ---------------------------------------------------------------------------


def lower_rms_norm(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x = graph.tensor(op.inputs[0])
    gamma = graph.tensor(op.inputs[1])
    y = graph.tensor(op.outputs[0])
    if len(x.shape) == 3:
        B, H, D = x.shape
        domain = TileDomain(((B, 1), (H, 1)))

        def reads3(coords):
            b, h = coords
            return (_tile_region(x, ((b, b + 1), (h, h + 1), (0, D))), _whole(gamma))

        def writes3(coords):
            b, h = coords
            return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, D))),)

        reads, writes, width = reads3, writes3, D
    else:
        rows, width = x.shape
        domain = TileDomain(((rows, 1), (width, width)))

        def reads2(coords):
            return (_tile_region(x, ((coords[0], coords[0] + 1), (0, width))), _whole(gamma))

        def writes2(coords):
            return (_tile_region(y, ((coords[0], coords[0] + 1), (0, width))),)

        reads, writes = reads2, writes2

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_rmsnorm",
        kind="rms_norm",
        op=op,
        domain=domain,
        inputs=(x.name, gamma.name),
        outputs=(y.name,),
        params={"eps": op.attributes.get("eps", 1e-6)},
        threads=THREADS_PER_WORKER,
        scratch_bytes=width * 4,
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# RoPE: one task per (batch, head); reads row p of the cos/sin tables
# ---------------------------------------------------------------------------


def lower_rope(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x = graph.tensor(op.inputs[0])
    cos_t = graph.tensor(op.inputs[1])
    sin_t = graph.tensor(op.inputs[2])
    y = graph.tensor(op.outputs[0])
    B, H, D = x.shape
    domain = TileDomain(((B, 1), (H, 1)))

    def reads(coords):
        b, h = coords
        return (
            _tile_region(x, ((b, b + 1), (h, h + 1), (0, D))),
            # Row p is symbolic: whole tables (conservative).
            _whole(cos_t),
            _whole(sin_t),
        )

    def writes(coords):
        b, h = coords
        return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, D))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_rope",
        kind="rope",
        op=op,
        domain=domain,
        inputs=(x.name, cos_t.name, sin_t.name),
        outputs=(y.name,),
        params={"layer": op.attributes.get("layer", 0), "which": op.attributes.get("which", "q"), "position": op.attributes.get("position", "p")},
        threads=THREADS_PER_WORKER,
        scratch_bytes=D * 4,
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# SwiGLU: elementwise silu(gate) * up over [B, F]
# ---------------------------------------------------------------------------


def lower_swiglu(op: Operator, graph: OperatorGraph) -> TaskFamily:
    gate = graph.tensor(op.inputs[0])
    up = graph.tensor(op.inputs[1])
    y = graph.tensor(op.outputs[0])
    domain = TileDomain(((gate.numel, ELEM_TILE),))

    def reads(coords):
        box = _flat_box(gate, coords[0] * ELEM_TILE, min((coords[0] + 1) * ELEM_TILE, gate.numel))
        return (_tile_region(gate, box), _tile_region(up, box))

    def writes(coords):
        box = _flat_box(y, coords[0] * ELEM_TILE, min((coords[0] + 1) * ELEM_TILE, y.numel))
        return (_tile_region(y, box),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_swiglu",
        kind="elementwise",
        op=op,
        domain=domain,
        inputs=(gate.name, up.name),
        outputs=(y.name,),
        params={"op": "swiglu"},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Cache append: one task per (batch, head); writes row p (§4.3, §5.3)
# ---------------------------------------------------------------------------


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
        params={"layer": op.attributes.get("layer", 0), "position": p},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# GDN conv: decode-step FIR + state shift; one task per (batch, channel tile)
# ---------------------------------------------------------------------------


def _gdn_tile(C: int) -> int:
    """Channel-tile width for gdn_conv: ELEM_TILE when it divides C, else the
    largest power-of-two divisor that does (the device template requires an
    exact tiling, no channel masking)."""
    tile = ELEM_TILE
    while C % tile:
        tile //= 2
    return tile


def lower_gdn_conv(op: Operator, graph: OperatorGraph) -> TaskFamily:
    state = graph.tensor(op.inputs[0])
    x = graph.tensor(op.inputs[1])
    w = graph.tensor(op.inputs[2])
    out = graph.tensor(op.outputs[0])
    B, Km1, C = state.shape
    K = Km1 + 1
    if tuple(w.shape) != (C, K):
        raise ValueError(f"gdn_conv FIR weights {w.shape} do not match state pool {[B, Km1, C]} (need [{C}, {K}])")
    tile = _gdn_tile(C)
    domain = TileDomain(((B, 1), (C, tile)))

    def reads(coords):
        b, c = coords
        c0, c1 = c * tile, min((c + 1) * tile, C)
        return (
            _tile_region(state, ((b, b + 1), (0, Km1), (c0, c1))),
            _tile_region(x, ((b, b + 1), (c0, c1))),
            _tile_region(w, ((c0, c1), (0, K))),
        )

    def writes(coords):
        b, c = coords
        c0, c1 = c * tile, min((c + 1) * tile, C)
        # Read-modify-write: the task shifts its own state tile in place.
        return (
            _tile_region(state, ((b, b + 1), (0, Km1), (c0, c1))),
            _tile_region(out, ((b, b + 1), (c0, c1))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_gdn_conv",
        kind="gdn_conv",
        op=op,
        domain=domain,
        inputs=(state.name, x.name, w.name),
        outputs=(out.name,),
        params={"layer": op.attributes.get("layer", 0), "conv_kernel": K, "tile": tile},
        threads=THREADS_PER_WORKER,
        scratch_bytes=tile * 4,  # fp32 accumulator for one channel tile
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Attention: one task per (batch, head); valid prefix [0, p] (§5.3, §10.3)
# ---------------------------------------------------------------------------


def lower_gdn_delta(op: Operator, graph: OperatorGraph) -> TaskFamily:
    """One task per (batch, value head): the task owns the head's [HV, HK]
    state slice (read-modify-write, in registers on device) and its v/z
    rows; q/k are read per key head (group-expanded). The state WAW/WAR is
    phase-ordered like the KV append — two gdn_delta ops on one pool chain
    through the state-storage hazards."""
    state = graph.tensor(op.inputs[0])
    q = graph.tensor(op.inputs[1])
    k = graph.tensor(op.inputs[2])
    v = graph.tensor(op.inputs[3])
    z = graph.tensor(op.inputs[4])
    a = graph.tensor(op.inputs[5])
    b = graph.tensor(op.inputs[6])
    a_log = graph.tensor(op.inputs[7])
    dt_bias = graph.tensor(op.inputs[8])
    norm_w = graph.tensor(op.inputs[9])
    out = graph.tensor(op.outputs[0])
    B, NV, HV, HK = state.shape
    NK = q.shape[1]
    if NV % NK:
        raise ValueError(f"gdn_delta head group must divide: NV={NV} not divisible by NK={NK}")
    domain = TileDomain(((B, 1), (NV, 1)))

    def reads(coords):
        bb, h = coords
        kh = h // (NV // NK)
        return (
            _tile_region(state, ((bb, bb + 1), (h, h + 1), (0, HV), (0, HK))),
            _tile_region(q, ((bb, bb + 1), (kh, kh + 1), (0, HK))),
            _tile_region(k, ((bb, bb + 1), (kh, kh + 1), (0, HK))),
            _tile_region(v, ((bb, bb + 1), (h, h + 1), (0, HV))),
            _tile_region(z, ((bb, bb + 1), (h, h + 1), (0, HV))),
            _tile_region(a, ((bb, bb + 1), (h, h + 1))),
            _tile_region(b, ((bb, bb + 1), (h, h + 1))),
            _whole(a_log),
            _whole(dt_bias),
            _whole(norm_w),
        )

    def writes(coords):
        bb, h = coords
        # Read-modify-write: the task updates its own state slice in place.
        return (
            _tile_region(state, ((bb, bb + 1), (h, h + 1), (0, HV), (0, HK))),
            _tile_region(out, ((bb, bb + 1), (h, h + 1), (0, HV))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_gdn_delta",
        kind="gdn_delta",
        op=op,
        domain=domain,
        inputs=(state.name, q.name, k.name, v.name, z.name, a.name, b.name, a_log.name, dt_bias.name, norm_w.name),
        outputs=(out.name,),
        params={
            "layer": op.attributes.get("layer", 0),
            "scale": op.attributes.get("scale", 1.0),
            "eps": op.attributes.get("eps", 1e-6),
        },
        threads=THREADS_PER_WORKER,
        scratch_bytes=HV * HK * 4,  # the [HV, HK] fp32 state slice (registers/scratch budget)
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Attention: one task per (batch, head); valid prefix [0, p] (§5.3, §10.3)
# ---------------------------------------------------------------------------


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
        params={"scale": float(op.attributes.get("scale") or 0.0), "layer": op.attributes.get("layer", 0), "position": p, "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=S * 4,  # per-task score row
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
        params={"layer": op.attributes.get("layer", 0), "position": p},
        threads=THREADS_PER_WORKER,
        scratch_bytes=2 * S * 4,  # row + running max/sum
        read_regions=reads,
        write_regions=writes,
    )


def lower_attention_values(op: Operator, graph: OperatorGraph) -> TaskFamily:
    probs = graph.tensor(op.inputs[0])
    v_cache = graph.tensor(op.inputs[1])
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
        return (
            _tile_region(probs, ((b, b + 1), (h, h + 1), (0, f"{p}+1"))),
            _tile_region(v_cache, ((b, b + 1), (h // group, h // group + 1), (0, f"{p}+1"), (0, D))),
        )

    def writes(coords):
        b, h = coords
        return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, D))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_attn_values",
        kind="attention_values",
        op=op,
        domain=domain,
        inputs=(probs.name, v_cache.name),
        outputs=(y.name,),
        params={"layer": op.attributes.get("layer", 0), "position": p, "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=D * 4,  # per-task context accumulator
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Registry (§3.1)
# ---------------------------------------------------------------------------

LOWERINGS: dict[str, Callable[[Operator, OperatorGraph], TaskFamily]] = {
    "linear": lower_linear,
    "layer_norm": lower_layer_norm,
    "rms_norm": lower_rms_norm,
    "rope": lower_rope,
    "gelu": lower_gelu,
    "swiglu": lower_swiglu,
    "add": lower_add,
    "embedding": lower_embedding,
    "cache_append": lower_cache_append,
    "gdn_conv": lower_gdn_conv,
    "gdn_delta": lower_gdn_delta,
    "attention_scores": lower_attention_scores,
    "softmax": lower_softmax,
    "attention_values": lower_attention_values,
}

LOWERINGS_VERSION = "0.1.0"


def lower_op(op: Operator, graph: OperatorGraph) -> TaskFamily:
    factory = LOWERINGS.get(op.kind)
    if factory is None:
        raise KeyError(f"no task lowering registered for operator kind {op.kind!r}")
    return factory(op, graph)


def lower_graph(graph: OperatorGraph) -> list[TaskFamily]:
    """One family per recorded operator, in program (phase) order."""
    return [lower_op(op, graph) for op in graph.ops]


def check_thread_contract(families: list[TaskFamily]) -> list[str]:
    """All families must agree on the common thread count (§7.1)."""
    threads = {f.threads for f in families}
    if len(threads) != 1:
        return [f"thread-group contract mismatch: families use {sorted(threads)}; a persistent worker cannot change its physical block size between tasks"]
    return []
