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

from ..operator_ir import F32, F8_E4M3, Operator, OperatorGraph, Region, TensorValue
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
# fp8-blockwise linear (issue #91): y = x @ dequant(w_fp8, scale)^T
#
# Same tile shape as the dense linear lowering — one task per (m, 16-column)
# output tile with a full-K sweep — but the weight is fp8 e4m3 [N, K] row-major
# (checkpoint nn.Linear layout) with fp32 scales [ceil(N/128), K/128]
# block-major (DeepSeek-style 128x128 block-FP8; ragged trailing N block
# allowed). A 16-column tile always lies within a single 128-wide N block
# (128 = 8 * 16), so each task reads exactly one row of the scale tensor and
# one fp32 scale per 128-deep k-block.
# ---------------------------------------------------------------------------

GEMV_FP8_TILE_N = 16  # output columns per task (within one 128-wide scale block)


def lower_linear_fp8(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x, w, scale = (graph.tensor(op.inputs[i]) for i in range(3))
    y = graph.tensor(op.outputs[0])
    if op.attributes.get("weight_layout") != "fp8_block":
        raise ValueError(f"linear_fp8 {op.source_location!r}: unsupported weight_layout {op.attributes.get('weight_layout')!r}")
    qb = int(op.attributes.get("quant_block", 128))
    m, k = x.shape
    n, k_w = w.shape
    if k_w != k:
        raise ValueError(f"linear_fp8 {op.source_location!r}: weight contraction mismatch ({k_w} != {k})")
    if k % qb or n % GEMV_FP8_TILE_N or not w.is_contiguous():
        raise ValueError(
            f"linear_fp8 {op.source_location!r}: K={k} must be a multiple of the {qb}-wide quant block, N={n} a multiple of the {GEMV_FP8_TILE_N}-wide output tile, and w row-major contiguous"
        )
    if scale.dtype != F32 or w.dtype != F8_E4M3:
        raise ValueError(
            f"linear_fp8 {op.source_location!r}: dtypes must be e4m3 weights + fp32 scales, got {w.dtype}/{scale.dtype}"
        )
    domain = TileDomain(((m, GEMM_TILE_M), (n, GEMV_FP8_TILE_N)))
    kb = k // qb

    def reads(coords):
        (m0, m1), (n0, n1) = _pair_box(domain, coords)
        sb = n0 // qb  # the tile's 128-wide N block (tile is block-aligned)
        return (
            _tile_region(x, ((m0, m1), (0, k))),
            _tile_region(w, ((n0, n1), (0, k))),
            _tile_region(scale, ((sb, sb + 1), (0, kb))),
        )

    def writes(coords):
        (m0, m1), (n0, n1) = _pair_box(domain, coords)
        return (_tile_region(y, ((m0, m1), (n0, n1))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_linear_fp8",
        kind="gemv_fp8",
        op=op,
        domain=domain,
        inputs=(x.name, w.name, scale.name),
        outputs=(y.name,),
        params={"bias": False, "tile_m": GEMM_TILE_M, "tile_n": GEMV_FP8_TILE_N, "quant_block": qb},
        threads=THREADS_PER_WORKER,
        # In-register dequant: the 16-column weight tile + one x K-slice in fp32.
        scratch_bytes=(GEMV_FP8_TILE_N + 1) * qb * 4,
        read_regions=reads,
        write_regions=writes,
    )


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
        params={"position": op.attributes.get("position", "p"), "position_form": op.attributes.get("position_form", "scalar"), "pos_table": pos is not None},
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
        params={
            "layer": op.attributes.get("layer", 0),
            "which": op.attributes.get("which", "q"),
            "position": op.attributes.get("position", "p"),
            "position_form": op.attributes.get("position_form", "scalar"),
            "convention": op.attributes.get("convention", "rotate_half"),
            "rotary_dim": op.attributes.get("rotary_dim", D),
        },
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
        params={"layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar")},
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
        params={"layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar"), "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=D * 4,  # per-task context accumulator
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Paged decode (#94): pools addressed via external i32 [B, S] slot tables.
# Per-task pool regions are the whole pool: the addressed slot is runtime
# data (slot_table[b, t]), so overlaps are conservative — sound under
# phase order, mirroring the Region.indirect storage-span rule (§5.2).
# ---------------------------------------------------------------------------


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
        params={"layer": op.attributes.get("layer", 0), "position": p},
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )


# Lightning-indexer (issue #97): fused ReLU scoring over the compressed
# entries + per-head mix, then the fixed-count top-k selection.
#
# indexer_scores: one task per (batch, entry tile). Each task streams the
# row's full indexer query block [H, D] against its entry tile and reduces
# the head mix in registers — H is the small indexer head count (e.g. 4),
# so per-task reads stay bounded and no cross-task reduction is needed.
#
# index_topk: one task per batch row (M <= ~1k candidates fit in one
# block's sweep). The device template computes ranks by comparison
# counting (rank(j) = #{valid i : s_i > s_j} + #{valid i < j : s_i == s_j}),
# which yields the slot positions directly and tie-breaks to the lowest
# candidate index with no shared-memory sort.
# ---------------------------------------------------------------------------

INDEXER_TILE_M = 64  # compressed-entry candidates per indexer_scores task


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
        params={"scale": float(op.attributes.get("scale") or 0.0), "layer": op.attributes.get("layer", 0), "position": p, "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=S * 4,
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
        params={"layer": op.attributes.get("layer", 0), "position": p, "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=D * 4,
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


# ---------------------------------------------------------------------------
# Registry (§3.1)
# ---------------------------------------------------------------------------

LOWERINGS: dict[str, Callable[[Operator, OperatorGraph], TaskFamily]] = {
    "linear": lower_linear,
    "linear_fp8": lower_linear_fp8,
    "indexer_scores": lower_indexer_scores,
    "index_topk": lower_index_topk,
    "layer_norm": lower_layer_norm,
    "rms_norm": lower_rms_norm,
    "rope": lower_rope,
    "gelu": lower_gelu,
    "swiglu": lower_swiglu,
    "add": lower_add,
    "embedding": lower_embedding,
    "cache_append": lower_cache_append,
    "gdn_conv": lower_gdn_conv,
    "cache_append_paged": lower_cache_append_paged,
    "attention_scores": lower_attention_scores,
    "attention_scores_paged": lower_attention_scores_paged,
    "softmax": lower_softmax,
    "attention_values": lower_attention_values,
    "attention_values_paged": lower_attention_values_paged,
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
