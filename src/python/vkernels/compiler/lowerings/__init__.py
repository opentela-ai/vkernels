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
    gh = op.attributes.get("grouped_heads")
    domain = _gemm_domain(x, w)

    def reads(coords):
        (m0, m1), (n0, n1) = _pair_box(domain, coords)
        if gh:
            # Block-diagonal (issue #95 GroupedLinear): each head's output
            # rows read only that head's x/w diagonal blocks — off-block
            # storage is never observed (NaN-canary sound).
            k_g, n_g = k // gh, w.shape[1] // gh
            regs = []
            for h in range(n0 // n_g, min(gh, (n1 + n_g - 1) // n_g)):
                h_n0, h_n1 = max(n0, h * n_g), min(n1, (h + 1) * n_g)
                regs.append(_tile_region(x, ((m0, m1), (h * k_g, (h + 1) * k_g))))
                regs.append(_tile_region(w, ((h * k_g, (h + 1) * k_g),
                                             (h * n_g + (h_n0 - h * n_g), h * n_g + (h_n1 - h * n_g)))))
            if bias is not None:
                regs.append(_tile_region(bias, ((n0, n1),)))
            return tuple(regs)
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
        params={"bias": bias is not None, "tile_m": GEMM_TILE_M, "tile_n": GEMM_TILE_N,
                "grouped_heads": gh,
                **({"k_g": k // gh, "n_g": w.shape[1] // gh} if gh else {})},
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
# Gated RMSNorm (issue #100, GLM o_norm): same per-row/per-head task domain
# as rms_norm with a second elementwise input stream (the gate).
# ---------------------------------------------------------------------------


def lower_rms_norm_gated(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x = graph.tensor(op.inputs[0])
    gate = graph.tensor(op.inputs[1])
    gamma = graph.tensor(op.inputs[2])
    y = graph.tensor(op.outputs[0])
    activation = op.attributes.get("activation", "sigmoid")
    if activation not in ("sigmoid",):
        raise ValueError(f"rms_norm_gated {op.source_location!r}: unsupported activation {activation!r}")
    if gate.shape != x.shape:
        raise ValueError(f"rms_norm_gated {op.source_location!r}: gate {gate.shape} != x {x.shape}")
    if len(x.shape) == 3:
        B, H, D = x.shape
        if tuple(gamma.shape) != (D,):
            raise ValueError(f"rms_norm_gated {op.source_location!r}: weight {gamma.shape} != [{D}]")
        domain = TileDomain(((B, 1), (H, 1)))

        def reads3(coords):
            b, h = coords
            return (
                _tile_region(x, ((b, b + 1), (h, h + 1), (0, D))),
                _tile_region(gate, ((b, b + 1), (h, h + 1), (0, D))),
                _whole(gamma),
            )

        def writes3(coords):
            b, h = coords
            return (_tile_region(y, ((b, b + 1), (h, h + 1), (0, D))),)

        reads, writes, width = reads3, writes3, D
    else:
        rows, width = x.shape
        if tuple(gamma.shape) != (width,):
            raise ValueError(f"rms_norm_gated {op.source_location!r}: weight {gamma.shape} != [{width}]")
        domain = TileDomain(((rows, 1), (width, width)))

        def reads2(coords):
            r = coords[0]
            return (
                _tile_region(x, ((r, r + 1), (0, width))),
                _tile_region(gate, ((r, r + 1), (0, width))),
                _whole(gamma),
            )

        def writes2(coords):
            r = coords[0]
            return (_tile_region(y, ((r, r + 1), (0, width))),)

        reads, writes = reads2, writes2

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_rmsnorm_gated",
        kind="rms_norm_gated",
        op=op,
        domain=domain,
        inputs=(x.name, gate.name, gamma.name),
        outputs=(y.name,),
        params={"eps": op.attributes.get("eps", 1e-6), "activation": activation},
        threads=THREADS_PER_WORKER,
        # x row + gate row in fp32 registers alongside the fp32 accumulator.
        scratch_bytes=2 * width * 4,
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
# mHC hyper-connection mixing (issue #99): the data-dependent pre/post/comb
# weights and stream collapse (pre), then the composed stream update (post).
# One task per row (pre) / per (batch, stream) row (post) at every layer
# boundary. The fn projection is FOLDED into the pre task rather than lowered
# as a separate `linear` op: its [mix=(2+hc)·hc] output is consumed entirely
# by the same task's elementwise gate/softmax/Sinkhorn chain, so a separate
# op would only add a workspace round-trip plus a second task, while the
# unweighted-RMSNorm rsqrt over the flattened streams must sit between the
# stream load and the GEMV reduction anyway (floe's mhc_pre_gemm_sqrsum
# fusion precedent). No tl.dot on device: the GEMV is a K-reduction over
# hc·C with a tiny mix-row output, and hc² Sinkhorn is a register-resident
# [hc, hc] chain.
# ---------------------------------------------------------------------------


def lower_mhc_pre(op: Operator, graph: OperatorGraph) -> TaskFamily:
    """One task per batch row: the task owns the row's whole [hc, C] stream
    stack (the flat RMSNorm reduction and the fn GEMV both read the full
    hc·C extent), computes the [mix] projection logits, the sigmoid gates,
    the hc×hc Sinkhorn chain and the pre-weighted collapse. hc is small
    (2–8): the entire mixing state lives in registers/scratch."""
    streams = graph.tensor(op.inputs[0])
    fn = graph.tensor(op.inputs[1])
    base = graph.tensor(op.inputs[2])
    scale = graph.tensor(op.inputs[3])
    h_in = graph.tensor(op.outputs[0])
    post_w = graph.tensor(op.outputs[1])
    comb = graph.tensor(op.outputs[2])
    B, hc, C = streams.shape
    mix = (2 + hc) * hc
    if fn.shape != (mix, hc * C):
        raise ValueError(f"mhc_pre fn must be [{mix}, {hc * C}]; got {fn.shape}")
    domain = TileDomain(((B, 1),))

    def reads(coords):
        (bb,) = coords
        return (
            _tile_region(streams, ((bb, bb + 1), (0, hc), (0, C))),
            _whole(fn),
            _whole(base),
            _whole(scale),
        )

    def writes(coords):
        (bb,) = coords
        return (
            _tile_region(h_in, ((bb, bb + 1), (0, C))),
            _tile_region(post_w, ((bb, bb + 1), (0, hc))),
            _tile_region(comb, ((bb, bb + 1), (0, hc), (0, hc))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_mhc_pre",
        kind="mhc_pre",
        op=op,
        domain=domain,
        inputs=(streams.name, fn.name, base.name, scale.name),
        outputs=(h_in.name, post_w.name, comb.name),
        params={
            "layer": op.attributes.get("layer", 0),
            "hc": hc,
            "iters": op.attributes.get("iters", 1),
            "eps": float(op.attributes.get("eps", 1e-6)),
            "rms_eps": float(op.attributes.get("rms_eps", 1e-6)),
        },
        threads=THREADS_PER_WORKER,
        scratch_bytes=hc * C * 4,  # fp32 flattened-stream row
        read_regions=reads,
        write_regions=writes,
    )


def lower_mhc_post(op: Operator, graph: OperatorGraph) -> TaskFamily:
    """One task per (batch, stream j): the task reads the whole stream stack
    row (all hc sources feed stream j through comb[:, j]), the body output
    row and this token's post/comb weights, and writes exactly stream j of
    the fresh [B, hc, C] output stack. Narrow per-stream writes keep the
    family's tasks independent — layer chaining is the ordinary RAW hazard
    on the returned workspace view."""
    streams = graph.tensor(op.inputs[0])
    body_out = graph.tensor(op.inputs[1])
    post_w = graph.tensor(op.inputs[2])
    comb = graph.tensor(op.inputs[3])
    streams_post = graph.tensor(op.outputs[0])
    B, hc, C = streams.shape
    domain = TileDomain(((B, 1), (hc, 1)))

    def reads(coords):
        bb, j = coords
        return (
            _tile_region(streams, ((bb, bb + 1), (0, hc), (0, C))),
            _tile_region(body_out, ((bb, bb + 1), (0, C))),
            _tile_region(post_w, ((bb, bb + 1), (j, j + 1))),
            _tile_region(comb, ((bb, bb + 1), (0, hc), (j, j + 1))),
        )

    def writes(coords):
        bb, j = coords
        return (_tile_region(streams_post, ((bb, bb + 1), (j, j + 1), (0, C))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_mhc_post",
        kind="mhc_post",
        op=op,
        domain=domain,
        inputs=(streams.name, body_out.name, post_w.name, comb.name),
        outputs=(streams_post.name,),
        params={"layer": op.attributes.get("layer", 0), "hc": hc},
        threads=THREADS_PER_WORKER,
        scratch_bytes=C * 4,  # fp32 composed stream row
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

def lower_kda_delta(op: Operator, graph: OperatorGraph) -> TaskFamily:
    """One task per (batch, head): the task owns the head's [K, V] state
    slice (read-modify-write, in registers on device) plus its q/k/v/f/gate
    rows. KDA heads are square (K = V = head_dim) with one q/k/v head each
    — no group expansion, unlike gdn_delta. Two kda_delta ops on one pool
    chain through the state-storage hazards (RAW/WAR/WAW), phase-ordered
    like the KV append."""
    state = graph.tensor(op.inputs[0])
    q = graph.tensor(op.inputs[1])
    k = graph.tensor(op.inputs[2])
    v = graph.tensor(op.inputs[3])
    f = graph.tensor(op.inputs[4])
    b = graph.tensor(op.inputs[5])
    dt_bias = graph.tensor(op.inputs[6])
    A_log = graph.tensor(op.inputs[7])
    out = graph.tensor(op.outputs[0])
    B, H, K, V = state.shape
    domain = TileDomain(((B, 1), (H, 1)))

    def reads(coords):
        bb, h = coords
        return (
            _tile_region(state, ((bb, bb + 1), (h, h + 1), (0, K), (0, V))),
            _tile_region(q, ((bb, bb + 1), (h, h + 1), (0, K))),
            _tile_region(k, ((bb, bb + 1), (h, h + 1), (0, K))),
            _tile_region(v, ((bb, bb + 1), (h, h + 1), (0, V))),
            _tile_region(f, ((bb, bb + 1), (h, h + 1), (0, K))),
            _tile_region(b, ((bb, bb + 1), (h, h + 1))),
            _whole(dt_bias),
            _whole(A_log),
        )

    def writes(coords):
        bb, h = coords
        # Read-modify-write: the task updates its own state slice in place.
        return (
            _tile_region(state, ((bb, bb + 1), (h, h + 1), (0, K), (0, V))),
            _tile_region(out, ((bb, bb + 1), (h, h + 1), (0, V))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_kda_delta",
        kind="kda_delta",
        op=op,
        domain=domain,
        inputs=(state.name, q.name, k.name, v.name, f.name, b.name, dt_bias.name, A_log.name),
        outputs=(out.name,),
        params={
            "layer": op.attributes.get("layer", 0),
            "scale": op.attributes.get("scale", 1.0),
            "lower_bound": op.attributes.get("lower_bound"),
        },
        threads=THREADS_PER_WORKER,
        scratch_bytes=K * V * 4,  # the [K, V] fp32 state slice (registers/scratch budget)
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
        params={"layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar")},
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
        params={"scale": float(op.attributes.get("scale") or 0.0), "layer": op.attributes.get("layer", 0), "position": p, "position_form": op.attributes.get("position_form", "scalar"), "kv_heads": kvh, "group": group},
        threads=THREADS_PER_WORKER,
        scratch_bytes=S * 4,
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
# MLA decode (issue #95): shared-KV MQA over a latent cache — fused
# scores+softmax+sink over window ∪ compressed candidates, then the context
# gather; plus the conjugate (output-side, negative-angle) rope.
# ---------------------------------------------------------------------------


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



# ---------------------------------------------------------------------------
# Registry (§3.1)
# ---------------------------------------------------------------------------


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


def lower_moe_route(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x, router_w, ids, weights = (graph.tensor(op.inputs[i]) for i in range(4))
    b, h = x.shape
    e, h_w = router_w.shape
    k = ids.shape[1]
    if h_w != h:
        raise ValueError(f"moe_route {op.source_location!r}: contraction mismatch")
    mode = op.attributes["mode"]
    score_fn = op.attributes["score_fn"]
    extra: list[str] = []
    if mode == "hash":
        extra = [op.inputs[4], op.inputs[5]]  # tid2eid, token_ids
    elif score_fn == "sigmoid_noaux_tc":
        extra = [op.inputs[4]]  # e_score_correction_bias
    domain = TileDomain(((b, 1),))

    def reads(coords):
        regions = [
            _tile_region(x, ((coords[0], coords[0] + 1), (0, h))),
            _whole(router_w),
        ]
        for name in extra:
            regions.append(_whole(graph.tensor(name)))
        return tuple(regions)

    def writes(coords):
        r = coords[0]
        return (
            _tile_region(ids, ((r, r + 1), (0, k))),
            _tile_region(weights, ((r, r + 1), (0, k))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_moe_route",
        kind="moe_route",
        op=op,
        domain=domain,
        inputs=(x.name, router_w.name, ids.name, weights.name, *extra),
        outputs=(ids.name, weights.name),
        params=dict(op.attributes),
        threads=THREADS_PER_WORKER,
        # Per-worker score scratch: one E-wide fp32 choice-score vector.
        scratch_bytes=e * 4,
        read_regions=reads,
        write_regions=writes,
    )


def lower_moe_expert(op: Operator, graph: OperatorGraph) -> TaskFamily:
    x, gate_up, down, ids = (graph.tensor(op.inputs[i]) for i in range(4))
    partials = graph.tensor(op.outputs[0])
    b, h = x.shape
    e, two_i, h_w = gate_up.shape
    inter = two_i // 2
    k = ids.shape[1]
    if h_w != h or down.shape != (e, h, inter):
        raise ValueError(f"moe_expert {op.source_location!r}: expert stack shape mismatch")
    # The static k·B grid: one task per (row, slot); the weight base is read
    # from ids[b, slot] at run time (runtime indirection, #94 pattern).
    domain = TileDomain(((b, 1), (k, 1)))

    def reads(coords):
        r, s = coords
        return (
            _tile_region(x, ((r, r + 1), (0, h))),
            # Indirected weight pool: slots are runtime data — whole-pool
            # declaration is the conservative sound choice (#94).
            _whole(gate_up),
            _whole(down),
            _whole(ids),
        )

    def writes(coords):
        r, s = coords
        return (_tile_region(partials, ((r, r + 1), (s, s + 1), (0, h))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_moe_expert",
        kind="moe_expert",
        op=op,
        domain=domain,
        inputs=(x.name, gate_up.name, down.name, ids.name),
        outputs=(partials.name,),
        params=dict(op.attributes),
        threads=THREADS_PER_WORKER,
        # In-register gate/up rows: 2·inter fp32 values + the h-wide activation.
        scratch_bytes=(two_i + h) * 4,
        read_regions=reads,
        write_regions=writes,
    )


def lower_moe_combine(op: Operator, graph: OperatorGraph) -> TaskFamily:
    partials, weights = graph.tensor(op.inputs[0]), graph.tensor(op.inputs[1])
    shared = graph.tensor(op.inputs[2]) if len(op.inputs) > 2 else None
    y = graph.tensor(op.outputs[0])
    b, k, h = partials.shape
    domain = TileDomain(((b, 1),))

    def reads(coords):
        regions = [_whole(partials), _whole(weights)]
        if shared is not None:
            regions.append(_whole(shared))
        return tuple(regions)

    def writes(coords):
        r = coords[0]
        return (_tile_region(y, ((r, r + 1), (0, h))),)

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_moe_combine",
        kind="moe_combine",
        op=op,
        domain=domain,
        inputs=(partials.name, weights.name) + ((shared.name,) if shared is not None else ()),
        outputs=(y.name,),
        params=dict(op.attributes),
        threads=THREADS_PER_WORKER,
        read_regions=reads,
        write_regions=writes,
    )


# ---------------------------------------------------------------------------
# Registry (§3.1)
# ---------------------------------------------------------------------------


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
# DSA compressor append (issue #96): one RMW task per (batch, layer); masked
# per-row on the m-token boundary (issue #93 positions). Emission = gated
# softmax fold of the m-token window + rms_norm + rope rotated ONCE at the
# emitting row's position; series bookkeeping ping-pongs Ca/Cb slot roles
# at the r-boundary (width 2r stride r).
# ---------------------------------------------------------------------------


def lower_compressor_append(op: Operator, graph: OperatorGraph) -> TaskFamily:
    pool, state = graph.tensor(op.inputs[0]), graph.tensor(op.inputs[1])
    window, gates = graph.tensor(op.inputs[2]), graph.tensor(op.inputs[3])
    rms_w, cos, sin = graph.tensor(op.inputs[4]), graph.tensor(op.inputs[5]), graph.tensor(op.inputs[6])
    b, mw, d = window.shape
    m = op.attributes.get("m")
    r = op.attributes.get("r")
    if not isinstance(m, int) or m <= 0 or not isinstance(r, int) or r <= 0 or r % m:
        raise ValueError(f"compressor_append {op.source_location!r}: needs r % m == 0; got m={m!r}, r={r!r}")
    if mw != m:
        raise ValueError(f"compressor_append {op.source_location!r}: window token axis {mw} != m={m}")
    _, layers, _, _, _ = pool.shape
    domain = TileDomain(((b, 1), (layers, 1)))  # one task per (batch, layer)
    p = op.attributes.get("position", "p")

    def reads(coords):
        bi, _ = coords
        return (
            _tile_region(window, ((bi, bi + 1), (0, m), (0, d))),
            _tile_region(gates, ((bi, bi + 1), (0, m))),
            _tile_region(rms_w, ((0, d),)),
            _tile_region(cos, ((bi, bi + 1), (0, d // 2))),
            _tile_region(sin, ((bi, bi + 1), (0, d // 2))),
            _tile_region(state, ((bi, bi + 1), (0, layers), (0, 2))),
        )

    def writes(coords):
        bi, _ = coords
        # Conservative per-row slab: this task owns its row's whole pool slab
        # and series state (the slot/cb_len it writes are runtime values).
        return (
            _tile_region(pool, ((bi, bi + 1), (0, layers), (0, 2), (0, r // m), (0, d))),
            _tile_region(state, ((bi, bi + 1), (0, layers), (0, 2))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_compressor_append",
        kind="compressor_append",
        op=op,
        domain=domain,
        inputs=(pool.name, state.name, window.name, gates.name, rms_w.name, cos.name, sin.name),
        outputs=(),
        params={
            "m": m,
            "r": r,
            "eps": op.attributes.get("eps", 1e-6),
            "position": p,
            "position_form": op.attributes.get("position_form", "scalar"),
        },
        threads=THREADS_PER_WORKER,
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
    "rms_norm_gated": lower_rms_norm_gated,
    "rope": lower_rope,
    "gelu": lower_gelu,
    "swiglu": lower_swiglu,
    "add": lower_add,
    "embedding": lower_embedding,
    "cache_append": lower_cache_append,
    "gdn_conv": lower_gdn_conv,
    "gdn_delta": lower_gdn_delta,
    "kda_delta": lower_kda_delta,
    "cache_append_paged": lower_cache_append_paged,
    "mhc_pre": lower_mhc_pre,
    "mhc_post": lower_mhc_post,
    "attention_scores": lower_attention_scores,
    "attention_scores_paged": lower_attention_scores_paged,
    "mla_scores": lower_mla_scores,
    "mla_values": lower_mla_values,
    "conjugate_rope": lower_conjugate_rope,
    "softmax": lower_softmax,
    "attention_values": lower_attention_values,
    "attention_values_paged": lower_attention_values_paged,
    "moe_route": lower_moe_route,
    "moe_expert": lower_moe_expert,
    "moe_combine": lower_moe_combine,
    "compressor_append": lower_compressor_append,
    "moe_route": lower_moe_route,
    "moe_expert": lower_moe_expert,
    "moe_combine": lower_moe_combine,
    "indexer_scores": lower_indexer_scores,
    "index_topk": lower_index_topk,
    "compressor_append": lower_compressor_append,
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
