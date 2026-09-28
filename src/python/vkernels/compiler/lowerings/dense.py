"""Dense operator tile and memory-region rules."""
from __future__ import annotations
from ..operator_ir import F32, F8_E4M3, Operator, OperatorGraph
from ..task_ir import TaskFamily, TileDomain
from .common import GEMM_TILE_M, GEMM_TILE_N, ELEM_TILE, EMBED_TILE_C, THREADS_PER_WORKER, _whole, _tile_region, _gemm_domain, _pair_box, GEMV_FP8_TILE_N, _flat_box

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
