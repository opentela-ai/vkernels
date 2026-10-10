"""Recurrent operator tile and memory-region rules."""
from __future__ import annotations
from ..operator_ir import Operator, OperatorGraph
from ..task_ir import TaskFamily, TileDomain
from .common import THREADS_PER_WORKER, _whole, _tile_region, _gdn_tile

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

def lower_kda_fused_decode(op: Operator, graph: OperatorGraph) -> TaskFamily:
    """E1 fused KDA decode: ONE task per (batch row, head) — the fused
    kernel's CTA decomposition (the CUDA oracle maps one block per
    (token, value-head)). The task owns its head's channels in each of the
    q|k|v segments of the RAW row + the w-major conv pool, and its head's
    [V, K] V-major ssm slice (read-modify-write). Pool rows are addressed
    through the runtime slot table (Region.indirect — the #94 pattern):
    per-task pool regions are the conservative whole-pool slab, exact
    under the op's slot-disjointness obligation (duplicate live slots in
    one phase are UB, as in the fused kernel; -1 padded slots skip all
    pool accesses). Two kda_fused_decode ops on one pool chain through the
    pool-storage hazards (the cross-step recurrence), phase-ordered like
    the KV append."""
    from ..operator_ir import Region

    conv = graph.tensor(op.inputs[0])
    ssm = graph.tensor(op.inputs[1])
    slot_ids = graph.tensor(op.inputs[2])
    qkv_raw = graph.tensor(op.inputs[3])
    f_raw = graph.tensor(op.inputs[4])
    b_raw = graph.tensor(op.inputs[5])
    g_raw = graph.tensor(op.inputs[6])
    taps = graph.tensor(op.inputs[7])
    dt_bias = graph.tensor(op.inputs[8])
    A_log = graph.tensor(op.inputs[9])
    o_norm = graph.tensor(op.inputs[10])
    out = graph.tensor(op.outputs[0])
    Kw, Cc = conv.shape[1], conv.shape[2]
    H, V, K = ssm.shape[1:]
    B = out.shape[0]
    seg = Cc // 3
    domain = TileDomain(((B, 1), (H, 1)))

    def reads(coords):
        bb, h = coords
        conv_r = Region.indirect(conv, slot_ids, axis=0)
        ssm_r = Region.indirect(ssm, slot_ids, axis=0)
        return (
            conv_r,
            ssm_r,
            _tile_region(slot_ids, ((bb, bb + 1),)),
            # the task's q|k|v channel stripes of the RAW row (contiguous
            # [h·D, (h+1)·D) inside each third — the kernel's slicing)
            _tile_region(qkv_raw, ((bb, bb + 1), (h * K, (h + 1) * K))),
            _tile_region(qkv_raw, ((bb, bb + 1), (seg + h * K, seg + (h + 1) * K))),
            _tile_region(qkv_raw, ((bb, bb + 1), (2 * seg + h * K, 2 * seg + (h + 1) * K))),
            _tile_region(f_raw, ((bb, bb + 1), (h, h + 1), (0, K))),
            _tile_region(b_raw, ((bb, bb + 1), (h, h + 1))),
            _tile_region(g_raw, ((bb, bb + 1), (h, h + 1), (0, V))),
            _whole(taps),
            _whole(dt_bias),
            _whole(A_log),
            _whole(o_norm),
        )

    def writes(coords):
        bb, h = coords
        return (
            Region.indirect(conv, slot_ids, axis=0),
            Region.indirect(ssm, slot_ids, axis=0),
            _tile_region(out, ((bb, bb + 1), (h, h + 1), (0, V))),
        )

    return TaskFamily(
        family_id=f"ph{op.opid:02d}_kda_fused",
        kind="kda_fused_decode",
        op=op,
        domain=domain,
        inputs=(conv.name, ssm.name, slot_ids.name, qkv_raw.name, f_raw.name, b_raw.name,
                g_raw.name, taps.name, dt_bias.name, A_log.name, o_norm.name),
        outputs=(out.name,),
        params={
            "layer": op.attributes.get("layer", 0),
            "scale": op.attributes.get("scale", 1.0),
            "eps": op.attributes.get("eps", 1e-6),
            "lower_bound": op.attributes.get("lower_bound"),
            "head_dim": K,
            "conv_taps": Kw + 1,
            "conv_dim": Cc,
        },
        threads=THREADS_PER_WORKER,
        scratch_bytes=V * K * 4 + 3 * K * 4,  # the [V, K] fp32 state slice + q/k/v rows
        read_regions=reads,
        write_regions=writes,
    )

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
