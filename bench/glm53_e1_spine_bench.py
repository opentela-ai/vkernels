"""E1 KDA-spine microbench (GB10): ONE persistent launch vs the incumbent
chains for GLM-5.3-Flash's 34 KDA layers, at real per-rank TP4 decode dims.

This prices megakernel Lever E1 (``.agents/runs/gap-levers/megakernel-
readiness.md`` §5) BEFORE any serving integration, per the readiness
decision rule: the spine megakernel must beat the graph-of-kernels baseline
by >= 0.3 ms/step at acceptable parity to fund E2 (the floe lane).

Paths measured (B=1 decode, fp32 state math everywhere — the KDA recurrence
is unconditionally fp32 in floe):

  A  eager decomposed chain     — the incumbent class floe runs today:
                                  per layer: mHC pre (torch ops) -> ln1 ->
                                  qkv/f_a/f_b/b/g_a/g_b bf16 GEMVs ->
                                  conv -> delta rule -> gated norm -> o_proj
                                  -> mHC post. ~30+ kernel launches/layer.
  A' CUDA-graph replay of A     — the honest strong baseline (host overhead
                                  gone; per-kernel boundary gaps remain).
  B  fused-CUDA chain (lane 3)  — the same body with
                                  ``vkernels.torch_ops.glm_kda_fused_decode``
                                  replacing conv+delta+norm (nvcc-JIT kernel,
                                  the fused-decode contract: raw bf16 dots,
                                  V-major pool). ~15 launches/layer.
  B' CUDA-graph replay of B.
  M  the E1 spine megakernel    — ONE persistent Triton launch per step
                                  (this bench's hand assembly; the compiler
                                  schedule is 11 phases/layer, the assembly
                                  coalesces the independent GEMV phases to
                                  7 barriers/layer = 238 for 34 layers),
                                  reusing the compiler's attested task
                                  bodies (_t_mhc_pre/_t_mhc_post/_t_rms2d/
                                  _t_gemv) + the new fused-contract body
                                  _t_kda_fused.

Correctness gates BEFORE any timing (a broken decode makes timings
meaningless): M vs A final-streams + ssm-pool agreement in the documented
bf16-ABI band; B's fused decode vs its own eager torch oracle; the
megakernel barrier counter soak (no drift over 500 steps).

Run:  python bench/glm53_e1_spine_bench.py   (needs CUDA + triton; B needs nvcc)
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from vkernels.compiler.device_triton import grid_barrier  # noqa: E402
from vkernels.compiler.triton_dense import _t_gemv, _t_rms2d  # noqa: E402
from vkernels.compiler.triton_recurrent import _t_kda_fused, _t_mhc_post, _t_mhc_pre  # noqa: E402

DEV = "cuda"
L = 34  # GLM-5.3-Flash KDA layers (per rank, TP4)
C, H, D = 4096, 16, 128  # hidden, KDA heads per rank, head dim (K = V = D)
HC = 4  # mHC stream count
MIX = (2 + HC) * HC  # 24
SEG = H * D  # 2048
SEG = tl.constexpr(SEG)  # captured by the jit'd kernel as a global
CC = 3 * SEG  # 6144 conv channels
KT = 4  # conv taps
ITERS = 20  # Sinkhorn iterations (real config)
EPS_NORM = 1e-5
EPS_HC = 1e-6
EPS_L2 = 1e-6
LB = -5.0
SCALE = D ** -0.5
MIXP = triton.next_power_of_2(MIX)  # 32
HCP = HC  # 4 (pow2)
BLOCK_K = 512
BLOCK_C = 1024
BARRIERS_PER_LAYER = 7


# ---------------------------------------------------------------------------
# Weights + pools (random; real per-rank TP4 shapes)
# ---------------------------------------------------------------------------


class SpineWeights:
    def __init__(self, seed: int = 20260917):
        g = torch.Generator(device=DEV).manual_seed(seed)
        bf = lambda *s: (torch.randn(*s, generator=g, device=DEV) * 0.02).to(torch.bfloat16)  # noqa: E731
        # torch-linear layout [out, in], precomputed ONCE (F.linear does x @ W.T;
        # transposing per step would pollute the incumbent timings)
        self.qkv_t = bf(L, CC, C)
        self.fa_t = bf(L, D, C)
        self.fb_t = bf(L, SEG, D)
        self.b_t = bf(L, H, C)
        self.ga_t = bf(L, D, C)
        self.gb_t = bf(L, SEG, D)
        self.o_t = bf(L, C, SEG)
        # gemv layout [K, N] for the megakernel's _t_gemv bodies — a
        # packed COPY OF THE SAME VALUES (transpose of the torch-linear
        # layout; not a view: _t_gemv needs dense rows)
        self.qkv = self.qkv_t.transpose(1, 2).contiguous()
        self.fa = self.fa_t.transpose(1, 2).contiguous()
        self.fb = self.fb_t.transpose(1, 2).contiguous()
        self.b = self.b_t.transpose(1, 2).contiguous()
        self.ga = self.ga_t.transpose(1, 2).contiguous()
        self.gb = self.gb_t.transpose(1, 2).contiguous()
        self.o = self.o_t.transpose(1, 2).contiguous()
        self.ln1 = (1 + 0.02 * torch.randn(L, C, generator=g, device=DEV)).to(torch.bfloat16)
        self.taps = (0.3 * torch.randn(L, KT, CC, generator=g, device=DEV)).float()
        self.dtb = (0.1 * torch.randn(L, H, D, generator=g, device=DEV)).float()
        self.alog = (0.2 * torch.randn(L, H, generator=g, device=DEV)).float()
        self.onorm = (1 + 0.02 * torch.randn(L, D, generator=g, device=DEV)).float()
        self.mfn = (0.05 * torch.randn(L, MIX, HC * C, generator=g, device=DEV)).float()
        self.mbase = (0.1 * torch.randn(L, MIX, generator=g, device=DEV)).float()
        self.mscale = torch.ones(L, 3, device=DEV)
        self.zero_bias = torch.zeros(CC, dtype=torch.float32, device=DEV)

    def fresh_pools(self, seed: int = 5):
        """Pools for the fp32-storage paths (A, M): conv values pre-rounded
        to the bf16 grid (the pool's declared value class) widened to fp32;
        ssm fp32. The fused-CUDA path keeps its own native-bf16 conv pool."""
        g = torch.Generator(device=DEV).manual_seed(seed)
        conv = (0.5 * torch.randn(L, 1, KT - 1, CC, generator=g, device=DEV)).to(torch.bfloat16).float()
        ssm = (0.5 * torch.randn(L, 1, H, D, D, generator=g, device=DEV)).float()
        conv_bf = conv.to(torch.bfloat16)  # same grid values, the kernel's dtype
        return conv, ssm, conv_bf


# ---------------------------------------------------------------------------
# Incumbent A: the eager decomposed chain (torch)
# ---------------------------------------------------------------------------


def _mhc_pre_t(streams, w: SpineWeights, l: int):
    B, hc, c = streams.shape
    flat = streams.reshape(B, hc * c).float()
    flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + EPS_NORM)
    logits = flat @ w.mfn[l].T
    pre_w, post_w, comb_w = logits[:, :hc], logits[:, hc : 2 * hc], logits[:, 2 * hc:].reshape(B, hc, hc)
    pre = torch.sigmoid(pre_w * w.mscale[l, 0] + w.mbase[l, :hc]) + EPS_HC
    post = 2 * torch.sigmoid(post_w * w.mscale[l, 1] + w.mbase[l, hc : 2 * hc])
    comb = torch.softmax(comb_w * w.mscale[l, 2] + w.mbase[l, 2 * hc:].reshape(hc, hc), -1) + EPS_HC
    comb = comb / (comb.sum(-2, keepdim=True) + EPS_HC)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS_HC)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS_HC)
    h_in = (pre[:, :, None] * streams.float()).sum(1)
    return h_in, post, comb


def _mhc_post_t(streams, body, post, comb):
    return post[:, :, None] * body[:, None, :] + torch.einsum("bkj,bkc->bjc", comb, streams.float())


def eager_spine_step(w: SpineWeights, streams, conv, ssm):
    """Incumbent A: one decode step over the 34-layer spine (bf16 GEMVs,
    fp32 state math — floe's eager class). Mutates the pools in place."""
    streams = streams.float()
    for l in range(L):
        h_in, post, comb = _mhc_pre_t(streams, w, l)
        h = h_in * torch.rsqrt(h_in.pow(2).mean(-1, keepdim=True) + EPS_NORM) * w.ln1[l].float()
        hb = h.to(torch.bfloat16)
        mixed = torch.nn.functional.linear(hb, w.qkv_t[l]).float()
        fm = torch.nn.functional.linear(hb, w.fa_t[l])
        f_raw = torch.nn.functional.linear(fm, w.fb_t[l]).float()
        b_raw = torch.nn.functional.linear(hb, w.b_t[l]).float()
        gm = torch.nn.functional.linear(hb, w.ga_t[l])
        g_raw = torch.nn.functional.linear(gm, w.gb_t[l]).float()
        # conv update (w-major window, fp32 taps)
        window = torch.cat([conv[l][0], mixed], dim=0)  # [KT, CC]
        acc = (window * w.taps[l]).sum(0)
        y = acc * torch.sigmoid(acc)
        conv[l][0] = window[1:]
        q = y[:SEG].view(H, D)
        k = y[SEG : 2 * SEG].view(H, D)
        v = y[2 * SEG :].view(H, D)
        xx = f_raw.view(H, D) + w.dtb[l]
        decay = torch.exp(LB * torch.sigmoid(torch.exp(w.alog[l])[:, None] * xx))
        beta = torch.sigmoid(b_raw.view(H))  # [H] (F.linear keeps the [1, H] row)
        qn = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + EPS_L2) * SCALE
        kn = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + EPS_L2)
        s = ssm[l][0] * decay[:, None, :]  # [H, V, K] V-major
        t = (s * kn[:, None, :]).sum(-1)  # [H, V]
        s = s + (beta[:, None] * (v - t))[:, :, None] * kn[:, None, :]
        ssm[l][0] = s
        o = (s * qn[:, None, :]).sum(-1)  # [H, V]
        on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS_NORM) * w.onorm[l] * torch.sigmoid(g_raw.view(H, D))
        body = torch.nn.functional.linear(on.reshape(1, SEG).to(torch.bfloat16), w.o_t[l]).float()
        streams = _mhc_post_t(streams, body, post, comb)
    return streams


# ---------------------------------------------------------------------------
# Incumbent B: the fused-CUDA chain (lane 3's glm_kda_fused_decode)
# ---------------------------------------------------------------------------


def fused_cuda_spine_step(w: SpineWeights, streams, conv_bf, ssm):
    """Incumbent B: the decomposed chain with conv+delta+norm replaced by
    the nvcc-JIT fused kernel (raw-dot bf16 ABI, V-major pool — the same
    contract the megakernel's _t_kda_fused implements)."""
    from vkernels.torch_ops.glm_kda_fused_decode import glm_kda_fused_decode

    streams = streams.float()
    ids = torch.zeros(1, dtype=torch.int32, device=DEV)
    for l in range(L):
        h_in, post, comb = _mhc_pre_t(streams, w, l)
        h = h_in * torch.rsqrt(h_in.pow(2).mean(-1, keepdim=True) + EPS_NORM) * w.ln1[l].float()
        hb = h.to(torch.bfloat16)
        mixed = torch.nn.functional.linear(hb, w.qkv_t[l])
        fm = torch.nn.functional.linear(hb, w.fa_t[l])
        a = torch.nn.functional.linear(fm, w.fb_t[l])
        b = torch.nn.functional.linear(hb, w.b_t[l])
        gm = torch.nn.functional.linear(hb, w.ga_t[l])
        g = torch.nn.functional.linear(gm, w.gb_t[l])
        on = glm_kda_fused_decode(
            mixed, a, b, conv_bf[l],
            w.taps[l, :, :SEG].contiguous(), w.taps[l, :, SEG : 2 * SEG].contiguous(),
            w.taps[l, :, 2 * SEG :].contiguous(),
            w.zero_bias,
            w.alog[l], w.dtb[l].reshape(-1).contiguous(), g, w.onorm[l],
            ssm[l], ids, SCALE, EPS_NORM, lower_bound=LB,
        )  # both pools shift/update IN PLACE
        body = torch.nn.functional.linear(on.reshape(1, SEG), w.o_t[l]).float()
        streams = _mhc_post_t(streams, body, post, comb)
    return streams


# ---------------------------------------------------------------------------
# M: the E1 spine megakernel (ONE persistent launch per decode step)
# ---------------------------------------------------------------------------

# workspace element offsets (fp32)
WS = {}
_o = 0


def _add(name, n):
    global _o
    WS[name] = _o
    _o += n


_add("st0", HC * C)  # mHC streams (ping)
_add("st1", HC * C)  # mHC streams (pong)
_add("hin", C)
_add("h", C)
_add("mixed", CC)
_add("fmid", D)
_add("fraw", SEG)
_add("braw", H)
_add("gmid", D)
_add("graw", SEG)
_add("postw", HC)
_add("comb", HC * HC)
_add("kdaout", SEG)
_add("body", C)
WS_TOTAL = _o


@triton.jit(do_not_specialize=["bar_base", "L"])
def glm53_kda_spine_megakernel(
    bar_ptr,
    bar_base: tl.int64,
    L,
    # weights (layer-uniform classes)
    qkv_w, fa_w, fb_w, bw, ga_w, gb_w, ow,  # bf16 GEMV weights [L, K, N]
    ln1_ptr,  # [L, C] bf16
    taps_ptr,  # [L, KT, CC] fp32 time-major
    dtb_ptr,  # [L, H, D] fp32
    alog_ptr,  # [L, H] fp32
    onorm_ptr,  # [L, D] fp32
    mfn_ptr,  # [L, MIX, HC*C] fp32
    mbase_ptr,  # [L, MIX] fp32
    mscale_ptr,  # [L, 3] fp32
    # persistent state
    conv_ptr,  # [L, 1, KT-1, CC] fp32 (bf16-grid values)
    ssm_ptr,  # [L, 1, H, V, K] fp32 V-major
    ids_ptr,  # i32 [1]
    ws_ptr,  # fp32 workspace
    out_ptr,  # final streams [HC, C] (copy of the live ping-pong buffer)
    C: tl.constexpr,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CC: tl.constexpr,
    KT: tl.constexpr,
    SCALE: tl.constexpr,
    LB: tl.constexpr,
    EPS: tl.constexpr,
    HCEPS: tl.constexpr,
    ITERS: tl.constexpr,
    MIXP: tl.constexpr,
    HCP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    worker = tl.program_id(0)
    P = tl.num_programs(0)
    for l in range(L):
        li = l.to(tl.int64)
        cur = ws_ptr + (l % 2) * (HC * C)  # ping-pong stream buffers
        nxt = ws_ptr + ((l + 1) % 2) * (HC * C)
        bi = (BARRIERS_PER_LAYER_C * l + 1) * P + bar_base
        # phase 1: mHC pre-mix (fp32: rms over the flattened streams, the
        # folded fn GEMV, sigmoid gates, 20-iter Sinkhorn, stream collapse)
        _t_mhc_pre(worker, P, cur, mfn_ptr + li * (MIX * HC * C), mbase_ptr + li * MIX,
                   mscale_ptr + li * 3, ws_ptr + WS_HIN, ws_ptr + WS_POSTW, ws_ptr + WS_COMB,
                   1, HC, C, MIX, HCEPS, EPS, ITERS, MIXP, HCP, BLOCK_K)
        grid_barrier(bar_ptr, bi)
        # phase 2: ln1 (weighted rms over the collapsed stream row)
        _t_rms2d(worker, P, ws_ptr + WS_HIN, ln1_ptr + li * C, ws_ptr + WS_H, 1, C, C, EPS)
        grid_barrier(bar_ptr, bi + P)
        # phase 3: the four independent h-GEMVs (qkv | f_a | b | g_a).
        # TILE=64 keeps each bf16 weight row-segment a full 128B transaction
        # (TILE=16 = 32B segments = 2x read amplification; measured 410 ->
        # 1041 GB/s effective on the qkv shape)
        _t_gemv(worker, P, ws_ptr + WS_H, qkv_w + li * (C * CC), ws_ptr + WS_MIXED, 1, C, CC, 64, 256)
        _t_gemv(worker, P, ws_ptr + WS_H, fa_w + li * (C * D), ws_ptr + WS_FMID, 1, C, D, 64, 256)
        _t_gemv(worker, P, ws_ptr + WS_H, bw + li * (C * H), ws_ptr + WS_BRAW, 1, C, H, 16, 256)
        _t_gemv(worker, P, ws_ptr + WS_H, ga_w + li * (C * D), ws_ptr + WS_GMID, 1, C, D, 64, 256)
        grid_barrier(bar_ptr, bi + 2 * P)
        # phase 4: the two chained small GEMVs (f_b | g_b)
        _t_gemv(worker, P, ws_ptr + WS_FMID, fb_w + li * (D * SEG), ws_ptr + WS_FRAW, 1, D, SEG, 64, 128)
        _t_gemv(worker, P, ws_ptr + WS_GMID, gb_w + li * (D * SEG), ws_ptr + WS_GRAW, 1, D, SEG, 64, 128)
        grid_barrier(bar_ptr, bi + 3 * P)
        # phase 5: the fused KDA decode (conv + delta rule + gated norm,
        # raw-dot bf16 ABI, V-major pool — the compiled op's contract)
        _t_kda_fused(worker, P, conv_ptr + li * ((KT - 1) * CC), ssm_ptr + li * (H * D * D),
                     ids_ptr, ws_ptr + WS_MIXED, ws_ptr + WS_FRAW, ws_ptr + WS_BRAW,
                     ws_ptr + WS_GRAW, taps_ptr + li * (KT * CC), dtb_ptr + li * (H * D),
                     alog_ptr + li * H, onorm_ptr + li * D, ws_ptr + WS_KDAOUT,
                     1, H, D, CC, KT - 1, SCALE, EPS, LB)
        grid_barrier(bar_ptr, bi + 4 * P)
        # phase 6: o_proj GEMV
        _t_gemv(worker, P, ws_ptr + WS_KDAOUT, ow + li * (SEG * C), ws_ptr + WS_BODY, 1, SEG, C, 64, 256)
        grid_barrier(bar_ptr, bi + 5 * P)
        # phase 7: mHC post-compose into the other stream buffer
        _t_mhc_post(worker, P, cur, ws_ptr + WS_BODY, ws_ptr + WS_POSTW, ws_ptr + WS_COMB,
                    nxt, 1, HC, C, HCP, BLOCK_C)
        grid_barrier(bar_ptr, bi + 6 * P)
    # final: expose the live streams buffer (after L layers: buffer L%2)
    task = worker
    while task < HC * C // 1024:
        offs = task * 1024 + tl.arange(0, 1024)
        m = offs < HC * C
        v = tl.load(ws_ptr + (L % 2) * (HC * C) + offs, mask=m, other=0.0)
        tl.store(out_ptr + offs, v, mask=m)
        task += P


# constexpr-friendly copies (globals captured by the jit'd kernel must be
# tl.constexpr instances — plain Python ints are rejected at compile time)
BARRIERS_PER_LAYER_C = tl.constexpr(BARRIERS_PER_LAYER)
WS_HIN = tl.constexpr(WS["hin"])
WS_POSTW = tl.constexpr(WS["postw"])
WS_COMB = tl.constexpr(WS["comb"])
WS_H = tl.constexpr(WS["h"])
WS_MIXED = tl.constexpr(WS["mixed"])
WS_FMID = tl.constexpr(WS["fmid"])
WS_FRAW = tl.constexpr(WS["fraw"])
WS_BRAW = tl.constexpr(WS["braw"])
WS_GMID = tl.constexpr(WS["gmid"])
WS_GRAW = tl.constexpr(WS["graw"])
WS_KDAOUT = tl.constexpr(WS["kdaout"])
WS_BODY = tl.constexpr(WS["body"])


class KdaSpineMegakernel:
    """Host side: one persistent launch per decode step over the spine."""

    def __init__(self, w: SpineWeights, *, workers: int):
        self.w = w
        self.workers = workers
        self.ws = torch.zeros(WS_TOTAL, device=DEV, dtype=torch.float32)
        self.out = torch.zeros(HC * C, device=DEV, dtype=torch.float32)
        self.ids = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.bar = torch.zeros(1, device=DEV, dtype=torch.int64)
        self._bar_base = 0

    def load_streams(self, streams):
        self.ws[0 : HC * C] = streams.reshape(-1).float()

    def run(self, *, check_counter: bool = False):
        w = self.w
        glm53_kda_spine_megakernel[(self.workers,)](
            self.bar, self._bar_base, L,
            w.qkv, w.fa, w.fb, w.b, w.ga, w.gb, w.o, w.ln1,
            w.taps, w.dtb, w.alog, w.onorm, w.mfn, w.mbase, w.mscale,
            self._conv, self._ssm, self.ids, self.ws, self.out,
            C=C, HC=HC, MIX=MIX, H=H, D=D, CC=CC, KT=KT,
            SCALE=SCALE, LB=LB, EPS=EPS_NORM, HCEPS=EPS_HC, ITERS=ITERS,
            MIXP=MIXP, HCP=HCP, BLOCK_K=BLOCK_K, BLOCK_C=BLOCK_C,
            num_warps=4,  # _t_kda_fused's [V, K] tile reductions are exact
            # at 4 warps on this triton/GB10 stack and mis-execute at 8
            # (attested: out/conv bit-exact vs the fp64 fused mirror at 4)
        )
        self._bar_base += BARRIERS_PER_LAYER * L * self.workers
        if check_counter:
            torch.cuda.synchronize()
            got = int(self.bar[0])
            if got != self._bar_base:
                raise RuntimeError(f"barrier counter {got} != {self._bar_base}")
        return self.out

    def attach_pools(self, conv, ssm):
        self._conv = conv
        self._ssm = ssm


# ---------------------------------------------------------------------------
# Timing + counting helpers (the bench_megakernel_qwen3.py pattern)
# ---------------------------------------------------------------------------


def time_steps(fn, steps: int = 100, warmup: int = 20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(steps):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / steps)
    return min(times), float(np.median(times))


def count_kernels(fn):
    """Count CUDA kernel events for one call (§15.2); memcpys excluded."""
    from torch.profiler import ProfilerActivity, profile

    fn()  # warm
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for e in prof.events()
        if str(getattr(e, "device_type", "")) == "DeviceType.CUDA"
        and "memcpy" not in str(getattr(e, "name", "")).lower()
        and "memset" not in str(getattr(e, "name", "")).lower()
    )


class Graphed:
    """CUDA-graph replay wrapper around a zero-arg step callable."""

    def __init__(self, step):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()  # warm on a side stream (allocator-safe)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            step()

    def __call__(self):
        self.g.replay()


def main():
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    print(f"GLM-5.3-Flash E1 KDA-spine microbench (per-rank TP4 dims, B=1)")
    print(f"GPU: {torch.cuda.get_device_name(0)}, SMs={props.multi_processor_count}")
    print(f"spine: L={L} KDA layers, C={C}, H={H} heads x D={D}, hc={HC}, "
          f"conv {KT}-tap, Sinkhorn {ITERS} iters")
    print()

    w = SpineWeights()

    # -- gate 1: incumbent B's fused decode vs its own eager oracle (1 layer)
    from vkernels.torch_ops.glm_kda_fused_decode import glm_kda_fused_decode_reference as oracle

    conv0, ssm0, conv0_bf = w.fresh_pools(seed=5)
    h0 = torch.randn(1, C, device=DEV, dtype=torch.bfloat16) * 0.5
    hb = h0
    mixed = torch.nn.functional.linear(hb, w.qkv_t[0])
    fm = torch.nn.functional.linear(hb, w.fa_t[0])
    a = torch.nn.functional.linear(fm, w.fb_t[0])
    b = torch.nn.functional.linear(hb, w.b_t[0])
    gm = torch.nn.functional.linear(hb, w.ga_t[0])
    g = torch.nn.functional.linear(gm, w.gb_t[0])
    out_o, conv_o, ssm_o = oracle(
        mixed, a, b, conv0_bf[0].clone(),
        w.taps[0, :, :SEG].contiguous(), w.taps[0, :, SEG : 2 * SEG].contiguous(),
        w.taps[0, :, 2 * SEG :].contiguous(),
        w.zero_bias,
        w.alog[0], w.dtb[0].reshape(-1).contiguous(), g, w.onorm[0],
        ssm0[0].clone(), torch.zeros(1, dtype=torch.int32, device=DEV), SCALE, EPS_NORM, lower_bound=LB,
    )
    from vkernels.torch_ops.glm_kda_fused_decode import glm_kda_fused_decode

    ssm_k = ssm0[0].clone()
    out_k = glm_kda_fused_decode(
        mixed, a, b, conv0_bf[0].clone(),
        w.taps[0, :, :SEG].contiguous(), w.taps[0, :, SEG : 2 * SEG].contiguous(),
        w.taps[0, :, 2 * SEG :].contiguous(),
        w.zero_bias,
        w.alog[0], w.dtb[0].reshape(-1).contiguous(), g, w.onorm[0],
        ssm_k, torch.zeros(1, dtype=torch.int32, device=DEV), SCALE, EPS_NORM, lower_bound=LB,
    ).reshape(H, D)
    d_out = (out_k.float() - out_o.float().reshape(H, D)).abs().max().item()
    d_ssm = (ssm_k - ssm_o).abs().max().item()
    print(f"gate 1 — fused CUDA kernel vs its eager oracle: out {d_out:.2e}, ssm {d_ssm:.2e}")
    assert d_out < 1e-2 and d_ssm < 1e-3, "fused CUDA kernel broken on this box"

    # -- gate 2: the megakernel vs incumbent A (final streams + ssm pools)
    mega = KdaSpineMegakernel(w, workers=min(32, props.multi_processor_count))
    streams0 = (0.5 * torch.randn(1, HC, C, generator=torch.Generator(device=DEV).manual_seed(9), device=DEV)).float()
    conv_a, ssm_a, _ = w.fresh_pools(seed=5)
    st_a = eager_spine_step(w, streams0.clone(), conv_a, ssm_a)
    conv_m, ssm_m, _ = w.fresh_pools(seed=5)
    mega.attach_pools(conv_m, ssm_m)
    mega.load_streams(streams0.clone())
    st_m = mega.run(check_counter=True).reshape(1, HC, C)
    rel = (st_m - st_a).abs().max().item() / st_a.abs().max().item()
    d_pool = (ssm_m - ssm_a).abs().max().item()
    print(f"gate 2 — megakernel vs eager decomposed (bf16-ABI band): "
          f"streams rel {rel:.2e}, ssm abs {d_pool:.2e}")
    assert rel < 5e-2, "megakernel diverges beyond the documented band"
    assert d_pool < 5e-2
    # second step: pool RMW chains across launches (barrier base advanced)
    st_a2 = eager_spine_step(w, st_a, conv_a, ssm_a)
    st_m2 = mega.run(check_counter=True).reshape(1, HC, C)
    rel2 = (st_m2 - st_a2).abs().max().item() / st_a2.abs().max().item()
    print(f"gate 2b — chained step 2: streams rel {rel2:.2e}")
    assert rel2 < 5e-2
    # barrier soak: 500 steps, counter must track exactly
    for _ in range(500):
        mega.run()
    mega.run(check_counter=True)
    print(f"gate 3 — barrier counter soak: 502 steps, counter == base (no drift)")

    k_mega = count_kernels(lambda: mega.run())
    print(f"launch check (megakernel): {k_mega} CUDA kernel event(s) per step")
    assert k_mega == 1, "the megakernel must be exactly one launch"
    print()

    # -- incumbent chains for timing (pools allocated once, mutated in
    # place per step — the realistic serving pattern)
    conv_a, ssm_a, _ = w.fresh_pools(seed=5)
    st_in = streams0.clone()
    def step_a():
        return eager_spine_step(w, st_in, conv_a, ssm_a)
    graph_a = Graphed(step_a)

    have_b = True
    try:
        conv_b, ssm_b, conv_b_bf = w.fresh_pools(seed=5)
        def step_b():
            return fused_cuda_spine_step(w, st_in, conv_b_bf, ssm_b)
        step_b()
    except Exception as exc:  # nvcc missing etc. — B is optional for pricing
        print(f"(fused-CUDA chain unavailable: {type(exc).__name__}: {str(exc)[:120]})")
        have_b = False
    if have_b:
        graph_b = Graphed(step_b)

    rows = []
    k_a = count_kernels(step_a)
    lo, med = time_steps(step_a)
    rows.append(("A  eager decomposed", lo, med, k_a))
    k_ag = count_kernels(graph_a)
    lo, med = time_steps(graph_a)
    rows.append(("A' CUDA-graph of A", lo, med, k_ag))
    if have_b:
        k_b = count_kernels(step_b)
        lo, med = time_steps(step_b)
        rows.append(("B  fused-CUDA chain", lo, med, k_b))
        k_bg = count_kernels(graph_b)
        lo, med = time_steps(graph_b)
        rows.append(("B' CUDA-graph of B", lo, med, k_bg))

    best = None
    for P in sorted({8, 16, 32, min(48, props.multi_processor_count)}):
        mega.workers = P
        mega._bar_base = 0
        mega.bar.zero_()
        lo, med = time_steps(lambda: mega.run(), steps=100)
        print(f"    megakernel P={P:2d}: {lo * 1e3:7.3f} us/step (median {med * 1e3:7.3f})")
        if best is None or lo < best[1]:
            best = (P, lo, med)
    mega.workers = best[0]
    mega._bar_base = 0
    mega.bar.zero_()
    lo, med = time_steps(lambda: mega.run(), steps=100)
    rows.append((f"M  spine megakernel P={best[0]}", lo, med, 1))

    print()
    print(f"{'path':28s} {'min us/step':>12s} {'median':>10s} {'launches':>9s}")
    for name, lo, med, k in rows:
        print(f"{name:28s} {lo * 1e3:12.3f} {med * 1e3:10.3f} {k:9d}")

    # -- the E1 decision arithmetic (readiness §5 rule) -------------------
    print()
    base = next((lo for name, lo, _, _ in rows if name.startswith("A'")), None)
    if base is not None:
        saving = base - best[1]
        print(f"E1 decision: M {best[1] * 1e3:.3f} us vs graph-of-kernels A' {base * 1e3:.3f} us"
              f" -> saving {saving:+.3f} ms/step over {k_ag} launches"
              f" (rule: >= +0.300 ms/step funds E2)")
    base_b = next((lo for name, lo, _, _ in rows if name.startswith("B'")), None)
    if base_b is not None:
        print(f"            vs the fused-CUDA graph B' {base_b * 1e3:.3f} us:"
              f" {base_b - best[1]:+.3f} ms/step")
    per_barrier = best[1] * 1e6 / (BARRIERS_PER_LAYER * L)  # us, incl. all phase work
    print(f"structure: 1 launch, {BARRIERS_PER_LAYER * L} in-kernel barriers "
          f"(coalesced from the compiler's {11 * L} op-phases)")


if __name__ == "__main__":
    main()
