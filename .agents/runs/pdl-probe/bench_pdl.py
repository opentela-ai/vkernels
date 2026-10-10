"""PDL (programmatic dependent launch) probe — GB10 (sm_121), Triton 3.8.0.

Questions this bench answers, in order:
  P1. Is the PDL launch attribute (launch_pdl=True ->
      CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) actually
      honored under CUDA-graph capture+replay on GB10/driver 580.95?
      -> functional probe: long spin producer + prologue-heavy consumer;
         if PDL works, the consumer's producer-independent prologue hides
         under the producer's execution.
  P2. Do back-to-back chained launches (N>=100) of the mHC kernels / a
      minimal elementwise pair compress their kernel-boundary gap with
      PDL? (donor's law: ~0.74 us dead time/boundary -> ~0.06 us with
      PDL.) Trigger placement (top-of-kernel vs after-last-store) is
      tested as a variable. Timing is A/B interleaved across variants to
      cancel DVFS drift.
  P3. Parity: copied+annotated variants must be bit-identical to the
      production kernels, eager AND graph-replayed.

COPIED kernel variants only — production kernels are NOT modified.

Run: PYTHONPATH=src/python python .agents/runs/pdl-probe/bench_pdl.py \
       --out .agents/runs/pdl-probe/bench_results.json [--only probe|minimal|mhc]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents


# --------------------------------------------------------------------------
# P1 functional probe: spin producer + prologue consumer
# --------------------------------------------------------------------------


@triton.jit
def _spin(X, iters):
    pid = tl.program_id(0)
    acc = pid.to(tl.float32) * 1e-30
    for _ in range(iters):
        acc = acc * 1.0000001 + 1e-7
    tl.store(X + pid, acc)


@triton.jit
def _spin_trig(X, iters):
    pid = tl.program_id(0)
    gdc_launch_dependents()  # earliest possible trigger
    acc = pid.to(tl.float32) * 1e-30
    for _ in range(iters):
        acc = acc * 1.0000001 + 1e-7
    tl.store(X + pid, acc)


@triton.jit
def _prologue_cons(X, Y, iters2):
    pid = tl.program_id(0)
    acc = pid.to(tl.float32) * 1e-30
    for _ in range(iters2):  # producer-independent work
        acc = acc * 1.0000001 + 1e-7
    x = tl.load(X + pid)  # depends on the producer
    tl.store(Y + pid, x + acc * 0.0)


@triton.jit
def _prologue_cons_pdl(X, Y, iters2):
    pid = tl.program_id(0)
    acc = pid.to(tl.float32) * 1e-30
    for _ in range(iters2):  # hides under the producer when PDL is honored
        acc = acc * 1.0000001 + 1e-7
    gdc_wait()
    x = tl.load(X + pid)
    tl.store(Y + pid, x + acc * 0.0)


def probe_functional(iters_prod=60000, iters_cons=30000, cta=48):
    x = torch.empty(cta, device="cuda", dtype=torch.float32)
    y = torch.empty(cta, device="cuda", dtype=torch.float32)
    grid = (cta,)

    def base():
        _spin[grid](x, iters_prod)
        _prologue_cons[grid](x, y, iters_cons)

    def pdl():
        _spin_trig[grid](x, iters_prod)
        _prologue_cons_pdl[grid](x, y, iters_cons, launch_pdl=True)

    res = timed_graph_ab({"pair_baseline": base, "pair_pdl": pdl})
    res["solo_consumer_us"] = timed_graph_ab(
        {"solo_consumer": lambda: _prologue_cons[grid](x, y, iters_cons)})
    res["pdl_parity"] = _parity(lambda: (pdl(), base()), lambda: torch.equal(y, x + 0.0))
    t_base = res["pair_baseline"]["min_us"]
    t_pdl = res["pair_pdl"]["min_us"]
    t_cons = res["solo_consumer_us"]["solo_consumer"]["min_us"]
    res["overlap_us"] = t_base - t_pdl
    res["pdl_honored"] = bool(res["overlap_us"] > 0.5 * t_cons)
    return res


def _parity(run, check):
    run()
    torch.cuda.synchronize()
    return bool(check())


# --------------------------------------------------------------------------
# Minimal elementwise pair: A fills x, B scales x -> y (B depends on A).
# Per-pair buffers eliminate WAR hazards so an un-waited PDL consumer (A)
# is memory-safe. Loop kernels give an L2-resident exec estimate.
# --------------------------------------------------------------------------


@triton.jit
def _fill_base(X, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offs, (offs * 0.001).to(tl.float32), mask=offs < n)


@triton.jit
def _scale_base(X, Y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n, other=0.0)
    tl.store(Y + offs, x * 2.0, mask=offs < n)


@triton.jit
def _fill_pdl(X, n, BLOCK: tl.constexpr, TRIG_LATE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if not TRIG_LATE:
        gdc_launch_dependents()  # earliest trigger: consumer CTAs launch now
    tl.store(X + offs, (offs * 0.001).to(tl.float32), mask=offs < n)
    if TRIG_LATE:
        gdc_launch_dependents()  # donor-style: trigger after the last store


@triton.jit
def _scale_pdl(X, Y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    gdc_wait()  # x was written by the previous grid
    x = tl.load(X + offs, mask=offs < n, other=0.0)
    tl.store(Y + offs, x * 2.0, mask=offs < n)


@triton.jit
def _fill_loop(X, n, R: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for r in range(R):
        tl.store(X + offs, (offs * 0.001 + r * 1e-9).to(tl.float32), mask=offs < n)


@triton.jit
def _scale_loop(X, Y, n, R: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for r in range(R):
        x = tl.load(X + offs, mask=offs < n, other=0.0) + r * 1e-9
        tl.store(Y + offs, x * 2.0, mask=offs < n)


def bench_minimal_pair(pairs=100, block=1024):
    n = 65536
    grid = (triton.cdiv(n, block),)
    xs = [torch.empty(n, device="cuda", dtype=torch.float32) for _ in range(pairs)]
    ys = [torch.empty(n, device="cuda", dtype=torch.float32) for _ in range(pairs)]
    big = torch.empty(n * 64, device="cuda", dtype=torch.float32)
    big2 = torch.empty(n * 64, device="cuda", dtype=torch.float32)

    def chain_base():
        for i in range(pairs):
            _fill_base[grid](xs[i], n, block)
            _scale_base[grid](xs[i], ys[i], n, block)

    def chain_pdl_early():
        for i in range(pairs):
            _fill_pdl[grid](xs[i], n, block, False, launch_pdl=True)
            _scale_pdl[grid](xs[i], ys[i], n, block, launch_pdl=True)

    def chain_pdl_late():
        for i in range(pairs):
            _fill_pdl[grid](xs[i], n, block, True, launch_pdl=True)
            _scale_pdl[grid](xs[i], ys[i], n, block, launch_pdl=True)

    def chain_consumer_only():
        for i in range(pairs):
            _fill_base[grid](xs[i], n, block)  # producer: untouched launch
            _scale_pdl[grid](xs[i], ys[i], n, block, launch_pdl=True)

    res = timed_graph_ab({
        "baseline": chain_base,
        "pdl_early": chain_pdl_early,
        "pdl_late": chain_pdl_late,
        "pdl_consumer_only": chain_consumer_only,
    })
    res["n_launches"] = 2 * pairs
    exec_est = timed_graph_ab({
        "exec_fill": lambda: _fill_loop[grid](big, n, 64, block),
        "exec_scale": lambda: _scale_loop[grid](big, big2, n, 64, block),
    })
    res["exec_fill_us"] = exec_est["exec_fill"]["min_us"] / 64
    res["exec_scale_us"] = exec_est["exec_scale"]["min_us"] / 64
    res["eager_baseline_us"] = timed_eager(chain_base)["median_us"]
    res["pdl_chain_parity"] = _parity(
        lambda: (chain_pdl_early(), chain_pdl_late(), torch.cuda.synchronize()),
        lambda: all(torch.equal(ys[i], xs[i] * 2.0) for i in range(pairs)))
    return res


# --------------------------------------------------------------------------
# mHC chain kernels — COPIES of the production bodies.
# baseline copies: verbatim from
#   src/python/vkernels/torch_ops/mhc_pre_gemv.py   (_pre_gemv)
#   src/python/vkernels/torch_ops/glm_mhc_big_fuse.py (_big_fuse)
#   src/python/vkernels/torch_ops/mhc_compose.py    (_compose)
# pdl copies: + gdc annotations (TRIG_LATE selects trigger placement),
# math untouched.
# Chain per site: pre_gemv(streams->logits) -> big_fuse(->post,comb,out)
#   -> compose(->streams') ; site i's compose output feeds site i+1.
# --------------------------------------------------------------------------


@triton.jit
def _pre_gemv_base(X, FN, OUT, K, MIX, EPS: tl.constexpr, BLOCK_K: tl.constexpr):
    token = tl.program_id(0)
    row = tl.program_id(1)
    ss = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X + token * K + offs, mask=mask, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(ss / K + EPS).to(OUT.dtype.element_ty).to(tl.float32)
    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X + token * K + offs, mask=mask, other=0.0).to(tl.float32)
        xn = (x * rstd).to(OUT.dtype.element_ty).to(tl.float32)
        w = tl.load(FN + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(xn * w, axis=0)
    tl.store(OUT + token * MIX + row, acc.to(OUT.dtype.element_ty))


@triton.jit
def _pre_gemv_pdl(X, FN, OUT, K, MIX, EPS: tl.constexpr, BLOCK_K: tl.constexpr,
                  TRIG_LATE: tl.constexpr):
    token = tl.program_id(0)
    row = tl.program_id(1)
    if not TRIG_LATE:
        gdc_launch_dependents()
    gdc_wait()  # X (streams) written by the previous compose grid
    ss = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X + token * K + offs, mask=mask, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(ss / K + EPS).to(OUT.dtype.element_ty).to(tl.float32)
    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X + token * K + offs, mask=mask, other=0.0).to(tl.float32)
        xn = (x * rstd).to(OUT.dtype.element_ty).to(tl.float32)
        w = tl.load(FN + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(xn * w, axis=0)
    tl.store(OUT + token * MIX + row, acc.to(OUT.dtype.element_ty))
    if TRIG_LATE:
        gdc_launch_dependents()


@triton.jit
def _big_fuse_base(L, B, S, STREAMS, NORM_W, POST, COMB, OUT, D,
                   HC: tl.constexpr, EPS: tl.constexpr, ITERS: tl.constexpr,
                   NORM_EPS: tl.constexpr, BLOCK_D: tl.constexpr):
    token = tl.program_id(0)
    k = tl.arange(0, HC)
    width: tl.constexpr = HC * (HC + 2)
    s0 = tl.load(S)
    s1 = tl.load(S + 1)
    s2 = tl.load(S + 2)
    pre = tl.load(L + token * width + k).to(tl.float32) * s0 + tl.load(B + k)
    post = tl.load(L + token * width + HC + k).to(tl.float32) * s1 + tl.load(
        B + HC + k
    )
    pre_gated = tl.sigmoid(pre) + EPS
    tl.store(POST + token * HC + k, 2.0 * tl.sigmoid(post))
    offset = k[:, None] * HC + k[None, :]
    logits = tl.load(L + token * width + 2 * HC + offset).to(tl.float32) * s2 + tl.load(
        B + 2 * HC + offset
    )
    value = tl.exp(logits - tl.max(logits, 1)[:, None])
    value = tl.div_rn(value, tl.sum(value, 1)[:, None]) + EPS
    value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        value = tl.div_rn(value, tl.sum(value, 1)[:, None] + EPS)
        value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
    tl.store(COMB + token * HC * HC + offset, value)
    offs = tl.arange(0, BLOCK_D)
    streams = tl.load(
        STREAMS + token * HC * D + k[:, None] * D + offs[None, :]
    ).to(tl.float32)
    acc = tl.sum(pre_gated[:, None] * streams, axis=0)
    collapsed = acc.to(OUT.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(collapsed * collapsed, axis=0) / D + NORM_EPS)
    w = tl.load(NORM_W + offs).to(tl.float32)
    y = (collapsed * rstd).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + token * D + offs, (w * y).to(OUT.dtype.element_ty))


@triton.jit
def _big_fuse_pdl(L, B, S, STREAMS, NORM_W, POST, COMB, OUT, D,
                  HC: tl.constexpr, EPS: tl.constexpr, ITERS: tl.constexpr,
                  NORM_EPS: tl.constexpr, BLOCK_D: tl.constexpr,
                  TRIG_LATE: tl.constexpr):
    token = tl.program_id(0)
    k = tl.arange(0, HC)
    width: tl.constexpr = HC * (HC + 2)
    if not TRIG_LATE:
        gdc_launch_dependents()
    # S/B/STREAMS/NORM_W are NOT written by the producer (pre_gemv writes
    # only L) — everything before the wait may overlap the producer.
    s0 = tl.load(S)
    s1 = tl.load(S + 1)
    s2 = tl.load(S + 2)
    gdc_wait()  # L (mix logits) written by the pre_gemv grid
    pre = tl.load(L + token * width + k).to(tl.float32) * s0 + tl.load(B + k)
    post = tl.load(L + token * width + HC + k).to(tl.float32) * s1 + tl.load(
        B + HC + k
    )
    pre_gated = tl.sigmoid(pre) + EPS
    tl.store(POST + token * HC + k, 2.0 * tl.sigmoid(post))
    offset = k[:, None] * HC + k[None, :]
    logits = tl.load(L + token * width + 2 * HC + offset).to(tl.float32) * s2 + tl.load(
        B + 2 * HC + offset
    )
    value = tl.exp(logits - tl.max(logits, 1)[:, None])
    value = tl.div_rn(value, tl.sum(value, 1)[:, None]) + EPS
    value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        value = tl.div_rn(value, tl.sum(value, 1)[:, None] + EPS)
        value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
    tl.store(COMB + token * HC * HC + offset, value)
    offs = tl.arange(0, BLOCK_D)
    streams = tl.load(
        STREAMS + token * HC * D + k[:, None] * D + offs[None, :]
    ).to(tl.float32)
    acc = tl.sum(pre_gated[:, None] * streams, axis=0)
    collapsed = acc.to(OUT.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(collapsed * collapsed, axis=0) / D + NORM_EPS)
    w = tl.load(NORM_W + offs).to(tl.float32)
    y = (collapsed * rstd).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + token * D + offs, (w * y).to(OUT.dtype.element_ty))
    if TRIG_LATE:
        gdc_launch_dependents()


@triton.jit
def _compose_base(POST, COMB, SUB, RESIDUAL, OUT, D,
                  HC: tl.constexpr, BLOCK_D: tl.constexpr):
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    j = tl.arange(0, HC)
    post = tl.load(POST + token * HC + j)
    sub = tl.load(SUB + token * D + offs, mask=mask, other=0.0)
    residual = RESIDUAL + token * HC * D
    acc = tl.zeros([HC, BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(HC):
        comb = tl.load(COMB + token * HC * HC + k * HC + j)
        value = tl.load(residual + k * D + offs, mask=mask, other=0.0)
        acc += comb[:, None] * value[None, :].to(tl.float32)
    acc += post[:, None] * sub[None, :].to(tl.float32)
    tl.store(
        OUT + token * HC * D + j[:, None] * D + offs[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=mask[None, :],
    )


@triton.jit
def _compose_pdl(POST, COMB, SUB, RESIDUAL, OUT, D,
                 HC: tl.constexpr, BLOCK_D: tl.constexpr, TRIG_LATE: tl.constexpr):
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    if not TRIG_LATE:
        gdc_launch_dependents()
    gdc_wait()  # POST/COMB/SUB/RESIDUAL all written by the big_fuse grid
    j = tl.arange(0, HC)
    post = tl.load(POST + token * HC + j)
    sub = tl.load(SUB + token * D + offs, mask=mask, other=0.0)
    residual = RESIDUAL + token * HC * D
    acc = tl.zeros([HC, BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(HC):
        comb = tl.load(COMB + token * HC * HC + k * HC + j)
        value = tl.load(residual + k * D + offs, mask=mask, other=0.0)
        acc += comb[:, None] * value[None, :].to(tl.float32)
    acc += post[:, None] * sub[None, :].to(tl.float32)
    tl.store(
        OUT + token * HC * D + j[:, None] * D + offs[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=mask[None, :],
    )
    if TRIG_LATE:
        gdc_launch_dependents()


def bench_mhc_chain(sites=40, tokens=1, hc=4, hidden=4096, sinkhorn_iters=20):
    k = hc * hidden
    width = hc * (hc + 2)
    d = hidden
    dev = "cuda"
    fn = (torch.randn(width, k, device=dev, dtype=torch.bfloat16) * 0.01)
    base = torch.randn(width, device=dev, dtype=torch.float32) * 0.1
    scale = torch.tensor([0.5, 0.5, 0.5], device=dev, dtype=torch.float32)
    norm_w = torch.randn(d, device=dev, dtype=torch.float32)
    eps = 1e-6
    block_k = min(16384, max(256, triton.next_power_of_2(k)))  # 16384
    warps_pre = 16
    block_d = min(4096, triton.next_power_of_2(d))  # 4096
    grid_pre = (tokens, width)
    grid_fuse = (tokens,)
    grid_comp = (tokens, triton.cdiv(d, block_d))

    streams = [torch.randn(tokens, hc, d, device=dev, dtype=torch.bfloat16) * 0.1
               for _ in range(sites + 1)]
    logits = [torch.empty(tokens, width, device=dev, dtype=torch.bfloat16)
              for _ in range(sites)]
    posts = [torch.empty(tokens, hc, device=dev, dtype=torch.float32)
             for _ in range(sites)]
    combs = [torch.empty(tokens, hc, hc, device=dev, dtype=torch.float32)
             for _ in range(sites)]
    outs = [torch.empty(tokens, d, device=dev, dtype=torch.bfloat16)
            for _ in range(sites)]

    def pre(i, kern, pdl, late):
        extra = (late,) if pdl else ()
        kern[grid_pre](streams[i], fn, logits[i], k, width, eps, block_k, *extra,
                       num_warps=warps_pre, enable_fp_fusion=False, launch_pdl=pdl)

    def fuse(i, kern, pdl, late):
        extra = (late,) if pdl else ()
        kern[grid_fuse](logits[i], base, scale, streams[i], norm_w,
                        posts[i], combs[i], outs[i], d, hc, eps,
                        sinkhorn_iters, eps, block_d, *extra,
                        num_warps=8, enable_fp_fusion=False, launch_pdl=pdl)

    def comp(i, kern, pdl, late):
        extra = (late,) if pdl else ()
        kern[grid_comp](posts[i], combs[i], outs[i], streams[i],
                        streams[i + 1], d, hc, block_d, *extra,
                        num_warps=4, enable_fp_fusion=False, launch_pdl=pdl)

    def chain_base():
        for i in range(sites):
            pre(i, _pre_gemv_base, False, False)
            fuse(i, _big_fuse_base, False, False)
            comp(i, _compose_base, False, False)

    def chain_pdl_early():
        for i in range(sites):
            pre(i, _pre_gemv_pdl, True, False)
            fuse(i, _big_fuse_pdl, True, False)
            comp(i, _compose_pdl, True, False)

    def chain_pdl_late():
        for i in range(sites):
            pre(i, _pre_gemv_pdl, True, True)
            fuse(i, _big_fuse_pdl, True, True)
            comp(i, _compose_pdl, True, True)

    def chain_consumer_only():
        for i in range(sites):
            pre(i, _pre_gemv_base, False, False)
            fuse(i, _big_fuse_pdl, True, False)
            comp(i, _compose_base, False, False)

    res = timed_graph_ab({
        "baseline": chain_base,
        "pdl_early": chain_pdl_early,
        "pdl_late": chain_pdl_late,
        "pdl_consumer_only": chain_consumer_only,
    })
    res["sites"] = sites
    res["tokens"] = tokens
    res["n_launches"] = 3 * sites
    res["eager_baseline_us"] = timed_eager(chain_base)["median_us"]

    # parity: copied-baseline chain vs production wrappers (validates copies)
    import sys
    if "src/python" not in sys.path:
        sys.path.insert(0, "src/python")
    from vkernels.torch_ops.mhc_pre_gemv import mhc_pre_gemv
    from vkernels.torch_ops.glm_mhc_big_fuse import mhc_pre_big_fuse
    from vkernels.torch_ops.mhc_compose import mhc_compose

    def check(i):
        rl = mhc_pre_gemv(streams[i].reshape(tokens, k), fn, hc=hc,
                          hidden_size=hidden, eps=eps)
        rp, rc, ro = mhc_pre_big_fuse(rl, base, scale, streams[i], norm_w, hc=hc,
                                      eps=eps, sinkhorn_iters=sinkhorn_iters,
                                      norm_eps=eps)
        rs = mhc_compose(rp, rc, ro, streams[i], hc=hc)
        return (torch.equal(logits[i], rl) and torch.equal(posts[i], rp)
                and torch.equal(combs[i], rc) and torch.equal(outs[i], ro)
                and torch.equal(streams[i + 1], rs))

    chain_base()
    torch.cuda.synchronize()
    res["copies_match_production"] = bool(check(sites - 1))
    res["pdl_chain_parity"] = _parity(
        lambda: (chain_pdl_early(), chain_pdl_late(), torch.cuda.synchronize()),
        lambda: check(sites - 1))
    return res


# --------------------------------------------------------------------------
# Harness: A/B interleaved graph-replay timing (cancels DVFS drift)
# --------------------------------------------------------------------------


def _sample_graph(graph, inner=8):
    best = float("inf")
    for _ in range(inner):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0)
    return best


def timed_graph_ab(builds, rounds=8, inner=8):
    """Capture one graph per variant; then interleave replays across variants
    in rounds. Reports per-variant min over all round-minima (drift-robust)
    and the median of round-minima."""
    graphs = {}
    for name, build in builds.items():
        build()  # eager warmup: compile before capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            build()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        graphs[name] = graph
    names = list(graphs)
    round_min = {name: [] for name in names}
    for _ in range(rounds):
        for name in names:
            round_min[name].append(_sample_graph(graphs[name], inner))
    return {name: {"min_us": min(v), "median_of_round_min_us": statistics.median(v)}
            for name, v in round_min.items()}


def timed_eager(build, repeats=5):
    build()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        build()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return {"median_us": statistics.median(samples), "min_us": min(samples)}


def summarize(res, n_launches, exec_keys=None):
    out = {}
    exec_avg = None
    if exec_keys:
        exec_avg = sum(res[k] for k in exec_keys) / len(exec_keys)
        out["exec_avg_est_us"] = exec_avg
    for variant in ("baseline", "pdl_early", "pdl_late", "pdl_consumer_only"):
        if variant not in res:
            continue
        per = res[variant]["min_us"] / n_launches
        entry = {"chain_min_us": res[variant]["min_us"],
                 "per_launch_us": per}
        if exec_avg:
            entry["implied_gap_us"] = per - exec_avg
        out[variant] = entry
    if "pdl_early" in out:
        out["compression_early_vs_base"] = out["baseline"]["per_launch_us"] / out["pdl_early"]["per_launch_us"]
    if "pdl_late" in out:
        out["compression_late_vs_base"] = out["baseline"]["per_launch_us"] / out["pdl_late"]["per_launch_us"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=["probe", "minimal", "mhc"])
    args = ap.parse_args()
    props = torch.cuda.get_device_properties(0)
    report = {}
    out_path = args.out
    if out_path.exists():
        try:
            report = json.loads(out_path.read_text())
        except Exception:
            report = {}
    report.update({
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": props.name,
        "compute_capability": f"sm_{props.major}{props.minor}",
        "sms": props.multi_processor_count,
        "driver_note": "PDL requires sm_90+; GB10 is sm_121 (Blackwell), "
                       "driver 580.95.05, torch 2.14.0+cu130 graph capture.",
    })
    which = args.only or "all"
    if which in ("all", "probe"):
        print("P1 functional probe ...", flush=True)
        report["functional_probe"] = fp = probe_functional()
        print(json.dumps(fp, indent=2), flush=True)
    if which in ("all", "minimal"):
        print("minimal pair ...", flush=True)
        report["minimal_pair"] = mp = bench_minimal_pair()
        print(json.dumps({"gaps": summarize(mp, mp["n_launches"],
                                            ["exec_fill_us", "exec_scale_us"]),
                          "parity": mp["pdl_chain_parity"]}, indent=2), flush=True)
    if which in ("all", "mhc"):
        for tokens in (1, 4):
            print(f"mhc chain tokens={tokens} ...", flush=True)
            key = f"mhc_chain_b{tokens}"
            report[key] = mc = bench_mhc_chain(tokens=tokens)
            print(json.dumps({key: summarize(mc, mc["n_launches"]),
                              "copies_match_production": mc["copies_match_production"],
                              "pdl_chain_parity": mc["pdl_chain_parity"]}, indent=2),
                  flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
