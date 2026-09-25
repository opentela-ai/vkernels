#!/usr/bin/env python3
"""Wrapper-level CONFIRM pass for the dense-m8 sweep finalists.

Interleaved round-robin re-measurement (same methodology as the M=1 lane's
confirm_gemv_tiles.py, but at the WRAPPER level — dense_gemv /
dense_gemv_fp8 calls with the production knobs, graph-captured — because
that is the level that rejected 3 of the M=1 sweep winners).

Per (shape, dtype, M) cell the candidates are:
  cublas  : F.linear / mm on the bf16 weight (fp8: dequant-on-use mm),
  default : the committed wrapper config (M=1 pin table / M>1 heuristic),
  pin     : the sweep finalist injected into _CFG_M / _TILES_M,
            PLUS the runner-up finalists (top-3) so a noisy pass-1 winner
            can be rejected in favor of a corroborated alternative.

Each candidate gets its own CUDA graph (N=32 wrapper calls); all graphs are
replayed in rotation for ROUNDS=30 rounds; per-config MEDIAN us/call wins.
A cell is confirm-corroborated GEMV-winning iff median(pin best) <
median(cublas) * 0.98.

Output: /ws/m8_confirm_results.json rows
  {o, i, dtype, m, kind: cublas|default|pin, cfg, median, gbps}
"""
import json
import os
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache_m8")
import statistics
import time

import torch
import triton

torch.cuda.set_device(0)

import vkernels.torch_ops.glm_gemv as gmod
import vkernels.torch_ops.glm_dense_fp8_gemv as fmod

R = json.load(open("/ws/m8_sweep_results.json"))
OUT = []
ROUNDS = 30
N = 32


def sweep_top(o, i, dtype, m, k=3):
    rs = [r for r in R if (r["o"], r["i"]) == (o, i) and r["dtype"] == dtype
          and r["m"] == m and r["tag"] == "sweep"]
    rs.sort(key=lambda r: r["us"])
    out = []
    for r in rs:
        c = (r["rows"], r["block_i"], r["warps"])
        if c not in out:
            out.append(c)
        if len(out) >= k:
            break
    return out


def cap(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(N):
            fn()
    return g


def round_robin(graphs, labels):
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    times = {lab: [] for lab in labels}
    for _ in range(ROUNDS):
        for lab, g in zip(labels, graphs):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            times[lab].append((time.perf_counter() - t0) / N * 1e6)
    return {lab: statistics.median(v) for lab, v in times.items()}


def confirm_bf16(o, i, m):
    gen = torch.Generator(device="cuda").manual_seed(o * 7 + i + m)
    w = (torch.randn(o, i, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    x = (torch.randn(m, i, generator=gen, device="cuda")).to(torch.bfloat16)
    b = o * i * 2 + m * (i * 2 + o * 2)
    labels, fns = [], []

    labels.append("cublas")
    fns.append(lambda: torch.nn.functional.linear(x, w))

    # default wrapper (ensure no stray pin for this cell)
    gmod._CFG_M.pop((o, i, m), None)
    labels.append("default")
    fns.append(lambda: gmod.dense_gemv(x, w, m_cap=8))

    graphs = [cap(fn) for fn in fns]
    for k, c in enumerate(sweep_top(o, i, "bf16", m)):
        gmod._CFG_M[(o, i, m)] = c
        labels.append(f"pin{k}:{c[0]},{c[1]},{c[2]}")
        graphs.append(cap(lambda: gmod.dense_gemv(x, w, m_cap=8)))
    gmod._CFG_M.pop((o, i, m), None)
    med = round_robin(graphs, labels)
    for lab, fn, g in zip(labels, fns, graphs):
        cfg = (0, 0, 0) if lab == "cublas" else (
            gmod._CFG_M.get((o, i, m), (0, 0, 0)) if lab == "default" else
            tuple(int(t) for t in lab.split(":")[1].split(",")))
        OUT.append(dict(o=o, i=i, dtype="bf16", m=m,
                        kind=lab.split(":")[0], cfg=list(cfg),
                        median=round(med[lab], 3),
                        gbps=round(b / med[lab] / 1e3, 0)))
        print(f"  bf16 m={m} ({o:6d},{i:6d}) {lab:22} median {med[lab]:8.2f} us"
              f"  {b/med[lab]/1e3:6.0f} GB/s", flush=True)
    for g in graphs:
        del g
    del w, x
    torch.cuda.empty_cache()


def confirm_fp8(o, i, m):
    gen = torch.Generator(device="cuda").manual_seed(o * 11 + i + m)
    w8 = ((torch.randn(o, i, generator=gen, device="cuda") * 0.1)
          .clamp(-0.4, 0.4)).to(torch.float8_e4m3fn)
    s = (torch.rand(o // 128, i // 128, generator=gen, device="cuda") * 0.4 + 0.8)
    wref = (w8.float().view(o // 128, 128, i // 128, 128)
            * s.float()[:, None, :, None]).reshape(o, i).to(torch.bfloat16)
    x = (torch.randn(m, i, generator=gen, device="cuda")).to(torch.bfloat16)
    b = o * i + (o // 128) * (i // 128) * 4 + m * (i * 2 + o * 2)
    labels, fns = [], []

    labels.append("cublas")
    fns.append(lambda: torch.mm(x, wref.t()))

    fmod._TILES_M.pop((o, i, m), None)
    labels.append("default")
    fns.append(lambda: fmod.dense_gemv_fp8(x, w8, s, m_cap=8))

    graphs = [cap(fn) for fn in fns]
    for k, c in enumerate(sweep_top(o, i, "fp8", m)):
        fmod._TILES_M[(o, i, m)] = c
        labels.append(f"pin{k}:{c[0]},{c[1]},{c[2]}")
        graphs.append(cap(lambda: fmod.dense_gemv_fp8(x, w8, s, m_cap=8)))
    fmod._TILES_M.pop((o, i, m), None)
    med = round_robin(graphs, labels)
    for lab, g in zip(labels, graphs):
        cfg = (0, 0, 0) if lab == "cublas" else (
            fmod._TILES_M.get((o, i, m), (0, 0, 0)) if lab == "default" else
            tuple(int(t) for t in lab.split(":")[1].split(",")))
        OUT.append(dict(o=o, i=i, dtype="fp8", m=m,
                        kind=lab.split(":")[0], cfg=list(cfg),
                        median=round(med[lab], 3),
                        gbps=round(b / med[lab] / 1e3, 0)))
        print(f"  fp8  m={m} ({o:6d},{i:6d}) {lab:22} median {med[lab]:8.2f} us"
              f"  {b/med[lab]/1e3:6.0f} GB/s", flush=True)
    for g in graphs:
        del g
    del w8, s, wref, x
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"confirm on {torch.cuda.get_device_name(0)} torch={torch.__version__}",
          flush=True)
    t0 = time.time()
    cells = sorted({(r["o"], r["i"], r["dtype"], r["m"]) for r in R
                    if r["tag"] == "sweep"})
    for (o, i, dt, m) in cells:
        if time.time() - t0 > 35 * 60:
            print("!! deadline hit, stopping confirm"); break
        if dt == "bf16":
            confirm_bf16(o, i, m)
        else:
            confirm_fp8(o, i, m)
    with open("/ws/m8_confirm_results.json", "w") as f:
        json.dump(OUT, f, indent=1)
    print(f"[confirm done in {time.time()-t0:.0f}s; {len(OUT)} rows"
          f" -> /ws/m8_confirm_results.json]", flush=True)
