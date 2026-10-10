#!/usr/bin/env python3
"""Config sweeps for the vendored tokenspeed-amd kernels on gfx942.

The vendored launch tables were tuned for MI355X; this script measures the
gfx942-optimal configs so the tables can carry a local override:

* routing softmax top-k: NUM_WARPS for the single-CTA kernel
  (upstream: ``min(next_pow2(M), 4)``)
* sampling argmax: block size / warps / split strategy per (M, N)
  (upstream ``_select_config``)

Emits JSON to stdout and ``--out``. Run on beverin with the triton-3.8
PYTHONPATH (see meta/benchmarks/artifacts/tokenspeed-amd-bucket1).
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from vkernels.torch_ops.tokenspeed_amd import routing as ts_routing
from vkernels.torch_ops.tokenspeed_amd._gluon import triton


def time_fn(fn, *, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    return round(statistics.median(times), 2)


def _launch_route(logits, bias, topk, ids, weights, num_warps):
    M, E = logits.shape

    def next_pow2(x):
        return 1 << (max(1, x) - 1).bit_length()

    ts_routing._softmax_topk_route_gluon_kernel[(1,)](
        logits, bias, ids, weights,
        logits.stride(0), logits.stride(1), bias.stride(0),
        ids.stride(0), ids.stride(1), weights.stride(0), weights.stride(1),
        M=M, E=E, TOPK=topk,
        MP=next_pow2(M), EP=next_pow2(E), TKP=next_pow2(topk),
        HAS_BIAS=True, NORMALIZE_TOPK_WEIGHTS=True,
        ROUTED_SCALING_FACTOR=2.5,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )


def sweep_routing(results):
    torch.manual_seed(0)
    shapes = [(1, 128, 16), (1, 384, 16), (8, 256, 16), (16, 128, 8), (64, 384, 2), (4, 256, 8), (32, 128, 4)]
    for m, e, topk in shapes:
        if m * topk > 128:
            continue
        logits = torch.randn(m, e, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(e, device="cuda", dtype=torch.float32)
        ids = torch.empty(m, topk, dtype=torch.int32, device="cuda")
        weights = torch.empty(m, topk, dtype=torch.float32, device="cuda")
        entry = {"op": "routing_softmax_topk", "shape": {"M": m, "E": e, "topk": topk}, "configs": {}}
        ref_ids, _ = ts_routing.invoke_softmax_topk_route_gluon(
            logits, topk, correction_bias=bias, routed_scaling_factor=2.5
        )
        for nw in (1, 2, 4, 8):
            try:
                _launch_route(logits, bias, topk, ids, weights, nw)
                torch.cuda.synchronize()
                if not torch.equal(ids, ref_ids):
                    entry["configs"][f"warps={nw}"] = "parity FAIL"
                    continue
                entry["configs"][f"warps={nw}"] = time_fn(
                    lambda: _launch_route(logits, bias, topk, ids, weights, nw)  # noqa: B023
                )
            except Exception as exc:  # noqa: BLE001
                entry["configs"][f"warps={nw}"] = f"error: {exc!r}"[:80]
        entry["upstream_warps"] = min(1 << (max(1, m) - 1).bit_length(), 4)
        results.append(entry)


def _launch_argmax(logits, out, block, num_warps, use_split, fixed_split=None):
    from vkernels.torch_ops.tokenspeed_amd import argmax as ts_argmax

    m, n = logits.shape
    load_elems = ts_argmax._load_elements_per_thread(logits.dtype)
    if fixed_split is not None:
        num_splits = fixed_split
        chunk = triton.cdiv(n, num_splits)
        pv, pi, cnt = ts_argmax._get_atomic_scratch(m, num_splits, logits.device)
        ts_argmax.gluon_argmax_split_fixed_count_gfx950[(m, num_splits)](
            logits, pv, pi, cnt, out,
            stride_m=logits.stride(0), out_stride=out.stride(0),
            N=n, CHUNK_SIZE=chunk, BLOCK=block, NUM_SPLITS=num_splits,
            REDUCE_BLOCK=num_splits, LOAD_ELEMS=load_elems, num_warps=num_warps,
        )
    elif use_split:
        num_splits = triton.cdiv(n, block)
        split_block = triton.next_power_of_2(num_splits)
        pv, pi, cnt = ts_argmax._get_atomic_scratch(m, num_splits, logits.device)
        ts_argmax.gluon_argmax_split_fixed_block_gfx950[(m, num_splits)](
            logits, pv, pi, cnt, out,
            stride_m=logits.stride(0), out_stride=out.stride(0),
            N=n, NUM_SPLITS=num_splits, BLOCK=block,
            REDUCE_BLOCK=split_block, LOAD_ELEMS=load_elems, num_warps=num_warps,
        )
    else:
        ts_argmax.gluon_argmax_gfx950[(m,)](
            logits, out,
            stride_m=logits.stride(0), out_stride=out.stride(0),
            N=n, BLOCK=block, LOAD_ELEMS=load_elems, num_warps=num_warps,
        )


def sweep_argmax(results):
    from vkernels.torch_ops.tokenspeed_amd import argmax as ts_argmax

    torch.manual_seed(1)
    shapes = [(1, 32768), (1, 131072), (8, 131072), (64, 151936), (8, 262144), (16, 65536)]
    for m, n in shapes:
        logits = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        ref = torch.argmax(logits, dim=-1)
        out = torch.empty(m, dtype=torch.int64, device="cuda")
        entry = {"op": "sampling_argmax", "shape": {"M": m, "N": n}, "configs": {}}
        entry["configs"]["torch"] = time_fn(lambda: torch.argmax(logits, dim=-1))  # noqa: B023
        variants = []
        if m <= 4 and n >= 65536:
            variants += [("split-fixed", b, w, s) for s in (8, 16, 32) for b, w in ((4096, 2), (8192, 4), (8192, 2))]
        variants += [("split", b, w, None) for b in (8192, 16384) for w in (2, 4)]
        variants += [("plain", b, w, None) for b in (4096, 8192, 16384) for w in (2, 4)]
        for kind, block, warps, fixed in variants:
            key = f"{kind} b={block} w={warps}" + (f" s={fixed}" if fixed else "")
            try:
                _launch_argmax(logits, out, block, warps, kind == "split", fixed)
                torch.cuda.synchronize()
                if not torch.equal(out, ref):
                    entry["configs"][key] = "parity FAIL"
                    continue
                entry["configs"][key] = time_fn(
                    lambda: _launch_argmax(logits, out, block, warps, kind == "split", fixed)  # noqa: B023
                )
            except Exception as exc:  # noqa: BLE001
                entry["configs"][key] = f"error: {exc!r}"[:80]
        entry["upstream"] = str(ts_argmax._select_config(m, n))
        results.append(entry)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sweep_tokenspeed_amd_gfx942.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    env = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    try:
        import triton

        env["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        env["triton"] = "unavailable"
    results: list = []
    for name, fn in (("routing", sweep_routing), ("argmax", sweep_argmax)):
        try:
            fn(results)
            print(f"[{name}] ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"op": name, "error": repr(exc)})
    report = {"env": env, "results": results}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
