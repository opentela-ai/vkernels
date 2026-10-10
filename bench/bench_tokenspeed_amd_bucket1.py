#!/usr/bin/env python3
"""Benchmark the vendored TokenSpeed-Kernel-AMD "bucket 1" ops on AMD GPUs.

Measures each vendored component against a baseline on the current device
(beverin / MI300A is the reference target), after asserting parity once so
the timings describe verified-correct paths:

  routing (softmax gluon top-k)   vs torch softmax->topk->gather->normalize
  moe sorting (4 gluon stages)    vs aiter moe_align_block_size (if importable)
                                   and a pure-torch reference sort
  hadamard-128 (gluon butterfly)  vs torch fp32 matmul with H128
  mhc pre reduce/apply (gluon)    vs the torch reference chain
  argmax (gluon split-atomic)     vs torch.argmax — gluon paths need
                                   ``gluon.language.amd.cdna4``; recorded as
                                   unavailable where the host triton lacks it

Emits a JSON report to stdout and to ``--out``. Run inside an env with ROCm
torch + triton (beverin: the glm5-smoke environment via sbatch), with
``PYTHONPATH=<repo>/src/python``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

from vkernels.torch_ops import tokenspeed_amd as ts_amd
from vkernels.torch_ops.tokenspeed_amd import hadamard as ts_hadamard
from vkernels.torch_ops.tokenspeed_amd import moe_sorting as ts_sorting
from vkernels.torch_ops.tokenspeed_amd import mhc as ts_mhc
from vkernels.torch_ops.tokenspeed_amd import routing as ts_routing
from vkernels.torch_ops.tokenspeed_amd import scale_layout as ts_scale


def _triton():
    import triton

    return triton


def bench_env() -> dict:
    env = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    try:
        env["triton"] = _triton().__version__
    except Exception as exc:  # noqa: BLE001
        env["triton"] = f"unavailable: {exc}"
    env["gluon"] = ts_amd.gluon_available()
    env["gluon_amd_cdna4"] = ts_amd.cdna4_available()
    env["gluon_features"] = ts_amd.gluon_features()
    try:
        import aiter  # noqa: F401

        env["aiter"] = True
    except Exception:  # noqa: BLE001
        env["aiter"] = False
    return env


def time_fn(fn, *, warmup: int = 10, iters: int = 50) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # us
    return {
        "us_median": round(statistics.median(times), 2),
        "us_p10": round(sorted(times)[iters // 10], 2),
        "us_p90": round(sorted(times)[(9 * iters) // 10], 2),
    }


# --- routing -------------------------------------------------------------------


def torch_softmax_route(logits, topk, bias, scale, normalize=True):
    probs = torch.softmax(logits.float(), dim=-1)
    choice = probs + bias if bias is not None else probs
    sel = torch.topk(choice, topk, dim=-1).indices
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None]
    w = probs[rows, sel]
    if normalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return sel.to(torch.int32), (w * scale).to(torch.float32)


def parity_routing() -> None:
    torch.manual_seed(0)
    for m, e, topk in ((1, 128, 8), (8, 256, 16)):
        logits = torch.randn(m, e, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(e, device="cuda", dtype=torch.float32)
        ids, weights = ts_routing.invoke_softmax_topk_route_gluon(
            logits, topk, correction_bias=bias, routed_scaling_factor=2.5
        )
        sel, w = torch_softmax_route(logits, topk, bias, 2.5)
        assert torch.equal(ids.to(torch.int64), sel.to(torch.int64)), (m, e, topk)
        assert torch.allclose(weights, w, atol=1e-2, rtol=1e-2), (m, e, topk)


def bench_routing(results: list) -> None:
    parity_routing()
    for m, e, topk in ((1, 128, 16), (1, 384, 16), (8, 256, 16), (16, 128, 8), (64, 384, 2)):
        if m * topk > 128 or e > 1024:
            continue
        logits = torch.randn(m, e, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(e, device="cuda", dtype=torch.float32)
        gluon = time_fn(
            lambda: ts_routing.invoke_softmax_topk_route_gluon(  # noqa: B023
                logits, topk, correction_bias=bias, routed_scaling_factor=2.5
            )
        )
        ref = time_fn(lambda: torch_softmax_route(logits, topk, bias, 2.5))  # noqa: B023
        results.append(
            {
                "op": "routing_softmax_topk",
                "shape": {"M": m, "E": e, "topk": topk},
                "gluon": gluon,
                "torch": ref,
            }
        )


# --- moe sorting -----------------------------------------------------------------


def torch_reference_sort(ids, weights, e, block):
    flat = ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=e)
    starts = [0]
    for c in counts.tolist():
        starts.append(starts[-1] + -(-c // block) * block)
    order = torch.argsort(flat, stable=True)
    return flat[order], weights.reshape(-1)[order], starts


def parity_sorting() -> None:
    torch.manual_seed(1)
    m, e, topk, block = 512, 128, 4, 16
    ids = torch.randint(0, e, (m, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand(m, topk, device="cuda", dtype=torch.float32)
    (sorted_ids, sorted_weights, _, num_valid, _) = ts_sorting.gluon_moe_sorting(
        ids, weights, e, 64, torch.bfloat16, block, compact_route_programs=False
    )
    # per-expert multiset comparison via the block-aligned slot ranges
    flat = ids.reshape(-1)
    counts = torch.bincount(flat, minlength=e)
    starts = [0]
    for c in counts.tolist():
        starts.append(starts[-1] + -(-c // block) * block)
    for exp in range(0, e, 31):
        s, c = starts[exp], int(counts[exp])
        got_tokens = sorted((sorted_ids[s : s + c] & 0xFFFFFF).tolist())
        ref_tokens = sorted(((flat == exp).nonzero().flatten() // topk).tolist())
        assert got_tokens == ref_tokens, exp
        for slot in range(s, s + c):
            k = int(sorted_ids[slot] >> 24)
            tok = int(sorted_ids[slot] & 0xFFFFFF)
            assert abs(float(sorted_weights[slot]) - float(weights[tok, k])) < 1e-6, (
                exp,
                slot,
            )
    assert int(num_valid[0]) == starts[-1]


def bench_sorting(results: list) -> None:
    parity_sorting()
    aiter_align = None
    if bench_env_cached.get("aiter"):
        try:
            from vkernels.torch_ops.aiter_ops import moe_align_block_size as aiter_align  # type: ignore
        except Exception:  # noqa: BLE001
            aiter_align = None
    for m, e, topk, block, compact in (
        (8, 128, 4, 16, True),
        (8, 256, 8, 16, True),
        (64, 128, 8, 64, True),
        (512, 128, 4, 64, False),
        (4096, 256, 8, 128, False),
    ):
        ids = torch.randint(0, e, (m, topk), device="cuda", dtype=torch.int32)
        weights = torch.rand(m, topk, device="cuda", dtype=torch.float32)
        gluon = time_fn(
            lambda: ts_sorting.gluon_moe_sorting(  # noqa: B023
                ids, weights, e, 64, torch.bfloat16, block,
                compact_route_programs=compact,
            )
        )
        entry = {
            "op": "moe_sorting",
            "shape": {"M": m, "E": e, "topk": topk, "block": block, "compact": compact},
            "gluon": gluon,
            "torch_ref": time_fn(lambda: torch_reference_sort(ids, weights, e, block)),  # noqa: B023
        }
        if aiter_align is not None:
            entry["aiter"] = time_fn(
                lambda: aiter_align(ids, e, block)  # noqa: B023
            )
        results.append(entry)


# --- hadamard ---------------------------------------------------------------------


def hadamard_matrix(device):
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    mat = base
    for _ in range(6):
        mat = torch.kron(mat, base)
    return mat.to(device=device, dtype=torch.float32)


def parity_hadamard() -> None:
    torch.manual_seed(2)
    x = torch.randn(1024, 128, device="cuda", dtype=torch.bfloat16)
    scale = 128 ** -0.5
    out = ts_hadamard.launch_gluon_hadamard_transform_128_gfx950(x, scale=scale)
    ref = (x.float() @ hadamard_matrix(x.device).T) * scale
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=1e-2)


def bench_hadamard(results: list) -> None:
    parity_hadamard()
    mat = hadamard_matrix("cuda")
    scale = 128 ** -0.5
    for rows in (4096, 32768, 262144):
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        gluon = time_fn(
            lambda: ts_hadamard.launch_gluon_hadamard_transform_128_gfx950(x, scale=scale)  # noqa: B023
        )

        def torch_fn(x=x, mat=mat):
            return torch.mm(x.float(), mat.T, out=torch.empty(rows, 128, device="cuda")).to(torch.bfloat16)

        ref = time_fn(torch_fn)
        bytes_moved = rows * 128 * 2 * 2
        results.append(
            {
                "op": "hadamard_128",
                "shape": {"rows": rows},
                "gluon": gluon,
                "torch_matmul_fp32": ref,
                "effective_GBps": round(bytes_moved / (gluon["us_median"] * 1e-6) / 1e9, 1),
            }
        )


# --- mhc ---------------------------------------------------------------------------


def torch_mhc_reference(mul, sqrsum, hc_scale, hc_base, residual, eps_rms, eps_hc, iters):
    tokens, hidden = residual.shape[0], residual.shape[2]
    inv_rms = torch.rsqrt(sqrsum.sum(0) / (4 * hidden) + eps_rms)
    proj = mul.sum(0)
    pre = (
        1.0 / (1.0 + torch.exp(-(proj[:, :4] * inv_rms[:, None] * hc_scale[0] + hc_base[:4])))
        + eps_hc
    )
    layer = (pre[:, :, None] * residual).sum(1).to(torch.bfloat16)
    post = 2.0 / (1.0 + torch.exp(-(proj[:, 4:8] * inv_rms[:, None] * hc_scale[1] + hc_base[4:8])))
    comb = proj[:, 8:].reshape(tokens, 4, 4) * inv_rms[:, None, None] * hc_scale[2]
    comb = comb + hc_base[8:].reshape(4, 4)
    comb = torch.exp(comb - comb.max(dim=-1, keepdim=True).values)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps_hc
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps_hc)
    # Verified on MI300A (triton 3.8): the gluon kernel places HC_EPS in the
    # denominator during the Sinkhorn loop (matching the upstream Triton
    # reference), not as a post-division add.
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps_hc)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps_hc)
    return layer, post, comb.reshape(tokens, 16)


def parity_mhc() -> None:
    torch.manual_seed(3)
    tokens, hidden, splits = 8, 4096, 64
    mul = torch.randn(splits, tokens, 24, device="cuda", dtype=torch.float32)
    sqrsum = torch.rand(splits, tokens, device="cuda", dtype=torch.float32)
    hc_scale = torch.randn(3, device="cuda")
    hc_base = torch.randn(24, device="cuda")
    residual = torch.randn(tokens, 4, hidden, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    post_mix = torch.empty(tokens, 4, device="cuda")
    comb_mix = torch.empty(tokens, 16, device="cuda")
    ts_mhc.gluon_mhc_pre_reduce_apply_gfx950(
        mul, sqrsum, hc_scale, hc_base, residual, layer_input,
        post_mix, comb_mix, hidden, 1e-6, 1e-4, 20, splits, tokens,
    )
    layer, post, comb = torch_mhc_reference(mul, sqrsum, hc_scale, hc_base, residual, 1e-6, 1e-4, 20)
    assert torch.allclose(layer_input.float(), layer.float(), atol=2e-2, rtol=1e-2)
    assert torch.allclose(post_mix, post, atol=2e-3, rtol=1e-3)
    assert torch.allclose(comb_mix, comb, atol=5e-3, rtol=1e-2)


def bench_mhc(results: list) -> None:
    parity_mhc()
    for tokens in (1, 8, 64):
        hidden, splits = 4096, 64
        mul = torch.randn(splits, tokens, 24, device="cuda")
        sqrsum = torch.rand(splits, tokens, device="cuda")
        hc_scale = torch.randn(3, device="cuda")
        hc_base = torch.randn(24, device="cuda")
        residual = torch.randn(tokens, 4, hidden, device="cuda")
        layer_input = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        post_mix = torch.empty(tokens, 4, device="cuda")
        comb_mix = torch.empty(tokens, 16, device="cuda")

        def gluon_fn():
            ts_mhc.gluon_mhc_pre_reduce_apply_gfx950(
                mul, sqrsum, hc_scale, hc_base, residual, layer_input,
                post_mix, comb_mix, hidden, 1e-6, 1e-4, 20, splits, tokens,
            )

        def ref_fn():
            torch_mhc_reference(mul, sqrsum, hc_scale, hc_base, residual, 1e-6, 1e-4, 20)

        results.append(
            {
                "op": "mhc_pre_reduce_apply",
                "shape": {"tokens": tokens, "hidden": hidden, "splits": splits},
                "gluon": time_fn(gluon_fn),
                "torch_ref": time_fn(ref_fn),
            }
        )


# --- argmax -------------------------------------------------------------------------


def bench_argmax(results: list) -> None:
    from vkernels.torch_ops.tokenspeed_amd import argmax as ts_argmax

    for m, n in ((1, 32768), (8, 131072), (64, 151936), (8, 262144)):
        logits = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        entry = {
            "op": "sampling_argmax",
            "shape": {"M": m, "N": n},
            "torch": time_fn(lambda: torch.argmax(logits, dim=-1)),  # noqa: B023
        }
        if ts_amd.cdna4_available() and n >= 4096:
            got = ts_argmax.launch_gluon_argmax_gfx950(logits)
            assert torch.equal(got, torch.argmax(logits, dim=-1))
            entry["gluon"] = time_fn(
                lambda: ts_argmax.launch_gluon_argmax_gfx950(logits)  # noqa: B023
            )
        else:
            entry["gluon"] = "unavailable: gluon amd.cdna4 missing on this triton"
        results.append(entry)


# --- host-side preprocessing (torch, any device) --------------------------------------


def bench_scale_swizzle(results: list) -> None:
    torch.manual_seed(4)
    scale = torch.randint(100, 130, (128 * 2880, 2880 // 32), device="cuda", dtype=torch.uint8)
    t = time_fn(lambda: ts_scale.swizzle_cdna4_mxfp4_scale(scale), warmup=3, iters=10)
    results.append(
        {
            "op": "scale_swizzle_cdna4 (host/torch)",
            "shape": {"rows": int(scale.shape[0]), "k_scale": int(scale.shape[1])},
            "torch": t,
        }
    )


bench_env_cached: dict = {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench_tokenspeed_amd_bucket1.json")
    parser.add_argument(
        "--skip", nargs="*", default=[], help="bench sections to skip"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/ROCm device required", file=sys.stderr)
        return 2
    torch.cuda.init()
    bench_env_cached.update(bench_env())

    results: list = []
    sections = {
        "routing": bench_routing,
        "sorting": bench_sorting,
        "hadamard": bench_hadamard,
        "mhc": bench_mhc,
        "argmax": bench_argmax,
        "scale": bench_scale_swizzle,
    }
    for name, fn in sections.items():
        if name in args.skip:
            continue
        t0 = time.time()
        try:
            fn(results)
            print(f"[{name}] ok ({time.time() - t0:.1f}s)", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — record and continue
            print(f"[{name}] FAILED: {exc!r}", file=sys.stderr)
            results.append({"op": name, "error": repr(exc)})

    report = {
        "env": bench_env_cached,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
