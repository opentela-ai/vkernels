"""Issue #148 MI300A bench: hardware-fp8 MFMA decode on the fnuz storage.

Run with PYTHONPATH=src/python python meta/benchmarks/bench_glm_issue148_decode.py
on a gfx942 node (see meta/scripts/issue148_decode_mfma_mi300.sh).

Measures, at the GLM decode expert shapes (gate/up [4096,4096], down
[4096,2048], M in {1,2}):

  * glm_decode_fp8_gemm (new): fnuz weights -> CDNA3 v_mfma fp8 dot, split-K
    over the 128-wide scale blocks, fused one-launch activation quantize.
    Config sweep (BN x warps x stages x split-K), kernel-only and
    end-to-end (with quantize + partial reduce).
  * Baselines in the same process: the torch-blockwise fp8 GEMM backend on
    the same fnuz operands, and expert_gemv's manual fnuz bit-decode GEMV
    with a GLM53_GEMV_FNUZ_NATIVE=1 A/B (hardware float8e4b8 convert).

Parity before timing:
  * decode GEMM vs the fp32 dequant reference (fnuz-rounded class), and
    vs the torch blockwise oracle on the SAME quantized operands.
  * expert_gemv FNUZ_NATIVE output must be BIT-identical to the manual
    decode (the fnuz->f32 convert is exact).

The HIP fused-kernel records (22.4/16.4 us, job 629781) are re-measured
on the same node by the sbatch wrapper via glm_fp8_gemv_bench.
"""

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch

from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
    _torch_blockwise_gemm,
    _triton_decode_kernel,
    e4m3fn_to_fnuz_inplace,
    glm_decode_fp8_gemm,
    quantize_activations_fnuz,
    quantize_activations_fnuz_fused,
)  # _torch_blockwise_gemm: BM=128 default-backend baseline only
from vkernels.torch_ops.glm_expert_gemv import expert_gemv

SHAPES = {
    "gate_up": (4096, 4096),  # [N, K]
    "down": (4096, 2048),
}


def make_panel(torch, n, k, e=8, seed=3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randint(0, 256, (e, n, k), dtype=torch.uint8, generator=g)
    raw.masked_fill_((raw & 127) == 127, 0)  # drop the fn NaN encodings
    w = raw.view(torch.float8_e4m3fn)
    scales = (torch.rand((e, n // 128, k // 128), generator=g) * 0.01 + 0.002).float()
    return w.to("cuda"), scales.to("cuda")


def dequant_ref(w_fnuz, scales2):
    """fp32 dequant reference: fnuz payload x doubled scale, per 128-block."""
    n, k = w_fnuz.shape
    w32 = w_fnuz.to(torch.float32).view(n // 128, 128, k // 128, 128)
    w32 = w32 * scales2.view(n // 128, 1, k // 128, 1)
    return w32.view(n, k)


def time_fn(fn, warmup=10, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, stops))
    return statistics.median(ms) * 1e3  # us


def kernel_only_decode(x, w_fnuz, w_scales2, m, n, k, bn, warps, stages, sk, kp):
    """Fused (in-GEMM quantize) launch only, output buffer preallocated —
    comparable to the HIP fused kernel + its reduce."""
    import triton

    part = torch.empty((sk, m, n), device="cuda", dtype=torch.float32)
    _triton_decode_kernel()[(triton.cdiv(n, bn), sk)](
        x, w_scales2, w_fnuz, w_scales2, part, m, n, k,
        w_scales2.stride(0), w_scales2.stride(1),
        w_scales2.stride(0), w_scales2.stride(1),
        NKB=k // 128, SK=sk, BM=16, BN=bn, KP=kp, FUSE_Q=True,
        num_warps=warps, num_stages=stages,
    )
    return part.sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="work/glm53/issue148_decode.json")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--sweep", action="store_true", help="full config sweep")
    args = ap.parse_args()

    torch.manual_seed(0)
    results = {"device": torch.cuda.get_device_name(0), "shapes": {}}

    for name, (n, k) in SHAPES.items():
        w_fn, s_fn = make_panel(torch, n, k)
        w_fnuz, s2 = e4m3fn_to_fnuz_inplace(w_fn[0].clone(), s_fn[0].clone())
        wref = dequant_ref(w_fnuz, s2)
        shape_res = {}
        best_m1 = {}  # shape -> best M=1 decode config, reused for M=2

        for m in (1, 2):
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.5
            dref = x.float() @ wref.T  # fp32 dequant reference [m, n]

            # ---- parity ------------------------------------------------
            out = glm_decode_fp8_gemm(x, w_fnuz, s2, split_k=4)
            assert torch.isfinite(out).all()
            rel_ref = ((out.float() - dref).norm() / dref.norm().clamp_min(1e-12)).item()
            # fp32 oracle on the SAME quantized operands (per-row scales)
            a_nz, asc2 = quantize_activations_fnuz(x)
            orc = torch.zeros((m, n), device="cuda")
            a32o = a_nz.to(torch.float32)
            w32o = w_fnuz.to(torch.float32).view(n // 128, 128, k // 128, 128)
            for kb in range(k // 128):
                ww = (w32o[:, :, kb] * s2[:, kb].view(-1, 1, 1)).reshape(n, 128)
                aa = a32o[:, kb * 128 : (kb + 1) * 128] * asc2[:, kb : kb + 1]
                orc += aa @ ww.T
            rel_orc = ((out.float() - orc).norm() / orc.norm().clamp_min(1e-12)).item()
            # fused quantize vs torch-chain quantize
            qf, sf = quantize_activations_fnuz_fused(x)
            vf = qf.view(torch.uint8).view(torch.float8_e4m3fnuz).to(torch.float32) * sf.repeat_interleave(128, dim=1)
            vt = a_nz.to(torch.float32) * asc2.repeat_interleave(128, dim=1)
            q_rel = ((vf - vt).norm() / vt.norm().clamp_min(1e-12)).item()
            assert rel_ref < 0.05, rel_ref
            assert rel_orc < 0.01, rel_orc
            assert q_rel < 0.01, q_rel

            # ---- sweep (M=1 only; best config reused for M=2) -----------
            cfgs = []
            if args.sweep and m == 1:
                grid = itertools.product((64, 128), (4,), (2, 3), (2, 4, 8))
            elif m == 2 and best_m1[name]:
                b1 = best_m1[name]
                grid = [(b1["bn"], b1["warps"], b1["stages"], b1["sk"])]
            else:
                # MI300A-feasible default (BN=128 s=3 exceeds LDS on gfx942)
                grid = [(128, 4, 2, 4)]
            best = None
            for bn, warps, stages, sk in grid:
                kp = 2 if (k // 128 // sk) % 2 == 0 else 1
                try:
                    candidate = kernel_only_decode(x, w_fnuz, s2, m, n, k, bn, warps, stages, sk, kp)
                    error = (candidate.float() - orc).norm() / orc.norm().clamp_min(1e-12)
                    assert torch.isfinite(candidate).all() and error < .01, error
                    t = time_fn(
                        lambda: kernel_only_decode(x, w_fnuz, s2, m, n, k, bn, warps, stages, sk, kp),
                        iters=args.iters,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  cfg BN={bn} w={warps} s={stages} sk={sk}: FAILED {type(exc).__name__}: {exc}")
                    continue
                gbps = n * k / t / 1e3
                print(f"  M={m} BN={bn} warps={warps} stages={stages} sk={sk}: {t:8.2f} us  {gbps:7.0f} GB/s")
                cfgs.append({"bn": bn, "warps": warps, "stages": stages, "sk": sk, "us": t, "GBps": gbps})
                if best is None or t < best["us"]:
                    best = cfgs[-1]
            assert best, "all decode configs failed"
            if m == 1:
                best_m1[name] = best

            bbn, bw, bs, bsk = best["bn"], best["warps"], best["stages"], best["sk"]
            os.environ["VK_DECODE_GEMM_TILES"] = f"{bbn},{bw},{bs}"
            selected = glm_decode_fp8_gemm(x, w_fnuz, s2, split_k=bsk)
            assert torch.isfinite(selected).all()
            assert (selected.float() - orc).norm() / orc.norm().clamp_min(1e-12) < .01
            # end-to-end (quantize + gemm + reduce) at the best config
            t_e2e = time_fn(lambda: glm_decode_fp8_gemm(x, w_fnuz, s2, split_k=bsk), iters=args.iters)
            t_q = time_fn(lambda: quantize_activations_fnuz_fused(x), iters=args.iters)

            # ---- baselines ---------------------------------------------
            # torch-op default backend (BM=128 tile) on the same operands
            t_torch_backend = time_fn(
                lambda: _torch_blockwise_gemm(a_nz, asc2, w_fnuz, s2, torch.empty((m, n), device="cuda", dtype=torch.bfloat16)),
                iters=max(args.iters // 4, 50),
            ) if m == 1 else None
            # expert_gemv manual fnuz decode, single (token, expert) pair
            idx = torch.tensor([[0]], device="cuda", dtype=torch.int64)
            w_fnuz_stack, s2_stack = e4m3fn_to_fnuz_inplace(w_fn.clone(), s_fn.clone())
            a_nz, asc2 = quantize_activations_fnuz_fused(x)
            t_gemv_manual = time_fn(
                lambda: expert_gemv(x[:1], w_fnuz_stack, s2_stack, idx, storage="e4m3fnuz"),
                iters=args.iters,
            )
            os.environ["GLM53_GEMV_FNUZ_NATIVE"] = "1"
            t_gemv_native = time_fn(
                lambda: expert_gemv(x[:1], w_fnuz_stack, s2_stack, idx, storage="e4m3fnuz"),
                iters=args.iters,
            )
            # bit-identity of the fnuz-native decode
            os.environ.pop("GLM53_GEMV_FNUZ_NATIVE")
            out_manual = expert_gemv(x[:1], w_fnuz_stack, s2_stack, idx, storage="e4m3fnuz")
            os.environ["GLM53_GEMV_FNUZ_NATIVE"] = "1"
            out_native = expert_gemv(x[:1], w_fnuz_stack, s2_stack, idx, storage="e4m3fnuz")
            os.environ.pop("GLM53_GEMV_FNUZ_NATIVE")
            bit_identical = bool(torch.equal(out_manual, out_native))
            assert bit_identical, "native FNUZ conversion changed expert GEMV output"

            shape_res[f"M={m}"] = {
                "parity_vs_f32ref_rel": rel_ref,
                "parity_vs_torch_oracle_rel": rel_orc,
                "fused_quantize_rel": q_rel,
                "best_cfg": best,
                "decode_kernel_only_us": best["us"],
                "decode_kernel_only_GBps": best["GBps"],
                "decode_e2e_us": t_e2e,
                "fused_quantize_us": t_q,
                "torch_backend_us": t_torch_backend,
                "expert_gemv_manual_us": t_gemv_manual,
                "expert_gemv_fnuz_native_us": t_gemv_native,
                "expert_gemv_fnuz_native_bit_identical": bit_identical,
                "weight_stream_roof_GBps_at_copy": 2868.0,
            }
            print(f"[{name} M={m}] best {best}  e2e {t_e2e:.2f} us  quant {t_q:.2f} us")
            print(f"  expert_gemv manual {t_gemv_manual:.2f} us vs FNUZ_NATIVE {t_gemv_native:.2f} us (bit-identical={bit_identical})")
        results["shapes"][name] = shape_res

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
