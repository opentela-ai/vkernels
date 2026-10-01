"""Lane-31 GB10 microbench matrix: T in {1,2,4,8} x the glm53-tp4 decode MoE
shapes (census: E=288 experts, top_k=8, H=4096, I_tp=512) — DeepGEMM masked
grouped borrow vs the incumbents (expert_gemv, sgl_fused_moe).

On GB10 (capability 12,1) the DeepGEMM column is structurally N/A: ptxas
rejects both wgmma-fp8 and tcgen05 for sm_121 and DeepGEMM's host dispatch
is unreachable at arch major 12 — this script PRINTS the gate's reason and
measures the incumbents anyway, so the rig A/B (sm90, sglang-glm53 image)
has the exact same-harness baseline to beat. Parity vs the eager fp8-w8a8
oracle is tolerance-gated (the runs/moegrp-micro halfway-rounding class is
inside the band by design).

Run:  python bench/deepgemm_moe_matrix.py [--iters 200]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from vkernels.torch_ops.deepgemm_moe import (  # noqa: E402
    deepgemm_grouped_moe,
    deepgemm_moe_eligible,
    deepgemm_unavailable_reason,
)
from vkernels.torch_ops.glm_expert_gemv import expert_gemv  # noqa: E402
from vkernels.torch_ops.moe_combine import moe_weighted_sum  # noqa: E402
from vkernels.torch_ops.sgl_moe import per_token_group_quant_fp8, sgl_fused_moe  # noqa: E402

SWIGLU_LIMIT = 7.0
E, H, I_TP, TOPK = 288, 4096, 512, 8  # glm53 tp4 census shapes (per rank)


def make_case(t: int, seed: int = 0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(t, H, device=device, dtype=torch.bfloat16, generator=gen)
    w13 = (torch.randn(E, 2 * I_TP, H, device=device, generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I_TP, device=device, generator=gen) * 0.02).to(torch.bfloat16)

    def quant(w, o, i):
        wb = w.float().view(E, o // 128, 128, i // 128, 128)
        s = wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0
        return (wb / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(E, o, i), \
            s.squeeze(2).squeeze(3).contiguous()

    w13_8, s13 = quant(w13, 2 * I_TP, H)
    w2_8, s2 = quant(w2, H, I_TP)
    idx = torch.stack([torch.randperm(E, device=device, generator=gen)[:TOPK]
                       for _ in range(t)]).to(torch.int64)
    w = torch.softmax(torch.randn(t, TOPK, device=device, generator=gen), dim=-1)
    return x, w13_8, s13, w2_8, s2, idx, w


def step_expert_gemv(x, w13, s13, w2, s2, idx, w):
    gate_up = expert_gemv(x, w13, s13, idx, t_cap=8)
    gate, up = gate_up.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate.clamp(max=SWIGLU_LIMIT)) \
        * up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    out = expert_gemv(act, w2, s2, idx, t_cap=8)
    return moe_weighted_sum(out, w.to(torch.float32))


def step_sgl_fused(x, w13, s13, w2, s2, idx, w):
    return sgl_fused_moe(x, w13, s13, w2, s2, idx, w, swiglu_limit=SWIGLU_LIMIT)


def step_deepgemm(x, w13, s13, w2, s2, idx, w):
    """Called only where the gate is ON; on GB10 the matrix skips it."""
    return deepgemm_grouped_moe(x, w13, s13, w2, s2, idx, w, swiglu_limit=SWIGLU_LIMIT)


def eager_oracle(x, w13, s13, w2, s2, idx, w):
    def deq(w8, s):
        e, o, i = w8.shape
        return (w8.float().view(e, o // 128, 128, i // 128, 128)
                * s.float()[:, :, None, :, None]).reshape(e, o, i)

    W13, W2 = deq(w13, s13).to(torch.bfloat16), deq(w2, s2).to(torch.bfloat16)
    t = x.shape[0]
    acc = torch.zeros(t, H, device=x.device, dtype=torch.float32)
    for tok in range(t):
        for k in range(TOPK):
            j = int(idx[tok, k])
            xq, xs = per_token_group_quant_fp8(x[tok: tok + 1])
            a = (xq.float() * xs.repeat_interleave(128, dim=-1)).to(torch.bfloat16) \
                @ W13[j].t()
            g, u = a[0, :I_TP], a[0, I_TP:]
            act = (torch.nn.functional.silu(g.clamp(max=SWIGLU_LIMIT))
                   * u.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT))
            aq, as_ = per_token_group_quant_fp8(act.unsqueeze(0))
            b = (aq.float() * as_.repeat_interleave(128, dim=-1)).to(torch.bfloat16) \
                @ W2[j].t()
            acc[tok] += float(w[tok, k]) * b[0].float()
    return acc.to(torch.bfloat16)


def bench(fn, args, iters: int) -> float:
    """Median of do_bench reps (GB10 wall-clock is noisy run-to-run; the
    CUDA-event mean over 200 iters swung ±30% between invocations)."""
    from triton.testing import do_bench

    return do_bench(lambda: fn(*args), warmup=25, rep=iters, return_mode="median")


def parity(out, ref) -> dict:
    d = (out.float() - ref.float()).abs()
    rel = d / ref.float().abs().clamp_min(1e-3)
    rms = ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
    return {
        "max_abs": round(d.max().item(), 5),
        "max_abs_over_rms": round((d.max() / rms).item(), 4),
        "median_rel": f"{rel.median().item():.2e}",
        "p999_abs_over_rms": f"{torch.quantile(d.flatten().float(), 0.999).item() / rms.item():.2e}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    name = torch.cuda.get_device_name(0).replace(" ", "_")
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} capability {cap}")
    gate = deepgemm_unavailable_reason()
    print(f"deepgemm gate: {'AVAILABLE' if gate is None else 'OFF — ' + gate}")
    print(f"shapes: E={E} H={H} I_tp={I_TP} topk={TOPK}\n")

    rows, parities = [], {}
    for t in (1, 2, 4, 8):
        case = make_case(t)
        oracle = eager_oracle(*case)
        row = {"T": t}
        for label, fn in (("expert_gemv", step_expert_gemv),
                          ("sgl_fused_moe", step_sgl_fused),
                          ("deepgemm_moe", step_deepgemm)):
            if label == "deepgemm_moe":
                if not deepgemm_moe_eligible(*case):
                    row[label] = None  # gated OFF on this arch (reason printed above)
                    continue
                out = fn(*case)
                parities[(t, label)] = parity(out, oracle)
                row[label] = round(bench(fn, case, args.iters), 4)
                continue
            out = fn(*case)
            parities[(t, label)] = parity(out, oracle)
            row[label] = round(bench(fn, case, args.iters), 4)
        rows.append(row)
        print(f"T={t}: " + "  ".join(
            f"{k}={v if v is not None else 'N/A'}" for k, v in row.items()))

    print("\nparity vs eager fp8-w8a8 oracle (tolerance-gated):")
    for (t, label), p in parities.items():
        print(f"  T={t} {label}: {p}")

    out_path = Path(__file__).parent / "deepgemm_moe_matrix.gb10.json"
    out_path.write_text(json.dumps({
        "device": {"name": name, "capability": list(cap)},
        "gate_reason": gate,
        "shapes": {"E": E, "H": H, "I_tp": I_TP, "topk": TOPK},
        "iters": args.iters,
        "rows": rows,
        "parity": {f"T{t}/{k}": v for (t, k), v in parities.items()},
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
