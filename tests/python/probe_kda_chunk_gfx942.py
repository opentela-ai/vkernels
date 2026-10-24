#!/usr/bin/env python3
"""MI300A gate: the fused Triton KDA chunk pipeline on gfx942.

Parity vs the eager fp32 reference + per-shape timing at the serving
shapes (TP4: H=16 sharded heads, D=128; solo H=64 for reference), the
spec-verify small-T spans (S=4/8) and the prefill chunk (S=2048).
The kernel is the same fla chunk_kda Triton port that is default-on for
NVIDIA (GH200 parity 7.7e-07, job 3469211); this probe decides whether
VK_GLM_KDA_CHUNK_TRITON_HIP flips to trusted for beverin's serving.
"""
import os
import time

os.environ.setdefault("VK_GLM_KDA_CHUNK_TRITON_HIP", "1")

import torch

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.glm_kda_chunk import kda_chunk, kda_chunk_reference


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def run_case(bsz, heads, S, D=128, chunk=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(bsz, S, heads, D, generator=g, device="cuda", dtype=torch.float32) * 0.5
    k = torch.randn(bsz, S, heads, D, generator=g, device="cuda", dtype=torch.float32) * 0.5
    v = torch.randn(bsz, S, heads, D, generator=g, device="cuda", dtype=torch.float32) * 0.5
    gg = -torch.rand(bsz, S, heads, D, generator=g, device="cuda", dtype=torch.float32) * 5.0  # log-space in [-5, 0]
    beta = torch.rand(bsz, S, heads, generator=g, device="cuda", dtype=torch.float32)

    try:
        fused_out, fused_state = kda_chunk(q, k, v, gg, beta, chunk_size=chunk,
                                           initial_state=None, output_final_state=True)
    except OpNotEligible as exc:
        return ("DECLINED", f"OpNotEligible: {exc}", None, None, None, None)
    except Exception as exc:  # OutOfResources / compile failures: name them
        import traceback

        tb = traceback.format_exc()
        # the failing triton kernel: the last @persistent_autotune/JITFunction
        # frame in the stack names it
        frames = [ln for ln in tb.splitlines() if "vllm_kda" in ln or "kernel_name" in ln]
        return ("DECLINED", f"{type(exc).__name__}: {exc} | site: {frames[-2:]}", None, None, None, None)
    ref_out, ref_state = kda_chunk_reference(q, k, v, gg, beta, chunk_size=chunk,
                                             initial_state=None, output_final_state=True)
    out_diff = (fused_out - ref_out).abs().max().item()
    st_diff = (fused_state - ref_state).abs().max().item()
    out_scale = ref_out.abs().max().item()
    st_scale = ref_state.abs().max().item()
    out_rel = out_diff / max(out_scale, 1e-30)
    st_rel = st_diff / max(st_scale, 1e-30)

    def f():
        kda_chunk(q, k, v, gg, beta, chunk_size=chunk, initial_state=None,
                  output_final_state=True)

    def r():
        kda_chunk_reference(q, k, v, gg, beta, chunk_size=chunk, initial_state=None,
                            output_final_state=True)

    t_fused = bench(f) if S >= 4 else bench(f, iters=50)
    t_ref = bench(r) if S >= 4 else bench(r, iters=50)
    return ("OK", out_diff, st_diff, t_fused, t_ref, out_rel, st_rel)


def main():
    print(f"hip: {torch.version.hip is not None}, device: {torch.cuda.get_device_name(0)}")
    cases = [
        (1, 16, 4, 64),    # spec verify block (TP4 sharded heads, MTP block 4)
        (1, 16, 8, 64),    # spec verify block (DFlash2 block 8)
        (1, 16, 64, 64),   # mid chunk
        (1, 16, 2048, 64), # prefill chunk (TP4 sharded)
        (1, 64, 2048, 64), # prefill chunk (solo heads)
    ]
    all_ok = True
    for bsz, heads, S, chunk in cases:
        status, a, b, tf, tr, orel, srel = run_case(bsz, heads, S, chunk=chunk, seed=S)
        if status != "OK":
            all_ok = False
            print(f"B{bsz} H{heads} S{S}: DECLINED -- {a}")
            continue
        # fp32 accumulation contract: ABSOLUTE for the small spans, RELATIVE
        # for long recurrences (chunk boundaries double at HIP chunk 32)
        ok = (a < 5e-4 or orel < 1e-4) and (b < 5e-4 or srel < 1e-4)
        all_ok = all_ok and ok
        print(f"B{bsz} H{heads} S{S:>4}: out max|d|={a:.2e} (rel {orel:.1e}) "
              f"state max|d|={b:.2e} (rel {srel:.1e}) {'PASS' if ok else 'FAIL'} | "
              f"triton {tf:.3f} ms vs eager {tr:.3f} ms")
    print("KDA_TRITON_HIP_GATE", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
