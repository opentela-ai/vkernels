"""Decode A/B: vendored Triton sparse-MLA vs floe's HIP split-key decode
(vkernels.hip_dsa_mhc, per-request batch-1 ABI, exactly floe's dispatch) and
the eager fp32 reference. W=2051, H=64, K=512, NoPE absorbed form.
"""
import math
import sys

import torch

from vkernels.torch_ops.vllm_sparse_mla import dsa_sparse_fwd  # noqa: E402
from vkernels import hip_dsa_mhc as hip_mod  # noqa: E402

DEV = "cuda"
H, K = 64, 512
SCALING = 256.0 ** -0.5
SM_SCALE = SCALING * math.log2(math.e)
W = 2051
S_KV = 4096

print("hip available:", hip_mod.available() if hasattr(hip_mod, "available") else "n/a")


def ref_eager(q_abs, latent, indices):
    b, s_q, h, k = q_abs.shape
    idx = indices.reshape(b, s_q, -1)
    lf = latent.reshape(latent.shape[0], latent.shape[1], -1).float()
    safe = idx.clamp(min=0)
    gathered = lf[torch.arange(b, device=lf.device)[:, None, None], safe]
    scores = torch.einsum("bshk,bswk->bshw", q_abs.float(), gathered) * SCALING
    scores = scores.masked_fill(idx[:, :, None, :] < 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where((idx >= 0).any(dim=-1)[:, :, None, None], weights, 0.0)
    return torch.einsum("bshw,bswk->bshk", weights, gathered).to(q_abs.dtype)


def hip_decode(q_abs, latent, indices):
    split_kv = hip_mod.dsa_split_for(q_abs.shape[0] * q_abs.shape[1], H, W)
    outs = []
    for i in range(q_abs.shape[0]):
        outs.append(hip_mod.dsa_sparse_fwd_split(
            q_abs[i:i + 1].contiguous(),
            latent[i:i + 1].contiguous(),
            indices[i:i + 1, :, None, :].contiguous(),
            dim=K, tail_dim=0, topk=W, sm_scale=SM_SCALE, split_kv=split_kv))
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def make_case(b, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q_abs = (torch.randn(b, 1, H, K, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    latent = (torch.randn(b, S_KV, 1, K, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    indices = torch.rand(b, 1, 1, W, generator=g, device=DEV).argsort(dim=-1)[..., :W]
    indices = (indices % S_KV).to(torch.int32)
    return q_abs, latent, indices


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


print(f"{'batch':>6} {'triton':>9} {'hip-loop':>9} {'eager':>10}   parity(k|h|e)")
for b in (1, 4, 16, 64):
    q_abs, latent, indices = make_case(b, seed=b)
    ref = ref_eager(q_abs, latent, indices) if b <= 4 else None
    out_t = dsa_sparse_fwd(q_abs, latent, indices, dim=K, tail_dim=0, topk=W, sm_scale=SM_SCALE)
    try:
        out_h = hip_decode(q_abs, latent, indices)
        t_h = bench(lambda: hip_decode(q_abs, latent, indices))
        d_h = (out_h.float() - ref.float()).abs().max().item() if ref is not None else float("nan")
    except Exception as ex:
        out_h, t_h, d_h = None, float("nan"), float("nan")
        import traceback
        traceback.print_exc()
        print(f"  hip[{b}] FAILED: {type(ex).__name__}: {ex}")
    t_t = bench(lambda: dsa_sparse_fwd(q_abs, latent, indices, dim=K, tail_dim=0,
                                       topk=W, sm_scale=SM_SCALE))
    d_t = (out_t.float() - ref.float()).abs().max().item() if ref is not None else float("nan")
    t_e = bench(lambda: ref_eager(q_abs, latent, indices)) if b <= 4 else float("nan")
    d_e = (ref.float() - ref_eager(q_abs, latent, indices, )).abs().max().item() if ref is not None else float("nan")
    print(f"{b:>6} {t_t:9.1f} {t_h:9.1f} {t_e:10.1f}   d_t {d_t:.2e} d_h {d_h:.2e}")
