"""Parity + timing gate: vendored vLLM Triton sparse-MLA kernels vs floe's
eager reference (Glm53Attention._sparse_attention_torch math).

Reference (fp32 compute, floe-exact):
    safe   = indices.clamp(min=0)
    g      = latent[b, safe]                      # [B, S_q, W, K]
    scores = einsum(q, g) * scaling               # scaling = qk_head_dim**-0.5
    scores.masked_fill(indices < 0, -inf); softmax
    rows with no valid index -> all-zero weights -> zero output
    out    = einsum(weights, g).to(bf16)

The kernels receive sm_scale = scaling * log2(e) (the vkernels dsa_sparse_fwd
ABI, log2 units) — dsa_sparse_fwd converts to natural-exp units internally.
W = index_topk + index_kpool - 1 (kpool tail); H = 64 heads; K = 512; NoPE
absorbed form (tail_dim 0). Also covers a fully-masked row and fp64 truth
deltas of both sides.

Timing: CUDA events, 20 iters + 5 warmup (kernels JIT on first call).
"""
import math
import sys

import torch

sys.path.insert(0, "/iopsstor/scratch/cscs/xyao/floe-beverin/aiter-mig/vkernels-py")
from vkernels.torch_ops.vllm_sparse_mla import dsa_sparse_fwd  # noqa: E402

DEV = "cuda"
H, K = 64, 512
SCALING = 256.0 ** -0.5  # floe: qk_head_dim**-0.5 (absorbed form)
SM_SCALE = SCALING * math.log2(math.e)  # vkernels ABI log2 units


def ref_eager(q_abs, latent, indices, dtype=torch.float32):
    """floe's _sparse_attention_torch verbatim math at ``dtype`` compute."""
    b, s_q, h, k = q_abs.shape
    if indices.dim() == 4:  # dsa_sparse_fwd ABI [B, S_q, 1, W] -> [B, S_q, W]
        indices = indices.reshape(b, s_q, -1)
    if latent.dim() == 4:  # [B, S_kv, 1, K] -> [B, S_kv, K]
        latent = latent.reshape(latent.shape[0], latent.shape[1], latent.shape[-1])
    qf = q_abs.to(dtype)
    lf = latent.to(dtype)
    safe = indices.clamp(min=0)
    gathered = lf[torch.arange(b, device=lf.device)[:, None, None], safe]
    scores = torch.einsum("bshk,bswk->bshw", qf, gathered) * SCALING
    scores = scores.to(dtype).masked_fill(indices[:, :, None, :] < 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(
        (indices >= 0).any(dim=-1)[:, :, None, None], weights, torch.zeros((), dtype=dtype))
    return torch.einsum("bshw,bswk->bshk", weights, gathered).to(q_abs.dtype)


def make_case(b, s_q, s_kv, w, seed=0, fully_masked_row=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q_abs = (torch.randn(b, s_q, H, K, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    latent = (torch.randn(b, s_kv, 1, K, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    indices = torch.rand(b, s_q, 1, w, generator=g, device=DEV).argsort(dim=-1)[..., :w]
    indices = (indices % s_kv).to(torch.int32)
    if fully_masked_row is not None and b > 0 and s_q > 0:
        indices[:, fully_masked_row] = -1
    return q_abs, latent, indices


def bench(fn, iters=20, warmup=5):
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


def check(tag, b, s_q, s_kv, w):
    q_abs, latent, indices = make_case(b, s_q, s_kv, w, seed=hash(tag) % 2**31)
    ref = ref_eager(q_abs, latent, indices)                     # fp32, bf16 out
    ref64 = ref_eager(q_abs, latent, indices, dtype=torch.float64)
    out = dsa_sparse_fwd(q_abs, latent, indices, dim=K, tail_dim=0,
                         topk=w, sm_scale=SM_SCALE)
    d_k = (out.float() - ref.float()).abs().max().item()
    d_k64 = (out.float() - ref64.float()).abs().max().item()
    d_ref64 = (ref.float() - ref64.float()).abs().max().item()
    rel = d_k / ref.float().abs().max().item()
    # fully-masked row must come out zero
    masked_ok = ""
    if indices.dim() and (indices < 0).all(dim=-1).any():
        mrow = (indices < 0).all(dim=-1)[0].nonzero()[0, 0].item()
        masked_ok = f" | masked-row max {out[0, mrow].float().abs().max().item():.1e}"
    print(f"[{tag:16s}] d_vs_ref32 {d_k:.3e} (rel {rel:.2e})  "
          f"kern|fp64 {d_k64:.3e}  ref32|fp64 {d_ref64:.3e}{masked_ok}")
    return out, ref


print("=== parity (W=2051: index_topk 2048 + kpool tail 3) ===")
W = 2051
check("decode-b1", 1, 1, 4096, W)
check("decode-b4", 4, 1, 4096, W)
check("decode-b16", 16, 1, 4096, W)
check("decode-b64", 64, 1, 8192, W)
check("prefill-512", 1, 512, 512, W)
check("prefill-2048", 1, 2048, 2048, W)
check("prefill-8192", 1, 8192, 8192, W)
check("decode-masked-row", 1, 4, 2048, W, )  # + explicit fully-masked row below
q_abs, latent, indices = make_case(1, 4, 2048, W, seed=7)
indices[0, 2] = -1  # one fully-masked query row
out = dsa_sparse_fwd(q_abs, latent, indices, dim=K, tail_dim=0, topk=W, sm_scale=SM_SCALE)
print("fully-masked query row max:", out[0, 2].float().abs().max().item(),
      "(expect 0)")

print("\n=== timing (us, kernel incl. ragged build vs eager fp32) ===")
for tag, b, s_q, s_kv, w in [
    ("decode-b1", 1, 1, 4096, W),
    ("decode-b4", 4, 1, 4096, W),
    ("decode-b16", 16, 1, 4096, W),
    ("decode-b64", 64, 1, 8192, W),
    ("prefill-512", 1, 512, 512, W),
    ("prefill-2048", 1, 2048, 2048, W),
    ("prefill-8192", 1, 8192, 8192, W),
]:
    q_abs, latent, indices = make_case(b, s_q, s_kv, w, seed=1)
    t_k = bench(lambda: dsa_sparse_fwd(q_abs, latent, indices, dim=K,
                                       tail_dim=0, topk=w, sm_scale=SM_SCALE))
    t_e = bench(lambda: ref_eager(q_abs, latent, indices))
    print(f"[{tag:16s}] kernel {t_k:9.1f} | eager {t_e:9.1f} | x{t_e / t_k:6.2f}")
