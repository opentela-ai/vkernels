"""Config-bisect repro for the kkt intra_sub_inter memory fault (job 644327).

GPU faults are fatal, so the sweep driver runs one process per autotune
config: monkeypatch both kkt kernels' configs to a single config, then run
kda_chunk_floe on (a) a tiny case and (b) the faulting serving case.
"""
import sys

import torch

sys.path.insert(0, "/iopsstor/scratch/cscs/xyao/floe-beverin/aiter-mig/vkernels-py")
import triton  # noqa: E402
import vkernels.torch_ops.vllm_kda as K  # noqa: E402

BK = int(sys.argv[1])
NW = int(sys.argv[2])
NS = int(sys.argv[3])
TINY = len(sys.argv) > 4 and sys.argv[4] == "tiny"

cfg = [triton.Config({"BK": BK}, num_warps=NW, num_stages=NS)]


def _pin(jit_fn):
    """Rebuild heuristics(autotune(jit)) with a single pinned config —
    patching .configs on the live Autotuner corrupts Triton 3.4's cached
    launch tuples. jit_fn is the outer Heuristics wrapper; .fn is the
    Autotuner; .fn.fn is the JITFunction."""
    core = jit_fn.fn.fn if hasattr(jit_fn.fn, "fn") else jit_fn.fn
    autotuned = triton.autotune(configs=list(cfg), key=["BC"])(core)
    return triton.heuristics(
        {"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})(autotuned)


K.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter = _pin(
    K.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter)
K.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra = _pin(
    K.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra)

torch.manual_seed(0)
DEV = "cuda"
B, H, D = (1, 4, 32) if TINY else (1, 64, 128)
S = 64 if TINY else 512

q = torch.randn(B, H, S, D, device=DEV, dtype=torch.float32)
k = torch.randn(B, H, S, D, device=DEV, dtype=torch.float32)
q = q / q.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
k = k / k.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
v = torch.randn(B, H, S, D, device=DEV, dtype=torch.float32)
g = torch.rand(B, H, S, D, device=DEV, dtype=torch.float32) * -5.0
beta = torch.rand(B, H, S, device=DEV, dtype=torch.float32)

out, st = K.kda_chunk_floe(q, k, v, g, beta, chunk_size=64, output_final_state=True)
torch.cuda.synchronize()
print(f"OK BK={BK} NW={NW} NS={NS} tiny={TINY} "
      f"out_abs={out.abs().max().item():.3e} state_abs={st.abs().max().item():.3e}")
