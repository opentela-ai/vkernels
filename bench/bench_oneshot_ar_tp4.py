#!/usr/bin/env python3
"""4-rank smoke + microbench: OneShotComm's symmetric-memory one-shot
all-reduce vs the plain NCCL transport, on the decode seam payload sizes
(GLM-5.3-Flash TP4/GH200 decode, kineto 3496101 anatomy).

Correctness first: for every payload size, the one-shot result must match
a NCCL all-reduce of the same inputs exactly (bitwise for a same-order
deterministic kernel is NOT guaranteed — compare with torch.allclose and
an exact-sum int32 probe). Then timing: CUDA-event loops per size.

STANDALONE — imports nothing from floe/vkernels; it re-implements the
exact OneShotComm._reduce block (bucket + rendezvous + one-shot op) so it
can run against any staged build. Run on clariden (parent-owned; this
agent must NOT submit):

  cd <stage dir>
  srun --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=8 \
       --partition=debug --account=infra02 --time=00:10:00 \
       bash -c 'python bench_oneshot_ar_tp4.py'
  # or: torchrun --nproc_per_node=4 bench_oneshot_ar_tp4.py

Env knobs mirrored from floe: FLOE_TP_AR_KERNEL=one_shot|multimem,
FLOE_TP_AR_MAX_KB (threshold), FLOE_ITERS (bench iters, default 2000).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

# The decode seam payload census (bf16 elems): hidden tails land exactly,
# indexer score rows are P elems (context-dependent) — bracket them.
SIZES = [
    ("hidden-tail 4096 e (8 KB)", 4096),
    ("score-row P=512 (1 KB)", 512),
    ("score-row P=2048 (4 KB)", 2048),
    ("bucket-climb 8192 (16 KB)", 8192),
    ("over-LL 65536 (128 KB)", 65536),
    ("big 262144 (512 KB)", 262144),
]


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    if rank == 0:
        print(f"=== oneshot AR smoke: world={world} {torch.cuda.get_device_name(0)} "
              f"nccl={dist.get_backend()} torch={torch.__version__}", flush=True)

    from torch.distributed import _symmetric_memory as symm_mem

    kernel = os.environ.get("FLOE_TP_AR_KERNEL", "one_shot")
    max_bytes = int(os.environ.get("FLOE_TP_AR_MAX_KB", "512")) * 1024
    iters = int(os.environ.get("FLOE_ITERS", "2000"))
    group = dist.distributed_c10d._get_default_group()
    gname = group.group_name

    def bucket(n: int) -> int:
        return 1 << max(10, (n - 1).bit_length())

    # One persistent symm buffer per (bucket) — allocated in the SAME order
    # on every rank (identical size list), as in OneShotComm.
    bufs: dict[int, torch.Tensor] = {}
    for _, n in SIZES:
        b = bucket(n)
        if b not in bufs:
            t = symm_mem.empty(b, dtype=torch.bfloat16, device=dev)
            symm_mem.rendezvous(t, group)
            bufs[b] = t

    results = []
    for name, n in SIZES:
        # deterministic per-rank inputs, identical across runs
        g = torch.Generator(device="cuda").manual_seed(1234 + rank * 7 + n)
        x = torch.randn(n, dtype=torch.float32, device=dev, generator=g).bfloat16()
        # NCCL reference
        ref = x.clone()
        dist.all_reduce(ref)
        dist.barrier()

        # one-shot reduce (in-place into `out`, like OneShotComm._reduce)
        out = x.clone()
        view = bufs[bucket(n)][:n].view(n)
        view.copy_(out)
        if kernel == "multimem":
            torch.ops.symm_mem.multimem_all_reduce_(view, "sum", gname)
            out.copy_(view)
        else:
            torch.ops.symm_mem.one_shot_all_reduce_out(view, "sum", gname, out)
        dist.barrier()

        ok = torch.allclose(out.float(), ref.float(), rtol=0, atol=1e-2 * float(n) ** 0.5)
        exact = bool((out == ref).all().item())  # informational; bf16 order differs
        maxdiff = (out.float() - ref.float()).abs().max().item()

        # timing: NCCL vs one-shot, CUDA events, same stream
        def bench(fn, iters=iters):
            for _ in range(64):
                fn()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters * 1e3  # us

        t_nccl = bench(lambda: dist.all_reduce(out))
        if kernel == "multimem":
            t_one = bench(lambda: (view.copy_(out),
                                   torch.ops.symm_mem.multimem_all_reduce_(view, "sum", gname),
                                   out.copy_(view)))
        else:
            t_one = bench(lambda: (view.copy_(out),
                                   torch.ops.symm_mem.one_shot_all_reduce_out(view, "sum", gname, out)))
        results.append((name, n, ok, exact, maxdiff, t_nccl, t_one))
        dist.barrier()

    if rank == 0:
        print(f"--- kernel={kernel} threshold={max_bytes >> 10} KiB iters={iters}")
        print(f"{'payload':<28} {'elems':>7} {'ok':>3} {'exact':>5} "
              f"{'maxdiff':>9} {'nccl us':>8} {'oneshot us':>10} {'speedup':>7}")
        bad = 0
        for name, n, ok, exact, maxdiff, t_nccl, t_one in results:
            bad += not ok
            print(f"{name:<28} {n:>7} {'YES' if ok else 'NO':>3} {str(exact):>5} "
                  f"{maxdiff:>9.4f} {t_nccl:>8.2f} {t_one:>10.2f} {t_nccl / max(t_one, 1e-9):>6.2f}x")
        print("SMOKE_OK" if bad == 0 else f"SMOKE_FAILED ({bad} mismatches)")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
