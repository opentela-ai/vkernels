#!/usr/bin/env python3
"""Compare symmetric-memory reductions with NCCL on four colocated GPUs.

Run: torchrun --standalone --nproc_per_node=4 bench/bench_oneshot_ar_tp4.py
For Slurm, run torchrun once inside a four-GPU allocation. PyTorch's private
symmetric-memory API and kernel support must be available on the target build.
The measured path includes staging and output copies, as the serving seam does.
"""
from __future__ import annotations

import argparse
import os

SIZES = [("score P=512", 512), ("score P=2048", 2048), ("hidden 4096", 4096),
         ("bucket 8192", 8192), ("large 65536", 65536), ("large 262144", 262144)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("one_shot", "multimem"),
                        default=os.environ.get("FLOE_TP_AR_KERNEL", "one_shot"))
    parser.add_argument("--max-kb", type=int, default=int(os.environ.get("FLOE_TP_AR_MAX_KB", "512")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("FLOE_ITERS", "2000")))
    args = parser.parse_args()
    if args.iters <= 0 or args.max_kb <= 0:
        parser.error("--iters and --max-kb must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4 or "LOCAL_RANK" not in os.environ:
        parser.error("launch with torchrun --nproc_per_node=4 on one node")
    sizes = [(name, n) for name, n in SIZES if n * 2 <= args.max_kb * 1024]
    if not sizes:
        parser.error("threshold excludes every payload")

    import torch
    import torch.distributed as dist
    from torch.distributed import _symmetric_memory as symm_mem

    # Select the rank's device before NCCL initializes its communicator.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.distributed_c10d._get_default_group()
        gname = group.group_name
        if rank == 0:
            print(f"world={world} gpu={torch.cuda.get_device_name(device)} torch={torch.__version__}", flush=True)
            print(f"kernel={args.kernel} max_kb={args.max_kb} iters={args.iters}", flush=True)

        def bucket(n):
            return 1 << max(10, (n - 1).bit_length())

        buffers = {}
        for _, n in sizes:
            b = bucket(n)
            if b not in buffers:
                t = symm_mem.empty(b, dtype=torch.bfloat16, device=device)
                symm_mem.rendezvous(t, group)
                buffers[b] = t

        failed = torch.zeros((), dtype=torch.int32, device=device)
        for name, n in sizes:
            view = buffers[bucket(n)][:n]
            gen = torch.Generator(device=device).manual_seed(1234 + rank * 7 + n)
            x = torch.randn(n, device=device, generator=gen).bfloat16()
            out = torch.empty_like(x)
            ref = torch.empty_like(x)

            def symmetric(source):
                view.copy_(source)
                if args.kernel == "multimem":
                    torch.ops.symm_mem.multimem_all_reduce_(view, "sum", gname)
                    out.copy_(view)
                else:
                    torch.ops.symm_mem.one_shot_all_reduce_out(view, "sum", gname, out)

            def nccl():
                ref.copy_(x)
                dist.all_reduce(ref)

            # Small integers sum exactly in BF16; compare against an int32
            # NCCL oracle before allowing floating-point reassociation error.
            integer = torch.full((n,), rank + 1, device=device, dtype=torch.int32)
            dist.all_reduce(integer)
            symmetric(torch.full_like(x, rank + 1))
            exact_ok = torch.equal(out.float(), integer.float())
            nccl()
            symmetric(x)
            finite = bool(torch.isfinite(out).all() & torch.isfinite(ref).all())
            ok = exact_ok and finite and torch.allclose(out.float(), ref.float(), rtol=.03, atol=.03125)
            maxdiff = (out.float() - ref.float()).abs().max().item()
            if not ok:
                failed.fill_(1)

            def bench(fn):
                for _ in range(64):
                    fn()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    fn()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) * 1000 / args.iters

            # Every iteration restores the immutable source, avoiding overflow
            # from repeatedly summing the previous all-reduce's output.
            t_nccl, t_symmetric = bench(nccl), bench(lambda: symmetric(x))
            if rank == 0:
                print(f"{name:16} n={n:7} ok={ok} maxdiff={maxdiff:.5f} "
                      f"nccl_us={t_nccl:.3f} symmetric_us={t_symmetric:.3f} "
                      f"speedup={t_nccl / max(t_symmetric, 1e-9):.2f}", flush=True)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        bad = bool(failed.item())
        if rank == 0:
            print("SMOKE_FAILED" if bad else "SMOKE_OK", flush=True)
        return int(bad)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
