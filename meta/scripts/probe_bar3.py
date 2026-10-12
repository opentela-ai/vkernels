"""Barrier variants with the spin INLINED in the kernel (no @triton.jit helper).

Same race-free per-round-slot judge as probe_bar2.  The only difference is
that the barrier body lives in the kernel, not in a helper call: probe_bar2
(helper) let blocks through early on MI300A, and an earlier probe showed the
inlined copy passes where the outlined one hangs.  If inlining is what makes
the difference, one of these variants will be exact on both vendors.

Variants:
  1  .cv poll + trailing acquire RMW          (the shipped CDNA body, inlined)
  2  acquire-RMW poll                          (the SM body, inlined)
  6  .cv poll + trailing acquire RMW + s_waitcnt only
  7  .cv poll + trailing acquire RMW + buffer_inv (gfx10+ L1 invalidate)
"""
import sys

import triton
import triton.language as tl
import torch

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 48
VLIST = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [1, 2, 6, 7])


@triton.jit
def _spin_cv(bar_ptr, target):
    c = tl.load(bar_ptr, cache_modifier=".cv")
    while c < target:
        c = tl.load(bar_ptr, cache_modifier=".cv")


@triton.jit
def judge(out_ptr, bar_ptr, P, ROUNDS, base: tl.int64, V: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    r = 0
    while r < ROUNDS:
        tl.store(out_ptr + r * P + pid, pid * 1000.0 + r)
        # ---- barrier, inlined: every variant is written out in full ----
        if V == 1:
            tl.debug_barrier()
            tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
            c = tl.load(bar_ptr, cache_modifier=".cv")
            while c < base + (r + 1) * P:
                c = tl.load(bar_ptr, cache_modifier=".cv")
            tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            tl.debug_barrier()
        elif V == 2:
            tl.debug_barrier()
            tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
            c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            while c < base + (r + 1) * P:
                c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            tl.debug_barrier()
        elif V == 6:
            tl.debug_barrier()
            tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
            c = tl.load(bar_ptr, cache_modifier=".cv")
            while c < base + (r + 1) * P:
                c = tl.load(bar_ptr, cache_modifier=".cv")
            tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            tl.inline_asm_elementwise(
                asm="s_waitcnt vmcnt(0) lgkmcnt(0)",
                constraints="=r,r", args=[tl.full((1,), 0, tl.int32)],
                dtype=tl.int32, is_pure=False, pack=1,
            )
            tl.debug_barrier()
        elif V == 7:
            tl.debug_barrier()
            tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
            c = tl.load(bar_ptr, cache_modifier=".cv")
            while c < base + (r + 1) * P:
                c = tl.load(bar_ptr, cache_modifier=".cv")
            tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            tl.inline_asm_elementwise(
                asm="buffer_inv; s_waitcnt vmcnt(0) lgkmcnt(0)",
                constraints="=r,r", args=[tl.full((1,), 0, tl.int32)],
                dtype=tl.int32, is_pure=False, pack=1,
            )
            tl.debug_barrier()
        nb = (pid + 1) % P
        acc += tl.load(out_ptr + r * P + nb)
        r += 1
    tl.store(out_ptr + ROUNDS * P + pid, acc)


def run(P, R, V):
    out = torch.full((R * P + P + 1,), -1.0, device="cuda")
    bar = torch.zeros(1, dtype=torch.int64, device="cuda")
    judge[(P,)](out, bar, P, R, 0, V=V, num_warps=8)
    torch.cuda.synchronize()
    got = out[R * P : R * P + P].cpu()
    bad = []
    for pid in range(P):
        nb = (pid + 1) % P
        expect = float(sum(nb * 1000 + r for r in range(R)))
        d = float(got[pid]) - expect
        if d != 0.0:
            bad.append((pid, d))
    return bar, bad


NAMES = {1: "inlined shipped CDNA", 2: "inlined SM RMW poll",
         6: "inlined CDNA + s_waitcnt", 7: "inlined CDNA + buffer_inv"}

for V in VLIST:
    for P, R in [(1, 32), (8, 32), (48, 32), (228, 32)]:
        if P > CAP:
            continue
        try:
            bar, bad = run(P, R, V)
        except Exception as exc:
            print(f"  V={V} P={P:3d}: {type(exc).__name__}: {str(exc)[:110]}", flush=True)
            continue
        tail = f" first={bad[:3]}" if bad else ""
        print(f"  V={V} [{NAMES[V]:26s}] P={P:3d} R={R} counter={int(bar[0])}/{P * R} "
              f"bad_pids={len(bad)}{tail}", flush=True)
    print("", flush=True)
