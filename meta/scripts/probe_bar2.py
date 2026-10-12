"""Barrier variants judged by the race-free per-round-slot microprogram.

Judge (V-independent): every round r, block pid stores `pid*1000 + r` to
slot r*P+pid, barriers, then reads slot r*P + (pid+1)%P.  Slots are per-round
and per-block, so no block can clobber a value another block is about to
read: any deviation from the exact sum is a barrier failure.

Variants (V):
  0  no barrier (control -- shows what free running looks like)
  1  shipped CDNA: .cv poll + trailing acquire RMW
  2  SM form: acquire RMW poll
  3  .cv poll + trailing acq_rel RMW
  4  shipped CDNA + an L1-invalidate / fence inline asm after release
"""
import sys

import triton
import triton.language as tl
import torch

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 48
VLIST = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [0, 1, 2, 3, 4])


@triton.jit
def _bar(bar_ptr, target, V: tl.constexpr):
    if V == 0:
        tl.debug_barrier()
    elif V == 1:
        tl.debug_barrier()
        tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
        c = tl.load(bar_ptr, cache_modifier=".cv")
        while c < target:
            c = tl.load(bar_ptr, cache_modifier=".cv")
        tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        tl.debug_barrier()
    elif V == 2:
        tl.debug_barrier()
        tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        while c < target:
            c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        tl.debug_barrier()
    elif V == 3:
        tl.debug_barrier()
        tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
        c = tl.load(bar_ptr, cache_modifier=".cv")
        while c < target:
            c = tl.load(bar_ptr, cache_modifier=".cv")
        tl.atomic_add(bar_ptr, 0, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
    elif V == 4:
        tl.debug_barrier()
        tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
        c = tl.load(bar_ptr, cache_modifier=".cv")
        while c < target:
            c = tl.load(bar_ptr, cache_modifier=".cv")
        tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        # invalidate L1 so post-barrier plain loads cannot hit a stale line
        tl.inline_asm_elementwise(
            asm="buffer_wbinvl1_vol; s_waitcnt vmcnt(0) lgkmcnt(0)",
            constraints="=r,r", args=[tl.full((1,), 0, tl.int32)],
            dtype=tl.int32, is_pure=False, pack=1,
        )
        tl.debug_barrier()


@triton.jit
def judge(out_ptr, bar_ptr, P, ROUNDS, base: tl.int64, V: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    r = 0
    while r < ROUNDS:
        tl.store(out_ptr + r * P + pid, pid * 1000.0 + r)
        _bar(bar_ptr, base + (r + 1) * P, V)
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


NAMES = {0: "no barrier (control)", 1: "shipped CDNA (.cv + acq RMW)",
         2: "SM (RMW poll)", 3: ".cv + acq_rel RMW",
         4: "CDNA + L1 invalidate"}

for V in VLIST:
    for P, R in [(1, 32), (8, 32), (48, 32), (228, 32)]:
        if P > CAP:
            continue
        try:
            bar, bad = run(P, R, V)
        except Exception as exc:
            print(f"  V={V} P={P:3d}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        ctr = int(bar[0])
        tail = f" first={bad[:3]}" if bad else ""
        print(f"  V={V} [{NAMES[V]:26s}] P={P:3d} R={R} counter={ctr}/{P * R} "
              f"bad_pids={len(bad)}{tail}", flush=True)
    print("", flush=True)
