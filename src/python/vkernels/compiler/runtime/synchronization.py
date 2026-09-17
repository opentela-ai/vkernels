"""Synchronization: the grid-barrier contract (§8.4) and a CPU model.

The design is explicit that current CuTe launch documentation exposes
``cooperative=True`` as a *launch* capability, not as a grid barrier inside
the generated device body, and that ``cute.arch.grid_sync()`` is not an
established API. Before the multi-block backend is complete, Milestone 0
must demonstrate a two-phase multi-block program on the pinned CuTe version
using a supported grid-synchronization path (cooperative launch + a
verified software barrier, or a backend extension).

This module therefore separates:

* :class:`GridBarrierSpec` — the *protocol obligations* any real barrier
  must satisfy (arrival/departure phases, release/acquire visibility,
  reuse protocol). A plain counter plus polling loop is explicitly *not* a
  sufficient specification (§8.4).
* :class:`SimulatedGridBarrier` — a CPU model used by the reference
  executor to validate *schedule logic* (uniform participation, phase
  progress). Per §15.1, randomized CPU execution checks graph logic; it
  cannot validate compiled GPU synchronization or memory ordering.

The capability flag :data:`GRID_SYNC_BACKEND_VERIFIED` records whether the
device primitive has been demonstrated on the target stack. It starts
``False`` and is flipped only by an explicit Milestone-0 validation call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "GridBarrierSpec",
    "SimulatedGridBarrier",
    "BarrierError",
    "GRID_SYNC_BACKEND_VERIFIED",
    "mark_grid_sync_verified",
    "require_grid_sync_backend",
]

# Milestone 0 status flag (§8.4). Do not flip this without the
# producer-barrier-consumer capability test on the pinned stack.
GRID_SYNC_BACKEND_VERIFIED = False


def mark_grid_sync_verified() -> None:
    global GRID_SYNC_BACKEND_VERIFIED
    GRID_SYNC_BACKEND_VERIFIED = True


def verify_milestone0(rounds: int = 200, worker_counts=(1, 8, 48), invocations: int = 3) -> dict:
    """Run the §8.4 producer-barrier-consumer microprogram and verify it.

    Exercises, on the actual Triton/GB10 stack:

    * repeated barriers (``rounds`` per invocation) across several legal
      worker counts, including P=1;
    * cross-block memory visibility: each block reads a neighbour's
      round-tagged write that only the barrier can have published;
    * idle-worker participation (odd blocks skip the work but still arrive
      at every barrier, §8.3);
    * repeated invocations against the monotonic counter (no reset kernel,
      base handed in deterministically by the host).

    On success this flips :data:`GRID_SYNC_BACKEND_VERIFIED` (strict mode's
    device gate) and returns the observed counter bases. Raises on any
    checksum mismatch.
    """
    import torch

    from ..device_triton import milestone0_barrier_test

    if not torch.cuda.is_available():
        raise RuntimeError("Milestone-0 verification requires the GPU")
    report = {}
    for P in worker_counts:
        for idle in (False, True):
            out = torch.full((2 * P,), -1.0, device="cuda")
            bar = torch.zeros(1, dtype=torch.int64, device="cuda")
            base = 0
            for _ in range(invocations):
                milestone0_barrier_test[(P,)](out, bar, P, rounds, base, IDLE=idle, num_warps=8)
                base += rounds * P
                torch.cuda.synchronize()
            # checksum: acc[pid] = sum over rounds of v(nb), nb = (pid+1) % P,
            # v = nb*1000 + r when the neighbour produced that round.
            got = out[P:].cpu()
            for pid in range(P):
                nb = (pid + 1) % P
                expect = 0.0 if (idle and nb % 2 == 1) else float(sum(nb * 1000 + r for r in range(rounds)))
                if abs(float(got[pid]) - expect) > 1e-2 * max(1.0, abs(expect)):
                    raise RuntimeError(f"Milestone-0 FAILED: P={P} idle={idle} pid={pid}: checksum {float(got[pid])} != {expect} (visibility/ordering violation)")
            counter = int(bar[0])
            if counter != base:
                raise RuntimeError(f"Milestone-0 FAILED: P={P} idle={idle}: counter {counter} != {base}")
            report[f"P={P},idle={idle}"] = counter
    mark_grid_sync_verified()
    return report


def require_grid_sync_backend() -> None:
    if not GRID_SYNC_BACKEND_VERIFIED:
        raise CapabilityError("grid synchronization backend not verified: Milestone 0 (producer-barrier-consumer microprogram on the pinned CuTe version) has not been executed on this stack. The emitted megakernel source is illustrative until then (§8.4).")


class CapabilityError(Exception):
    """A required backend capability is not established (§8.4)."""


@dataclass(frozen=True)
class GridBarrierSpec:
    """Protocol obligations for the device grid barrier (§8.4, §11.3).

    A legal implementation must provide, at minimum:

    * **Two-phase arrival/departure** semantics with reuse safety: the
      counter used for barrier ``k`` must not be reset while any block can
      still arrive at ``k`` (the classic sense-reversing barrier or an
      arrival-count + generation-tag scheme).
    * **Release/acquire visibility**: writes before the barrier are visible
      to every block after it, at device scope. On CUDA this means the
      barrier's release store and subsequent acquire load must be ordered
      with the memory fence the platform documents for cooperative groups;
      a relaxed counter increment alone does not establish it.
    * **Uniform participation**: every resident block arrives exactly once
      per barrier instance; idle workers arrive with zero tasks (§8.3).
    * **Resident-grid progress**: correctness must not depend on blocks
      becoming resident later — the launch must guarantee co-residency
      (cooperative launch bound per §7.4).
    """

    workers: int
    name: str = "grid_barrier"

    def describe(self) -> str:
        return f"{self.name}: two-phase arrival/departure, release/acquire at device scope, uniform participation by {self.workers} resident blocks, reuse-safe generations"


class BarrierError(Exception):
    """The simulated barrier observed a protocol violation."""


@dataclass
class SimulatedGridBarrier:
    """CPU model of one grid barrier instance (schedule-logic checks).

    Tracks arrivals per worker and enforces the participation contract of
    §8.3: every worker arrives exactly once before the barrier releases.
    This validates *schedules*; it proves nothing about GPU memory order.
    """

    workers: int
    generation: int = 0
    arrivals: set = field(default_factory=set)

    def arrive(self, worker: int) -> None:
        if worker in self.arrivals:
            raise BarrierError(f"worker {worker} arrived twice at barrier generation {self.generation}")
        if not (0 <= worker < self.workers):
            raise BarrierError(f"unknown worker {worker} (grid is {self.workers} blocks)")
        self.arrivals.add(worker)

    def release(self) -> int:
        """All workers arrived -> barrier passes; returns the generation."""
        missing = set(range(self.workers)) - self.arrivals
        if missing:
            raise BarrierError(f"barrier generation {self.generation} released with missing workers {sorted(missing)} (a worker exited or skipped a barrier — §8.3 violation)")
        gen = self.generation
        self.generation += 1
        self.arrivals = set()
        return gen

    def sync(self) -> int:
        """Convenience: all workers already arrived -> release."""
        return self.release()
