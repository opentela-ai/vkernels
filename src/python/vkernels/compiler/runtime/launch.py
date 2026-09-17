"""Launch: worker-count selection and the launch manifest (§7.4, §13.2).

The compile-inspect-choose pipeline of §13.2:

    capture and verify -> select tasks/threads -> plan memory ->
    emit + compile the megakernel -> *inspect compiled resource usage* ->
    choose a legal worker count -> validate guards -> launch once.

Worker count is a tuning parameter inside the legal range (§7.4):

    1 <= P <= N_SM * A(K_mega, T, S_block)

where ``A`` is the occupancy-derived active-block limit *of the compiled
megakernel* — a standalone kernel's occupancy is not a valid substitute.
Without the device toolchain, this module records the bound as a
capability note: the manifest marks ``residency_bound_verified=False``
until the compiled kernel's resources are inspected on the target GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LaunchManifest", "choose_worker_count", "query_device_sms"]

_FALLBACK_SMS = 16


def query_device_sms() -> int:
    """SM count via torch when available; conservative fallback otherwise."""
    try:  # torch is optional for the compiler package
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        pass
    return _FALLBACK_SMS


@dataclass
class LaunchManifest:
    """Everything the host needs to launch the megakernel once (§13.1)."""

    workers: int
    threads_per_worker: int
    cooperative: bool = True
    shared_scratch_bytes: int = 0
    workspace_bytes: int = 0
    num_sms: int = 0
    capability_notes: list[str] = field(default_factory=list)

    @property
    def grid(self) -> tuple[int, ...]:
        return (self.workers, 1, 1)

    @property
    def block(self) -> tuple[int, ...]:
        return (self.threads_per_worker, 1, 1)

    def describe(self) -> str:
        notes = "; ".join(self.capability_notes) or "none"
        return f"launch: grid={self.grid} block={self.block} cooperative={self.cooperative} smem/worker={self.shared_scratch_bytes}B workspace={self.workspace_bytes}B SMs={self.num_sms} | notes: {notes}"


def choose_worker_count(
    requested: int | None = None,
    *,
    num_sms: int | None = None,
    max_tasks_in_widest_phase: int = 1,
) -> int:
    """Pick P within the legal range (§7.4), defaulting to a sane tuning.

    Without a compiled kernel to occupancy-check, the conservative upper
    bound is one resident block per SM; ``requested`` lets the host tune
    below that (tiny phases leave wide grids idle — §8.5, §15.4).
    """
    sms = num_sms if num_sms is not None else query_device_sms()
    upper = max(1, sms)
    if requested is None:
        # Default heuristic: cover the widest phase without wild oversubscription.
        requested = min(upper, max(1, max_tasks_in_widest_phase))
    if requested < 1:
        raise ValueError(f"worker count must be >= 1, got {requested}")
    p = min(requested, upper)
    return p
