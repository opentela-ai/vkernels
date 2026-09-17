"""Runtime subpackage for the megakernel compiler.

* :mod:`synchronization` — the grid-barrier protocol obligations (§8.4)
  plus the CPU simulation used by the reference executor.
* :mod:`launch` — worker-count selection and the launch manifest (§7.4, §13.2).
* :mod:`task_queue` — the later ready-task runtime, specified and
  CPU-modeled (§11).
"""

from .launch import LaunchManifest, choose_worker_count, query_device_sms
from .synchronization import (
    CapabilityError,
    GRID_SYNC_BACKEND_VERIFIED,
    BarrierError,
    GridBarrierSpec,
    SimulatedGridBarrier,
    mark_grid_sync_verified,
    require_grid_sync_backend,
    verify_milestone0,
)
from .task_queue import ProgressError, ReadyTaskState, TaskDeps, simulate_fifo_execution

__all__ = [
    "GRID_SYNC_BACKEND_VERIFIED",
    "BarrierError",
    "CapabilityError",
    "GridBarrierSpec",
    "LaunchManifest",
    "ProgressError",
    "ReadyTaskState",
    "SimulatedGridBarrier",
    "TaskDeps",
    "choose_worker_count",
    "mark_grid_sync_verified",
    "query_device_sms",
    "require_grid_sync_backend",
    "simulate_fifo_execution",
    "verify_milestone0",
]
