"""Compilation orchestration: the §13.1 API and the capability report.

    executable = compile_model(
        model_body=build_forward,
        model_config=config,
        input_spec=decode_spec,
        target="cute_persistent",
        schedule="phase",
        strict_single_kernel=True,
    )
    workspace = executable.allocate_workspace()
    logits = executable(ids, weights, cache, position, workspace=workspace)

The pipeline is the §13.2 host sequence:

    capture and verify operator IR -> select task implementations and
    thread count -> plan scratch and global workspace -> emit the
    megakernel source -> inspect resource usage -> choose a legal worker
    count -> validate guards -> launch once.

Without the CuTe DSL toolchain, the compilation still completes (emission
is host-side) but the *device* path refuses to run: strict mode turns an
unverified grid-sync backend into a capability error rather than silently
falling back to multi-kernel execution (§3.3). The reference path executes
the identical schedule on CPU for validation (§15.1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .capture import RecordingBackend, capture_model
from .codegen_cute import emit_megakernel_source
from .legality import Diagnostic, LegalityError, check_graph
from .lowerings import (
    ELEM_TILE,
    GEMM_TILE_M,
    GEMM_TILE_N,
    LOWERINGS_VERSION,
    THREADS_PER_WORKER,
    check_thread_contract,
    lower_graph,
)
from .memory import kv_cache_bytes, plan_memory
from .model_gpt2 import GPT2Config, GPT2Weights, KVCache, SymbolicModelArgs, build_forward, random_weights
from .model_qwen3 import (
    Qwen3Config,
    Qwen3ModelArgs,
    build_qwen3_forward,
    random_qwen3_weights,
)
from .model_qwen35 import Qwen35ModelArgs, build_qwen35_forward
from .qwen35_arch import Qwen35Config, random_qwen35_weights
from .operator_ir import I32, OperatorGraph
from .reference_exec import ExecutionTrace, ReferenceExecutor
from .runtime.launch import LaunchManifest, choose_worker_count, query_device_sms
from .runtime.synchronization import CapabilityError, require_grid_sync_backend
from .schedule_phase import PhaseSchedule

__all__ = [
    "CompilationReport",
    "CompiledExecutable",
    "compile_model",
    "DeviceUnavailable",
]


class DeviceUnavailable(CapabilityError):
    """The strict device path cannot run on this stack (§3.3, §8.4)."""


@dataclass
class CompilationReport:
    """§13.1: what compilation produced, assumed, and transformed."""

    target: str
    schedule_kind: str
    num_phases: int = 0
    total_tasks: int = 0
    workspace_bytes: int = 0
    workspace_naive_bytes: int = 0
    scratch_bytes_per_worker: int = 0
    threads_per_worker: int = THREADS_PER_WORKER
    kv_cache_bytes: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)
    capability_notes: list[str] = field(default_factory=list)
    specialization_key: str = ""
    generated_source: str = ""

    def describe(self) -> str:
        lines = [
            f"compilation report: target={self.target} schedule={self.schedule_kind}",
            f"  phases={self.num_phases} tasks={self.total_tasks}",
            f"  workspace={self.workspace_bytes} B (naive {self.workspace_naive_bytes} B, reuse saves {self.workspace_naive_bytes - self.workspace_bytes} B)",
            f"  scratch/worker={self.scratch_bytes_per_worker} B  threads/worker={self.threads_per_worker}",
            f"  kv cache={self.kv_cache_bytes} B (persistent, caller-owned)",
            f"  specialization key={self.specialization_key[:16]}...",
        ]
        for d in self.diagnostics:
            lines.append(f"  [{d.severity}] {d.code}: {d.message} ({d.location})")
        for n in self.capability_notes:
            lines.append(f"  capability: {n}")
        return "\n".join(lines)


class CompiledExecutable:
    """The §13.1 executable: launch manifest, workspace, and run paths."""

    def __init__(
        self,
        graph: OperatorGraph,
        config: GPT2Config,
        weights: GPT2Weights,
        schedule: PhaseSchedule,
        families,
        workspace_plan,
        scratch_plan,
        source: str,
        report: CompilationReport,
        workers: int,
        manifest: LaunchManifest | None = None,
    ):
        self.graph = graph
        self.config = config
        self.weights = weights
        self.schedule = schedule
        self.families = families
        self.workspace_plan = workspace_plan
        self.scratch_plan = scratch_plan
        self.source = source
        self.report = report
        self.workers = workers
        self.manifest = manifest or LaunchManifest(
            workers=workers,
            threads_per_worker=THREADS_PER_WORKER,
            shared_scratch_bytes=scratch_plan.per_worker_bytes,
            workspace_bytes=workspace_plan.total_bytes,
            num_sms=query_device_sms(),
            capability_notes=list(report.capability_notes),
        )

    # -- workspace -----------------------------------------------------------

    def allocate_workspace(self, dtype=np.float64) -> np.ndarray:
        """Preallocate the intermediate workspace, NaN-initialized (§9.1).

        NaN initialization is deliberate: it makes any read of unwritten
        storage observable, which the canary protocol and the §5.3 tests
        rely on. Reference mode runs in float64 for oracle stability.
        """
        return np.full(self.workspace_plan.total_elements, np.nan, dtype=dtype)

    # -- execution -----------------------------------------------------------

    def __call__(
        self,
        ids: np.ndarray,
        cache: KVCache,
        position: int,
        *,
        workspace: np.ndarray | None = None,
        mode: str = "reference",
        workers: int | None = None,
        canary: bool = True,
    ) -> tuple[np.ndarray, ExecutionTrace]:
        return self.run(ids, cache, position, workspace=workspace, mode=mode, workers=workers, canary=canary)

    def run(
        self,
        ids: np.ndarray,
        cache: KVCache,
        position: int,
        *,
        workspace: np.ndarray | None = None,
        mode: str = "reference",
        workers: int | None = None,
        canary: bool = True,
    ) -> tuple[np.ndarray, ExecutionTrace]:
        if mode == "device":
            require_grid_sync_backend()  # raises until Milestone 0 passes (§8.4)
            raise DeviceUnavailable("CuTe DSL device backend is not available on this stack; run with mode='reference' for schedule validation (§15.1)")
        if mode != "reference":
            raise ValueError(f"unknown execution mode {mode!r}")

        # Host-checked preconditions (§13.4): position guard, shape guards.
        position = int(position)
        self.graph.scalars["p"].guard(position)
        ids = np.asarray(ids)
        if ids.shape != (self.config.batch,):
            raise ValueError(f"ids shape {ids.shape} != ({self.config.batch},)")

        workers = workers if workers is not None else self.workers
        # Re-specialize the schedule for a different worker count: legal for
        # any P within the residency bound (§13.2).
        schedule = self.schedule if workers == self.schedule.workers else PhaseSchedule.from_families([f for p in self.schedule.phases for f in p.families], workers=workers)

        if workspace is None:
            workspace = self.allocate_workspace()
        elif workspace.shape != (self.workspace_plan.total_elements,):
            raise ValueError(f"workspace has {workspace.shape[0]} elements; plan requires {self.workspace_plan.total_elements}")

        # Storage arrays: caller-owned externals + workspace slices per
        # fresh buffer (placement follows the plan exactly).
        storage_arrays = self._external_storage_arrays(ids, cache)
        for buf in self.workspace_plan.buffers:
            storage_arrays[buf.storage_id] = workspace[buf.offset : buf.offset + buf.numel]

        executor = ReferenceExecutor(
            schedule,
            workers=workers,
            storage_arrays=storage_arrays,
            graph=self.graph,
            workspace_plan=self.workspace_plan,
            canary=canary,
        )
        trace = executor.run({"p": position})

        if trace.kernel_launches != 1:
            raise DeviceUnavailable(f"reference execution used {trace.kernel_launches} launches; the single-launch contract was violated (§15.2)")

        logits = self._final_output(executor)
        cache.valid_len = position + 1  # host bookkeeping AFTER success (§13.4)
        return logits, trace

    def _external_storage_arrays(self, ids: np.ndarray, cache: KVCache) -> dict[int, np.ndarray]:
        """Map external storages to flat numpy arrays by registered name."""
        arrays = self.weights.arrays()
        arrays["ids"] = np.ascontiguousarray(ids).reshape(-1)
        arrays["k_cache"] = cache.k.reshape(-1)
        arrays["v_cache"] = cache.v.reshape(-1)
        storage_arrays: dict[int, np.ndarray] = {}
        for sid, name in self.graph.storage_names.items():
            if sid in self.graph.fresh_storages:
                continue
            if name not in arrays:
                raise DeviceUnavailable(f"external storage {name!r} has no runtime array")
            arr = np.ascontiguousarray(arrays[name]).reshape(-1)
            storage_arrays[sid] = arr
        return storage_arrays

    def _final_output(self, executor: ReferenceExecutor) -> np.ndarray:
        # The final phase's output tensor is the logits buffer.
        out_name = self.graph.ops[-1].outputs[0]
        out = executor.tensor(out_name)
        return np.array(out, copy=True)


def _specialization_key(config, target: str, schedule: str) -> str:
    """§13.3: dimensions, dtypes, layouts, tasks, schedule, numerical mode."""
    import dataclasses

    payload = (
        target,
        schedule,
        type(config).__name__,
        tuple(sorted((k, repr(v)) for k, v in dataclasses.asdict(config).items())),
        LOWERINGS_VERSION,
        GEMM_TILE_M,
        GEMM_TILE_N,
        ELEM_TILE,
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def _kv_width(config) -> int:
    """Per-side cache element width per (layer, batch, kv-head, position)."""
    if isinstance(config, Qwen3Config):
        return config.kv_heads * config.head_dim
    return config.hidden  # GPT-2: KVH*D == C


def compile_model(
    model_body=None,
    model_config=None,
    weights=None,
    *,
    target: str = "cute_persistent",
    schedule: str = "phase",
    strict_single_kernel: bool = True,
    workers: int | None = None,
    seed: int = 0,
) -> CompiledExecutable:
    """Compile one single-token decode step into a persistent schedule.

    The frontend is selected by config type: :class:`GPT2Config` captures
    the GPT-2 body, :class:`Qwen3Config` the dense-Qwen3 body (RMSNorm,
    RoPE, QK-norm, GQA, SwiGLU, tied head).
    """
    if schedule != "phase":
        raise ValueError(f"schedule={schedule!r}: only the phase-synchronous target is implemented; static/dynamic fine-grained schedules are Milestone 5 (§10, §11)")
    if target not in ("cute_persistent", "reference"):
        raise ValueError(f"unknown target {target!r}")

    if model_config is None:
        model_config = GPT2Config()
    config = model_config
    is_qwen3 = isinstance(config, Qwen3Config)
    is_qwen35 = isinstance(config, Qwen35Config)
    config.validate()
    if model_body is None:
        if is_qwen35:
            model_body = build_qwen35_forward
        else:
            model_body = build_qwen3_forward if is_qwen3 else build_forward
    if weights is None:
        if is_qwen35:
            weights = random_qwen35_weights(config, seed=seed)
        elif is_qwen3:
            weights = random_qwen3_weights(config, seed=seed)
        else:
            weights = random_weights(config, seed=seed)

    # ---- capture (§4, M1) ----------------------------------------------
    recorder = RecordingBackend()
    if is_qwen35:
        # #93: the hybrid step is ragged-row native — per-row positions.
        position = recorder.define_row_positions(
            "row_positions", config.batch, config.cache_capacity, storage_id=10**6 + 5
        )
    else:
        position = recorder.define_position(config.cache_capacity)
    ids = recorder.external_tensor("ids", (config.batch,), I32, storage_id=10**6)
    if is_qwen35:
        args = Qwen35ModelArgs(recorder, config)
    elif is_qwen3:
        args = Qwen3ModelArgs(recorder, config)
    else:
        args = SymbolicModelArgs(recorder, config, weights)
    graph, recorder = capture_model(model_body, args, ids, position, config, backend=recorder)

    # ---- legality (§3.3) --------------------------------------------------
    diags = check_graph(graph)
    errors = [d for d in diags if d.severity == "error"]
    if errors:
        raise LegalityError(errors)

    # ---- task lowering (§6, M2) -------------------------------------------
    families = lower_graph(graph)
    thread_problems = check_thread_contract(families)
    if thread_problems:
        raise LegalityError([Diagnostic("error", "thread-contract", p, "lowering") for p in thread_problems])

    # ---- schedule + memory (§8, §9, M3/M4) ---------------------------------
    widest_phase = max(f.task_count for f in families)
    chosen_workers = choose_worker_count(workers, max_tasks_in_widest_phase=widest_phase)
    phase_schedule = PhaseSchedule.from_families(families, workers=chosen_workers)
    workspace_plan, scratch_plan = plan_memory(graph, phase_schedule, families)

    # ---- codegen (§8) -------------------------------------------------------
    source = emit_megakernel_source(
        phase_schedule,
        workspace_plan,
        families,
        workers_hint=chosen_workers,
        threads_per_worker=THREADS_PER_WORKER,
        scratch_bytes_per_worker=scratch_plan.per_worker_bytes,
    )

    report = CompilationReport(
        target=target,
        schedule_kind=schedule,
        num_phases=phase_schedule.num_phases,
        total_tasks=phase_schedule.total_tasks,
        workspace_bytes=workspace_plan.total_bytes,
        workspace_naive_bytes=workspace_plan.naive_bytes,
        scratch_bytes_per_worker=scratch_plan.per_worker_bytes,
        kv_cache_bytes=kv_cache_bytes(config.layers, config.batch, config.cache_capacity, _kv_width(config)),
        diagnostics=diags,
        capability_notes=[
            "grid synchronization backend unverified: Milestone 0 producer-barrier-consumer microprogram required before any device execution (§8.4)",
            "worker count is a launch-time quantity within the cooperative-residency bound; occupancy of the *compiled* megakernel must be inspected on target (§7.4, §13.2)",
            "position p and cache length remain runtime parameters; shapes specialize (§13.3)",
        ],
        specialization_key=_specialization_key(config, target, schedule),
        generated_source=source,
    )
    if strict_single_kernel:
        report.capability_notes.append("strict_single_kernel=True: no fallback launches; device execution refuses to run until the grid-sync primitive is verified (§3.3)")

    return CompiledExecutable(
        graph=graph,
        config=config,
        weights=weights,
        schedule=phase_schedule,
        families=families,
        workspace_plan=workspace_plan,
        scratch_plan=scratch_plan,
        source=source,
        report=report,
        workers=chosen_workers,
    )
