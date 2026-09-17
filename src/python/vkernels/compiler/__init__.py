"""The CuTe DSL megakernel model compiler (design doc implementation).

Implements the compiler proposed in ``cutedsl_megakernel_compiler_design.md``:
it transforms a captured GPT-2 single-token decode step into device-callable
tasks plus a phase-synchronous schedule executed by persistent workers —
one kernel launch instead of a sequence of operator launches.

Pipeline (§14 milestones):

    capture.py            M1  recording backend, symbolic values, shadow cache
    operator_ir.py            tensor semantics, views, regions, effects
    model_gpt2.py             GPT-2 frontend + independent NumPy oracle
    legality.py           §3.3 supported-subset checks and diagnostics
    lowerings/            M2  task-template registry (tile domains, regions)
    task_ir.py                TaskFamily contracts
    schedule_phase.py     M3  operator-ordered persistent schedule (§8)
    memory.py             M4  workspace reuse + per-worker scratch (§9)
    schedule_static.py    M5  tile-region dependencies (§10, static option)
    codegen_cute.py           CuTe DSL megakernel source emission (§8)
    reference_exec.py         CPU validation of the compiled schedule (§15.1)
    compile.py                the §13.1 ``compile_model`` API + report
    runtime/                  launch manifest, grid-barrier contract, §11 spec

The package is importable without torch/cutlass/numpy at the IR level;
:mod:`.model_gpt2` and :mod:`.reference_exec` need numpy (tests only).

Status (per the design doc): the schedule and memory planning are validated
on CPU against an independent oracle; device execution additionally
requires the Milestone-0 grid-synchronization primitive (§8.4), which is
*not* claimed to exist on any particular CuTe version.
"""

from __future__ import annotations

from .capture import CaptureError, RecordingBackend, SymbolicTensor, UnsupportedOperator, capture_model
from .legality import Diagnostic, LegalityError, check_graph
from .operator_ir import (
    DType,
    F32,
    F8_E4M3,
    Operator,
    OperatorGraph,
    Region,
    SymbolicScalar,
    TensorValue,
    ValidLength,
    compute_hazards,
)
from .schedule_phase import PhaseSchedule, ScheduleInvariantError
from .task_ir import TaskFamily, TileDomain

__all__ = [
    "CaptureError",
    "CompiledExecutable",
    "CompilationReport",
    "Diagnostic",
    "DType",
    "F32",
    "F8_E4M3",
    "GPT2Config",
    "GPT2Weights",
    "KVCache",
    "LegalityError",
    "Operator",
    "OperatorGraph",
    "PhaseSchedule",
    "QWEN3_06B",
    "Qwen3Config",
    "Qwen3KVCache",
    "Qwen3Weights",
    "RecordingBackend",
    "Region",
    "ScheduleInvariantError",
    "SymbolicScalar",
    "SymbolicTensor",
    "TaskFamily",
    "TensorValue",
    "TileDomain",
    "UnsupportedOperator",
    "ValidLength",
    "TritonMegakernel",
    "attach_megakernel_pool",
    "triton_available",
    "build_forward",
    "build_qwen3_forward",
    "capture_model",
    "check_graph",
    "compile_model",
    "compute_hazards",
    "qwen3_reference_forward",
    "random_weights",
    "random_qwen3_weights",
    "reference_forward",
    "tiny_qwen3_config",
]


def __getattr__(name: str):
    """Lazy re-exports that pull heavier dependencies (numpy) on demand."""
    if name in ("GPT2Config", "GPT2Weights", "KVCache", "build_forward", "random_weights", "reference_forward"):
        from . import model_gpt2

        return getattr(model_gpt2, name)
    if name in (
        "Qwen3Config",
        "Qwen3Weights",
        "Qwen3KVCache",
        "QWEN3_06B",
        "build_qwen3_forward",
        "random_qwen3_weights",
        "qwen3_reference_forward",
        "tiny_qwen3_config",
        "rope_tables",
    ):
        from . import model_qwen3

        return getattr(model_qwen3, name)
    if name in ("compile_model", "CompiledExecutable", "CompilationReport"):
        from .compile import CompiledExecutable, CompilationReport, compile_model

        return {"compile_model": compile_model, "CompiledExecutable": CompiledExecutable, "CompilationReport": CompilationReport}[name]
    if name in ("TritonMegakernel", "triton_available", "attach_megakernel_pool"):
        from . import device_triton

        return getattr(device_triton, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
