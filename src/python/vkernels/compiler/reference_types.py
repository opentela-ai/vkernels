"""Execution diagnostics and scalar numerical helpers."""
from __future__ import annotations

from dataclasses import dataclass, field


import numpy as np



def _stable_sigmoid(g: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, fp64 mirror of the device epilogue
    ``1 / (1 + exp(-g))`` (issue #92). Overflow-safe for large |g|: the
    device computes exp(-g) directly, so huge negative g overflows to inf
    and the ratio still rounds to 0; huge positive g underflows to 0 and
    the ratio rounds to 1 — same saturating semantics, no NaN."""
    g = np.asarray(g, dtype=np.float64)
    out = np.empty_like(g)
    pos = g >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-g[pos]))
    eg = np.exp(g[~pos])
    out[~pos] = eg / (1.0 + eg)
    return out


def bf16_round(x):
    """Round float values to the bf16 grid (round-to-nearest-even), the
    ABI rounding of the fused-decode contract (E1).

    Mirrors CUDA ``__float2bfloat16`` / torch ``.to(torch.bfloat16)``:
    keep the top 16 bits of the fp32 representation, rounding to nearest
    with ties-to-even via the standard ``+0x7FFF + lsb`` increment. NaN
    payloads survive; the result is widened back to the input dtype's
    float64/float32 numpy type so executor storage stays fp32 (the
    compiled-pool convention) while VALUES sit exactly on the bf16 grid
    the device kernel exchanges (``glm_kda_fused_decode``'s documented
    "raw dot / sigmoid-after-bf16-round" semantics)."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim and not a.flags["C_CONTIGUOUS"]:
        a = a.copy()  # strided views (executor storage) cannot be .view-ed
    u = a.view(np.uint32).astype(np.uint64)
    r = u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))
    out = (r & np.uint64(0xFFFF0000)).astype(np.uint32).view(np.float32)
    want = np.asarray(x).dtype
    return out.astype(want) if want.kind == "f" else out.astype(np.float64)


class ExecutorError(Exception):
    """The executed schedule violated an obligation (§12)."""


def decode_e4m3(bytes_u8):
    """Decode fp8 e4m3fn bytes (uint8 array) to float64, vectorized.

    e4m3fn: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits; no
    infinities — exponent 0xF with mantissa 0x7 encodes NaN, and the
    maximum finite magnitude is 448. Subnormals (exponent 0) are
    mant/8 * 2^-6. This is the numpy-side counterpart of the checkpoint's
    ``torch.float8_e4m3fn`` weights and of ``_fp8_e4m3fn_encode``.
    """
    b = np.asarray(bytes_u8, dtype=np.uint8).astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    is_nan = (exp == 0xF) & (mant == 0x7)
    normal = np.exp2((exp - 7).astype(np.float64)) * (1.0 + mant / 8.0)
    sub = np.exp2(-6.0) * (mant / 8.0)
    val = sign * np.where(exp == 0, sub, normal)
    return np.where(is_nan, np.nan, val)


@dataclass
class PhaseStat:
    index: int
    kind: str
    tasks: int
    per_worker: list[int]
    barrier_generation: int

    @property
    def idle_workers(self) -> int:
        return sum(1 for c in self.per_worker if c == 0)

    @property
    def max_tasks_per_worker(self) -> int:
        return max(self.per_worker)


@dataclass
class ExecutionTrace:
    kernel_launches: int = 0
    grid_barriers: int = 0
    phases: list[PhaseStat] = field(default_factory=list)
    task_executions: int = 0

    def summary(self) -> str:
        lines = [f"execution: {self.kernel_launches} kernel launch(s), {self.grid_barriers} grid barriers, {self.task_executions} task executions"]
        for st in self.phases:
            lines.append(f"  phase {st.index:2d} {st.kind:18s} tasks={st.tasks:3d} per-worker={st.per_worker} idle={st.idle_workers}")
        return "\n".join(lines)

_decode_e4m3 = decode_e4m3
