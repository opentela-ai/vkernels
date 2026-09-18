"""Canonical direct-launch helper for batched megakernel task bodies (tests).

Task-body templates in ``vkernels.compiler.device_triton`` take two leading
RUNTIME args — ``worker`` and ``P`` — and stride tasks ``worker, worker+P,
...`` over the task count. The megakernel runtime composes them with
``worker = tl.program_id(0)`` and ``P = grid`` (see the program-id wrappers
in device_triton.py). A direct single-program test launch covers every task
with ``grid=(1,), worker=0, P=1``: the stride loop makes the one program
execute the whole task range.

Hand-rolling the leading args has a five-incident history — rope (pr-112),
gdn_conv (pr-114), kda_delta, and the mla scores/values + gdn_delta +
indexer launches this helper fixed — so always launch task bodies through
here. For a multi-program launch (only needed to exercise cross-program
behaviour), wrap the body in a ``@triton.jit`` launcher that injects
``tl.program_id(0), tl.num_programs(0)`` — see ``_mhc_triton_launchers.py``.
"""

from __future__ import annotations


def launch_task_body(kern, *args, num_warps=4, **kwargs) -> None:
    """Direct-launch a batched task body: one program strided-covers all tasks."""
    kern[(1,)](0, 1, *args, num_warps=num_warps, **kwargs)
