"""Inspect a compiled kernel before permitting a persistent spin barrier."""

from __future__ import annotations

import ctypes
from functools import lru_cache


@lru_cache(maxsize=1)
def _driver():
    driver = ctypes.CDLL("libcuda.so.1")
    fn = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return driver


def validate_residency(kernel, workers: int, device) -> int:
    import torch

    if workers < 1:
        raise ValueError("workers must be positive")
    if torch.version.hip:
        raise RuntimeError("Triton persistent residency has not been implemented for HIP")
    kernel._init_handles()
    active = ctypes.c_int()
    status = _driver().cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(active),
        ctypes.c_void_p(kernel.function),
        kernel.metadata.num_warps * 32,
        kernel.metadata.shared,
    )
    if status or active.value < 1:
        raise RuntimeError(f"compiled kernel residency query failed (CUDA status {status})")
    # One block per SM is deliberately conservative. It avoids filling every
    # available slot with polling blocks when other kernels are in flight.
    bound = torch.cuda.get_device_properties(device).multi_processor_count
    if workers > bound:
        raise ValueError(f"workers={workers} exceeds compiled persistent residency bound {bound}")
    return bound
