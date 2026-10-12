"""Inspect a compiled kernel before permitting a persistent spin barrier.

The megakernel's grid barrier is a monotonic-counter spin (see
``device_triton.grid_barrier``): *every* worker must be co-resident or the
spin can never be satisfied and the launch hangs until the wall clock stops
it.  The launch therefore has to guarantee co-residency, and the only sound
bound is the hardware occupancy limit of the **compiled** kernel -- not of a
look-alike kernel nor of the source it came from.

Both vendor drivers expose that limit per SM/CU:

* NVIDIA -- ``cuOccupancyMaxActiveBlocksPerMultiprocessor`` on the cubin the
  triton driver already loaded (``kernel.function`` after ``_init_handles``),
  block size ``num_warps * 32``.
* AMD -- ``hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`` on the
  kernel's own HSACO.  The triton HIP launcher keeps its ``hipFunction_t``
  inside a C++ closure, so this module re-materialises the code object from
  ``kernel.asm["hsaco"]`` -- byte-for-byte what the launcher loads -- through
  ``libamdhip64`` and queries that function.  Block size is
  ``num_warps * warp_size``; triton reports ``warp_size = 64`` on CDNA
  (gfx9xx, e.g. MI300A/gfx942) and 32 from gfx10 on, so the value is read
  from the kernel metadata rather than assumed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache


@lru_cache(maxsize=1)
def _driver():
    driver = ctypes.CDLL("libcuda.so.1")
    fn = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return driver


@lru_cache(maxsize=1)
def _hip_driver():
    """``libamdhip64`` bound for the module-occupancy query (ROCm)."""
    name = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
    lib = ctypes.CDLL(name)
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.restype = ctypes.c_int
    lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
    lib.hipModuleUnload.restype = ctypes.c_int
    return lib


def _warp_size(kernel) -> int:
    """Lanes per wave/warp as triton compiled this kernel (64 on CDNA)."""
    return int(getattr(kernel.metadata, "warp_size", 32) or 32)


def _block_size(kernel) -> int:
    return int(kernel.metadata.num_warps) * _warp_size(kernel)


def _hip_active_blocks(kernel, block_size: int) -> int:
    asm = getattr(kernel, "asm", None) or {}
    hsaco = asm.get("hsaco") or getattr(kernel, "kernel", None)
    if not hsaco:
        raise RuntimeError(
            "compiled kernel carries no HSACO (kernel.asm['hsaco']); refusing to "
            "guess a cooperative-residency bound for a spin-barrier launch"
        )
    if isinstance(hsaco, str):
        from pathlib import Path

        hsaco = Path(hsaco).read_bytes()
    hsaco = bytes(hsaco)
    lib = _hip_driver()
    buf = ctypes.create_string_buffer(hsaco, len(hsaco))
    module = ctypes.c_void_p()
    status = lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p))
    if status or not module.value:
        raise RuntimeError(f"compiled kernel residency query failed (HIP code-object load status {status})")
    function = ctypes.c_void_p()
    name = kernel.metadata.name if hasattr(kernel, "metadata") else kernel.name
    try:
        status = lib.hipModuleGetFunction(ctypes.byref(function), module, name.encode())
        if status or not function.value:
            raise RuntimeError(
                f"compiled kernel residency query failed (HIP symbol {name!r} status {status})"
            )
        active = ctypes.c_int()
        status = lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(active), function, int(block_size), int(kernel.metadata.shared)
        )
        if status or active.value < 1:
            raise RuntimeError(
                f"compiled kernel residency query failed (HIP status {status}, "
                f"block={block_size}, shared={int(kernel.metadata.shared)})"
            )
        return active.value
    finally:
        lib.hipModuleUnload(module)


def validate_residency(kernel, workers: int, device, *, blocks_per_sm: int = 1) -> int:
    import torch

    if workers < 1:
        raise ValueError("workers must be positive")
    if not isinstance(blocks_per_sm, int) or blocks_per_sm < 1:
        raise ValueError("blocks_per_sm must be positive")
    active = ctypes.c_int()
    if torch.version.hip:
        active_blocks = _hip_active_blocks(kernel, _block_size(kernel))
    else:
        kernel._init_handles()
        status = _driver().cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(active),
            ctypes.c_void_p(kernel.function),
            _block_size(kernel),
            kernel.metadata.shared,
        )
        if status or active.value < 1:
            raise RuntimeError(f"compiled kernel residency query failed (CUDA status {status})")
        active_blocks = active.value
    # Existing callers retain the conservative one-block-per-SM limit.
    # An explicitly isolated compute region may request more, but only up to
    # the residency of THIS compiled kernel (registers and shared memory).
    if blocks_per_sm > active_blocks:
        raise ValueError(f"blocks_per_sm={blocks_per_sm} exceeds compiled occupancy {active_blocks}")
    bound = torch.cuda.get_device_properties(device).multi_processor_count * blocks_per_sm
    if workers > bound:
        raise ValueError(f"workers={workers} exceeds compiled persistent residency bound {bound}")
    return bound
