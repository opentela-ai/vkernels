"""Pinned SGLang0.5.20 one-shot push kernel, TP4 Hopper, in-place SUM.

Apache-2.0 source and provenance live in _vendor_sgl_push. The donor's device
kernel arithmetic, two-phase Lamport slots, phase counters and FP32 rank-order reduction
are preserved. The host entry writes into the caller's input. Each input vector
is pushed before its disjoint output vector is overwritten. No input IPC or
SGLang scheduler/cache dependency is needed. This plan owns all storage/views
and counters for its whole lifetime, including every captured graph replay.

Requires optional apache-tvm-ffi and a CUDA compiler on first eager use; both
compilation and symmetric rendezvous are forbidden during stream capture.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import torch

MAX_BYTES = 64 * 1024


def eligible(tensor: torch.Tensor, world_size: int) -> bool:
    return (
        world_size == 4
        and tensor.is_cuda
        and torch.version.hip is None
        and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and tensor.is_contiguous()
        and 0 < tensor.nbytes <= MAX_BYTES
        and tensor.nbytes % 16 == 0
        and tensor.data_ptr() % 16 == 0
        and torch.cuda.get_device_capability(tensor.device) == (9, 0)
    )


@lru_cache(maxsize=1)
def _module():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SGL push compilation must be warmed before capture")
    import tvm_ffi.cpp

    root = Path(__file__).with_name("_vendor_sgl_push")
    # Header contents participate in the JIT name, so package updates cannot
    # accidentally reuse a binary compiled against older donor headers.
    digest = hashlib.sha256()
    for header in sorted(root.rglob("*")):
        if header.suffix in (".h", ".cuh"):
            digest.update(str(header.relative_to(root)).encode())
            digest.update(header.read_bytes())
    module = tvm_ffi.cpp.load_inline(
        name="vkernels_sgl_push_tp4_sm90_" + digest.hexdigest()[:16],
        cuda_sources=(root / "inplace.cu").read_text(),
        extra_cflags=["-std=c++20", "-O3"],
        extra_cuda_cflags=["-std=c++20", "-O3", "-DSGL_CUDA_ARCH=900", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"],
        extra_include_paths=[str(root), str(root / "include")],
    )
    module.register_communicator()
    return module


@lru_cache(maxsize=1)
def _types():
    import tvm_ffi

    _module()

    @tvm_ffi.register_object("sgl.distributed.PushPlane")
    class PushPlane(tvm_ffi.Object):
        rank: int
        world_size: int
        num_blocks: int
        slot_bytes: int

        def __init__(self, rank, world, workspaces, counter):
            self.__ffi_init__(rank, world, workspaces, counter, 0)

    @tvm_ffi.register_object("sgl.distributed.Communicator")
    class Communicator(tvm_ffi.Object):
        def __init__(self, push):
            self.__ffi_init__(push, None)

    return PushPlane, Communicator


class SglPushPlan:
    """One stream-ordered mailbox plane for the rank-uniform TP seam sequence."""

    def __init__(self, rank, world_size, device, group, *, num_blocks=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SGL push rendezvous must be warmed before capture")
        if world_size != 4 or torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("SGL push plan requires TP4 Hopper")
        self.module = _module()
        PushPlane, Communicator = _types()
        from torch.distributed import _symmetric_memory as symm
        import torch.distributed as dist

        blocks = num_blocks or torch.cuda.get_device_properties(device).multi_processor_count
        self.storage = symm.empty(2 * world_size * MAX_BYTES, dtype=torch.uint8, device=device)
        self.handle = symm.rendezvous(self.storage, group)
        self.workspaces = [self.handle.get_buffer(i, [2 * world_size, MAX_BYTES], torch.uint8)
                           for i in range(world_size)]
        self.counter = torch.zeros((blocks, 4), dtype=torch.uint8, device=device)
        self.workspaces[rank].zero_()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        self.plane = PushPlane(rank, world_size, self.workspaces, self.counter)
        self.communicator = Communicator(self.plane)
        self.functions = {torch.bfloat16: self.module.push_bf16, torch.float16: self.module.push_fp16,
                          torch.float32: self.module.push_fp32}
        self.device = torch.device(device)

    def reduce_(self, tensor):
        if not eligible(tensor, 4) or tensor.device != self.device:
            raise ValueError("ineligible tensor or wrong device for SGL push plan")
        self.functions[tensor.dtype](self.communicator, tensor)
        return tensor
