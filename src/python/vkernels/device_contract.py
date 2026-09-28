"""Shared GPU tensor, stream, ordering, and storage-lifetime boundary."""

from contextlib import contextmanager


@contextmanager
def launch_context(tensors, stream):
    """Prepare and launch on one device/stream, retaining allocator lifetimes.

    Inputs must be ready on the caller's current stream. An explicit stream
    waits for it; callers consuming outputs elsewhere must wait for the
    launch stream. Raw handles belong to the tensor device and must remain
    valid through completion. ExternalStream wraps them without ownership.
    """
    import torch

    tensors = tuple(t for t in tensors if t is not None)
    if not tensors or any(not isinstance(t, torch.Tensor) or not t.is_cuda for t in tensors):
        raise ValueError("device inputs and outputs must be CUDA/HIP tensors")
    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("device tensors must share one GPU device")
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        if stream is None:
            launch = current
        elif isinstance(stream, torch.cuda.Stream):
            launch = stream
            if launch.device != device:
                raise ValueError("device stream must belong to the tensor device")
        else:
            launch = torch.cuda.ExternalStream(int(stream), device=device)
        if launch != current:
            launch.wait_stream(current)
        with torch.cuda.stream(launch):
            # Record original tensors too: a conversion may read one and
            # return a different allocation. Temporaries are created on
            # launch itself, so their normal allocator lifetime is sufficient.
            for tensor in tensors:
                tensor.record_stream(launch)
            yield launch.cuda_stream
