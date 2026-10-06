"""Vectorized selected latent copies; all attention arithmetic stays with the caller."""

from functools import lru_cache

from ._dispatch import OpNotEligible, require, same_gpu_contiguous


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["SQ", "SKV", "W", "COUNT"])
    def _dsa_latent_gather(SRC, IDX, OUT, SQ, SKV, W, COUNT,
                          D: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
        positions = tl.program_id(0) * BR + tl.arange(0, BR)
        dims = tl.arange(0, BD)
        live = positions < COUNT
        ids = tl.load(IDX + positions, live, other=0).to(tl.int64)
        ids = tl.maximum(ids, 0)  # Same clamp as the Torch reference.
        batches = (positions // (SQ * W)).to(tl.int64)
        source = (batches * SKV + ids)[:, None] * D + dims[None, :]
        values = tl.load(SRC + source, live[:, None] & (dims[None, :] < D), other=0)
        target = positions.to(tl.int64)[:, None] * D + dims[None, :]
        tl.store(OUT + target, values, live[:, None] & (dims[None, :] < D))

    return _dsa_latent_gather


def dsa_latent_gather(latent, indices):
    """Copy latent[B,N,D] at indices[B,S,W], clamping negatives to row zero.

    Contiguous CUDA inference tensors only. Positive out-of-range IDs raise a
    device assertion as in Torch advanced indexing, rather than reading another
    request's rows. Raw integer views preserve NaN payloads and signed zero.
    """
    import torch

    require(latent.ndim == 3 and indices.ndim == 3, "expected latent [B,N,D], indices [B,S,W]")
    require(latent.shape[0] == indices.shape[0], "batch dimensions must match")
    require(latent.dtype in (torch.bfloat16, torch.float16, torch.float32), "unsupported latent dtype")
    require(indices.dtype in (torch.int32, torch.int64), "indices must be int32 or int64")
    require(not latent.requires_grad, "inference only")
    require(torch.version.hip is None, "CUDA qualification only")
    same_gpu_contiguous(latent, indices)
    batch, skv, dim = latent.shape
    _, seq, width = indices.shape
    require(skv > 0 and 0 < dim <= 512, "nonempty cache and at most 512 latent dimensions required")
    try:
        import triton
        import triton.language as language
    except ImportError as exc:
        raise OpNotEligible("Triton is unavailable") from exc
    out = torch.empty((batch, seq, width, dim), dtype=latent.dtype, device=latent.device)
    count = indices.numel()
    if count == 0:
        return out
    # Checking on the device avoids a request-path host synchronization.
    torch._assert_async((indices < skv).all(), "selected latent row exceeds cache bounds")
    raw_type = language.int32 if latent.element_size() == 4 else language.int16
    with torch.cuda.device(latent.device):
        _kernel()[(triton.cdiv(count, 8),)](
            triton.reinterpret(latent, raw_type), indices, triton.reinterpret(out, raw_type),
            seq, skv, width, count, dim, triton.next_power_of_2(dim), 8, num_warps=4)
    return out
