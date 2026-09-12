"""V4.1 fp8 block GEMM + UE8M0 quantization: reference oracle + GPU parity."""

import importlib.util
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.v41_fp8_gemm as m; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _dequant(torch, q, s, block, r, c):
    full = s.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
    return q.float() * full[:r, :c]


def test_ue8m0_scales_are_powers_of_two(torch):
    from vkernels.torch_ops.v41_fp8_gemm import quantize_fp8_ue8m0

    torch.manual_seed(0)
    x = torch.randn(70, 128) * 3.0  # M not a multiple of block
    q, s = quantize_fp8_ue8m0(x, block=32)
    assert s.shape == ((70 + 31) // 32, 128 // 32)
    log2s = torch.log2(s)
    assert torch.allclose(log2s, log2s.round())  # exact powers of two (UE8M0)


def test_quant_dequant_roundtrip(torch):
    from vkernels.torch_ops.v41_fp8_gemm import quantize_fp8_ue8m0

    torch.manual_seed(1)
    x = torch.randn(64, 128) * 2.0
    q, s = quantize_fp8_ue8m0(x, block=32)
    deq = _dequant(torch, q, s, 32, 64, 128)
    rel = (deq - x).abs().mean() / x.abs().mean()
    assert rel < 0.1, rel  # e4m3 block quant: a few percent


def test_gemm_reference_equals_dequant_matmul(torch):
    from vkernels.torch_ops.v41_fp8_gemm import fp8_block_gemm_reference, quantize_fp8_ue8m0

    torch.manual_seed(2)
    a = torch.randn(48, 128)
    b = torch.randn(32, 128)  # weights, transposed-use
    qa, sa = quantize_fp8_ue8m0(a, block=32)
    qb, sb = quantize_fp8_ue8m0(b, block=32)
    out = fp8_block_gemm_reference(qa, sa, qb, sb, block=32)
    manual = (_dequant(torch, qa, sa, 32, 48, 128) @ _dequant(torch, qb, sb, 32, 32, 128).t()).to(torch.bfloat16)
    assert torch.equal(out, manual)


def test_gemm_close_to_bf16(torch):
    from vkernels.torch_ops.v41_fp8_gemm import fp8_block_gemm_reference, quantize_fp8_ue8m0

    torch.manual_seed(3)
    a = torch.randn(64, 256)
    b = torch.randn(48, 256)
    ref = (a @ b.t()).to(torch.bfloat16)
    qa, sa = quantize_fp8_ue8m0(a, block=32)
    qb, sb = quantize_fp8_ue8m0(b, block=32)
    got = fp8_block_gemm_reference(qa, sa, qb, sb, block=32)
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.1, rel


def test_block128_gemm_matches_reference(torch):
    from vkernels.torch_ops.v41_fp8_gemm import fp8_block_gemm, fp8_block_gemm_reference, quantize_fp8_ue8m0

    torch.manual_seed(4)
    a = torch.randn(256, 256)
    b = torch.randn(128, 256)
    qa, sa = quantize_fp8_ue8m0(a, block=128)
    qb, sb = quantize_fp8_ue8m0(b, block=128)
    # CPU: fp8_block_gemm degrades to the reference (GLM Hopper kernel is GPU-only)
    assert torch.equal(
        fp8_block_gemm(qa, sa, qb, sb, block=128),
        fp8_block_gemm_reference(qa, sa, qb, sb, block=128),
    )



def test_wrapper_routes_to_reference_off_gpu(torch):
    from vkernels.torch_ops.v41_fp8_gemm import fp8_block_gemm, fp8_block_gemm_reference, quantize_fp8_ue8m0

    torch.manual_seed(5)
    a, b = torch.randn(64, 128), torch.randn(48, 128)
    qa, sa = quantize_fp8_ue8m0(a, block=32)
    qb, sb = quantize_fp8_ue8m0(b, block=32)
    # CPU: the wrapper is always the reference (bitwise)
    assert torch.equal(
        fp8_block_gemm(qa, sa, qb, sb, block=32),
        fp8_block_gemm_reference(qa, sa, qb, sb, block=32),
    )


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_decode_gemm_matches_reference(torch):
    """Decode-shape Triton block-scaled GEMM vs the torch oracle at real V4.1
    backbone geometries. Covers the two bugs the first bring-up hit: the
    per-tile (not per-row) B scale grid, and the padded-M masking (padded rows
    must neither load nor store)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import os

    from vkernels.torch_ops.v41_fp8_gemm import fp8_block_gemm, fp8_block_gemm_reference, quantize_fp8_ue8m0

    torch.manual_seed(5)
    shapes = [
        ("wq_b", 32768, 1280, 1, 32),
        ("wkv", 512, 5120, 1, 32),  # below the auto-gate: oracle by default
        ("wo_b", 5120, 32768, 1, 32),
        ("m8", 5120, 32768, 8, 32),
        ("odd_n", 300, 1280, 4, 32),
        ("block16", 512, 1280, 16, 16),
        ("m32b32", 5120, 5120, 32, 32),
    ]
    for tag, n, k, m, block in shapes:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        qa, sa = quantize_fp8_ue8m0(a, block=block)
        qb, sb = quantize_fp8_ue8m0(b, block=block)
        ref = fp8_block_gemm_reference(qa, sa, qb, sb, block=block)
        # forced: the kernel runs on every eligible shape (correct, sometimes slower)
        os.environ["VKERNELS_V41_FP8_GEMM_BACKEND"] = "triton"
        try:
            got = fp8_block_gemm(qa, sa, qb, sb, block=block)
        finally:
            os.environ.pop("VKERNELS_V41_FP8_GEMM_BACKEND", None)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-3, (tag, rel)
        # auto: shape-gated dispatch never loses — N >= 2048 takes the kernel
        auto = fp8_block_gemm(qa, sa, qb, sb, block=block)
        auto_rel = ((auto.float() - ref.float()).norm() / ref.float().norm()).item()
        assert auto_rel < 1e-3, (tag, auto_rel)
        if n >= 2048 and m <= block:
            assert torch.equal(auto, got), tag  # auto picked the kernel

