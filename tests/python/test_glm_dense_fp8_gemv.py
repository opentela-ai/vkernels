"""Dense block-fp8 decode GEMV: contract checks + GPU parity vs the
loader-dequant oracle (``dense_gemv_fp8_reference``).

The oracle reproduces floe's serving reference exactly (dequant_block_fp8's
fp32-product-then-bf16-cast, then a bf16 F.linear); the kernel differs only
in fp32 reduction order, so parity tolerances are tight.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[2] / "src" / "python")


def test_import_is_lazy():
    # the lazy-import probe runs in a clean interpreter pinned to THIS
    # checkout's src tree (the venv's editable install may be another branch)
    env = dict(os.environ, PYTHONPATH=_SRC)
    subprocess.run([sys.executable, "-c",
                    "import sys; import vkernels.torch_ops.glm_dense_fp8_gemv; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'triton' not in sys.modules"], check=True, env=env)


def test_opnoteligible_cpu():
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_dense_fp8_gemv import dense_gemv_fp8

    x = torch.randn(1, 256, dtype=torch.bfloat16)
    w8 = torch.zeros(128, 256, dtype=torch.float8_e4m3fn)
    s = torch.ones(1, 2)
    with pytest.raises(OpNotEligible):
        dense_gemv_fp8(x, w8, s)


def test_opnoteligible_bad_shapes():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("contract checks below need CUDA residency first")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_dense_fp8_gemv import dense_gemv_fp8

    x = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
    w8 = torch.zeros(128, 256, dtype=torch.float8_e4m3fn, device="cuda")
    cuda = dict(device="cuda")
    with pytest.raises(OpNotEligible):
        dense_gemv_fp8(x, w8, torch.ones(1, 3, **cuda))  # wrong scale grid
    with pytest.raises(OpNotEligible):
        dense_gemv_fp8(x[:, :100], w8[:, :100], torch.ones(1, 1, **cuda))  # I % 128
    with pytest.raises(OpNotEligible):
        dense_gemv_fp8(torch.randn(16, 256, dtype=torch.bfloat16, **cuda), w8,
                       torch.ones(1, 2, **cuda))  # M above the cap


def make_inputs(torch, o=256, i=256, seed=7):
    torch.manual_seed(seed)
    raw = torch.randint(0, 256, (o, i), dtype=torch.uint8)
    raw.masked_fill_((raw & 127) == 127, 0)  # drop NaN encodings
    w8 = raw.view(torch.float8_e4m3fn)
    scales = 0.0005 + torch.rand((o // 128, i // 128)) * 0.001
    return w8, scales


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("shape", [(128, 128), (256, 512), (4096, 1536), (512, 4096)])
def test_gpu_parity_vs_oracle(monkeypatch, m, shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    # m > 2 is the DFlash2 verify regime — widen the cap the same way
    # floe's moe_decode_max_tokens=8 bridge does
    monkeypatch.setenv("GLM53_MOE_DECODE_MAX_TOKENS", str(max(2, m)))
    from vkernels.torch_ops.glm_dense_fp8_gemv import (
        dense_gemv_fp8, dense_gemv_fp8_reference)

    o, i = shape
    w8, scales = make_inputs(torch, o=o, i=i)
    x = (torch.randn(m, i) * 0.1).to(torch.bfloat16)
    ref = dense_gemv_fp8_reference(x, w8, scales)
    out = dense_gemv_fp8(x.cuda(), w8.cuda(), scales.cuda()).cpu()
    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    # the dequantized weight values are bit-identical by construction; the
    # only divergence is fp32 reduction order — bounded at bf16-rounding
    # size relative to the OUTPUT magnitude (element-wise relative error is
    # unbounded at cancellation points, so the check is scale-aware).
    scale = ref.abs().float().max().clamp_min(1e-3)
    rel = ((out.float() - ref.float()).abs().max() / scale).item()
    assert rel < 5e-2, f"relative deviation {rel}"


def test_gpu_weight_values_bit_identical_to_loader_dequant():
    """In-kernel (v*s).to(bf16) must equal floe's dequant_block_fp8 output
    bitwise — the parity claim's foundation."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_dense_fp8_gemv import (
        dense_gemv_fp8, dense_gemv_fp8_reference)

    w8, scales = make_inputs(torch, o=256, i=512)
    x = torch.zeros(1, 512, dtype=torch.bfloat16)
    # with x = one-hot columns the kernel output reads back individual
    # dequantized weights, so bitwise equality is directly observable
    for col in (0, 129, 511):
        x.zero_()
        x[0, col] = 1.0
        got = dense_gemv_fp8(x.cuda(), w8.cuda(), scales.cuda()).cpu()[0]
        ref_w = dense_gemv_fp8_reference(x, w8, scales)[0]
        assert torch.equal(got, ref_w)
