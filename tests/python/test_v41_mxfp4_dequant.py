"""MXFP4 (E2M1 + microscale) dequant: reference oracle + optional GPU parity."""

import importlib.util
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.v41_mxfp4_dequant as m; "
            "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_known_codes(torch):
    from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant_reference

    # low nibble first: 0x10 -> [0.0, 0.5], 0x32 -> [1.0, 1.5];
    # 0x09 -> low 9 (-0.5), high 0 (0.0); 0xEF -> low 15 (-6.0), high 14 (-4.0)
    packed = torch.tensor([[0x10, 0x32], [0x09, 0xEF]], dtype=torch.uint8)
    scale = torch.ones(2, 4)  # group=1: per-element scale
    out = mxfp4_dequant_reference(packed, scale, group=1, dtype=torch.float32)
    assert out.tolist() == [[0.0, 0.5, 1.0, 1.5], [-0.5, 0.0, -6.0, -4.0]]


def test_reference_group_microscale(torch):
    from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant_reference

    packed = torch.tensor([[0x10, 0x32]], dtype=torch.uint8)  # -> [0,0.5,1,1.5]
    scale = torch.tensor([[2.0]])  # one scale for the whole width-4 group
    out = mxfp4_dequant_reference(packed, scale, group=4, dtype=torch.float32)
    assert out.tolist() == [[0.0, 1.0, 2.0, 3.0]]


def test_reference_uint8_e8m0_scale(torch):
    from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant_reference

    packed = torch.tensor([[0x32]], dtype=torch.uint8)  # -> [1.0, 1.5]
    scale = torch.tensor([[128]], dtype=torch.uint8)  # E8M0 code 128 -> 2**(128-127)=2
    out = mxfp4_dequant_reference(packed, scale, group=2, dtype=torch.float32)
    assert out.tolist() == [[2.0, 3.0]]


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_matches_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant, mxfp4_dequant_reference

    g = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (8, 64), dtype=torch.uint8)  # O=8, I=128
    scale = torch.rand(8, 4, generator=g) + 0.5  # group=32 -> I//32 = 4
    ref = mxfp4_dequant_reference(packed, scale, group=32, dtype=torch.bfloat16)
    got = mxfp4_dequant(packed.cuda(), scale.cuda(), group=32, dtype=torch.bfloat16).cpu()
    # elementwise, identical op order -> bitwise equality is deterministic
    assert torch.equal(got, ref)



def test_wrapper_falls_back_on_cpu(torch):
    from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant, mxfp4_dequant_reference

    g = torch.Generator().manual_seed(1)
    packed = torch.randint(0, 256, (4, 32), dtype=torch.uint8, generator=g)  # O=4, I=64
    scale = torch.rand(4, 2, generator=g) + 0.5  # group=32 -> I//32 = 2
    got = mxfp4_dequant(packed, scale, group=32, dtype=torch.bfloat16)  # CPU -> reference
    assert torch.equal(got, mxfp4_dequant_reference(packed, scale, group=32, dtype=torch.bfloat16))


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_full_stack_no_int32_overflow(torch):
    """The flattened V4.1 expert stack (E*O = 884_736 rows x I 5120, ~9 GB out)
    overflows int32 row offsets (2.26e9 packed bytes > 2**31) — regression for
    the illegal-memory-access the bench harness hit; the arch itself never
    materializes the full stack (dequant-on-use), so only big-shape callers
    see it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_properties(0).total_memory < 20 * 2**30:
        pytest.skip("needs >20 GB device memory")
    from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant, mxfp4_dequant_reference

    e, o, i = 384, 2304, 5120
    packed = torch.randint(0, 256, (e, o, i // 2), device="cuda", dtype=torch.uint8)
    scale = (torch.rand(e, o, i // 32, device="cuda") + 0.5).float()
    got = mxfp4_dequant(packed, scale, group=32, dtype=torch.bfloat16)
    # Spot-check the LAST row (the one whose int32 offset would wrap) against
    # the reference on the same rows; a full 9 GB reference round-trip is
    # unnecessary to pin the overflow.
    tail_ref = mxfp4_dequant_reference(
        packed[-4:].cpu(), scale[-4:].cpu(), group=32, dtype=torch.bfloat16)
    assert torch.equal(got[-4:].cpu(), tail_ref)

