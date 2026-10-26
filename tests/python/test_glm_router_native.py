"""The native-order router has a pinned stack and exact output contract."""
import subprocess
import sys
import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import vkernels.torch_ops.glm_router_native; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.mark.parametrize("change", ["cpu", "experts", "rows", "dtype", "bias_dtype", "groups", "normalization"])
def test_unsupported_contract_declines(change):
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_router_native import fused_router_native
    x, bias = torch.zeros(2, 288), torch.zeros(288)
    kwargs = {}
    if change == "experts":
        x, bias = torch.zeros(2, 16), torch.zeros(16)
    if change == "rows":
        x = torch.zeros(8, 288)
    if change == "dtype":
        x = x.half()
    if change == "bias_dtype":
        bias = bias.half()
    if change == "groups":
        kwargs["num_group"] = 2
    if change == "normalization":
        kwargs["norm_topk_prob"] = False
    before_x, before_bias = x.clone(), bias.clone()
    expected = "HIP device" if change == "cpu" else "geometry" if change in ("experts", "rows") else "FP32 logits" if change in ("dtype", "bias_dtype") else "normalized top8"
    with pytest.raises(OpNotEligible, match=expected):
        fused_router_native(x, bias, 8, 2.5, **kwargs)
    assert torch.equal(x, before_x) and torch.equal(bias, before_bias)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 16, 128])
def test_exact_native_order_and_weights_on_qualified_stack(rows):
    torch = pytest.importorskip("torch")
    if str(torch.__version__) != "2.9.0a0+git7bcbafe" or torch.version.hip != "7.0.51831-a3e329ad8":
        pytest.skip("pinned gfx942 Torch/HIP stack required")
    triton = pytest.importorskip("triton")
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx942" or triton.__version__ != "3.4.0":
        pytest.skip("pinned gfx942/Triton stack required")
    from vkernels.torch_ops.glm_router_native import fused_router_native
    with torch.no_grad():
        for seed in range(16):
            torch.manual_seed(seed)
            x = torch.randn(rows, 288, device="cuda") * (0.1, 1., 10., 100.)[seed % 4]
            bias = torch.randn(288, device="cuda", dtype=torch.bfloat16) * .01
            if seed == 0:
                x.fill_(-88.5)
                bias.zero_()
            if seed == 1:
                x.zero_()
                bias.zero_()
                bias[[32, 103, 163, 187, 259, 282, 285]] = 1.
            score = x.sigmoid()
            expected_idx = (score + bias).topk(8, dim=-1, sorted=False).indices
            w = score.gather(1, expected_idx)
            expected_w = w / (w.sum(-1, keepdim=True) + 1.e-20) * 2.5
            before_x, before_bias = x.clone(), bias.clone()
            idx, weight = fused_router_native(x, bias, 8, 2.5)
            assert torch.equal(idx, expected_idx)
            assert torch.equal(weight.view(torch.int32), expected_w.view(torch.int32))
            assert torch.equal(x, before_x) and torch.equal(bias, before_bias)
