"""Explicit CDNA3 eligibility and changed-input graph coverage for Issue 148."""
import pytest

torch = pytest.importorskip("torch")
from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.glm_fp8_blockwise_gemm import glm_decode_fp8_gemm, quantize_activations_fnuz_fused


def test_cpu_declines_before_compilation():
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    w = torch.zeros(128, 128, dtype=torch.float8_e4m3fnuz)
    s = torch.ones(1, 1)
    with pytest.raises(OpNotEligible, match="gfx942"):
        glm_decode_fp8_gemm(x, w, s)
    with pytest.raises(OpNotEligible, match="gfx942"):
        quantize_activations_fnuz_fused(x)


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("fused", [False, True])
def test_cdna3_decode_graph_reads_changed_operands(batch, fused):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm gfx942")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("requires ROCm gfx942")
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import quantize_activations_fnuz

    torch.manual_seed(148)
    x = torch.randn(batch, 512, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(256, 512, device="cuda") * .5).to(torch.float8_e4m3fnuz)
    s = torch.rand(2, 4, device="cuda") * .02

    def reference():
        q, a = quantize_activations_fnuz(x)
        lhs = q.float() * a.repeat_interleave(128, dim=1)
        rhs = w.float() * s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        return lhs @ rhs.T

    def run():
        return glm_decode_fp8_gemm(x, w, s, split_k=4, fused_quantize=fused)

    original = run().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    x.mul_(.75)
    w.copy_((-w.float()).to(w.dtype))
    s.mul_(1.25)
    graph.replay()
    expected = reference()
    assert torch.isfinite(output).all()
    assert not torch.equal(output, original)
    assert (output.float() - expected).norm() / expected.norm().clamp_min(1e-12) < .01
