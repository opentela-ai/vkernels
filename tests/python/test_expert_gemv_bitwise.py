"""Native conversion/traversal must preserve portable GEMV rounding."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("tokens,experts,width,outputs,broadcast", [
    (1, 8, 512, 384, True), (4, 9, 512, 256, True),
    (4, 9, 384, 384, False), (2, 2, 128, 128, False),
])
def test_native_matches_portable_bits_and_reads_replayed_inputs(
    tokens, experts, width, outputs, broadcast,
):
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("Native E4M3FN conversion requires a NVIDIA GPU")
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv

    gen = torch.Generator(device="cuda").manual_seed(842)
    shape = (tokens, width) if broadcast else (tokens, experts, width)
    x = torch.randn(shape, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn((12, outputs, width), generator=gen, device="cuda") * .3).to(torch.float8_e4m3fn)
    s = torch.exp(torch.randn((12, outputs // 128, width // 128), generator=gen, device="cuda")) * .02
    idx = torch.randint(12, (tokens, experts), generator=gen, device="cuda")
    idx[:, -1] = idx[:, 0]
    if tokens > 1:
        idx[1] = idx[0]

    def run(implementation):
        return expert_gemv(x, w, s, idx, t_cap=tokens, implementation=implementation)

    for implementation in ("portable", "cuda_native"):
        run(implementation)
    graphs, results = [], []
    for implementation in ("portable", "cuda_native"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run(implementation)
        graphs.append(graph)
        results.append(result)
    original = results[0].clone()
    assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))
    x.mul_(.75)
    s.mul_(1.25)
    for graph in graphs:
        graph.replay()
    assert not torch.equal(original, results[0])
    assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))
