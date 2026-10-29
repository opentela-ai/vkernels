"""Prefix state agrees with the independent chunk-WY oracle, including replay."""
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c",
                    "import sys; import vkernels.torch_ops.glm_kda_prefix_state; "
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


def inputs(torch, *, device, batch=3, heads=2, tokens=4, kd=64, vd=32):
    rng = torch.Generator(device=device).manual_seed(81)
    key = torch.randn(batch, heads, tokens, kd, device=device, generator=rng)
    key = torch.nn.functional.normalize(key, dim=-1)
    value = torch.randn(batch, heads, tokens, vd, device=device, generator=rng)
    gate = -torch.rand(key.shape, device=device, generator=rng)
    beta = torch.rand(batch, heads, tokens, device=device, generator=rng)
    state = torch.randn(batch, heads, kd, vd, device=device, generator=rng) * 0.1
    lengths = torch.tensor([0, 1, tokens], dtype=torch.int32, device=device)
    return key, value, gate, beta, state, lengths


def oracle(torch, args):
    from vkernels.torch_ops.glm_kda_chunk import kda_chunk_reference

    key, value, gate, beta, state, lengths = args
    rows = []
    for row, count in enumerate(lengths.tolist()):
        count = max(0, min(key.shape[2], count))
        if count == 0:
            rows.append(state[row:row + 1].clone())
        else:
            _, final = kda_chunk_reference(
                key[row:row + 1, :, :count], key[row:row + 1, :, :count],
                value[row:row + 1, :, :count], gate[row:row + 1, :, :count],
                beta[row:row + 1, :, :count], chunk_size=16,
                initial_state=state[row:row + 1], output_final_state=True)
            rows.append(final)
    return torch.cat(rows)


def test_cpu_rejected_before_triton_import():
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_kda_prefix_state import kda_prefix_state

    with pytest.raises(OpNotEligible, match="CUDA"):
        kda_prefix_state(*inputs(torch, device="cpu"))


@pytest.mark.parametrize("tokens,kd,vd", [(1, 32, 64), (4, 64, 32), (8, 128, 128)])
def test_gpu_matches_chunk_oracle_and_preserves_inputs(tokens, kd, vd):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    from vkernels.torch_ops.glm_kda_prefix_state import kda_prefix_state

    args = inputs(torch, device="cuda", tokens=tokens, kd=kd, vd=vd)
    before = [tensor.clone() for tensor in args]
    actual = kda_prefix_state(*args)
    expected = oracle(torch, args)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    for value, original in zip(args, before):
        assert torch.equal(value, original)
    assert torch.equal(actual[0], args[4][0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_prefix_state(*args)
    for counts in ([tokens, 0, 1], [tokens + 1, -1, tokens], [0, 0, 0]):
        args[-1].copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(captured, oracle(torch, args), atol=1e-4, rtol=1e-4)


def test_conv_cpu_rejected_before_triton_import():
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_kda_prefix_state import kda_prefix_conv_state

    with pytest.raises(OpNotEligible, match="CUDA"):
        kda_prefix_conv_state(torch.zeros(2, 5, 4), torch.zeros(2, 5, 3),
                              torch.tensor([1, 2], dtype=torch.int32))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("tokens,history", [(1, 3), (4, 3), (8, 0), (0, 3)])
def test_conv_history_matches_rolling_append_on_replay(dtype, tokens, history):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    from vkernels.torch_ops.glm_kda_prefix_state import kda_prefix_conv_state

    # Non-power-of-two channels exercise the final masked block.
    raw = torch.randn(3, 37, tokens, device="cuda", dtype=getattr(torch, dtype))
    initial = torch.randn(3, 37, history, device="cuda", dtype=raw.dtype)
    lengths = torch.tensor([0, 1, tokens], device="cuda", dtype=torch.int64)
    initial_before = initial.clone()
    raw_before = raw.clone()

    def expected():
        rows = []
        for row, length in enumerate(lengths.tolist()):
            count = max(0, min(tokens, length))
            combined = torch.cat((initial[row], raw[row, :, :count]), dim=-1)
            rows.append(combined[:, combined.shape[-1] - history:])
        return torch.stack(rows)

    assert torch.equal(kda_prefix_conv_state(raw, initial, lengths), expected())
    if history == 0:
        assert torch.equal(raw, raw_before)
        assert torch.equal(initial, initial_before)
        return  # no kernel to capture for an empty history
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_prefix_conv_state(raw, initial, lengths)
    for counts in ([tokens + 2, -1, 0], [1, tokens, tokens - 1], [0, 0, 0]):
        lengths.copy_(torch.tensor(counts, device="cuda", dtype=lengths.dtype))
        graph.replay()
        assert torch.equal(captured, expected())
    assert torch.equal(raw, raw_before)
    assert torch.equal(initial, initial_before)
