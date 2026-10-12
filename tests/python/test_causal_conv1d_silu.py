"""Single-round convolution, arbitrary input strides and read-only state."""

import pytest
import torch
import torch.nn.functional as F

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.causal_conv1d_silu import causal_conv1d_silu


@pytest.mark.parametrize("batch,channels,sequence", [(1, 17, 1), (4, 257, 19), (1, 768, 128)])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_single_round_matches_donor_arithmetic(batch, channels, sequence, width, with_state, transposed):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(293)
    x = torch.randn(batch, channels, sequence, device="cuda", dtype=torch.bfloat16)
    if transposed:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    weight = torch.randn(channels, 1, width, device="cuda", dtype=x.dtype)
    state = torch.randn(batch, channels, width - 1, device="cuda", dtype=x.dtype) if with_state else None
    if state is not None and transposed:
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    before = [t.clone() for t in (x, weight, state) if t is not None]
    padded = F.pad(x, (width - 1, 0)) if state is None else torch.cat([state, x], -1)
    accumulated = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(width):
        accumulated += padded[:, :, tap : tap + sequence].float() * weight[:, 0, tap][None, :, None].float()
    reference = F.silu(accumulated).to(x.dtype)
    with torch.inference_mode():
        output = causal_conv1d_silu(x, weight, state)
    torch.testing.assert_close(output, reference, rtol=0.008, atol=0.0001)
    for tensor, original in zip((t for t in (x, weight, state) if t is not None), before):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_cpu_is_ineligible():
    with pytest.raises(OpNotEligible):
        causal_conv1d_silu(torch.zeros(1, 3, 5), torch.zeros(3, 1, 4))


def test_silu_precedes_output_rounding():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.tensor([-0.10986328125, 0.84765625, 1.0234375, -1.046875], device="cuda", dtype=torch.bfloat16).reshape(
        1, 1, 4
    )
    weight = torch.tensor([-0.4140625, 0.103515625, -1.71875, 0.72265625], device=x.device, dtype=x.dtype).reshape(
        1, 1, 4
    )
    with torch.inference_mode():
        output = causal_conv1d_silu(x, weight)
    # The intermediate BF16 conv store would give -0.2021484375 instead.
    assert output[0, 0, -1].item() == -0.201171875


def test_chunked_prefill_keeps_raw_input_tail():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.randn(2, 31, 29, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(31, 1, 4, device=x.device, dtype=x.dtype)
    state = torch.zeros(2, 31, 3, device=x.device, dtype=x.dtype)
    pieces = []
    with torch.inference_mode():
        whole = causal_conv1d_silu(x, weight)
        for begin, end in ((0, 1), (1, 3), (3, 17), (17, 29)):
            chunk = x[:, :, begin:end]
            pieces.append(causal_conv1d_silu(chunk, weight, state))
            state = torch.cat([state, chunk], -1)[:, :, -3:]
        torch.testing.assert_close(torch.cat(pieces, -1), whole, rtol=0, atol=0)


def test_large_prefill_uses_unbounded_grid_axis():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # GLM warmup: 98,304 channel/token tiles exceed grid-Y's 65,535 limit.
    x = torch.ones(1, 3072, 8192, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(3072, 1, 4, device=x.device, dtype=x.dtype)
    with torch.inference_mode():
        output = causal_conv1d_silu(x, weight)
    assert bool((output[:, :, 3:] == 3.921875).all())


@pytest.mark.parametrize("invalid", ["dtype", "width", "state", "grad"])
def test_unsupported_inputs_decline(invalid):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.zeros(1, 3, 5, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(3, 1, 4, device=x.device, dtype=x.dtype)
    state = None
    if invalid == "dtype":
        x = x.float()
    elif invalid == "width":
        weight = weight[:, :, :1]
    elif invalid == "state":
        state = torch.zeros(1, 3, 4, device=x.device, dtype=x.dtype)
    else:
        x.requires_grad_()
    with pytest.raises(OpNotEligible):
        causal_conv1d_silu(x, weight, state)


def test_graph_replay_reads_new_input_and_state():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.randn(2, 19, 13, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(19, 1, 4, device=x.device, dtype=x.dtype)
    state = torch.randn(2, 19, 3, device=x.device, dtype=x.dtype)
    with torch.inference_mode():
        causal_conv1d_silu(x, weight, state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = causal_conv1d_silu(x, weight, state)
        x.add_(1)
        state.add_(2)
        graph.replay()
        torch.testing.assert_close(output, causal_conv1d_silu(x, weight, state), rtol=0, atol=0)
