"""Exact integer staging, padding, dynamic rosters and graph replay safety."""

import pytest
import torch

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.paged_decode_inputs import stage_paged_decode_inputs, warmup_paged_decode_inputs


def buffers(batch, width, device):
    return (torch.full((batch, 1), -19, dtype=torch.int64, device=device),
            torch.full((batch, 1), -23, dtype=torch.int64, device=device),
            torch.full((batch,), -29, dtype=torch.int64, device=device),
            torch.full((batch, width), -31, dtype=torch.int32, device=device),
            torch.full((batch,), -37, dtype=torch.int64, device=device))


def eager(inputs, outputs, scratch):
    ids, pos, lens, table, pick = inputs
    batch, width = ids.shape[0], min(table.shape[1], outputs[3].shape[1])
    for index in [0, 1, 2, 4]:
        outputs[index].zero_()
        outputs[index][:batch] = inputs[index]
    outputs[3].fill_(scratch)
    outputs[3][:batch, :width] = table[:, :width]


def test_cpu_miss_leaves_outputs_untouched():
    inputs, outputs = buffers(1, 16, "cpu"), buffers(4, 32, "cpu")
    before = [t.clone() for t in outputs]
    with pytest.raises(OpNotEligible, match="one GPU"):
        stage_paged_decode_inputs(*inputs, outputs, scratch_slot=7)
    assert all(torch.equal(a, b) for a, b in zip(before, outputs))


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Triton GPU")


@gpu
@pytest.mark.parametrize("live,static", [(1, 1), (3, 4), (4, 4)])
@pytest.mark.parametrize("source_width,graph_width", [(17, 32), (512, 256), (3328, 3328)])
def test_exact_staging_with_strided_table_and_live_shape_changes(live, static, source_width, graph_width):
    outputs = buffers(static, graph_width, "cuda")
    expected = buffers(static, graph_width, "cuda")
    before = [t.clone() for t in outputs]
    warmup_paged_decode_inputs(outputs, scratch_slot=127)
    assert all(torch.equal(a, b) for a, b in zip(before, outputs))
    # Same graph buffers survive a smaller next roster/table. Previously
    # written rows and columns must become zeros/scratch, never stale slots.
    for batch, width in [(live, source_width), (1, min(source_width, 13))]:
        inputs = list(buffers(batch, width, "cuda"))
        for index in [0, 1, 2, 4]:
            inputs[index].copy_(torch.arange(inputs[index].numel(), device="cuda").reshape_as(inputs[index]) + (1 << 35) + index)
        backing = torch.arange(batch * (source_width + 19), device="cuda", dtype=torch.int32).reshape(batch, -1)
        inputs[3] = backing[:, :width]
        sources = [t.clone() for t in inputs]
        eager(inputs, expected, 127)
        stage_paged_decode_inputs(*inputs, outputs, scratch_slot=127)
        assert all(torch.equal(a, b) for a, b in zip(expected, outputs))
        assert all(torch.equal(a, b) for a, b in zip(sources, inputs))


@gpu
def test_staging_rejects_aliases_and_invalid_dtype_before_writes():
    outputs = buffers(4, 32, "cuda")
    with pytest.raises(OpNotEligible, match="must not overlap"):
        stage_paged_decode_inputs(*outputs, outputs, scratch_slot=0)
    inputs = list(buffers(2, 16, "cuda"))
    inputs[0] = inputs[0].float()
    before = [t.clone() for t in outputs]
    with pytest.raises(OpNotEligible, match="layout/dtypes"):
        stage_paged_decode_inputs(*inputs, outputs, scratch_slot=0)
    assert all(torch.equal(a, b) for a, b in zip(before, outputs))


@gpu
def test_staging_capture_replay_uses_current_values():
    inputs, outputs = buffers(3, 64, "cuda"), buffers(4, 128, "cuda")
    stage_paged_decode_inputs(*inputs, outputs, scratch_slot=61)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage_paged_decode_inputs(*inputs, outputs, scratch_slot=61)
    for step in range(2):
        for tensor in inputs:
            tensor.add_(step + 3)
        graph.replay()
        expected = buffers(4, 128, "cuda")
        eager(inputs, expected, 61)
        assert all(torch.equal(a, b) for a, b in zip(expected, outputs))


@gpu
@pytest.mark.parametrize('live,static', [(65, 128), (127, 128), (128, 128), (129, 256), (255, 256), (256, 256)])
@pytest.mark.parametrize('source_width,graph_width', [(17, 4096), (4099, 4096)])
def test_large_batch_staging_keeps_exact_metadata_after_roster_shrink(live, static, source_width, graph_width):
    outputs, expected = [buffers(static, graph_width, 'cuda') for _ in range(2)]
    before = [tensor.clone() for tensor in outputs]
    warmup_paged_decode_inputs(outputs, scratch_slot=113)
    assert all(torch.equal(left, right) for left, right in zip(before, outputs))
    for batch, width in [(live, source_width), (3, 13)]:
        inputs = list(buffers(batch, width, 'cuda'))
        for index in [0, 1, 2, 4]:
            inputs[index].copy_(torch.arange(inputs[index].numel(), device='cuda').reshape_as(inputs[index])+(1 << 35)+index)
        backing = torch.arange(batch*(width+19), device='cuda', dtype=torch.int32).reshape(batch, -1)
        inputs[3] = backing[:, :width]
        eager(inputs, expected, 113)
        stage_paged_decode_inputs(*inputs, outputs, scratch_slot=113)
        assert all(torch.equal(left, right) for left, right in zip(outputs, expected))


@gpu
def test_batch_above_256_still_rejects_without_writes():
    outputs = buffers(257, 16, 'cuda')
    before = [tensor.clone() for tensor in outputs]
    with pytest.raises(OpNotEligible, match='geometry'):
        stage_paged_decode_inputs(*buffers(1, 16, 'cuda'), outputs, scratch_slot=0)
    assert all(torch.equal(left, right) for left, right in zip(outputs, before))
