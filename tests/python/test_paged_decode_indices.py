"""CSR slot packing is exact for scattered slots, ragged rows and graph replay."""
import pytest
import torch

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.paged_decode_indices import pack_paged_decode_indices


def test_cpu_miss_never_writes():
    out = torch.full((6,), -1, dtype=torch.int32)
    with pytest.raises(OpNotEligible, match='one GPU'):
        pack_paged_decode_indices(torch.zeros(2, 4, dtype=torch.int32), torch.tensor([0, 2, 6], dtype=torch.int32), out)
    assert out.tolist() == [-1]*6


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a Triton GPU')


@gpu
@pytest.mark.parametrize('batch', [1, 3, 8, 64, 128, 256])
@pytest.mark.parametrize('width', [17, 256, 4096, 8192])
def test_exact_scattered_metadata_with_noncontiguous_rows(batch, width):
    torch.manual_seed(73)
    backing = torch.randint(0, 893952, (batch, width+19), dtype=torch.int32, device='cuda')
    table = backing[:, :width]
    lengths = [width if lane == 0 else (lane*17)%width+1 for lane in range(batch)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1]+length)
    indptr = torch.tensor(offsets, dtype=torch.int32, device='cuda')
    guarded = torch.full((offsets[-1]+2,), -31, dtype=torch.int32, device='cuda')
    before = table.clone()
    pack_paged_decode_indices(table, indptr, guarded[1:-1])
    expected = torch.cat([table[lane, :length] for lane, length in enumerate(lengths)])
    assert torch.equal(guarded[1:-1], expected)
    assert guarded[0].item() == guarded[-1].item() == -31
    assert torch.equal(table, before)


@gpu
def test_capture_replay_reads_current_slot_ids():
    table = torch.arange(128, dtype=torch.int32, device='cuda').reshape(4, 32)
    indptr = torch.tensor([0, 1, 8, 40, 42], dtype=torch.int32, device='cuda')
    out = torch.empty(42, dtype=torch.int32, device='cuda')
    pack_paged_decode_indices(table, indptr, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pack_paged_decode_indices(table, indptr, out)
    for increment in [13, 71]:
        table.add_(increment)
        graph.replay()
        assert torch.equal(out, torch.cat([table[0, :1], table[1, :7], table[2], table[3, :2]]))


@gpu
@pytest.mark.parametrize('fault', ['alias', 'dtype', 'shape', 'batch', 'width'])
def test_invalid_shapes_or_aliases_reject_before_writes(fault):
    table = torch.zeros(4, 32, dtype=torch.int32, device='cuda')
    indptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device='cuda')
    out = torch.full((4,), -1, dtype=torch.int32, device='cuda')
    if fault == 'alias':
        out = table[0, :4]
    elif fault == 'dtype':
        table = table.to(torch.int64)
    elif fault == 'shape':
        indptr = indptr[:3]
    elif fault == 'batch':
        table = torch.zeros(257, 32, dtype=torch.int32, device='cuda')
    else:
        table = torch.zeros(4, 8193, dtype=torch.int32, device='cuda')
    before = out.clone()
    with pytest.raises(OpNotEligible):
        pack_paged_decode_indices(table, indptr, out)
    assert torch.equal(out, before)


@gpu
def test_zero_length_rows_preserve_csr_offsets():
    table = torch.arange(32, dtype=torch.int32, device='cuda').reshape(4, 8)
    indptr = torch.tensor([0, 0, 1, 3, 3], dtype=torch.int32, device='cuda')
    out = torch.empty(3, dtype=torch.int32, device='cuda')
    pack_paged_decode_indices(table, indptr, out)
    assert torch.equal(out, torch.cat([table[1, :1], table[2, :2]]))
