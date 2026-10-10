"""The raw transfer ABI masks invalid live rows and preserves storage bits."""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.state_transfer import masked_state_transfer  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_invalid_live_rows_are_masked_in_graph_replay(dtype):
    pool = torch.empty((5, 2, 17), dtype=dtype, device="cuda")
    pool.view(torch.uint8).random_(0, 256)
    stage = torch.zeros((3, 2, 17), dtype=dtype, device="cuda")
    pointers = torch.tensor([stage[:, i].data_ptr() for i in range(2)], device="cuda")
    strides = torch.full((2,), stage.stride(0), device="cuda", dtype=torch.int64)
    layers = torch.tensor([1, 0], device="cuda")
    rows = torch.tensor([4, 999, -1], device="cuda")
    live = torch.tensor([3], device="cuda")
    raw = pool.view(torch.uint16 if dtype == torch.float16 else torch.uint32)
    def transfer(gather):
        masked_state_transfer(pointers, strides, layers, raw, live, rows,
                              size=17, pool_stride=pool.stride(0), pool_rows=5,
                              bits=pool.element_size() * 8, batch=3, gather=gather)
    transfer(True)
    transfer(False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer(True)
        transfer(False)
    before = pool.view(torch.uint8).clone()
    for row in (4, 0, 2):
        rows[0] = row
        stage.fill_(1)
        graph.replay()
        assert torch.equal(pool.view(torch.uint8), before)
        expected = pool[row, torch.tensor([1, 0], device="cuda")]
        assert torch.equal(stage[0].view(torch.uint8), expected.view(torch.uint8))
        assert torch.count_nonzero(stage[1:]) == 0
