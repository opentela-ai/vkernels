"""Small-batch BF16 matrix eligibility, FP64 oracle and graph input lifetimes."""
import subprocess
import sys

import pytest
import torch

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.bf16_small_batch import dense_bf16_small_batch
from vkernels.torch_ops.glm_gemv import _is_gfx942

SHAPES = [(21504, 4096), (4096, 21504), (10752, 8192), (8192, 10752),
          (6144, 4096), (4096, 4096), (2560, 8192), (8192, 2048),
          (131072, 4096), (131072, 8192)]
GFX942 = torch.cuda.is_available() and _is_gfx942(torch.device("cuda"))


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import vkernels.torch_ops.bf16_small_batch; "
        "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.mark.parametrize("shape", [(1, 8), (5, 8), (2, 8)])
def test_cpu_and_unlisted_contracts_raise_for_caller_fallback(shape):
    with pytest.raises(OpNotEligible):
        dense_bf16_small_batch(torch.ones(shape, dtype=torch.bfloat16), torch.ones(16, 8, dtype=torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mapped_matrices_require_cuda_without_materializing_weights(dtype):
    x = torch.empty(2, 4096, device="meta", dtype=dtype)
    weight = torch.empty(21504, 4096, device="meta", dtype=dtype)
    with pytest.raises(OpNotEligible, match="CUDA"):
        dense_bf16_small_batch(x, weight)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or GFX942, reason="requires an unsupported CUDA architecture")
def test_mapped_matrix_rejects_other_gpu_architectures():
    x = torch.empty(2, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(21504, 4096, device=x.device, dtype=x.dtype)
    with pytest.raises(OpNotEligible, match="gfx942"):
        dense_bf16_small_batch(x, weight)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason="MFMA projection pins require gfx942")
@pytest.mark.parametrize("outputs,width", SHAPES)
@pytest.mark.parametrize("batch", [2, 3, 4])
def test_fp64_oracle_read_only_operands_and_updated_graph_inputs(outputs, width, batch):
    torch.manual_seed(415)
    x = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(outputs, width, device="cuda") * .02).bfloat16()
    saved_x, saved_w = x.clone(), w.clone()
    actual = dense_bf16_small_batch(x, w)
    expected = x.double() @ w.double().T
    torch.testing.assert_close(actual.double(), expected, rtol=1/128, atol=2e-5)
    assert torch.equal(x, saved_x) and torch.equal(w, saved_w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = dense_bf16_small_batch(x, w)
    for change in ("weight", "input"):
        if change == "weight":
            w[:, -1].add_(.125)
            saved_w[:, -1].add_(.125)
        else:
            x.copy_(torch.randn_like(x))
            saved_x.copy_(x)
        expected = x.double() @ w.double().T
        graph.replay()
        torch.testing.assert_close(replayed.double(), expected, rtol=1/128, atol=2e-5)
        assert torch.equal(x, saved_x) and torch.equal(w, saved_w)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason="MFMA projection pins require gfx942")
@pytest.mark.parametrize("outputs,width", SHAPES)
def test_last_reduction_column_and_batch_independence(outputs, width):
    x = torch.zeros(4, width, device="cuda", dtype=torch.bfloat16)
    x[:, -1] = torch.tensor([1., -1., .5, -2.], device="cuda", dtype=x.dtype)
    w = torch.zeros(outputs, width, device="cuda", dtype=x.dtype)
    w[:, -1] = torch.linspace(-1, 1, outputs, device="cuda", dtype=x.dtype)
    full = dense_bf16_small_batch(x, w)
    torch.testing.assert_close(full, x[:, -1:] * w[:, -1][None], rtol=0, atol=0)
    for batch in (2, 3):
        torch.testing.assert_close(dense_bf16_small_batch(x[:batch], w), full[:batch], rtol=0, atol=0)
    x.normal_()
    w.normal_(std=.02)
    full = dense_bf16_small_batch(x, w)
    for batch in (2, 3):
        torch.testing.assert_close(dense_bf16_small_batch(x[:batch], w), full[:batch], rtol=0, atol=0)
    with pytest.raises(OpNotEligible):
        dense_bf16_small_batch(x[:1], w)
    with torch.enable_grad(), pytest.raises(OpNotEligible):
        dense_bf16_small_batch(x.requires_grad_(), w)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_unlisted_matrix_is_rejected_for_the_caller_fallback():
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 128, device="cuda", dtype=x.dtype)
    with pytest.raises(OpNotEligible):
        dense_bf16_small_batch(x, w)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason="high-concurrency matrices require gfx942")
@pytest.mark.parametrize("outputs,width", [shape for shape in SHAPES if shape != (21504,4096)] + [(5376,4096),(4096,5376),(1536,4096),(4096,1024)])
@pytest.mark.parametrize("batch", [5,8,16,17,32])
def test_high_batch_fp64_oracle_and_graph_inputs(outputs, width, batch):
    test_fp64_oracle_read_only_operands_and_updated_graph_inputs(outputs, width, batch)


@pytest.mark.parametrize("batch", [1,33])
def test_high_batch_limits_preserve_fallback(batch):
    x = torch.empty(batch,4096,device="meta",dtype=torch.bfloat16)
    weight = torch.empty(5376,4096,device="meta",dtype=x.dtype)
    with pytest.raises(OpNotEligible, match="no measured"):
        dense_bf16_small_batch(x,weight)
