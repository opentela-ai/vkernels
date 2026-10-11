"""Offline capability planning and actual MoE dispatch use the same contract."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vkernels.registry import KernelImplementation, KernelRegistry, KernelRequest, KernelSelectionError, TensorMetadata
from vkernels.torch_ops.moe_combine_registry import MOE_COMBINE_REGISTRY


def _request(**kwargs):
    return KernelRequest(
        "moe_weighted_sum",
        (TensorMetadata((2, 4, 128), "bfloat16", "cuda:0"),
         TensorMetadata((2, 4), "float32", "cuda:0")),
        **kwargs,
    )


def test_import_and_explain_without_gpu_dependencies():
    script = '''
import sys
class BlockGPU:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton'}:
            raise AssertionError('unexpected GPU import: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from vkernels.torch_ops.moe_combine_registry import MOE_COMBINE_REGISTRY, KernelRequest, TensorMetadata
request = KernelRequest('moe_weighted_sum', (
    TensorMetadata((2, 4, 128), 'bfloat16', 'cuda:0'),
    TensorMetadata((2, 4), 'float32', 'cuda:0')),
    backends=frozenset({'triton'}), graph_capture=True)
assert MOE_COMBINE_REGISTRY.select(request).name == 'triton'
assert not {'torch', 'triton'}.intersection(sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src/python"))
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True)


def test_cache_and_explicit_override():
    request = _request(backends=frozenset({"triton"}), graph_capture=True)
    first = MOE_COMBINE_REGISTRY.explain(request)
    assert MOE_COMBINE_REGISTRY.explain(request) is first
    assert first.selected.name == "triton"
    assert MOE_COMBINE_REGISTRY.select(request, override="triton") is first.selected
    with pytest.raises(KernelSelectionError, match="unknown implementation: absent"):
        MOE_COMBINE_REGISTRY.select(request, override="absent")
    with pytest.raises(KernelSelectionError, match="backend 'triton' unavailable"):
        MOE_COMBINE_REGISTRY.select(replace(request, backends=frozenset()))


def test_reference_is_never_implicit_fallback():
    request = _request(backends=frozenset({"torch"}))
    with pytest.raises(KernelSelectionError, match="requires allow_reference=True"):
        MOE_COMBINE_REGISTRY.select(request)
    with pytest.raises(KernelSelectionError):
        MOE_COMBINE_REGISTRY.select(request, override="torch_reference")
    assert MOE_COMBINE_REGISTRY.select(request, override="torch_reference", allow_reference=True).reference


@pytest.mark.parametrize("tensors,reason", [
    ((TensorMetadata((2, 4, 128), "bfloat16", "cuda:0"),
      TensorMetadata((2, 4), "float32", "cpu")), "share a device"),
    ((TensorMetadata((2, 4, 128), "bfloat16", "cuda:0"),
      TensorMetadata((2, 3), "float32", "cuda:0")), "activations"),
    ((TensorMetadata((2, 4, 128), "int8", "cuda:0"),
      TensorMetadata((2, 4), "float32", "cuda:0")), "bfloat16"),
])
def test_metadata_contract_misses(tensors, reason):
    request = replace(_request(backends=frozenset({"triton"})), tensors=tensors)
    with pytest.raises(KernelSelectionError, match=reason):
        MOE_COMBINE_REGISTRY.select(request)


def test_priority_ties_architecture_workspace_and_capture():
    def check(_):
        return ()
    base = KernelImplementation("test", "z", "unimportable:launcher", check, "custom")
    impls = (base, replace(base, name="a"), replace(base, name="fast", priority=10,
             architectures=("sm100",), workspace_bytes=64, graph_capture=True))
    registry = KernelRegistry(impls)
    request = KernelRequest("test", (), backends=frozenset({"custom"}))
    assert registry.select(request).name == "a"
    assert KernelRegistry(tuple(reversed(impls))).select(request).name == "a"
    assert "architecture 'unknown' unsupported" in str(registry.explain(request))
    assert registry.select(replace(request, architecture="sm100", workspace_bytes=64)).name == "fast"
    with pytest.raises(KernelSelectionError, match="requires 64 workspace bytes"):
        registry.select(replace(request, graph_capture=True))
    with pytest.raises(ValueError, match="duplicate"):
        KernelRegistry((base, base))


def test_reference_numerics_and_validation():
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops.moe_combine import moe_weighted_sum_reference
    from vkernels.torch_ops import OpNotEligible
    out = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    weights = torch.randn(2, 4)
    assert torch.equal(moe_weighted_sum_reference(out, weights),
                       (out * weights.to(out.dtype).unsqueeze(-1)).sum(dim=1))
    with pytest.raises(OpNotEligible):
        moe_weighted_sum_reference(out, weights[:, :2])


def test_cuda_dispatch_override_and_graph_replay():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.moe_combine import _kernel, moe_weighted_sum, moe_weighted_sum_reference
    from vkernels.torch_ops import OpNotEligible
    out = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(2, 4, device="cuda")
    result = torch.empty(2, 128, device="cuda", dtype=out.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            moe_weighted_sum(out, weights, result=result, implementation="triton")
    torch.cuda.current_stream().wait_stream(stream)
    assert _kernel() is _kernel()
    with pytest.raises(KernelSelectionError, match="unknown implementation"):
        moe_weighted_sum(out, weights, implementation="bad")
    with pytest.raises(OpNotEligible):
        moe_weighted_sum(out, weights.cpu())
    with pytest.raises(OpNotEligible, match="result must match"):
        moe_weighted_sum(out, weights, result=result.float())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = moe_weighted_sum(out, weights, result=result)
    out.normal_()
    graph.replay()
    assert captured is result
    torch.testing.assert_close(result, moe_weighted_sum_reference(out, weights), rtol=2e-2, atol=2e-2)


def test_pinned_input_contract_miss_is_not_a_fallback():
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops.moe_combine import moe_weighted_sum
    from vkernels.torch_ops import OpNotEligible
    out, weights = torch.empty(2, 4, 128), torch.empty(2, 4)
    with pytest.raises(KernelSelectionError) as exc:
        moe_weighted_sum(out, weights, implementation="triton")
    assert not isinstance(exc.value, OpNotEligible)
    with pytest.raises(OpNotEligible):
        moe_weighted_sum(out, weights)


def test_cuda_pinned_unavailable_backend_is_not_a_fallback(monkeypatch):
    import importlib
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    combine = importlib.import_module("vkernels.torch_ops.moe_combine")
    from vkernels.torch_ops import OpNotEligible
    monkeypatch.setattr(combine, "_backends", lambda: frozenset({"torch"}))
    out, weights = torch.empty(2, 4, 128, device="cuda"), torch.empty(2, 4, device="cuda")
    with pytest.raises(KernelSelectionError, match="unavailable") as exc:
        combine.moe_weighted_sum(out, weights, implementation="triton")
    assert not isinstance(exc.value, OpNotEligible)
    with pytest.raises(OpNotEligible):
        combine.moe_weighted_sum(out, weights)


def test_cuda_output_views_are_validated_before_launch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.moe_combine import moe_weighted_sum
    from vkernels.torch_ops import OpNotEligible
    out = torch.randn(2, 4, 128, device="cuda")
    weights = torch.randn(2, 4, device="cuda")
    # Expanded output rows and overlapping as_strided output would race stores.
    expanded = torch.empty(1, 128, device="cuda").expand(2, -1)
    overlapping = torch.empty(129, device="cuda").as_strided((2, 128), (1, 1))
    transposed = torch.empty(128, 2, device="cuda").T
    for result in (expanded, overlapping, transposed):
        with pytest.raises(OpNotEligible, match="contiguous"):
            moe_weighted_sum(out, weights, result=result)
    # Contiguous output may still overwrite inputs read by other CTAs.
    with pytest.raises(OpNotEligible, match="must not overlap"):
        moe_weighted_sum(out, weights, result=out.reshape(-1)[:256].view(2, 128))
    weights_workspace = torch.empty(264, device="cuda")
    alias_weights = weights_workspace[128:136].view(2, 4)
    with pytest.raises(KernelSelectionError, match="must not overlap"):
        moe_weighted_sum(out, alias_weights, result=weights_workspace[:256].view(2, 128), implementation="triton")


def test_cuda_disjoint_workspace_views_and_strided_inputs_replay():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.moe_combine import moe_weighted_sum
    # Offsets must be honored: allocation identity alone would reject these.
    workspace = torch.randn(1024 + 8 + 256, device="cuda")
    out = workspace[:1024].view(4, 2, 128).transpose(0, 1)
    weights = workspace[1024:1032].view(4, 2).T
    result = workspace[1032:].view(2, 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            moe_weighted_sum(out, weights, result=result)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_weighted_sum(out, weights, result=result)
    out.normal_()
    graph.replay()
    torch.testing.assert_close(result, (out * weights.unsqueeze(-1)).sum(1), rtol=1e-5, atol=1e-5)


def test_priority_bands_validate_and_classify():
    from vkernels.registry import Priority

    def check(_):
        return ()

    base = KernelImplementation("test", "z", "unimportable:launcher", check, "custom")
    # Bands are contiguous and ordered; PLUGIN leaves headroom above SPECIALIZED.
    assert [int(b) for b in Priority] == [0, 4, 8, 12, 16]
    assert Priority.band_of(0) is Priority.REFERENCE
    assert Priority.band_of(10) is Priority.PERFORMANT
    assert Priority.band_of(19) is Priority.PLUGIN
    # In-band offsets are legal...
    KernelRegistry((replace(base, name="a", priority=Priority.PERFORMANT + 3),))
    # ...but leaving [0, 20) is rejected at registry construction.
    with pytest.raises(ValueError, match="Priority band"):
        KernelRegistry((replace(base, name="a", priority=100),))
    with pytest.raises(ValueError, match="Priority band"):
        KernelRegistry((replace(base, name="a", priority=-1),))


def test_format_signature_gating():
    from vkernels.signature import FormatSignature

    # Construction validation.
    with pytest.raises(ValueError, match="align"):
        FormatSignature(("a", "b"), (frozenset({"float32"}),))
    with pytest.raises(ValueError, match="no dtype"):
        FormatSignature(("a",), (frozenset(),))

    sig = FormatSignature(
        ("x", "w"), (frozenset({"bfloat16"}), frozenset({"float32", "float8_e4m3fn"}))
    )
    assert sig.matches((TensorMetadata((2,), "bfloat16", "cpu"),
                        TensorMetadata((2,), "float32", "cpu"))) == ()
    assert sig.matches((TensorMetadata((2,), "bfloat16", "cpu"),)) == (
        "expected 2 tensors (x, w), got 1",
    )
    reasons = sig.matches((TensorMetadata((2,), "float32", "cpu"),
                           TensorMetadata((2,), "int64", "cpu")))
    assert reasons == ("x requires bfloat16", "w requires float32/float8_e4m3fn")
    assert sig.describe() == "dense: x<bfloat16> w<float32/float8_e4m3fn>"

    # A signature-bearing registration surfaces those reasons in explain().
    def check(_):
        return ()

    impl = KernelImplementation(
        "sig_op", "x", "unimportable:launcher", check, "custom",
        signature=FormatSignature(("x",), (frozenset({"bfloat16"}),)),
    )
    registry = KernelRegistry((impl,))
    request = KernelRequest("sig_op", (TensorMetadata((4,), "float32", "cpu"),),
                            backends=frozenset({"custom"}))
    assert "x requires bfloat16" in str(registry.explain(request))
    ok_request = KernelRequest("sig_op", (TensorMetadata((4,), "bfloat16", "cpu"),),
                               backends=frozenset({"custom"}))
    assert registry.select(ok_request).name == "x"
