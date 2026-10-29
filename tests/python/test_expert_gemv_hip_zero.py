"""Explicit HIP/FNUZ skip contract; active arithmetic remains ordinary GEMV."""
from types import SimpleNamespace

import pytest
import torch

from vkernels.torch_ops import glm_expert_gemv as op


def cpu_inputs(storage="fnuz"):
    dtype = torch.float8_e4m3fnuz if storage == "fnuz" else torch.float8_e4m3fn
    return (torch.ones(2, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, 128, dtype=torch.uint8).view(dtype),
            torch.ones(2, 1, 1), torch.zeros(2, 2, dtype=torch.int64),
            torch.ones(2, 2))


@pytest.mark.parametrize("hip,arch,storage,selected,accepted", [
    ("7", "gfx942", "fnuz", "portable_fnuz", True),
    ("7", "gfx942:xnack-", "fnuz", "portable", True),
    ("7", "gfx950", "fnuz", "portable_fnuz", False),
    ("7", "gfx942", "fn", "portable", False),
    (None, "unused", "fnuz", "portable", False),
    (None, "unused", "fn", "cuda_native", True),
    (None, "unused", "fn", "portable", False),
])
def test_arch_storage_and_registered_backend_contract(monkeypatch, hip, arch, storage, selected, accepted):
    values = cpu_inputs(storage)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName=arch))
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return "launched"

    monkeypatch.setattr(op, "_select_implementation", lambda *args: SimpleNamespace(name=selected, load=lambda: launch))
    call = lambda: op.expert_gemv_skip_zero(*values, storage="e4m3" + storage)
    if accepted:
        assert call() == "launched"
        assert calls[0][1]["route_weights"] is values[-1]
    else:
        with pytest.raises(op.OpNotEligible):
            call()
        assert not calls


@pytest.mark.parametrize("invalid", ("shape", "dtype", "stride"))
def test_hip_routes_keep_original_metadata_contract(monkeypatch, invalid):
    values = list(cpu_inputs())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "hip", "7")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName="gfx942"))
    if invalid == "shape":
        values[-1] = torch.ones(1, 2)
    elif invalid == "dtype":
        values[-1] = values[-1].to(torch.bfloat16)
    else:
        values[-1] = torch.ones(2, 4)[:, ::2]
    with pytest.raises(op.OpNotEligible, match="contiguous same-device FP32"):
        op.expert_gemv_skip_zero(*values, storage="e4m3fnuz")


@pytest.mark.parametrize("batch", (1, 2, 4, 8))
@pytest.mark.parametrize("per_route_input", (False, True))
@pytest.mark.parametrize("native_decode", (False, True))
def test_actual_gfx942_active_exact_and_changed_routes(monkeypatch, batch, per_route_input, native_decode):
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("Requires actual gfx942")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("Requires actual gfx942")
    pytest.importorskip("triton")
    monkeypatch.setenv("GLM53_GEMV_FNUZ_NATIVE", "1" if native_decode else "0")
    torch.manual_seed(528)
    x = torch.randn((batch, 4, 128) if per_route_input else (batch, 128),
                    device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn((4, 256, 128), device="cuda") * .2).to(torch.float8_e4m3fnuz)
    scales = torch.full((4, 2, 1), .05, device="cuda")
    ids = torch.arange(batch * 4, device="cuda", dtype=torch.int64).reshape(batch, 4) % 4
    routes = torch.ones((batch, 4), device="cuda")

    def candidate():
        return op.expert_gemv_skip_zero(x, weight, scales, ids, routes, storage="e4m3fnuz", t_cap=8)

    def expected():
        ordinary = op.expert_gemv(x, weight, scales, ids, storage="e4m3fnuz", t_cap=8)
        return torch.where(routes[..., None] == 0, torch.zeros_like(ordinary), ordinary)

    for _ in range(3):
        candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    for iteration in range(5):
        x.normal_()
        ids.copy_((ids + 1) % 4)
        routes.fill_(1.)
        if iteration == 1:
            routes.zero_()
        elif iteration in (2, 3):
            routes[:, iteration % 2::2] = 0.
        else:
            ids.zero_()  # Repeated expert indices remain valid.
        eager, reference = candidate(), expected()
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for value in (eager, captured):
            assert torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
