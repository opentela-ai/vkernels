"""SGL decode numeric mode: the single-launch region must match the donor glue.

The prefill region and the decode glue share block-FP8 GEMM arithmetic, but
not a precision boundary. ``numerics='sgl_decode'`` reproduces the SGLang
decode donor's boundaries (amax floor 1e-10, reciprocal multiply, BF16
round-trips at every clamp/SiLU/product stage, route weight in the down store,
plain fp32 reduction) with a small-row tile and EP zero-route skipping. These
tests pin it bitwise to the incumbent five-launch glue, which is itself
checked against the fused donor.
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.persistent_moe import PersistentMoE
from vkernels.torch_ops.sgl_moe import sgl_fused_moe
from vkernels.torch_ops.sgl_moe_decode_glue import sgl_fused_moe_decode_glue

gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                         reason="CUDA required")


def case(t, h, i, e, k, skip, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16, generator=g)
    gu = torch.randn(e, 2 * i, h, device="cuda", generator=g).to(torch.float8_e4m3fn)
    dn = torch.randn(e, h, i, device="cuda", generator=g).to(torch.float8_e4m3fn)
    gs = torch.rand(e, 2 * i // 128, h // 128, device="cuda", generator=g) * .05
    ds = torch.rand(e, h // 128, i // 128, device="cuda", generator=g) * .05
    ids = torch.randint(e, (t, k), device="cuda", generator=g)
    routes = torch.rand(t, k, device="cuda", generator=g)
    if skip:
        routes[:, ::2] = 0
    return x, gu, gs, dn, ds, ids, routes


@gpu
@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("limit", [0.01, 7.0])
@pytest.mark.parametrize("t", [1, 2, 4, 8])
def test_sgl_decode_matches_donor_glue_bitwise(t, limit, skip):
    values = case(t, 256, 128, 12, 8, skip, seed=417 + t)
    kwargs = dict(swiglu_limit=limit, skip_zero_weights=skip)
    oracle = sgl_fused_moe(*values, route_weight_in_down=True, **kwargs)
    glue = sgl_fused_moe_decode_glue(*values, **kwargs)
    torch.testing.assert_close(glue, oracle, rtol=0, atol=0)
    region = PersistentMoE(*values, numerics="sgl_decode", **kwargs)
    for _ in range(2):
        actual = region.run(values[0], values[-2], values[-1])
        torch.testing.assert_close(actual, glue, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("t", [1, 4])
def test_serving_geometry_matches_glue_and_issues_one_launch(t, tmp_path):
    import json

    values = case(t, 4096, 2048, 72, 8, True, seed=99 + t)
    kwargs = dict(swiglu_limit=7.0, skip_zero_weights=True)
    glue = sgl_fused_moe_decode_glue(*values, **kwargs)
    region = PersistentMoE(*values, numerics="sgl_decode", **kwargs)
    region.run(values[0], values[-2], values[-1])
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        actual = region.run(values[0], values[-2], values[-1])
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, glue, rtol=0, atol=0)
    path = tmp_path / "trace.json"
    profile.export_chrome_trace(str(path))
    kernels = [e for e in json.loads(path.read_text())["traceEvents"]
               if e.get("cat") == "kernel"]
    assert len(kernels) == 1
    assert kernels[0]["name"].startswith("region")


@gpu
def test_sgl_decode_rejects_conflicting_options(tmp_path):
    values = case(1, 256, 128, 12, 8, True, seed=7)
    with pytest.raises(ValueError, match="numerics"):
        PersistentMoE(*values, numerics="fused")
    with pytest.raises(ValueError, match="double-round"):
        PersistentMoE(*values, numerics="sgl_decode", round_products_to_bf16=True)
