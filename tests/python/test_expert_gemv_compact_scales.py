"""Default-off compact scale loads retain original expert GEMV bits."""
from types import SimpleNamespace

import pytest
import torch

from vkernels.torch_ops import glm_expert_gemv as gemv


@pytest.mark.parametrize('hip,arch,storage,setting,expected', [
    (None, 'gfx942', 'float8_e4m3fnuz', '1', False),
    ('7.2', 'gfx942', 'float8_e4m3fnuz', None, False),
    ('7.2', 'gfx942', 'float8_e4m3fnuz', '0', False),
    ('7.2', 'gfx942', 'float8_e4m3fnuz', '1', True),
    ('7.2', 'gfx942:sramecc+:xnack-', 'float8_e4m3fnuz', '1', True),
    ('7.2', 'gfx950', 'float8_e4m3fnuz', '1', False),
    ('7.2', 'gfx942', 'float8_e4m3fn', '1', False),
])
def test_compact_scales_are_explicit_and_target_bounded(monkeypatch, hip, arch, storage, setting, expected):
    monkeypatch.setattr(torch.version, 'hip', hip)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(gcnArchName=arch))
    if setting is None:
        monkeypatch.delenv('GLM53_GEMV_FNUZ_COMPACT_SCALES', raising=False)
    else:
        monkeypatch.setenv('GLM53_GEMV_FNUZ_COMPACT_SCALES', setting)
    assert gemv._compact_scales_enabled(SimpleNamespace(device='cuda:0'), SimpleNamespace(dtype=getattr(torch, storage))) is expected


def _gfx942():
    return (torch.version.hip and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx942')


def _exact(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.skipif(not _gfx942(), reason='requires gfx942')
@pytest.mark.parametrize('batch', [1, 2, 4, 8])
@pytest.mark.parametrize('output,width,broadcast', [(1024, 4096, True), (4096, 512, False)])
@pytest.mark.parametrize('native', [False, True])
def test_compact_scale_stage_bits_and_changed_poisoned_replay(monkeypatch, batch, output, width, broadcast, native):
    device = torch.device('cuda', torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(8107 + batch + output)
    weights = (torch.randn(17, output, width, generator=generator, device=device) * .03).to(torch.float8_e4m3fnuz)
    scales = torch.rand(17, output // 128, width // 128, generator=generator, device=device) * .5 + .1
    x = torch.randn((batch, width) if broadcast else (batch, 9, width), generator=generator, device=device, dtype=torch.bfloat16)
    ids = torch.arange(batch * 9, device=device, dtype=torch.int64).reshape(batch, 9) % 17

    def original():
        monkeypatch.setenv('GLM53_GEMV_FNUZ_NATIVE', '0')
        monkeypatch.setenv('GLM53_GEMV_FNUZ_COMPACT_SCALES', '0')
        return gemv._launch_portable(x, weights, scales, ids)

    def candidate():
        monkeypatch.setenv('GLM53_GEMV_FNUZ_NATIVE', str(int(native)))
        monkeypatch.setenv('GLM53_GEMV_FNUZ_COMPACT_SCALES', '1')
        return gemv._launch_portable(x, weights, scales, ids)

    for _ in range(3):
        _exact(original(), candidate())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    for phase in range(3):
        x.copy_(torch.randn(x.shape, generator=generator, device=device, dtype=torch.bfloat16) * (phase + 1))
        ids.copy_((ids + phase * 7 + 1) % 17)
        expected = original()
        _exact(candidate(), expected)
        captured.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        _exact(captured, expected)


@pytest.mark.skipif(not _gfx942(), reason='requires gfx942')
@pytest.mark.parametrize('native,compact', [(False, False), (False, True), (True, False), (True, True)])
def test_all_256_fnuz_encodings_preserve_finite_bits_and_nan_class(monkeypatch, native, compact):
    device = torch.device('cuda', torch.cuda.current_device())
    raw = torch.arange(256, dtype=torch.int16, device=device).to(torch.uint8)
    weights = raw[:, None].expand(256, 128).contiguous().reshape(1, 256, 128).view(torch.float8_e4m3fnuz)
    scales = torch.ones(1, 2, 1, device=device, dtype=torch.float32)
    x = torch.zeros(1, 128, device=device, dtype=torch.bfloat16)
    x[0, 0] = 1
    ids = torch.zeros(1, 1, device=device, dtype=torch.int64)
    reference = []
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        value = mantissa * 2. ** -10 if exponent == 0 else (1 + mantissa / 8) * 2. ** (exponent - 8)
        reference.append(float('nan') if byte == 128 else (-value if byte & 128 else value))
    expected = torch.tensor(reference, dtype=torch.bfloat16, device=device).reshape(1, 1, 256)
    monkeypatch.setenv('GLM53_GEMV_FNUZ_NATIVE', str(int(native)))
    monkeypatch.setenv('GLM53_GEMV_FNUZ_COMPACT_SCALES', str(int(compact)))
    actual = gemv._launch_portable(x, weights, scales, ids)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.isnan(actual).sum().item() == 1
    finite = torch.isfinite(expected)
    _exact(actual[finite], expected[finite])
