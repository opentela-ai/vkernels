"""CPU mock checks for global TunableOp configuration/flag lifecycle."""

import importlib
import subprocess
import sys
from contextlib import nullcontext

import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import vkernels.torch_ops.qkv_tuned_blas; "
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import torch.cuda.tunable as tunable

    module = importlib.import_module("vkernels.torch_ops.qkv_tuned_blas")
    state = {"enabled": False, "tuning": True, "read_ok": True, "reads": 0}
    signature = "tn_8192_1_4096_ld_4096_4096_8192"
    state["results"] = [("GemmTunableOp_BFloat16_TN", signature, "Gemm_Hipblaslt_208015", 0.02)]
    monkeypatch.setattr(module, "_CONFIGURED", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(tunable, "is_enabled", lambda: state["enabled"])
    monkeypatch.setattr(tunable, "tuning_is_enabled", lambda: state["tuning"])
    monkeypatch.setattr(tunable, "enable", lambda value: state.update(enabled=value))
    monkeypatch.setattr(tunable, "tuning_enable", lambda value: state.update(tuning=value))

    def read_file(path):
        state["reads"] += 1
        assert state["enabled"] and not state["tuning"]
        if isinstance(state["read_ok"], Exception):
            raise state["read_ok"]
        return state["read_ok"]

    monkeypatch.setattr(tunable, "read_file", read_file)
    monkeypatch.setattr(tunable, "get_results", lambda: state["results"])
    artifact = tmp_path / "tuning.csv"
    # The validator set ``torch.cuda.tunable.write_file`` actually emits on
    # an AMD/HIP target (HIP_VERSION encodes ROCm major*100+minor; there is
    # no ROCM_VERSION validator). See the loader in qkv_tuned_blas.py and the
    # job-628932 regression test_accepts_genuine_pytorch_validator_set.
    artifact.write_text("".join(f"Validator,{key},test\n" for key in
                              ("PT_VERSION", "HIP_VERSION", "HIPBLASLT_VERSION", "GCN_ARCH_NAME", "ROCBLAS_VERSION"))
                        + f"GemmTunableOp_BFloat16_TN,{signature},Gemm_Hipblaslt_208015,0.02\n")
    return module, torch, state, artifact


@pytest.mark.parametrize("enabled,tuning", [(False, True), (True, False), (True, True), (False, False)])
def test_configuration_restores_flags(setup, enabled, tuning):
    module, _, state, artifact = setup
    state.update(enabled=enabled, tuning=tuning)
    assert module.configure_qkv_tuned_blas(artifact) == (1,)
    assert (state["enabled"], state["tuning"]) == (enabled, tuning)


@pytest.mark.parametrize("read_ok", [False, RuntimeError("read failed")])
def test_configuration_failure_restores_flags(setup, read_ok):
    module, _, state, artifact = setup
    state["read_ok"] = read_ok
    with pytest.raises(RuntimeError):
        module.configure_qkv_tuned_blas(artifact)
    assert not state["enabled"] and state["tuning"]
    assert module._CONFIGURED == {}


@pytest.mark.parametrize("kind", ["default", "validators", "loaded_mismatch", "missing_hip_version"])
def test_rejects_invalid_artifact(setup, kind):
    module, _, state, artifact = setup
    if kind == "default":
        artifact.write_text(artifact.read_text().replace("Gemm_Hipblaslt_208015", "Default"))
    elif kind == "validators":
        artifact.write_text(artifact.read_text().replace("Validator,PT_VERSION,test\n", ""))
    elif kind == "missing_hip_version":
        # A genuine PyTorch artifact MUST carry HIP_VERSION; the loader's
        # pre-check rejects a CSV that omits it (the old ROCM_VERSION bug).
        artifact.write_text(artifact.read_text().replace("Validator,HIP_VERSION,test\n", ""))
    else:
        state["results"] = []
    with pytest.raises((ValueError, RuntimeError)):
        module.configure_qkv_tuned_blas(artifact)
    assert not state["enabled"] and state["tuning"]
    assert module._CONFIGURED == {}


def test_accepts_genuine_pytorch_validator_set(setup):
    """Regression for job 628932 section 5: a real ``tunable.write_file``
    artifact carries HIP_VERSION (ROCm major*100+minor) and NO ROCM_VERSION.
    The loader must accept it; the prior required={...,ROCM_VERSION,...}
    rejected every genuine artifact on the MI300A node."""
    module, _, state, artifact = setup
    state["results"] = [("GemmTunableOp_BFloat16_TN",
                          "tn_8192_1_4096_ld_4096_4096_8192",
                          "Gemm_Rocblas_621284994", 0.0196817),
                         ("GemmTunableOp_BFloat16_TN",
                          "tn_8192_2_4096_ld_4096_4096_8192",
                          "Gemm_Rocblas_621284994", 0.0198381)]
    artifact.write_text(
        "Validator,PT_VERSION,2.9.1\n"
        "Validator,HIP_VERSION,603\n"
        "Validator,HIPBLASLT_VERSION,1000-b4e5042b\n"
        "Validator,GCN_ARCH_NAME,gfx942:sramecc+:xnack-\n"
        "Validator,ROCBLAS_VERSION,4.3.0.8ebd6c11\n"
        "GemmTunableOp_BFloat16_TN,tn_8192_1_4096_ld_4096_4096_8192,Gemm_Rocblas_621284994,0.0196817\n"
        "GemmTunableOp_BFloat16_TN,tn_8192_2_4096_ld_4096_4096_8192,Gemm_Rocblas_621284994,0.0198381\n")
    assert module.configure_qkv_tuned_blas(artifact) == (1, 2)
    assert module._CONFIGURED == {1: "Gemm_Rocblas_621284994", 2: "Gemm_Rocblas_621284994"}
    assert state["reads"] == 1            # read the genuine artifact exactly once
    assert (state["enabled"], state["tuning"]) == (False, True)   # flags restored


def test_rejects_configuration_inside_capture(setup, monkeypatch):
    module, torch, state, artifact = setup
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before capture"):
        module.configure_qkv_tuned_blas(artifact)
    assert state["reads"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_forward_restores_flags_and_does_not_read_or_tune(setup, monkeypatch, fail):
    module, torch, state, artifact = setup
    module.configure_qkv_tuned_blas(artifact)
    monkeypatch.setattr(module, "_validate", lambda *args, **kwargs: 1)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    x = torch.ones((1, 2), dtype=torch.bfloat16)
    w = torch.ones((2, 2), dtype=torch.bfloat16)

    def linear(x, weight):
        assert state["enabled"] and not state["tuning"]
        if fail:
            raise RuntimeError("projection failed")
        return x @ weight.T

    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    if fail:
        with pytest.raises(RuntimeError, match="projection failed"):
            module.qkv_tuned_blas(x, w, w, w)
    else:
        assert torch.equal(module.qkv_tuned_blas(x, w, w, w), torch.full((1, 6), 2, dtype=torch.bfloat16))
    assert state["reads"] == 1
    assert not state["enabled"] and state["tuning"]


def test_forward_requires_configured_row_count(setup, monkeypatch):
    module, torch, _, artifact = setup
    monkeypatch.setattr(module, "_validate", lambda *args, **kwargs: 1)
    x = torch.ones(1)
    with pytest.raises(RuntimeError, match="configure"):
        module.qkv_tuned_blas(x, x, x, x)
    module.configure_qkv_tuned_blas(artifact)
    monkeypatch.setattr(module, "_validate", lambda *args, **kwargs: 2)
    with pytest.raises(RuntimeError, match="row count"):
        module.qkv_tuned_blas(x, x, x, x)
