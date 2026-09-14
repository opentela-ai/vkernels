"""CPU mock checks for global TunableOp configuration/flag lifecycle.

Every artifact used here carries its sidecar manifest (the loader requires
one by default); the ``require_manifest=False`` legacy path has its own tests.
"""

import importlib
import subprocess
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import vkernels.torch_ops.qkv_tuned_blas; "
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_SOURCES = [str(REPO_ROOT / "src/python/vkernels/torch_ops/qkv_projection.py"),
                    str(REPO_ROOT / "meta/benchmarks/bench_qkv_projection.py")]


def _write_manifest(artifact, **overrides):
    """Build a truthful manifest for the current artifact bytes."""
    from vkernels.torch_ops import tuning_manifest

    manifest = tuning_manifest.build_manifest(
        artifact, kernel="qkv_projection", op="GemmTunableOp_BFloat16_TN",
        shapes={"tn_8192_1_4096_ld_4096_4096_8192":
                {"M": 1, "K": 4096, "N": 8192, "dtype": "bf16", "layout": "TN"}},
        producer_paths=PRODUCER_SOURCES, notes="test fixture")
    manifest.update(overrides)
    tuning_manifest.write_manifest(artifact, manifest)
    return manifest


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
    _write_manifest(artifact)
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


@pytest.mark.parametrize("kind", ["validators", "loaded_mismatch", "missing_hip_version"])
def test_rejects_invalid_artifact(setup, kind):
    module, _, state, artifact = setup
    if kind == "validators":
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
    _write_manifest(artifact)
    assert module.configure_qkv_tuned_blas(artifact) == (1, 2)
    assert module._CONFIGURED == {1: "Gemm_Rocblas_621284994", 2: "Gemm_Rocblas_621284994"}
    assert state["reads"] == 1            # read the genuine artifact exactly once
    assert (state["enabled"], state["tuning"]) == (False, True)   # flags restored


def test_default_winner_is_recorded_for_reproducibility_but_gated(setup):
    """Issue #67 Finding B: the autotuner picks ``Default`` for M=1
    non-deterministically across runs. Reproducibility records that choice
    (the loader configures it); quality gates it behind ``accept_default``."""
    module, _, state, artifact = setup
    artifact.write_text(artifact.read_text().replace("Gemm_Hipblaslt_208015", "Default"))
    _write_manifest(artifact)
    state["results"] = [("GemmTunableOp_BFloat16_TN", "tn_8192_1_4096_ld_4096_4096_8192",
                          "Default", 0.02)]
    assert module.configure_qkv_tuned_blas(artifact) == (1,)
    assert module._CONFIGURED == {1: "Default"}
    module._CONFIGURED.clear()
    with pytest.raises(ValueError, match="issue #67"):
        module.configure_qkv_tuned_blas(artifact, accept_default=False)
    assert module._CONFIGURED == {}


def test_rejects_stale_manifest_when_csv_changed(setup):
    module, _, _, artifact = setup
    artifact.write_text(artifact.read_text().replace("0.02", "0.020001"))
    with pytest.raises(ValueError, match="sha256"):
        module.configure_qkv_tuned_blas(artifact)


def test_rejects_manifest_disagreeing_with_csv_rows(setup):
    module, _, _, artifact = setup
    _write_manifest(artifact, algos={"tn_8192_1_4096_ld_4096_4096_8192": "Gemm_Rocblas_999"})
    with pytest.raises(ValueError, match="algorithms disagree"):
        module.configure_qkv_tuned_blas(artifact)


def test_rejects_stale_producer_fingerprint(setup):
    module, _, _, artifact = setup
    manifest = _write_manifest(artifact)
    manifest["producer"]["fingerprints"][PRODUCER_SOURCES[0]] = "sha256:" + "0" * 64
    from vkernels.torch_ops import tuning_manifest
    rows = tuning_manifest.read_csv_block(artifact)
    with pytest.raises(ValueError, match="changed since the artifact was produced"):
        tuning_manifest.validate_manifest(manifest, csv_path=artifact, csv_rows=rows)


def test_rejects_absent_fingerprinted_source(setup):
    from vkernels.torch_ops import tuning_manifest
    module, _, _, artifact = setup
    manifest = _write_manifest(artifact)
    manifest["producer"]["fingerprints"]["gone.py"] = "sha256:" + "1" * 64
    rows = tuning_manifest.read_csv_block(artifact)
    with pytest.raises(ValueError, match="absent from this checkout"):
        tuning_manifest.validate_manifest(manifest, csv_path=artifact, csv_rows=rows)


def test_environment_mismatch_is_rejected(monkeypatch, setup):
    from vkernels.torch_ops import tuning_manifest
    _, _, _, artifact = setup
    manifest = _write_manifest(artifact, device={"arch": "gfx90a", "cu_count": 104},
                               software={"torch": "2.9.1", "hip": 603})
    rows = tuning_manifest.read_csv_block(artifact)
    live = {"arch": "gfx942:sramecc+:xnack-", "cu_count": 228,
            "torch": "2.9.1", "hip": 603}
    with pytest.raises(ValueError, match="device architecture"):
        tuning_manifest.validate_manifest(manifest, csv_path=artifact, csv_rows=rows,
                                          environment=live)
    manifest["device"]["arch"] = live["arch"]
    with pytest.raises(ValueError, match="CU count"):
        tuning_manifest.validate_manifest(manifest, csv_path=artifact, csv_rows=rows,
                                          environment=live)
    manifest["device"]["cu_count"] = 228
    manifest["software"]["torch"] = "2.8.0"
    with pytest.raises(ValueError, match="PyTorch"):
        tuning_manifest.validate_manifest(manifest, csv_path=artifact, csv_rows=rows,
                                          environment=live)
    manifest["software"]["torch"] = live["torch"]
    manifest["software"]["hip"] = 602
    with pytest.raises(ValueError, match="HIP"):
        tuning_manifest.validate_manifest(manifest, csv_path=artifact, csv_rows=rows,
                                          environment=live)
    manifest["software"]["hip"] = 603
    tuning_manifest.validate_manifest(manifest, csv_path=artifact, csv_rows=rows,
                                      environment=live)  # now consistent


def test_missing_manifest_is_rejected_by_default(setup):
    module, _, state, artifact = setup
    artifact.with_suffix(".manifest.json").unlink()
    with pytest.raises(ValueError, match="missing tuning manifest"):
        module.configure_qkv_tuned_blas(artifact)
    assert module._CONFIGURED == {}
    # Legacy escape hatch stays available for interactive debugging only.
    assert module.configure_qkv_tuned_blas(artifact, require_manifest=False) == (1,)


def test_explicit_manifest_argument(setup):
    module, _, _, artifact = setup
    from vkernels.torch_ops import tuning_manifest
    manifest = tuning_manifest.build_manifest(
        artifact, kernel="qkv_projection", op="GemmTunableOp_BFloat16_TN",
        shapes={}, producer_paths=PRODUCER_SOURCES)
    assert module.configure_qkv_tuned_blas(artifact, manifest=manifest) == (1,)
    state = module.qkv_tuned_blas_state()
    assert state["configured"] == {1: "Gemm_Hipblaslt_208015"}
    assert state["manifest"]["kernel"] == "qkv_projection"


def test_warm_reconfiguration_keeps_exact_winners(setup):
    """Cold (first) and warm (second, results table already populated) loads
    must both verify the exact winners against the artifact."""
    module, _, state, artifact = setup
    assert module.configure_qkv_tuned_blas(artifact) == (1,)
    assert module.configure_qkv_tuned_blas(artifact) == (1,)
    assert state["reads"] == 2
    assert module._CONFIGURED == {1: "Gemm_Hipblaslt_208015"}


def test_configuration_is_lock_serialized(setup):
    module, _, state, artifact = setup
    errors = []

    def run():
        try:
            assert module.configure_qkv_tuned_blas(artifact) == (1,)
        except Exception as error:  # pragma: no cover
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and state["reads"] == 4


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
