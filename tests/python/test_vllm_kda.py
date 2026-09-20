"""Tests for the vendored fla/vLLM KDA chunked prefill (torch_ops/vllm_kda*).

Covers:

* the NV/AMD shared-kernel extraction (``_kda_kernels_common.py``): both
  vendored modules must bind the SAME kernel objects — guards against silent
  re-copy drift between the two backends;
* ``solve_tril`` against an independent torch ``(I+A)^-1`` (BT in {16, 32,
  64}; the 64x64 merge kernel carries an unrolled store sequence because
  triton 3.8 cannot lower tuple-iteration loops);
* ``recompute_w_u_fwd`` against closed-form formulas for ``u``, ``w`` and
  ``kg`` — regression-guards two NV-side fixes: the stale heuristics dict
  (missing STORE_QG/STORE_KG constexprs made every launch fail on triton 3.8)
  and the fp32-A dtype cast at the wrapper boundary (the kernel's ``tl.dot``
  cannot mix fp32 A with bf16 operands);
* the dense ``chunk_kda`` pipeline end to end (shapes + finiteness).

Kernel tests require CUDA and are skipped elsewhere (the module imports are
CPU-safe; the parity lock runs anywhere torch+triton import).
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")  # noqa: F841  (module import needs it)


@pytest.fixture(scope="module")
def nv():
    import vkernels.torch_ops.vllm_kda as mod

    return mod


@pytest.fixture(scope="module")
def amd():
    import vkernels.torch_ops.vllm_kda_amd as mod

    return mod


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for KDA kernel tests")


def _make_inputs(B=2, T=64, H=3, K=64, V=96, device="cuda"):
    """Realistic magnitudes: scaled q/k, decaying log2 gates, beta in (0,1)."""
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * K**-0.5
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * K**-0.5
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    gk = -torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 3.0
    beta = torch.rand(B, T, H, device=device, dtype=torch.bfloat16)
    return q, k, v, gk, beta


# ---------------------------------------------------------------------------
# dedup parity lock
# ---------------------------------------------------------------------------


def test_shared_kernels_are_the_same_objects(nv, amd):
    """The two vendored backends must share ONE kernel object per byte-identical unit.

    A re-copied per-file diverged copy would break object identity and fail
    here before the backends drift apart numerically.
    """
    import vkernels.torch_ops._kda_kernels_common as common

    assert (
        nv.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra
        is amd.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra
        is common.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra
    )
    assert (
        nv.recompute_w_u_fwd_kernel
        is amd.recompute_w_u_fwd_kernel
        is common.recompute_w_u_fwd_kernel
    )


# ---------------------------------------------------------------------------
# solve_tril vs independent torch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("BT", [16, 32, 64])
def test_solve_tril_matches_torch_inverse(nv, cuda, BT):
    """solve_tril(A) == (I+A)^-1 for strictly-lower A, across merge paths."""
    B, T, H = 2, 64, 3
    M = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.float32)
    for t0 in range(0, T, BT):
        r = torch.randn(B, H, BT, BT, device="cuda")
        M[:, t0 : t0 + BT] = torch.tril(r, -1).permute(0, 2, 1, 3) * 0.05

    Ai = nv.solve_tril(A=M, output_dtype=torch.float32)

    # [B,T,H,BT] -> [B,NT,H,BT,BT]: transpose first, reshape scrambles H
    NT = T // BT
    Mc = M.view(B, NT, BT, H, BT).permute(0, 1, 3, 2, 4)
    I = torch.eye(BT, device="cuda").expand(B, NT, H, BT, BT)
    ref = torch.linalg.solve_triangular(I + Mc, I, upper=False)
    ref = ref.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)
    assert torch.allclose(Ai, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# recompute_w_u_fwd closed forms (and the two NV regression guards)
# ---------------------------------------------------------------------------


def test_recompute_matches_closed_form(nv, cuda):
    q, k, v, gk, beta = _make_inputs()
    A, _ = nv.chunk_kda_scaled_dot_kkt_fwd(q, k, gk, beta, scale=1.0)
    Ai = nv.solve_tril(A=A, output_dtype=k.dtype)  # production pipeline step
    w, u, kg = nv.recompute_w_u_fwd(k, v, beta, Ai, gk)

    vB = (v.float() * beta.unsqueeze(-1).float()).to(v.dtype).float()
    kB = (k.float() * beta.unsqueeze(-1).float() * torch.exp2(gk)).to(v.dtype).float()
    for t0 in range(0, 64, 64):
        u_ref = torch.einsum("bthj,bjhd->bthd", Ai.float()[:, t0 : t0 + 64], vB[:, t0 : t0 + 64])
        w_ref = torch.einsum("bthj,bjhd->bthd", Ai.float()[:, t0 : t0 + 64], kB[:, t0 : t0 + 64])
        assert torch.allclose(u[:, t0 : t0 + 64].float(), u_ref, atol=2e-2, rtol=2e-2)
        assert torch.allclose(w[:, t0 : t0 + 64].float(), w_ref, atol=2e-2, rtol=2e-2)
        # kg = k * exp2(gn - gk) with gn the chunk-last cumulative gate
        gn = gk[:, t0 + 63]
        kg_ref = k[:, t0 : t0 + 64].float() * torch.exp2(gn.unsqueeze(1) - gk[:, t0 : t0 + 64])
        assert torch.allclose(kg[:, t0 : t0 + 64].float(), kg_ref, atol=2e-2, rtol=2e-2)


def test_recompute_accepts_fp32_A(nv, cuda):
    """Regression guard: fp32 A must be cast at the wrapper, not crash in tl.dot."""
    q, k, v, gk, beta = _make_inputs()
    A, _ = nv.chunk_kda_scaled_dot_kkt_fwd(q, k, gk, beta, scale=1.0)
    assert A.dtype == torch.float32  # wrapper default output
    w, u, kg = nv.recompute_w_u_fwd(k, v, beta, A, gk)  # crashed before the cast fix
    assert w.isfinite().all() and u.isfinite().all() and kg.isfinite().all()


# ---------------------------------------------------------------------------
# dense pipeline end to end
# ---------------------------------------------------------------------------


def test_chunk_kda_end_to_end(nv, cuda):
    q, k, v, gk, beta = _make_inputs(T=128)
    o, final_state = nv.chunk_kda(q, k, v, gk, beta, scale=1.0)
    assert o.shape == (2, 128, 3, 96)
    assert o.dtype == torch.bfloat16
    assert o.isfinite().all()
    assert final_state is None  # output_final_state defaults to False
