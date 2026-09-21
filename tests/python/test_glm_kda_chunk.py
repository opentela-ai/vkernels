"""Fused NVIDIA Triton KDA chunked prefill (per-dim-gated delta rule):
contract checks and GPU parity vs the eager fp32 chunked oracle.

The kernel (``vkernels.torch_ops.glm_kda_chunk``) adapts the vendored fla
pipeline (``vllm_kda``) to floe's ``_kda_chunk`` contract. Parity bar, in
order of strictness:

* **Oracle validity** (CPU, no GPU needed): the chunked WY oracle must
  equal an independent per-token recurrent delta-rule rollout — this
  pins the oracle itself, so the GPU comparison below is against math
  proven twice.
* **GPU parity** vs the chunked fp32 oracle within the tolerance the
  kda_decode GPU tests use (atol/rtol 1e-4) at the rig dims (H=64,
  D=128) across S=512/1024/2048 plus ragged lengths. The kernel is NOT
  bit-exact vs the reference: the blockwise ``(I+A)^{-1}`` reorders fp32
  sums against the reference's row-by-row forward substitution, and the
  gate decay is evaluated exp2(g·log2e) — bounded ~1e-5 abs at these
  dims, two orders under the bar.
* **Drift probe**: many trials at one shape; the max |kernel − oracle|
  must stay pinned under the same bound (a wandering max means a
  numerics regression, not noise).
"""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.glm_kda_chunk; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules; "
            "assert 'vkernels.torch_ops.vllm_kda' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _rand_inputs(torch, b, h, s, d, seed, decay=True, nonzero_state=False):
    """floe-caller-shaped inputs: fp32, q/k L2-normalized (unscaled)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, s, d, generator=g)
    k = torch.randn(b, h, s, d, generator=g)
    v = torch.randn(b, h, s, d, generator=g)
    # log gates: negative (decaying), like floe's -decay·softplus(·); the
    # zero-decay case exercises the g == 0 corner (exp2(0) == 1 paths).
    shape_g = (b, h, s, d)
    gate = -torch.rand(shape_g, generator=g) if decay else torch.zeros(shape_g)
    beta = torch.rand(b, h, s, generator=g)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    state = None
    if nonzero_state:
        state = torch.randn(b, h, d, d, generator=g) * 0.1
    return q, k, v, gate, beta, state


# ---------------------------------------------------------------------------
# CPU: the oracle is valid math (two independent derivations agree)
# ---------------------------------------------------------------------------
def test_oracle_matches_recurrent_rollout(torch):
    """Chunked WY oracle == per-token recurrent rollout (small dims)."""
    from vkernels.torch_ops.glm_kda_chunk import (
        kda_chunk_reference,
        kda_recurrent_reference,
    )

    torch.manual_seed(4)
    for chunk_size, b, h, s, d in [(64, 1, 2, 96, 16), (16, 2, 3, 40, 8), (32, 1, 1, 33, 8)]:
        q, k, v, gate, beta, state0 = _rand_inputs(torch, b, h, s, d, 7 + s)
        out, state = kda_chunk_reference(
            q, k, v, gate, beta, chunk_size, state0, output_final_state=True
        )
        # recurrent contract is [B, S, H, …] with pre-scaled q; both oracles
        # keep the state in [B, H, K, V] orientation
        rec_out, rec_state = kda_recurrent_reference(
            q.transpose(1, 2) * d**-0.5,
            k.transpose(1, 2),
            v.transpose(1, 2),
            gate.transpose(1, 2),
            beta.transpose(1, 2),
            state0,
            output_final_state=True,
        )
        torch.testing.assert_close(out, rec_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state, rec_state, atol=1e-4, rtol=1e-4)


def test_reference_shape_contract(torch):
    from vkernels.torch_ops.glm_kda_chunk import kda_chunk_reference

    torch.manual_seed(5)
    b, h, s, d = 2, 3, 100, 16
    q, k, v, gate, beta, state0 = _rand_inputs(
        torch, b, h, s, d, 11, nonzero_state=True
    )
    out, state = kda_chunk_reference(
        q, k, v, gate, beta, 64, state0, output_final_state=True
    )
    assert out.shape == (b, s, h, d) and out.dtype == torch.float32
    assert state.shape == (b, h, d, d)
    out2, no_state = kda_chunk_reference(q, k, v, gate, beta, 64, None)
    assert no_state is None
    # fresh prefill from zero state differs from a random initial state
    assert not torch.allclose(out, out2)


# ---------------------------------------------------------------------------
# contract (CPU tensors — rejected before any GPU work)
# ---------------------------------------------------------------------------
def test_contract(torch):
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_kda_chunk import kda_chunk

    q, k, v, gate, beta, state0 = _rand_inputs(torch, 1, 2, 64, 128, 3)
    with pytest.raises(OpNotEligible, match="CUDA"):
        kda_chunk(q, k, v, gate, beta)  # CPU tensors rejected
    with pytest.raises(OpNotEligible, match="fp32"):
        # fp16 hits the fp32-by-contract dtype gate before the device gate
        kda_chunk(q.half(), k.half(), v.half(), gate.half(), beta.half())
    with pytest.raises(OpNotEligible, match="head dimensions"):
        d48 = torch.randn(1, 2, 64, 48)
        kda_chunk(d48, d48, d48, d48, torch.rand(1, 2, 64))
    with pytest.raises(OpNotEligible, match="chunk sizes"):
        kda_chunk(q, k, v, gate, beta, chunk_size=48)
    with pytest.raises(OpNotEligible, match="beta"):
        kda_chunk(q, k, v, gate, torch.rand(1, 2, 63))
    with pytest.raises(OpNotEligible, match="initial_state"):
        kda_chunk(q, k, v, gate, beta, initial_state=torch.randn(1, 2, 64, 64))
    wide = torch.randn(1, 2, 64, 130)[:, :, :, :128]
    assert not wide.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        kda_chunk(wide, k, v, gate, beta)


# ---------------------------------------------------------------------------
# GPU parity vs the eager fp32 chunked oracle
# ---------------------------------------------------------------------------
RIG_SHAPES = [
    # (B, H, S, chunk): the rig prefill dims are B=1, H=64, D=128; S covers
    # the bench lengths plus a ragged tail and a single-token edge.
    (1, 64, 512, 64),
    (1, 64, 1024, 64),
    (1, 64, 2048, 64),
    (1, 8, 1537, 64),  # ragged: pad path must match the reference's padding
    (1, 8, 1, 64),  # single-token chunked prefill edge
]


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_kda_chunk import kda_chunk, kda_chunk_reference

    for b, h, s, cs in RIG_SHAPES:
        for seed, nonzero_state in [(41, False), (42, True)]:
            q, k, v, gate, beta, state0 = _rand_inputs(
                torch, b, h, s, 128, seed, nonzero_state=nonzero_state
            )
            q, k, v, gate, beta = (x.cuda() for x in (q, k, v, gate, beta))
            state0 = state0.cuda() if state0 is not None else None
            out, state = kda_chunk(
                q, k, v, gate, beta, cs, state0, output_final_state=True
            )
            ref_out, ref_state = kda_chunk_reference(
                q, k, v, gate, beta, cs, state0, output_final_state=True
            )
            assert out.shape == ref_out.shape and out.dtype == torch.float32
            assert state.shape == ref_state.shape and state.dtype == torch.float32
            torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(state, ref_state, atol=1e-4, rtol=1e-4)


def test_gpu_final_state_parity(torch):
    """Round-7 lane I regression pin: the fused pipeline's FINAL recurrent
    state must track the fp32 chunked oracle to <= 1e-5 max-abs at the rig
    sequence lengths (S=512/1024/2048, chunk 64; chunk 32 on one shape).

    History: lane C saw 7.3e-4 here on GH200 and attributed it to sm90
    codegen. The lane I bisect (bench/kda_state_bisect.py) showed the drift
    was the NGC container's tf32 matmul default degrading the eager oracle
    itself; with the oracle pinned to true fp32 (ieee), both sides are
    exact-fp32 and the residual is dot-ordering/exp2 rounding only (~1e-6).
    The state hands off to the torch recurrent decode path, so this is the
    number that must never silently regress.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_kda_chunk import kda_chunk, kda_chunk_reference

    cases = [
        (1, 64, 512, 128, 64, 41),
        (1, 64, 1024, 128, 64, 42),
        (1, 64, 2048, 128, 64, 43),
        (1, 8, 512, 128, 32, 44),  # chunk 32: solve_tril merge path differs
    ]
    for b, h, s, d, cs, seed in cases:
        q, k, v, gate, beta, state0 = _rand_inputs(
            torch, b, h, s, d, seed, nonzero_state=True
        )
        q, k, v, gate, beta, state0 = (
            x.cuda() for x in (q, k, v, gate, beta, state0)
        )
        out, state = kda_chunk(
            q, k, v, gate, beta, cs, state0, output_final_state=True
        )
        ref_out, ref_state = kda_chunk_reference(
            q, k, v, gate, beta, cs, state0, output_final_state=True
        )
        d_out = (out - ref_out).abs().max().item()
        d_state = (state - ref_state).abs().max().item()
        print(f"\nfinal-state parity B={b} H={h} S={s} cs={cs}: "
              f"max|Δout|={d_out:.3e} max|Δstate|={d_state:.3e}")
        assert d_out < 1e-4, f"output diff {d_out:.3e} breached the 1e-4 bar"
        assert d_state <= 1e-5, (
            f"final-state diff {d_state:.3e} breached the 1e-5 lane I bar "
            "(dot-precision or oracle-matmul regression)"
        )


def test_gpu_no_mutation_and_ragged_exact_length(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_kda_chunk import kda_chunk, kda_chunk_reference

    torch.manual_seed(9)
    b, h, s, d = 1, 4, 2511, 128  # rig-like ragged length (pad -> 2544)
    q, k, v, gate, beta, _ = _rand_inputs(torch, b, h, s, d, 17)
    q, k, v, gate, beta = (x.cuda() for x in (q, k, v, gate, beta))
    origs = [x.clone() for x in (q, k, v, gate, beta)]
    out, _ = kda_chunk(q, k, v, gate, beta, 64, None, output_final_state=False)
    assert out.shape == (b, s, h, d)
    for x, o in zip((q, k, v, gate, beta), origs):
        assert torch.equal(x, o), "kernel must not mutate inputs"
    ref_out, _ = kda_chunk_reference(q, k, v, gate, beta, 64, None)
    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    # the padded tail must not leak into the sliced output:
    # positions beyond the last real token stay finite
    assert torch.isfinite(out).all()


def test_gpu_drift_probe(torch):
    """Repeated trials at one shape: max |kernel − oracle| stays pinned.

    A wandering max-diff means a numerics regression (a config started
    taking a tf32 dot or a reordered accumulate), not run-to-run noise.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_kda_chunk import kda_chunk, kda_chunk_reference

    b, h, s, d = 1, 8, 512, 128
    worst_out = 0.0
    worst_state = 0.0
    for trial in range(25):
        q, k, v, gate, beta, state0 = _rand_inputs(
            torch, b, h, s, d, 100 + trial, nonzero_state=True
        )
        q, k, v, gate, beta, state0 = (x.cuda() for x in (q, k, v, gate, beta, state0))
        out, state = kda_chunk(q, k, v, gate, beta, 64, state0, output_final_state=True)
        ref_out, ref_state = kda_chunk_reference(
            q, k, v, gate, beta, 64, state0, output_final_state=True
        )
        worst_out = max(worst_out, (out - ref_out).abs().max().item())
        worst_state = max(worst_state, (state - ref_state).abs().max().item())
    print(f"\ndrift probe: max|Δout|={worst_out:.3e} max|Δstate|={worst_state:.3e}")
    assert worst_out < 1e-4, f"output drift {worst_out:.3e} breached the 1e-4 bar"
    assert worst_state < 1e-4, f"state drift {worst_state:.3e} breached the 1e-4 bar"
