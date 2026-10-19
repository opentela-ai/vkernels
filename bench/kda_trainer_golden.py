"""Lane-C oracle harness (perf3/kda-hip): golden tensors for the K3
per-key-dim gated delta rule, generated on CPU by the TRAINER's reference
math and laid out for the vkernels device-parity tests.

What this is for
----------------
The native HIP/CK ladder (docs/candidates/kda-native-hip-ck-ladder.md)
replaces the scalar chunked-WY launches in ``kda.hip`` with matrix-core
kernels. Every rung needs golden tensors that are independent of the
kernel under test: this script imports the trainer's ``vtrainer.model.kda``
(read-only; the trainer repo is the reference-math owner) and runs its
delta-rule step functions on deterministic CPU inputs, saving both the
END-TO-END outputs (any correct kernel must reproduce them) and the
PIECE tensors of the trainer's own chunked decomposition (useful for
stage-local debugging -- see the convention warning below).

Run (any python with torch >= 2.1; the trainer's .venv works):
    python3 bench/kda_trainer_golden.py                 # 8B-ish + small shape
    python3 bench/kda_trainer_golden.py --out FILE.pt   # custom output
    VTRAINER_SRC=/path/to/trainer/src python3 ...       # non-default trainer

The trainer repo is located via $VTRAINER_SRC (a path to its ``src/``
directory) or, by default, the sibling checkout ../../trainer/src relative
to this repo's root. The trainer is imported READ-ONLY; nothing under it
is written or executed on GPU.

Golden file contents (torch.save dict of fp32 CPU tensors)
----------------------------------------------------------
inputs (trainer contract, [B, S, H, D]; generated here, deterministic):
  q_norm, k_norm   L2-normalised queries/keys (trainer core fns apply the
                   D^-1/2 scale to q INTERNALLY; vkernels applies it at the
                   caller -- both layouts are exported, see below)
  v                values, N(0,1)
  g_log            per-(token, head, key-dim) LOG gates in [-5, 0]
                   (the trainer ForgetGate output contract)
  g_normal         exp(g_log) in (0,1] -- the vkernels normal-space gate
                   contract (kda.hpp takes g in (0,1] and logs internally;
                   do NOT re-log these)
  beta             scalar delta gate in (0,1), [B, S, H]
inputs_vk          the SAME inputs in vkernels layout ([B,H,S,D], beta
                   [B,H,S], q pre-scaled by D_k^-1/2) -- byte-for-byte what
                   a device-parity test feeds hip::kda_* entry points.
end-to-end (fp32 CPU):
  out_recurrent, state_recurrent    kda_recurrent  (the per-token oracle)
  out_chunked,  state_chunked       kda_chunked    (the mask-form oracle)
  out_fast,     state_fast          kda_chunked_fast (the _wy_grams path)
  out_vk, state_vk                  the end-to-end tensors re-expressed in
                                    vkernels layout ([B,H,S,D] out; state
                                    transposed to [v][k], see CONVENTIONS)
pieces (trainer decomposition, [B, H, nc, cs, ...] -- see CONVENTIONS):
  g_cum, attn (the raw strictly-lower WY matrix M), attn_intra, Ainv
  (= _wy_inverse(attn)), u, k_cumdecay
  per-chunk state trace state_trace[nc] (states entering each chunk + final)
meta: shapes, seed, tolerances measured between the three implementations.

CONVENTIONS -- read before comparing pieces to vkernels internals
-----------------------------------------------------------------
The trainer and vkernels use DIFFERENT (both self-consistent) WY beta
placements: the trainer folds beta_t into the ROW operand of the WY matrix
(M[t][j] = -sum_d b_t k_t k_j exp(g_t-g_j)) and into v/k before the solve;
vkernels folds b_j into the COLUMN side of its grams (M[t][j] =
b_j sum_k exp(L_t-L_j) k_j k_t, stored positive; U_v/W are beta-free). The
piece tensors are therefore NOT one-to-one comparable across the two
repos; only the END-TO-END out/state tensors are. The pieces exist to
bisect a future fused pipeline against the trainer's stages.
Gate space differs too: trainer gates are LOG-space (cumsum then exp),
vkernels takes normal-space g and logs in-kernel; g_cum here is in
trainer/natural-log units == the L of the vkernels cumsum launch.
State layout: trainer state[k][v] (key rows); vkernels state[v][k] with
S[v][k] = trainer_state.T — out_vk/state_vk do that transpose for you.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINER_SRC = REPO_ROOT.parent.parent / "trainer" / "src"


def _resolve_trainer(src_arg: str | None) -> Path:
    """Locate the trainer's src/ dir (env VTRAINER_SRC > --trainer > default)."""
    cand = Path(src_arg or os.environ.get("VTRAINER_SRC", str(DEFAULT_TRAINER_SRC)))
    if (cand / "vtrainer" / "model" / "kda.py").is_file():
        return cand
    sys.exit(
        f"trainer sources not found under {cand} (want <src>/vtrainer/model/kda.py); "
        "pass --trainer SRC or set VTRAINER_SRC"
    )


def _gen_inputs(
    B: int, S: int, H: int, D: int, seed: int, g_lower: float
) -> dict[str, torch.Tensor]:
    """Deterministic module-contract inputs on CPU (fp32).

    Mirrors KDALayer.forward's pre-core normalisation: q/k L2-normalised
    (per (b,s,h) row, fp32), g log-gates uniform in [g_lower, 0], beta
    sigmoid of N(0,1) logits.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=gen, dtype=torch.float32)
    k = torch.randn(B, S, H, D, generator=gen, dtype=torch.float32)
    v = torch.randn(B, S, H, D, generator=gen, dtype=torch.float32)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    g_log = torch.rand(B, S, H, D, generator=gen) * abs(g_lower) - abs(g_lower)
    beta = torch.sigmoid(torch.randn(B, S, H, generator=gen, dtype=torch.float32))
    return {"q_norm": q, "k_norm": k, "v": v, "g_log": g_log, "beta": beta}


def _vk_layout(t: dict[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    """Same inputs in vkernels layout: [B,H,S,D], beta [B,H,S], q*scale."""
    return {
        "q_vk": t["q_norm"].permute(0, 2, 1, 3).contiguous() * scale,
        "k_vk": t["k_norm"].permute(0, 2, 1, 3).contiguous(),
        "v_vk": t["v"].permute(0, 2, 1, 3).contiguous(),
        "g_vk": t["g_log"].permute(0, 2, 1, 3).contiguous().exp(),
        "beta_vk": t["beta"].permute(0, 2, 1).contiguous(),
    }


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="scratch/kda-trainer-golden.pt")
    ap.add_argument("--trainer", default=None, help="path to the trainer's src/")
    ap.add_argument(
        "--shape",
        action="append",
        default=None,
        metavar="B,S,H,D,CS",
        help="shape to generate (repeatable); default 1,256,4,128,64 and "
        "the 8B microbatch point 1,8192,32,128,64 (S=8192 is slow on CPU "
        "for the recurrent oracle -- it is skipped unless --with-8b)",
    )
    ap.add_argument("--with-8b", action="store_true")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    trainer_src = _resolve_trainer(args.trainer)
    sys.path.insert(0, str(trainer_src))
    import vtrainer.model.kda as kda  # noqa: E402  (read-only import)

    shapes = args.shape or ["1,256,4,128,64"]
    if args.with_8b:
        shapes.append("1,8192,32,128,64")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    golden: dict[str, object] = {"meta": {"trainer_src": str(trainer_src),
                                          "seed": args.seed,
                                          "shapes": shapes,
                                          "torch": torch.__version__}}

    for spec in shapes:
        B, S, H, D, CS = (int(x) for x in spec.split(","))
        tag = f"B{B}_S{S}_H{H}_D{D}_cs{CS}"
        assert S % CS == 0, f"S={S} must be a multiple of cs={CS}"
        scale = 1.0 / math.sqrt(D)
        t = _gen_inputs(B, S, H, D, args.seed + S + H, g_lower=5.0)
        nc = S // CS

        # -- end-to-end: the per-token oracle + both chunked forms ---------
        out_r, st_r = kda.kda_recurrent(
            t["q_norm"], t["k_norm"], t["v"], t["g_log"], t["beta"]
        )
        out_c, st_c = kda.kda_chunked(
            t["q_norm"], t["k_norm"], t["v"], t["g_log"], t["beta"], chunk_size=CS
        )
        out_f, st_f = kda.kda_chunked_fast(
            t["q_norm"], t["k_norm"], t["v"], t["g_log"], t["beta"], chunk_size=CS
        )

        # -- pieces: the trainer's own chunked decomposition ---------------
        q = t["q_norm"] * scale  # S % CS == 0 asserted above: no padding
        k = t["k_norm"]
        beta = t["beta"]
        k_beta = k * beta.unsqueeze(-1)
        chunkify = (
            lambda x: x.permute(0, 2, 1, 3)
            .reshape(B, H, nc, CS, x.shape[-1])
            .contiguous()
        )
        g_cum = chunkify(t["g_log"]).cumsum(dim=-2)  # == vkernels L (ln units)
        attn, attn_intra = kda._wy_grams(
            chunkify(k_beta), chunkify(q), chunkify(k), g_cum, CS, 16
        )
        ainv = kda._wy_inverse(attn.clone(), CS)
        v_beta = t["v"] * beta.unsqueeze(-1)
        u = ainv @ chunkify(v_beta)
        k_cumdecay = ainv @ (chunkify(k_beta) * g_cum.exp())

        # per-chunk state trace (states ENTERING each chunk, plus final) by
        # replaying the reference chunk loop on the piece tensors
        k_c = chunkify(k)
        st_trace = torch.zeros(B, H, nc + 1, D, D)
        st_in = torch.zeros(B, H, D, D)
        for c in range(nc):
            st_trace[:, :, c] = st_in
            kdec = k_c[:, :, c] * (g_cum[:, :, c, -1:, :] - g_cum[:, :, c]).exp()
            v_new = u[:, :, c] - k_cumdecay[:, :, c] @ st_in
            st_in = (
                st_in * g_cum[:, :, c, -1].exp().unsqueeze(-1)  # row decay
                + kdec.transpose(-1, -2) @ v_new
            )
        st_trace[:, :, nc] = st_in

        # -- vkernels-layout twins (out/state transposed to [v][k]) ---------
        out_vk = out_r.permute(0, 2, 1, 3).contiguous()
        state_vk = st_r.transpose(-1, -2).contiguous()

        tols = {
            "chunked_vs_recurrent_rel": _rel(out_c, out_r),
            "fast_vs_recurrent_rel": _rel(out_f, out_r),
            "fast_state_vs_recurrent_rel": _rel(st_f, st_r),
            # the piece replay (attn/Ainv/u/k_cumdecay + the chunk loop)
            # must reproduce the oracle's final state — the strongest
            # internal check that the exported pieces are self-consistent
            "piece_state_vs_recurrent_rel": _rel(st_trace[:, :, nc], st_r),
        }
        for name, tol in tols.items():
            if tol > 1e-3:
                raise SystemExit(f"{tag}: {name} = {tol:.2e} > 1e-3 (oracle drift)")

        p = f"{tag}/"
        p = f"{tag}/"
        vk = _vk_layout(t, scale)
        for key, val in {
            "q_norm": t["q_norm"],
            "k_norm": t["k_norm"],
            "v": t["v"],
            "g_log": t["g_log"],
            "g_normal": t["g_log"].exp(),
            "beta": t["beta"],
            "q_vk": vk["q_vk"],
            "k_vk": vk["k_vk"],
            "v_vk": vk["v_vk"],
            "g_vk": vk["g_vk"],
            "beta_vk": vk["beta_vk"],
            "out_recurrent": out_r,
            "state_recurrent": st_r,
            "out_chunked": out_c,
            "state_chunked": st_c,
            "out_fast": out_f,
            "state_fast": st_f,
            "out_vk": out_vk,
            "state_vk": state_vk,
            "g_cum": g_cum,
            "attn": attn,
            "attn_intra": attn_intra,
            "Ainv": ainv,
            "u": u,
            "k_cumdecay": k_cumdecay,
            "state_trace": st_trace,
            "tolerances": tols,
        }.items():
            golden[p + key] = val
        print(
            f"{tag}: ok  max|rel| recurrent-vs-chunked="
            f"{tols['chunked_vs_recurrent_rel']:.2e} "
            f"recurrent-vs-fast={tols['fast_vs_recurrent_rel']:.2e}"
        )

    torch.save(golden, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
