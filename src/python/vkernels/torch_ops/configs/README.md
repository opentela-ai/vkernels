# sgl_moe tuned configs

Sidecars for `sgl_moe.try_get_moe_config` (`floe/engine/runner/kernels/
sgl_moe.py`). Filename convention (SGLang's): the `device_name` token is
`torch.cuda.get_device_name(0)` with spaces as underscores, so the file must
match the SERVING host's reported name — H100s report either
`NVIDIA_H100_NVL` (sgs-gpu07 fleet, measured) or `NVIDIA_H100_80GB_HBM3`;
identical copies for both are shipped. Unknown device names fall back to the
in-code heuristic default, which is never worse than ~1% vs tuned at T>=4.

## `E=288,N=512,...` — GLM-5.3-Flash fp8 TP4 per rank (H=4096, I_tp=512,
topk=8, fp8-w8a8 block [128,128])

Tuned 2026-09-25 on sgs-gpu07 (H100 NVL, image `floe-sgs:glm53`, Triton
3.7.1) by sweeping BM/BN/BK/GROUP/warps/stages (coarse 48-config grid +
one-axis refinement + swap_ab forcing) at T in {2,4,8,16} under sustained
CUDA-graph load (150-400 timed iters per point, burn-in clock pins, two
independent rounds for finalists; round-to-round dev <= 2.4%).

Result: the heuristic decode default (BM=16/BN=128/BK=128/G=16/w=4/s=3,
swap_ab on SM90) is already within noise of best at T=4/8/16; only T=2
benefits (BM=32/BN=64, up to ~10% vs default, run-dependent 2-10%). The
two fused_moe_kernel launches are 85% of the per-layer call; the tuned
tiles sit ~1.2x above the HBM weight-read floor under uniform routing, so
tile choice is no longer the bottleneck. Full evidence + numbers:
`.agents/runs/sglang-borrow/fused-moe/tune.md` (lane report) — sweep
scripts `/pub/scratch/xiayao/workspace/moe-b4/{sweep288,followup288*}.py`.

Parity: per-T winners vs the eager per-expert oracle over seeds 0-2 —
quant-aware elementwise gate (rtol=atol=2e-2) clean at T=2; T>=4 shows
single-element quant-noise tails <= 0.006 beyond the elementwise allowance
with L2rel <= 0.048 < 5e-2 (the documented fp8-w8a8 activation-quant class;
config-independent, seed-0 CI gate unaffected).
