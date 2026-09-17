# issue #81 artifacts — dsa_topk_logits fp8-MFMA (gfx942 / MI300A)

Branch `issue-81-dsa-topk-tiled-keys`. Node: beverin MI300A, 228 CUs,
ROCm 6.3. Full record: `docs/performance/dsa-topk/gfx942.md` § "Issue #81".

## Raw logs

- `run-640511-correct-bench.out` — first full correctness (21/21 PASS) +
  bench. The split>=16 sweep rows and the auto/mfma8 variant rows read
  0.0 us: the ROCm 6.3 event-pair quirk persisting for whole bench points
  (see the doc's harness section), NOT a kernel property.
- `run-640544-split-scaling.out` — standalone event-timed harness
  (`probes/probe9_benchcheck.hip`): same launches as the 0.0 rows, with
  per-split output checksums. 21.3/14.6/12.0 us at split=16/32/64, 0/100
  zero-elapsed samples — proves the kernels run and the 0.0s are the
  measurement instrument.
- `run-640547-variant-first.out` — variant table run FIRST on a fresh
  GPU: auto/mfma8 measure fine (31.4/33.6 us vs bf16-MFMA 262.5), i.e.
  the 0.0 state is process-state-dependent.
- `run-640564-host-tests.out` — `vkernels_test_dsa` 25/25 (incl. the new
  `DsaTopk.FitsLdsMfmaFp8` admission gate).
- `run-640622-final-bench.out` — final consistent bench after the
  `bench_us` wall-clock fallback: zero 0.0 rows. Headline: msl=512
  30.8/33.0 us (82x vs fp32-Q, ~9x vs bf16-MFMA); msl=4096 unsplit
  214.3 us (4.2x vs old 893); split=64 5.9 us (old bf16-MFMA split=64:
  190 us).

## Probes (`probes/`)

The fp8 16x16x32 MFMA fragment layouts were not in any available ISA
table; they were decoded empirically on-device:

- `probe_f8mfma.hip` — first functional run; assumed layouts wrong.
- `probe2_f8map.hip`, `probe3_f8brute.hip` — structured input mapping,
  brute-force one-hot row/col response.
- `probe4_f8decode.hip` — information-theoretic decode (lane-id encoding,
  power-of-2 byte values): A fragment = lane `l` byte `j` =
  `X[l&15][(l>>4)*8+j]` (the H2 layout); `c[0]`/`c[2]` anomaly
  investigated.
- `probe5_f8cfrag.hip` — C-fragment double-accumulation investigation
  (was the layout mismatch, not the instruction).
- `probe6_f8lane.hip`, `probe7_f8rowmap.hip`, `probe8_f8h2.hip` —
  per-lane verification on odd rows; confirms the H2 mapping per lane.
- `probe9_benchcheck.hip` — the independent event-timed bench/verification
  harness used for the short-kernel numbers.
