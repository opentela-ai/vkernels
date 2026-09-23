# NOTES — issue #155 (dsa_sparse_fwd decode latency wall)

Branch: fix/issue-155-dsa-decode-parallel
Change: d95738d — auto decode split+combine in `dsa_sparse_fwd` (env kill-switch
`VK_DSA_DECODE_SPLIT=0` restores the serial dispatch). Dispatch tail factored
into `dsa_sparse_fwd_split_run`; internally managed grow-on-demand scratch.

## Baseline (BEFORE change) — job 649363, nid002936, 2026-09-23
- test_dsa_correct: **PASS (0 failures)**, out max_rel <= 3.9e-3 vs 2e-2 gate.
- Batched probe (baseline binary; 20 warmup + 1000 launches/event-pair x4
  batches; DVFS warmup ~15 s; sclk 2098–2103 MHz during run — boost confirmed):

| shape (H,S_q,dim,tail) | topk | plain dsa_sparse_fwd us/launch (4 batches) | bitsum | split ref us/launch |
|---|---|---|---|---|
| 64,1,256,0 | 256 | 296.1 / 296.2 / 296.3 / 296.3 | 531215618 | 35.6 (split=16) |
| 64,1,256,0 | 128 | 149.8 / 149.8 / 150.0 / 150.0 | 538227689 | 31.0 (split=16) |
| 64,1,256,0 | 2048 | 2347.1 / 2346.8 / 2346.9 / 2346.9 | 526300104 | 92.8 (split=32) |
| 16,1,576,64 | 256 | 539.1 / 539.2 / 539.4 / 539.3 | 260299025 | 80.3 (split=16) |
| 64,64,256,0 (short) | 128 | 140.8 / 139.9 / 140.1 / 140.2 | 33910152348 | — |
| 1,8192,256,0 (prefill) | 128 | 272.5 / 271.7 / 271.9 / 271.5 | 67625996934 | — |

Note: baseline absolute numbers (~296 us topk=256) differ from the issue's
397 us session — node/session variance; the A/B job re-runs the baseline
binary in the SAME job for the honest ratio.

## A/B job (AFTER change) — job 649392, nid002522, 2026-09-23
- Perf ran (sclk 2100–2108 MHz during run); correctness binaries had a
  wrong path in the script -> re-run separately as job 649402 (below).
- Same-job A/B (batched 1000-launch x4 batches; old binary / new
gate-off / new gate-on, us/launch):

| shape | topk | baseline | gate OFF | gate ON | speedup | bitsum(on) vs baseline |
|---|---:|---:|---:|---:|---:|---|
| 64,1,256,0 serving | 256 | 295.4 | 295.3 | **33.5** | **8.8x** | equal |
| 64,1,256,0 serving | 128 | 149.4 | 149.4 | **25.1** | **6.0x** | ±1 bit (RNE) |
| 64,1,256,0 full | 2048 | 2341.5 | 2340.8 | **85.2** | **27.5x** | ±1 bit (RNE) |
| 16,1,576,64 DSv3 | 256 | 537.6 | 537.3 | **80.1** | 6.7x | equal |
| 64,64,256,0 short | 128 | 141.0 | 141.1 | 141.1 | 1.00x (split_for=1) | equal |
| 1,8192,256,0 prefill | 128 | 266.1 | 265.5 | 266.6 | 1.00x (split_for=1) | equal |

- TARGET MET: >=4x at serving topk=256 (8.8x, 33.5 us <= 100 us); no
  regression at topk=128/2048; short/prefill bit-identical.
- Auto path even faster than explicit-split ref rows in the same
  binary (33.5 vs 34.9 @256; 25.1 vs 30.4 @128) — ref runs right after
  the serial sweep (colder L2/TLB).

## Correctness gates (AFTER) — job 649402, nid002664
- auto-split ON (default, decode split=0 rows exercise split path):
  **PASS (0 failures)**, max_rel identical to pre-#155 to 4 decimals.
- VK_DSA_DECODE_SPLIT=0 (plain path): **PASS (0 failures)**.
- reps=3 nondeterminism probe (default): **PASS (0 failures)**.

## Doc deliverables
- docs/performance/dsa/gfx942.md: new "#155" section, 5 new README-
  format evidence rows (AUTO-SPLIT), journal entry.
- docs/performance/dsa/beverin-155-decode-autosplit.log: raw A/B log
  + all gates + sclk trace.

## Split heuristic engaged by the auto path (dsa_sparse_fwd_split_for)
- topk=256 serving (H=64,S_q=1): blocks=64 < 228 -> split=16
- topk=128: split=16; topk=2048: split=32; DSv3 (H=16): split=16
- short S_q=64,H=64: blocks=1024 >= 228 -> split=1 (plain, untouched)
- prefill S_q=8192: split=1 (plain, untouched)
