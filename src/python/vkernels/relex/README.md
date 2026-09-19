# relex — information-theoretic foundations for relaxed correctness

`relex` is the home for techniques that relax kernel correctness in exchange
for speed, bytes, or FLOPs — **with a proven bound on the resulting error**,
not an empirically observed one.

This README covers the *information-theoretic half* of that story. The two
halves answer different questions:

* **Numerical analysis** (companion note, to be written): *how much error does
  this operation add?* — unit roundoff, Wilkinson `γ_n` bounds, quantization
  step bounds, Lipschitz composition.
* **Information theory** (this document): *what is the cheapest possible
  representation for a given error budget — and when are we already at the
  optimum?* Rate–distortion says what the floor is; lower-bound theorems say
  when no scheme can do better; distortion chains convert a compression ratio
  into a certified model-output error.

Together they give `relex` its contract shape: **(R, D)** — *N bits per
element, with output error ≤ D* — instead of "bit-identical to the CPU
oracle".

---

## 1. The master framework: rate–distortion

Rate–distortion theory (Shannon 1959) defines **R(D)**: the minimum average
bits per sample needed to represent a source within distortion D. Every
compression decision in a kernel stack is an R(D) problem in disguise — the
question is only whether the distortion is measured honestly.

Three results translate directly into kernel terms.

**The high-rate exchange rate.** For a scalar quantizer at R bits per
element, high-rate theory gives

```text
D ≈ c · σ² · 2^(−2R)
```

i.e. *one extra bit quarters the MSE* (the "≈ 6 dB per bit" SQNR rule).
This is the exchange rate between bits and error before any code is written:
cutting KV cache from 16 bits to 8 costs a factor ~256 in elementwise MSE —
whether that matters is decided downstream, in task-space distortion
(§2), not here.

**Reverse water-filling.** For independent Gaussian groups with variances
σᵢ², the R(D)-optimal rate allocation is

```text
Rᵢ = max(0, ½·log₂(σᵢ²/θ))        θ chosen so ΣRᵢ = R
```

Bits go to high-variance groups first, ∝ log-variance. Our MXFP4 per-group
ue8m0 scale bytes already carry a per-group σ estimate — a fixed E2M1 grid
everywhere is a fixed-rate allocation that reverse water-filling says is
suboptimal whenever group variances spread. The RD view licenses *mixed
4/6/8-bit group formats under one global D budget* and says how to allocate.

**Lloyd–Max.** The optimal scalar quantizer at a fixed rate (centroids at
conditional means, decision levels halfway between). This is the principled
upgrade path for activation/KV quantizers beyond uniform grids, and it
composes with the scale-byte machinery we already ship: per-group data,
better codebooks.

> Reading: Shannon, *Coding theorems for a discrete source with a fidelity
> criterion* (1959) · Cover & Thomas, *Elements of Information Theory*, ch. 10
> · Gray & Neuhoff, *Quantization*, IEEE T-IT (1998).

## 2. Distortion in task space: the bound chain

The single most useful idea to steal: **define D over model outputs, not
stored values.** A KV entry compressed to 8 bits has a large elementwise
error and possibly a tiny attention-output error — the second number is the
one that is a contract.

For attention the chain is explicit. Let P be the full attention
distribution and Q the compressed/sparse one; V the value rows:

```text
bits spent ──►  KL(P ‖ Q)  ──Pinsker──►  TV(P, Q) ≤ √(KL/2)
                                     ──convexity──►  |Δout| ≤ TV · diam(V)
```

* **Convexity step:** attention output is a convex combination of value rows,
  so replacing P by Q moves the output by at most `TV(P,Q) · max_{i,j}‖v_i − v_j‖`.
  No assumptions on V, no gap analysis, one line.
* **Data processing inequality (DPI):** for any downstream map T,
  `KL(T(P) ‖ T(Q)) ≤ KL(P ‖ Q)`. The chain composes layer by layer without
  new hypotheses — the information-theoretic counterpart of Lipschitz
  composition, and the reason an end-to-end bound can be stated once, at the
  representation, and trusted everywhere below.

Everything in §1 plugs in here: a quantizer's D (measured as KL between
attention logits/distributions, not as elementwise MSE) is what the budget
means, and Pinsker converts it back to the `‖Δout‖` norms the kernels and
tests already speak.

Related: the **rate–distortion–perception** tradeoff (Blau & Michaeli)
formalizes the choice of distortion metric itself — perceptual/task metrics
vs. signal metrics — which is exactly the decision when a contract should be
KL-in-distribution rather than L∞-in-value.

## 3. Information bottleneck: what to keep

Rate–distortion with a relevance variable: compress X while preserving
`I(X; Y)` (Tishby, Pereira & Bialek 1999). In kernel terms: **the KV tiles,
tokens, or experts we keep should maximize mutual information with the
next-token distribution**, with rate = bytes retained.

This is the principled version of the DSA indexer's top-k selection. The
*zero-error limit* of this idea is logit gap analysis (approximate logits
with margin > 2·bound select provably the same set); the *budgeted version*
is the information bottleneck — keep what carries information about Y, drop
what doesn't, at a rate the byte budget sets.

## 4. Probabilistic contracts: method of types and dithering

Every `(ε, δ)` guarantee we would ever print on a kernel — sampled
reductions, stochastic rounding, Monte-Carlo certification — is a
finite-sample shadow of large-deviation theory, and the exponents there are
*information divergences*:

* **Sanov / Hoeffding:** `P(empirical ≉ truth) ≈ 2^(−n·KL)`. The bounds are
  tight in the exponent, so these contracts are not merely valid — they are
  near-optimal in sample count. The gap between Hoeffding and the true
  exponent is itself measured by a KL term, which is why empirical-Bernstein
  refinements exist.
* **Dithered quantization** (Schuchman): adding known uniform dither makes
  quantization error exactly uniform and input-independent — the
  communications-engineering ancestor of stochastic rounding, with the
  cleanest proofs and a hard per-element bound `|Δ| ≤ Δ_step/2` that still
  holds on top of the stochastic ones.

## 5. Lower bounds: knowing when to stop

The other half of "proven" is knowing when nothing better exists.

* **JL is bits-optimal.** Preserving all pairwise dot products of n vectors
  to relative error ε requires `Ω(log(n)/ε²)` dimensions (packing/Fano
  argument). When we shrink the indexer head dimension with a random
  projection, we are provably at the floor — no scheme beats it in general.
* **Fano's inequality.** Below a certain bit budget, top-k selection *must*
  err: the information about the ranking simply isn't in the bits. The
  useful form: it tells us when a "2-bit indexer logits" proposal is not an
  engineering problem but an impossibility.

Together with §1 these make the `relex` claims two-sided: *this scheme
achieves D at rate R* AND *no scheme achieves D at much less than R*.

## 6. Distributed source coding: the KV restore/donate path

When the decoder holds correlated side information — stale peer pages in
cross-node KV reuse (docs/comm-cross-node-kv.md), or the previous layer's
cache — the encoder need not send what the receiver can reconstruct:

* **Slepian–Wolf:** lossless coding with decoder side information achieves
  `H(X | Y)` — the conditional entropy — without the encoder knowing Y.
* **Wyner–Ziv:** the lossy version; the rate penalty for keeping side
  information decoder-only is provably small (≤ ½ bit/sample for
  Gaussian–MSE sources; Zamir 1996).

Delta-encoding a donation against the receiver's stale copy is the practical
embodiment: rate ≈ `H(new | old)`, bound = the quantizer step. This is the
certified-lossy version of the currently byte-exact cross-node path: same
kernel interface, a `(R, D)` contract instead of byte-identity.

## 7. Synthesis: rate–distortion–compute

The operational takeaway for this repo: treat **bytes moved** and **FLOPs
done** as two rate currencies, output error as the distortion budget, and
allocate between them by marginal distortion reduction:

* spend precision on V before K when the output-Lipschitz constants differ
  (§2 makes both measurable — TV of the attention weights vs. value
  diameter);
* spend KV bytes before indexer FLOPs when the exponents in §5/§1 differ;
* spend bits where reverse water-filling points (§1), not uniformly.

That is the decision structure `relex` exists to support — a principled
answer to "where does the next marginal byte or FLOP buy the most fidelity?"

---

## Mapping to vkernels

| target (existing kernel/doc) | IT framework | contract you can print |
|---|---|---|
| `mxfp4_moe_quant` group formats | reverse water-filling, Lloyd–Max | global D budget over a mixed-bit allocation |
| `dsa_topk_logits` (fp8 indexer) | information bottleneck; JL optimality; Fano floor | selection-optimality objective; provable head-dim floor |
| `dsa_sparse_fwd` (sparse attention, τ) | Pinsker + convexity chain | dropped-mass budget ↔ certified `‖Δout‖` |
| cross-node KV restore/donate | R(D) + Wyner–Ziv vs stale pages | bits ↔ certified output error |
| lossy allreduce / stochastic rounding | dithering; Sanov exponents | `(ε, δ)` contracts with near-tight exponents |

## Reading order

1. Cover & Thomas, *Elements of Information Theory* — ch. 10 (rate–distortion),
   ch. 11 (method of types); skim ch. 7 (DPI, Pinsker).
2. Gray & Neuhoff, *Quantization* (1998) — everything about §1 in practice.
3. Blau & Michaeli — the rate–distortion–perception tradeoff (choosing D).
4. Tishby, Pereira & Bialek (1999) — the information bottleneck.
5. Zamir (1996) — Wyner–Ziv rate loss; then Slepian–Wolf/Wyner–Ziv originals
   as needed for §6.

A companion note will cover the numerical-analysis side (γ-bounds, quantization
step bounds, Lipschitz composition, gap analysis) and the `bounds.py` helpers
that evaluate both halves at runtime.
