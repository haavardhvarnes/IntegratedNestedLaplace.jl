# Brunei posterior parity — fix plan

> Tracks the work to lift `test/parity/brunei_test.jl`'s `@test_broken` on the
> per-area linear-predictor parity assertion. Promoted from CLAUDE.md "Phase 5
> deferred" since it's substantial enough to warrant its own document.

## What's broken

Two layers of failure on the Brunei (Besag + Poisson + sum-to-zero) parity
test, both rooted in the same place:

1. **My BFGS lands at τ ≈ 22 000 instead of R-INLA's 6.5.**
2. **At those wrong τ values, per-area linear predictors collapse to the
   intercept-only fit (≈ 0.115 everywhere) — a consequence of (1).**

The fix is at layer (1). Once we recover the correct τ, the latent posterior
follows.

## Diagnostic numbers (current `main`, commit 982c13f)

| θ = log τ | τ      | my obj  | R-INLA log-density (relative to its own mode) |
| ---:      | ---:   | ---:    | ---:                                          |
| 1.0       |    2.7 | −25.09  | −1.69                                         |
| 1.88      |    6.5 | −24.22  | 0  ← R-INLA's mode                            |
| 3.0       |   20   | −22.97  | −1.55                                         |
| 5.0       |  148   | −22.97  | −3.26                                         |
| 7.0       | 1097   | −24.56  | −∞ (out of grid)                              |
| 10.0      | 22 026 | −26.46  | −∞                                            |

R-INLA's posterior is a sharp Gaussian-ish peak near θ = 1.88; mine is
monotone-decreasing toward θ → ∞. The disagreement is **5+ nats at θ = 10** —
far beyond what the leading Edgeworth correction (∼0.4 nats at θ = 1.88,
∼0.001 at θ = 10) can fix.

## Root-cause hypotheses, ranked

1. **The Gaussian Laplace `π̂(x|θ,y) ≈ N(x*, H⁻¹)` is poor for Poisson on
   intrinsic GMRFs at small τ.**  R-INLA's default `strategy="simplified.laplace"`
   does *more* than the mean correction I implemented in Phase 4 — it
   refits per-`xᵢ` skewness-aware marginals and that changes
   `log π̂(y|θ)` at every θ.

2. **Possible bug in my constrained `log|H_c|` for full-rank H.**  I currently
   use `log|H + A_c'A_c|` (Rue & Held 2005 eq. 2.30) for *both* Q (intrinsic,
   correct) and H (full rank, may be wrong).  For full-rank H the textbook
   form is `log|H| − log(A_c H⁻¹ A_c')`.  Earlier (Phase 3) I switched both to
   the augmented form because the textbook form blew up against the 1e-9
   diagonal jitter in `BesagModel`.  That jitter has been removed since
   Phase 3, so the textbook form may now be correct and produce a different
   answer.

3. **A subtle τ-scaling or prior bug** that consistently underweights small-τ
   regions.

## Approach options

|     | What                                                                     | Cost           | Confidence in fix |
| :-: | ---                                                                      | ---            | ---               |
| A   | Switch the H branch of the constraint correction to the textbook form    | small (~½ d)   | medium-high       |
| B   | Importance-sampled correction to `log π̂(y|θ)` (N=50–200 constrained samples per evaluation) | medium (~1 d)  | high              |
| C   | Implement RMC09 §3.2.2 simplified-Laplace per-`xᵢ` skewness-corrected marginal + the resulting correction to `log π̂(y|θ)` | heavy (~2–3 d) | high — matches R-INLA exactly |
| D   | Implement R-INLA's `strategy="laplace"`: full Laplace re-run at each `xᵢ` held at quantile points | heavy (~3 d+)  | very high         |

## Recommendation: A → B → (C or D)

Cheap before expensive.

* **A is fast and falsifiable.**  Half-day's work. If it flips the obj
  curve so the mode moves toward θ ≈ 1.88, ship.
* **If A doesn't fix it, B is the right next step.** Importance sampling
  catches *all* higher-order corrections (not just leading Edgeworth) at
  bounded cost.  Brunei has 42 areas — N = 100 samples per evaluation is
  noise.
* **C/D are reserved for if both A and B reveal that R-INLA itself uses a
  more elaborate algorithm than direct importance sampling.**

## Phasing

### Phase 6a — diagnostic + (A)  *(~½ day)*

* [ ] `bench/brunei_obj_curve.jl` — evaluate `obj(θ)` and its components
  (`ll`, `lp_correct`, `−0.5 log|H_c|`, `prior`) over a θ grid; print and
  optionally save a CSV.  Verify the constraint stays satisfied at every θ.
* [ ] In `src/IntegratedNestedLaplace.jl::laplace_eval`, switch the H branch
  to the textbook form `log|H_c| = log|H| − log(A_c H⁻¹ A_c')`.  Q stays
  on the augmented form (it's intrinsic).
* [ ] Re-run the curve.  If the mode is now near θ ≈ 1.88: tighten
  `test/parity/brunei_test.jl`, drop `@test_broken`, run all parity suites.

### Phase 6b — (B) if (A) is insufficient  *(~1 day)*

* [ ] Add `_importance_correction(family, F_H, A_total, A_c, x*, η*, θ_y; N=100, rng)`
  returning `log E_Gauss[exp(R(δ))]` where `R` is the Taylor remainder of
  `log p(y|x* + δ)` beyond second order.  Sampling is on the constrained
  subspace via `δ = L⁻ᵀ z − A_cᵀ (A_c L⁻ᵀ z)` with `z ~ N(0, I)`.
* [ ] Wire into `laplace_eval` *replacing* the Phase-5 leading-Edgeworth
  correction — IS subsumes it.
* [ ] Confirm Brunei: per-area linear-predictor means agree with R-INLA
  within `max(0.05, 0.20 × R-INLA SD)`; SDs within 30 % rtol; τ mean within
  30 %.  Drop `@test_broken`.
* [ ] Re-run salamander, bivariate, SPDE parity — no regression.

### Phase 6c — only if A and B both fail

* [ ] Implement RMC09 §3.2.2 per-`xᵢ` simplified-Laplace marginal: for each
  i, refit the Laplace conditional on `xᵢ` held at `x*ᵢ ± k σᵢ`, fit a
  skew-normal to the resulting log-density values, recompute the marginal
  mean/sd of `xᵢ` and the contribution to `log π̂(y|θ)`.

## Acceptance criteria

* `test/parity/brunei_test.jl` no longer has `@test_broken`. Per-area
  linear-predictor means agree with R-INLA within
  `max(0.05, 0.20 × R-INLA SD)`. SDs within 30 % rtol. τ posterior mean
  within 30 % rtol.
* Salamander parity stays at 5 dp.
* Bivariate parity stays at 30 % rtol on precisions, 0.10 atol on ρ.
* SPDE smoke stays passing.
* Warm wall-time within 2× the current numbers in the README perf table.

## Notes / context

* Doing this fix tightens A (Salamander τ_F mean from 21 168 → R-INLA's
  21 981 — currently 3.7 % off) and likely D's `besag` part too as a side
  effect.  Brunei is the cleanest single-knob test of the marginal-of-θ
  approximation quality.
* Importance sampling makes the warm wall time grow proportionally to N.
  At N=100 and Brunei's small problem this is invisible.  For larger
  models a smarter sampler (antithetic, Halton) can shrink N.
* Once Brunei is green, the natural next item is **D** (Dengue stationary
  besag parity using a real-ish dataset) which exercises the same
  machinery on a bigger graph.
