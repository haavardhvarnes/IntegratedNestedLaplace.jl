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

### Phase 6a — diagnostic + (A)  ✅ landed; **does not fix Brunei**

* [x] `bench/brunei_obj_curve.jl` — prints `obj(θ)` and its components on
  a θ grid for the Brunei model. Constraint satisfied at every θ
  (`sum(u*) ~ 1e-15`).
* [x] Switched the H branch of `laplace_eval` to the textbook form
  `log|H_c| = log|H| − log(A_c H⁻¹ A_c')`. Mathematically correct for
  full-rank H; the previous augmented form `log|H + A_c' A_c|` is only
  correct for *singular* Q (Rue & Held 2005 eq. 2.30) and gives wrong
  full-rank H values by `2 log s` where `s = A_c H⁻¹ A_c'`. No
  regressions on Salamander, Bivariate, or SPDE parity suites.
* [x] **Hypothesis falsified.** The two formulas differ by an *exact
  constant* (Δ ≈ −21.3 nats ≈ `2 · log s` where `s ≈ 42 000`, dominated
  by intercept-vs-area-constant unidentifiability under our N(0, 10³)
  fixed-effect prior) across the entire θ grid. Both peak at τ → ∞.
  The fix is a correctness improvement (right answer for `log p̂(y|θ)`
  on intrinsic GMRF problems) but it doesn't move the optimum. Brunei
  posterior parity stays `@test_broken`.

### Phase 6b — (B) IS correction  ✅ landed; **does not fix Brunei**

* [x] `_importance_correction(family, A_total, F_H, x*, η*, θ_y, y_raw, o_vec, A_c; N=100, seed=…)`
  returning `log E_{N_c(0,H_c⁻¹)}[exp(R(δ))]`. Samples `δ = F_H.UP \ z`
  with `z ~ N(0, I)` (gives `δ ~ N(0, H⁻¹)`); projects to `ker(A_c)` for
  constrained problems.
* [x] Wired into `laplace_eval` after the constraint-corrected log-dets;
  replaces the Phase-5 leading-Edgeworth (IS subsumes it).
* [x] Deterministic seed for reproducibility. Try/catch fallback to 0.
* [x] Salamander, Bivariate, SPDE parity all stay green (no regression).
* [x] **Hypothesis falsified.** The IS correction is tiny across the
  entire θ grid for Brunei: −0.073 nats at θ = −1, +0.001 at θ = 10. Far
  too small to flip the 6-nat preference my Laplace gives to τ → ∞.
  Brunei mode does not move (still τ ≈ 16 000); per-area LPs stay
  collapsed at ≈ 0.115. `@test_broken` remains.

### Diagnostic that surprised me — Brunei is broken even with **Gaussian** likelihood

A pure Gaussian-likelihood + Besag fit on the same data, where the
Laplace approximation is *exact*, also shows a Julia/R-INLA τ
disagreement (though much smaller):

| Setup                          | Julia mode of log τ | R-INLA mode of log τ |
| ---                            | ---:                | ---:                  |
| Gaussian + besag (τ_y fixed)   | 1.23 (τ ≈ 3.4)      | 2.24 (τ ≈ 9.4)        |
| Poisson + besag (default)      | 9.65 (τ ≈ 16 000)   | 1.32 (τ ≈ 3.7)        |

The Gaussian gap is ~1 nat in log τ (factor 3 in τ). The Poisson gap is
~8 nats (factor 4 000). So:

1. There's a **structural disagreement** between my driver and R-INLA on
   the constrained-Laplace marginal *even where the Laplace is exact*.
   Small for Gaussian, compounds badly for non-Gaussian.
2. It's **not in the higher-order corrections** — IS handles those.
3. The likely culprit is somewhere in the constraint mechanics — possibly
   how the prior contributes to `log p(x*|θ)` on the (n−k)-dim
   constrained subspace, or how my mode-finder interacts with the
   constraint. Worth a focused investigation pass.

### Phase 6c.1 — Term-by-term diagnostic at fixed θ  ✅ landed; **found a real bug, but Brunei not yet green**

What landed:

* [x] `bench/brunei_dump.jl` — at fixed θ, dumps every scalar component of
  Julia's constrained-Laplace marginal-log-density formula (intercept,
  `u[1:5]`, `sum(u*)`, `log p(y|x*,θ)`, `½ x' Q x`, `log|Q + A_c'A_c|`,
  `log|H|`, `log(A_c H⁻¹ A_c')`, IS correction, prior, final obj). Writes
  JSON for cross-comparison.
* [x] `bench/brunei_rinla_dump.R` — mirror on the R-INLA side. Uses
  `inla(..., hyper=list(prec=list(initial=θ, fixed=TRUE)))` to fix θ.
  Extracts `cfg$mean` in R-INLA's `(u_1..u_42, β)` layout and the joint
  `cfg$Q`. Symmetrizes `cfg$Qprior` (R-INLA stores it upper-triangular).

What the diagnostic found:

1. **`BesagModel`'s `_besag_scale_factor` was inverted.** The function computed
   `geom_mean(diag(Σ_constrained))` correctly, but then returned
   `1.0 / geom_mean` — the *wrong* multiplier. With `Q_scaled = c · Q_unscaled`,
   the marginal variance scales as `1/c`, so to make
   `geom_mean(Var_scaled) = 1` we need `c = geom_mean(Var_unscaled)`, not its
   reciprocal. With the inverted factor, Julia's `Q[1,1] = 25.98` against
   R-INLA's `Qprior[1,1] = 8.41` — a factor-of-3 discrepancy, and
   `geom_mean(Var_scaled) = 0.32` instead of `1`. **Fixed in
   `dev/INLAModels/src/INLAModels.jl::_besag_scale_factor`.**

2. **After the fix, every component except one matches R-INLA exactly at fixed θ.**
   At θ = 2.0 (τ = 7.39), with the fix:
   * `β`, `u[1:5]`, `sum(u*)`, `||u*||₂` — match to 5 dp
   * `log p(y | x*, θ) = 51.518` — matches
   * `½ x*' Q x* = 5.624` — matches (after symmetrizing R-INLA's upper-
     triangular `Qprior` storage)
   * `log|Q + A_c'A_c|` — matches
   * `log|H|` — matches

3. **The one quantity that *doesn't* match is `log(A_c H⁻¹ A_c')`, and it's
   the τ-dependent slope that drives the wrong optimum.**

   | θ = log τ | Julia `log(A_c H⁻¹ A_c')` | R-INLA `log(A_c H⁻¹ A_c')` |
   | ---:      | ---:                       | ---:                        |
   | 2.0       | ≈ 10.65                    | −2.30                       |
   | 10.0      | ≈ 10.65                    |  9.21                       |

   Julia's value is **constant ≈ 10.65** across the entire θ grid; R-INLA's
   **varies by ≈ 11.5 nats**. Since the textbook constraint correction is
   `log|H_c| = log|H| − log(A_c H⁻¹ A_c')`, this is exactly the term
   that contributes a θ-dependent slope to `−½ log|H_c|`, and it's the
   one ingredient missing on Julia's side. Constant-in-θ on our side
   means our `H⁻¹` is being projected against `A_c` in a way that's
   dominated by an effectively-flat (intercept-driven) direction —
   plausibly because under our `N(0, 10³)` intercept prior the
   constrained `H` has a near-singular direction along
   `A_c = (0; 1/√n · 1)` that doesn't tighten with τ.

4. **`x*`-match is solid.** Same data, same θ, same constraint —
   `gmrf_newton_full` produces the same constrained mode as R-INLA at every
   θ tested. The mode-finder is not the problem.

What this **does** fix:
* Julia's `BesagModel` now reproduces R-INLA's `scale.model = TRUE` precision
  matrix exactly. This was a real correctness bug and the fix is independent
  of the rest of Phase 6c.
* No regressions on Salamander (13/13), Bivariate (10/10), SPDE (5/5), or
  the main test suite (15/15).

What this **does not** fix:
* Brunei BFGS still drifts to τ ≈ 17 700 (R-INLA τ ≈ 19). The
  `log(A_c H⁻¹ A_c')` slope discrepancy is the live bug.
  `test/parity/brunei_test.jl` stays `@test_broken` on the per-area
  linear-predictor parity assertion.

### Phase 6c.2 — Resolve the residual `log(A_c H⁻¹ A_c')` slope discrepancy

Hypothesis: our `H = Q + A' D(−h_η) A` differs from R-INLA's joint posterior
precision in how the (improper-prior) intercept couples with the
constraint direction `A_c`. R-INLA uses `prec.intercept = 0` (a
*genuinely* flat improper prior on the intercept, removed via an
augmented system that effectively projects it out before the constrained
Cholesky). We use `Q_fixed_block = 1e−3 · I`, a finite-precision
substitute that puts a non-zero penalty on the intercept and then asks
the constraint to absorb the difference.

* [ ] **6c.2.a Verify the hypothesis.** Run two variants of the dump:
   one with `prec_intercept = 1e−3` (current Julia), one with
   `prec_intercept = 0` (rebuild `Q_fixed_block` as the zero matrix on
   the intercept block). If the τ-dependent slope of `log(A_c H⁻¹ A_c')`
   reappears in the second variant and matches R-INLA, the hypothesis is
   confirmed and the fix is to remove the `1e−3` finite-precision
   substitute throughout the driver.
* [ ] **6c.2.b Implement.** If 6c.2.a confirms, implement
   `prec_intercept = 0` properly (it's an additional rank-1 augmentation
   of the constraint: stack the fixed-effect column onto `A_total` with
   a near-zero precision row in the augmented system, or equivalently
   move the intercept's null-space contribution into the constraint
   block). The mechanics already exist in `gmrf_newton_full` — only the
   formulation in `inla()` and `laplace_eval` need to change.
* [ ] **6c.2.c Drop `@test_broken`** once Brunei posterior-mean parity
   matches the acceptance criteria below.

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
