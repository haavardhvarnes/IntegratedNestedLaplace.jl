# IntegratedNestedLaplace.jl — Correctness & Parity Fix Plan

> Working branch: `worktree-inla-correctness-fix`
> Status: read-only review complete; this plan drives the fix-up work.

## 0. Mission

Bring `IntegratedNestedLaplace.jl` to a state where:

1. **Correctness:** for the official R-INLA reference examples (https://r-inla.org/examples/index.html), the Julia package reproduces R-INLA's posterior summaries within numerical tolerance.
2. **Performance:** warm (second-run) wall time is comparable to or better than R-INLA's `cpu.used["Total"]` on the same problem. Cold time (TTFX) is acknowledged as a Julia compilation cost and is *not* the success metric.
3. **No regressions in the math primitives** (`INLACore`, `INLASpatial`) which already pass their tests.

## 1. Reference targets — five R-INLA examples

These are the benchmarks every change must keep working. For each one we will commit a frozen R-INLA reference output (means, SDs, quantiles for fixed effects and hyperparameters) under `test/fixtures/<name>/rinla_reference.json` and a Julia comparison test that asserts agreement.

| # | Reference page | Existing Julia stub | Likelihood | Latent structure | Hypers |
|---|---|---|---|---|---|
| A | [Salamander mating](https://r-inla.org/examples/salamander/salamander/salamander.html) | `examples/04_salamander_mating/model.jl` | Bernoulli (binomial, n=1) | `iid2d` (experiments 1–2 female and male) + `iid` (experiment 3 female and male) | 6 (3 per `iid2d` block × 2 + 2 `iid`) |
| B | [Bivariate meta-analysis](https://r-inla.org/examples/bivariate-meta/bivariate-meta/bivmeta.html) | `examples/05_meta_analysis/model.jl` | Binomial (Ntrials=N) | `2diid` for study-level (sens, spec) | 3 (τ_μ, τ_ν, ρ) |
| C | [Brunei school disparities](https://r-inla.org/examples/AlvinBong/alvinbong-example.html) | `examples/06_brunei_school_disparities/model.jl` | Poisson with offset E | `bym2(graph)` + 3 fixed-effect covariates | 2 (BYM2 precision + φ mixing) |
| D | [Dengue Brazil non-stationary](https://r-inla.org/examples/fbesagExample/brazil_fbesag_workflow_pro_cited.html) | `examples/07_dengue_brazil/model.jl` | Poisson with offset E | Stationary `besag` baseline; non-stationary `fbesag` extension | 1 (besag) → many (fbesag) |
| E | [Joint longitudinal/spatial](https://r-inla.org/examples/Baghfalaki_et_al/Baghfalaki-et-al-example.html) | none yet | mixed: Gaussian + Weibull | B-spline + `iid2d` + `besag` + shared (`copy`) effects | many |

The current `examples/01_tokyo_rainfall`, `02_german_oral_cancer`, `03_swiss_rainfall` are *not* on the official index but are useful smoke tests; we will keep them but realign them to canonical R-INLA tutorials where one exists (Tokyo rainfall and German oral cancer are both in the standard R-INLA case study set).

**Per-example assets we will produce:**

```
examples/<id>/
  data/                  # the actual data (committed if small, fetched otherwise)
  rinla.R                # canonical R-INLA fit → writes JSON + RDS reference
  model.jl               # Julia INLA fit using same data and formula
  rinla_reference.json   # generated; checked in
test/parity/<id>_test.jl # asserts |Julia − R-INLA| under tolerance
```

Acceptance bar (per example):
- |posterior mean (fixed)| − R-INLA mean| ≤ max(1e-3, 1% × R-INLA SD)
- |posterior SD (fixed) − R-INLA SD| / R-INLA SD ≤ 5%
- log-precision posterior mean within 0.05 of R-INLA's (≈5% on the precision scale)
- ρ (when present) within 0.02 of R-INLA's
- warm wall time ≤ 2× R-INLA's `cpu.used["Total"]`

## 2. Findings to fix (from the read-only review)

These are **bugs**, not nice-to-haves. Each links to the line(s) where it lives in `main`.

### 2.1 Showstopper bugs in `inla()` driver (`src/IntegratedNestedLaplace.jl`)

1. **Empty-effects sum** at L158: `sum(e.n_params for e in effects)` errors on `y ~ 1`. Add `init=0` or restructure.
2. **`SPDEModel` precision dispatch** at L172: driver calls `precision_matrix(model, n_params, tau)` but `SPDEModel`'s method is `(model, kappa, tau)`. Same for `BivariateIIDModel`, `NonStationarySPDEModel`. Driver must dispatch by model type and pass the correct hypers.
3. **Hessian is just its diagonal** at L188 — `diag(A' diag(h) A)` instead of `A' diag(h) A`. Wrong by O(n²) entries. This alone explains the 1000× precision discrepancy on Salamander.
4. **Marginals not back-permuted** at L207–224. `S` is stored in the AMD-permuted basis (`F.p`) and `diag(S)` is read off without `vars[F.p[j]] = S[j,j]`. Per-coordinate variances are silently scrambled.
5. **No Gaussian observation precision.** L195 calls `INLAModels.log_pdf(family, y, eta, 1.0)` with τ_y hardcoded. R-INLA always estimates this. Must become a hyperparameter.
6. **No actual integration over θ.** Line 232 returns `[[theta_star[1]]]` and `[1.0]`. The CCD machinery (`INLACore.integration_nodes`) is never invoked. Empirical Bayes ≠ INLA.
7. **Gradient is a placeholder** (L217: `grad[i] = 0.0`). Forces NelderMead, which is why even cold runs on n=50 take seconds.
8. **`theta0` argument silently ignored** — driver always starts at `fill(1.0, n_h)` (L228).
9. **`n_h` collapses multi-hyper models to 1**: L161 sets `n_h = length(effects)`, so 3-hyper bivariate-IID and 4-hyper non-stationary SPDE are silently optimized as 1-D. Hypers must be enumerated per-model via a method `n_hyper(model)`.

### 2.2 Modeling layer issues (`dev/INLAModels/src/INLAModels.jl`)

10. **`RW1Model` adds a +0.1 ridge** at L86 to make the precision PSD. This changes the prior. Replace with the genuine intrinsic RW1 (rank-deficient) plus a sum-to-zero constraint or a tiny *documented* numerical diagonal (e.g. 1e-9) used only for factorization, not for the prior.
11. **Bivariate IID hyperparameters not plumbed.** `precision_matrix(::BivariateIIDModel, n_pairs, τ₁, τ₂, ρ)` exists, but the driver only ever passes one τ.
12. **No PC priors used in the driver.** `PCPrior` exists but the driver hard-codes a Gaussian prior on θ at L204 (`-0.5 * sum(theta.^2)`). Each model type needs to declare its prior.
13. **No fixed-effect prior.** Driver uses `1e-6 * I` (L169) — near-flat. R-INLA uses N(0, 1000). Match the standard or expose it.
14. **No constraint mechanism.** Intrinsic models (RW1, BYM, ICAR, besag) need sum-to-zero constraints to be identifiable alongside an intercept. Currently absent — explains the −1.7e8 latent values on Salamander.

### 2.3 Architectural gaps (vs. parity targets)

15. **Missing models needed for the five reference examples:**
    - `iid2d`, `2diid` (bivariate per-pair latent block) — required for A and B.
    - `bym2` (Riebler 2016 mixing parameterization) — required for C.
    - `besag` (proper CAR with sum-to-zero) — required for D.
    - `fbesag` (non-stationary besag) — required for D.
    - `copy` mechanism (shared effects across multiple linear predictors) — required for E.
    - Stacked multi-likelihood support (`family = c("gaussian", "weibullsurv")`) — required for E.
    - Offsets (the `E` argument in Poisson) and `Ntrials` (binomial size) — required for B, C, D.

### 2.4 Hygiene & infrastructure

16. Subpackage `Project.toml`s have no `[targets]` test block → `Pkg.test()` fails for INLACore/INLAModels/INLASpatial.
17. Examples use unseeded `rand()` — non-reproducible.
18. No CI configured.
19. `docs/make.jl` is empty.
20. `LICENSE` missing.
21. Several heavy `using`s in `src/IntegratedNestedLaplace.jl` (Enzyme, ImplicitDifferentiation, Metal, Meshes, RecipesBase) are unused — drop them.

## 3. Architectural decisions (binding for this branch)

These are conventions that the rest of the plan assumes. Revisit only with explicit user sign-off.

- **Joint mode via the augmented system.** Stack fixed and random effects into a single x of length n_latent. Use the **Schur complement / sparse Cholesky** form: `H = Q_x + A' D(τ_y) A` (no diagonal-only approximation). All examples assume `update_cache!` constructs this full `H`.
- **AMD ordering.** Wrap `cholesky(H; perm=...)` and **always** un-permute when reading marginals. A small `inverse_permutation(F.p)` helper goes in `INLACore`.
- **Hyperparameters.** Enumerate per model:
  ```
  n_hyper(::IIDModel) = 1                   # log τ
  n_hyper(::RW1Model) = 1
  n_hyper(::AR1Model) = 2                   # log τ, logit-shifted ρ
  n_hyper(::SPDEModel) = 2                  # log κ, log τ
  n_hyper(::ICARModel) = 1
  n_hyper(::BivariateIIDModel) = 3          # log τ₁, log τ₂, atanh ρ
  n_hyper(::NonStationarySPDEModel) = 2*p_κ + 2*p_τ  (configurable)
  n_hyper(::BYM2Model) = 2                  # log τ, logit φ
  n_hyper(::BesagModel) = 1
  ```
  Plus one per Gaussian likelihood: `n_hyper(::GaussianLikelihood) = 1`. Driver totals these.
- **Hyper prior interface.** `log_prior(model, θ_block) -> Real`. Default: PC where standard, else weakly informative. Driver sums them.
- **Optimizer.** Newton/BFGS with analytic gradients. The closed-form for ∂(−log π(θ|y))/∂θ uses the cached Cholesky and Takahashi marginals; specifically:
  ```
  ∂obj/∂θ = ½ tr(H⁻¹ ∂H/∂θ) − ½ tr(Q⁻¹ ∂Q/∂θ) − ½ x*' ∂Q/∂θ x* + ∂(−log π(θ))/∂θ
  ```
  Each ∂Q/∂θ for the supported models is closed-form sparse. `INLACore.sparse_trace_inverse` already exists for the trace term.
- **Integration over θ (the "I" in INLA).** Use `INLACore.integration_nodes` (CCD). Default to mode-only when `n_hyper == 1`; switch to CCD for ≥ 2. Recombine via Gaussian-weighted mixture for marginal latent posteriors.
- **Marginals.** Two levels: (a) Gaussian Laplace (mean = mode, var = Takahashi diag) — fast default; (b) simplified Laplace approximation (skewness correction) — opt-in `marginals=:simplified`. R-INLA's "simplified Laplace" is the gold standard; we implement (a) for v0 and add (b) before claiming parity on skewed posteriors.
- **Numeric type.** All math kernels parametric in `T<:Real`. Driver coerces inputs through `eltype(theta)` so AD works.
- **No silent diag ridges.** Any tiny shift used for factorization is named (`PRIOR_RIDGE = 1e-9`) and documented.
- **Constraints.** Implement A_constraint*x = e via the standard "soft constraint" trick (extending the system) for v0; revisit hard linear constraints if R-INLA's `extraconstr` is required.

## 4. Phased work plan

Each phase ends with green tests on the cumulative parity bench.

### Phase 0 — scaffolding (no behavior change)

- [ ] Add `[targets]` test blocks to `INLACore`, `INLAModels`, `INLASpatial` Project.toml so `Pkg.test()` works.
- [ ] Add `LICENSE` (MIT, matching the SciML default).
- [ ] Add a minimal CI: `julia --project=. test/runtests.jl` on macOS + linux for Julia LTS and stable. Run the parity bench at low frequency (nightly) since it depends on R.
- [ ] Add `test/fixtures/` and `test/parity/` directory layout. Add a small Julia helper (`test/parity/parity_helpers.jl`) that loads `rinla_reference.json` and exposes `assert_parity(julia_result, ref; tol)`.
- [ ] Add `bench/Rrun.sh` — runs `Rscript examples/<id>/rinla.R` and writes the JSON reference. Idempotent. CI publishes the JSONs as artifacts.
- [ ] Pin random seeds in all current example data generators.
- [ ] Drop unused `using`s in `src/IntegratedNestedLaplace.jl`.

### Phase 1 — driver correctness (single-hyper models)

Goal: all unit tests pass; example A (Salamander) within tolerance vs R-INLA.

- [ ] Fix bug 1 (empty effects), 2 (SPDE dispatch via `precision_matrix(model, θ_block, n)` with model-specific θ unpacking), 8 (`theta0` plumbing).
- [ ] Replace the diag-only Hessian with the full sparse `H = Q + A' D(−h_eta) A` (bug 3).
- [ ] Fix marginal back-permutation (bug 4); add a regression test that compares `marginals_latent` to `diag(inv(Matrix(H)))` on a small dense problem.
- [ ] Introduce `n_hyper(model)` and `precision_matrix(model, θ_block, n)`. Driver builds `Q = blockdiag(...)` from each model's slice of θ.
- [ ] Add Gaussian likelihood hyperparameter (bug 5); include it as θ[end] when `family isa GaussianLikelihood`.
- [ ] Add `log_prior(model, θ_block)` and `log_prior(::GaussianLikelihood, θ)`. Replace L204's hard-coded `0.5 * sum(θ.^2)`.
- [ ] Add fixed-effect prior `N(0, 1000)`; drop the `1e-6 * I` magic.
- [ ] Add sum-to-zero constraint for intrinsic models (RW1, ICAR, BYM, besag) when an intercept is present.
- [ ] Switch optimizer to BFGS via `OptimizationOptimJL.BFGS()`; supply analytical gradients for the hyperparameter posterior. Until that's in, leave NelderMead as the fallback.
- [ ] Stand up the parity test for example A (Salamander) using the simpler `Cross + f(Female, IID) + f(Male, IID)` formula with shared τ. Acceptance: τ_F, τ_M within 5% of R-INLA mean.

### Phase 2 — multi-hyper integration  ✅ landed

Goal: integration over θ happens; bivariate models work.

- [x] Wire CCD nodes into `inla()`. For every `n_hyper ≥ 1` we now evaluate the
  Laplace objective at every CCD node, normalize via softmax of the
  log-density gap, and return mixture means and variances for both the latent
  field and the hyperparameter vector. `INLAResult` exposes both `mode_latent`
  (joint mode at θ*) and `mean_latent` (CCD mixture mean), plus
  `hyper_precision_mean(res, i)` for the precision-scale mean of `exp(θ_i)`.
  See [src/IntegratedNestedLaplace.jl](src/IntegratedNestedLaplace.jl).
- [x] `BivariateIIDModel` reachable through the driver via `f(study, BivariateIID)`.
  3 hypers (log τ₁, log τ₂, atanh ρ) + an optional `type ∈ {1, 2}` covariate
  for which slot of the per-pair latent the observation hits. Smoke tested
  end-to-end with Gaussian likelihood.
- [x] `BesagModel(W; scale=true, constraint_precision=…)` + `loggamma` prior,
  with R-INLA-style scale.model normalization (geometric mean of marginal
  variance under sum-to-zero = 1).
- [x] Salamander parity stays green at Phase 2 tolerances (10 % × R-INLA SD on
  fixed-effect means, 10 % rtol on precisions). The remaining mode-vs-mean
  offset on Bernoulli fixed-effect posteriors is the simplified-Laplace
  skewness correction territory; lifted to Phase 3.

### Phase 2 deferred to Phase 3

* **Bivariate meta-analysis strict parity (example B)** — R-INLA's `2diid`
  uses a Wishart prior on the 2×2 precision matrix; aligning Julia's
  loggamma+gaussian prior on `(log τ₁, log τ₂, atanh ρ)` with R's Wishart
  parameterization is non-trivial. The Julia driver path is ready; the
  fixture is what's missing.
* **Brunei parity (example C)** — hard sum-to-zero is required. With the soft
  rank-1 constraint the Laplace objective has its global mode at τ → ∞
  (u → 0, intercept absorbs everything). R-INLA avoids this by removing the
  null-space degree of freedom from `Q` and `H` exactly. Implementing the
  augmented-system constrained Newton is the prerequisite. Documented
  diagnostics in `dev/notes/` if needed.
* **Simplified-Laplace marginals** — Phase 1/2 use the Gaussian Laplace at the
  mode; for Bernoulli the conditional posterior of η is mildly skew, which
  shifts the posterior mean by O(0.1 × SD). R-INLA's `strategy="simplified.laplace"`
  default applies a skewness correction. Implementing it tightens the
  fixed-effect parity tolerance from 10 % × SD to ~1 %.

### Phase 3 — constrained intrinsic models  ✅ landed (parity ⏳)

What landed:

- [x] **Hard sum-to-zero constraint** via the augmented (KKT) system in the
  inner Newton. New `gmrf_newton_full(...; constraint_A=…)` solves
  `[H A_c'; A_c 0] [dx; λ] = [-g; e_c − A_c x]` via Schur complement at every
  iteration. The constraint is enforced exactly — Brunei's area effect sums
  to zero to machine precision.
- [x] **`constraint_matrix(model, n_block)` interface** in `INLAModels`.
  `BesagModel` returns the normalized `1' / sqrt(n)` row; intrinsic models
  inheriting this interface (RW1, ICAR, BYM2 in Phase 4) just override.
- [x] **Determinant corrections in `laplace_eval`**: log-determinants on the
  constrained subspace use `log|M + A_c' A_c|` (Rue & Held 2005, eq. 2.30)
  for both `Q` and `H`. The standard `log|M| − log(A_c M⁻¹ A_c')` form
  silently over-counts when `M` is near-singular along the constraint
  direction (the Brunei pathology).
- [x] **Canonical log-det scaling**. `sparse_logdet(Q)` and the matching
  driver call previously each carried a stray ×2 factor that cancelled in
  Phase 1 but disagreed once the constrained corrections came in.
  Both call sites now use the single, canonical form.
- [x] **Multi-start BFGS**. The Laplace objective is non-convex in θ for
  IID precisions whose log-posterior is flat in tail regions; BFGS from a
  single seed routinely gets trapped. Driver now seeds at the user's
  `theta0`, plus `fill(5, n_h)` (near the log-Gamma prior mode) and
  `fill(-2, n_h)` (the weak-shrinkage corner), and keeps the best.
- [x] **Brunei R fixture + mechanics test**. Synthetic 42-area grid with rook
  adjacency. Parity test confirms (a) the constraint is enforced
  numerically, (b) the model dimensions are right, and (c) flags the
  per-area linear-predictor parity as `@test_broken` until simplified-
  Laplace lands.

### What's still blocked — promoted to Phase 4

* **Brunei posterior-mean parity (example C, the `@test_broken`)**. The Gaussian
  Laplace `π̂(x | θ, y)` is an O(0.1×SD) approximation for sharply log-concave
  likelihoods like Poisson on intrinsic GMRFs. The marginal posterior of θ
  computed from this Gaussian Laplace has its global minimum at τ → ∞ where
  R-INLA's marginal vanishes. The fix is the standard *simplified-Laplace*
  skewness correction (third-derivative term at the mode), which R-INLA
  applies by default. This is also what closes the residual mode-vs-mean
  offset on Salamander Bernoulli fixed-effect posteriors.
* **Bivariate meta-analysis strict parity (example B)**. Needs a Wishart
  prior on the 2×2 precision matrix to match R-INLA's `2diid` default;
  Julia's `BivariateIIDModel` currently only supports independent loggamma
  priors on `(τ₁, τ₂, atanh ρ)`.
* **SPDE PD fix + stationary parity**. The Phase 1 driver rewrite already
  exposed the `precision_matrix(SPDEModel, ...)` dispatch hole; SPDE smoke
  test in `test/runtests.jl` runs but the discretization in `INLASpatial`
  produces an indefinite Q on small meshes.
* **`fbesag` non-stationary Brazil parity**.

### Phase 5 — Edgeworth correction to `log π̂(y|θ)`  ✅ landed (insufficient for Brunei)

What landed:

- [x] **`fourth_deriv_eta(family, η, θ_y)`** dispatch alongside the existing
  third-derivative one (Bernoulli `−p(1−p)(1−6p(1−p))`, Poisson `−exp(η)`,
  Gaussian zero).
- [x] **Edgeworth correction inside `laplace_eval`** (so BFGS sees the
  corrected objective at every θ evaluation):
  ```
  correction(θ) = -⅛ ∑_k h⁽⁴⁾_k σ²²_k
                  + ⅛ ∑_{k,l} h⁽³⁾_k h⁽³⁾_l σ²_k σ²_l Σ_(k,l)
                  + ¹⁄₁₂ ∑_{k,l} h⁽³⁾_k h⁽³⁾_l Σ³_(k,l)
  ```
  where `Σ` is the η-marginal covariance under the *constrained* Gaussian
  Laplace. Cost is one extra CHOLMOD multi-RHS solve per `laplace_eval` call.
  Robust against degenerate cases (returns 0 if intermediate values are
  non-finite — matters when τ → ∞ pushes the constrained Σ near singular).

What this **does** fix:
- Nothing visibly on Salamander or Brunei. Salamander's residual `τ` precision
  gap (3.7%) was already CCD-grid discretization, not the Laplace expansion.

What this **doesn't** fix (and why):
- **Brunei**'s τ → ∞ drift in `log π̂(θ|y)` is several nats deep. The leading
  Edgeworth correction adds ~0.4 at θ=1.88 and ~0.001 at θ=10 — too small to
  flip my Laplace's 5-nat preference for τ → ∞. The pathology is structurally
  beyond the leading expansion: the joint Gaussian Laplace at the constrained
  mode misses the per-coordinate non-Gaussianity that R-INLA captures with
  its `strategy="laplace"` (running a full Laplace at each x_i marginal).
  Implementing that is real work — promoted again, this time to whatever the
  next "Phase 6 — full Laplace" lane will be.

### Phase 4 — simplified-Laplace marginal-mean correction  ✅ landed (Salamander)

What landed:

- [x] **`third_deriv_eta(family, η, θ_y)`** dispatch on each likelihood:
  Bernoulli `−p(1−p)(1−2p)`, Poisson `−exp(η)`, Gaussian zero (Laplace
  is exact, no correction needed).
- [x] **SLA correction inside the CCD loop**. At every CCD node `θ_k`:
  ```
  σ²_η = diag(A H⁻¹ Aᵀ)                          # via one CHOLMOD solve
  Δx = ½ · H⁻¹ Aᵀ · (h⁽³⁾(η*) ⊙ σ²_η)
  ```
  Then project `Δx` back onto the constraint set when
  `has_constraints` (otherwise the SLA correction would re-introduce
  sum-to-zero violations on intrinsic GMRFs). Mixture mean uses
  `x*_k + Δx_proj_k` weighted by the existing CCD weights.
- [x] **Salamander parity at strict tolerance (1 % × R-INLA SD)**.
  Posterior means on every fixed effect now match R-INLA to 5 decimal
  places (largest diff = 0.00035 × SD on `CrossW/R`). Warm wall 0.44 s
  vs R-INLA cpu.used 2.09 s (4.7× faster).

### What's still blocked — promoted to Phase 5

* **Brunei marginal-of-θ pathology (the `@test_broken`)**. The SLA mean
  correction on `x | θ` doesn't fix the τ → ∞ drift in `log π̂(θ|y)` itself.
  That correction needs the higher-order Laplace expansion (Edgeworth-style
  4th-derivative + 3rd-derivative cross-terms in `log π̂(y|θ)`, RMC09 eq. 3.4)
  or R-INLA's `strategy="laplace"` (full Laplace at each θ, expensive). For
  intrinsic GMRFs with low-information data, this is the real fix. Implementing
  it raises the same machinery as full Laplace and is best done together with
  the variance correction.
* **Bivariate meta-analysis strict parity (example B)** — Wishart prior on
  the 2×2 precision matrix.
* **SPDE PD fix + stationary parity**.
* **`fbesag` non-stationary Brazil parity**.
* **Joint multi-likelihood (example E)** — stacked `family::Vector{<:Likelihood}`,
  per-observation likelihood routing, `CopyEffect` adapter, Weibull likelihood
  with shape hyper.

### Phase 5 — perf parity

Goal: warm wall time ≤ R-INLA `cpu.used["Total"]` on every example.

- [ ] Profile cold vs warm with `BenchmarkTools.@btime` and `Profile`. Ensure `update_cache!` doesn't re-allocate on each call (prepare buffers once, in-place updates).
- [ ] Cache the symbolic Cholesky factor (`SuiteSparse.CHOLMOD.symbolic_factor`); only refactorize numerically each step. R-INLA effectively does this and it dominates the speedup.
- [ ] Cache `A'A` sparsity, the `Q` sparsity union, and the Takahashi pattern.
- [ ] Run `bench/parity_bench.jl` and assert warm-Julia ≤ 2× R-INLA Total CPU on each fixture. Publish a perf table in the README with both numbers (and explicit Julia version + commit) — replacing the current "matches or exceeds R-INLA" claim with measured numbers.

### Phase 6 — documentation

- [ ] Replace `docs/make.jl` with a Documenter setup; add a tutorial page per example linking to the R-INLA reference page.
- [ ] Update `README.md`: remove the broken examples; add the parity table from Phase 5; document supported models, priors, and limitations.
- [ ] `dev/INLA*/PLAN.md`: convert to status documents reflecting what's done.

## 5. R-INLA reference fixture pipeline

A single script per example, deterministic, committed. Example shape:

```r
# examples/A_salamander/rinla.R
suppressPackageStartupMessages({ library(INLA); library(jsonlite) })
df <- read.csv("examples/A_salamander/data/salamander.csv", stringsAsFactors = TRUE)
formula <- ...
res <- inla(formula, data = df, family = "binomial", Ntrials = rep(1, nrow(df)),
            control.compute = list(return.marginals = FALSE),
            silent = 2L)
ref <- list(
  fixed = res$summary.fixed[, c("mean","sd","0.025quant","0.5quant","0.975quant")],
  hyper = res$summary.hyperpar[, c("mean","sd","0.025quant","0.5quant","0.975quant")],
  cpu = as.list(res$cpu.used),
  inla_version = as.character(inla.version("version"))
)
write_json(ref, "examples/A_salamander/rinla_reference.json", pretty = TRUE, auto_unbox = TRUE)
```

Commit `rinla_reference.json`; rerun via `bench/Rrun.sh A_salamander` when R-INLA version pins change.

## 6. What we're explicitly NOT doing in v1

- Re-implementing R-INLA's full simplified-Laplace marginal correction. We accept Gaussian marginals where R-INLA also reports tight near-Gaussian posteriors; we flag deliberately when our marginals will diverge (e.g. very non-Gaussian Bernoulli posteriors at low n).
- GPU parity. Metal/CUDA paths can stay as opt-in but are not in the success criteria. The README's GPU table must come down or be re-measured before any reinstatement.
- Beating R-INLA. Parity (≤ 2×) is the bar.

## 7. Working in this worktree

- This worktree is on branch `worktree-inla-correctness-fix`.
- One feature/phase per PR; each PR must keep the parity bench green for completed examples.
- All R reference fixtures use INLA `25.10.19` (the version observed locally). Pin in CI via the `inla.version` string in the JSON.
- Never commit changes that regress an existing parity test. If a fix is incompatible, update the JSON intentionally (with a commit message explaining the R-INLA version bump or methodology change).

## 8. First concrete next step

Before writing any production code: stand up Phase 0's `bench/Rrun.sh` and the salamander `rinla.R` script so we have a stable, machine-readable target to fix against. Without that, "matches R-INLA" is unfalsifiable.
