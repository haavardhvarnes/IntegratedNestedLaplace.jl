# IntegratedNestedLaplace.jl

Native-Julia implementation of the Integrated Nested Laplace Approximation
(INLA) for approximate Bayesian inference in latent Gaussian models. The
goal is parity with [R-INLA](https://www.r-inla.org/) on its reference
examples, with comparable or better warm wall time.

> Status: under active development on the `worktree-inla-correctness-fix`
> branch. See [`CLAUDE.md`](CLAUDE.md) for the full plan and the current
> phase.

## Architecture

Three sub-packages plus a thin user-facing API:

* **`INLACore`** — sparse Cholesky / Newton GMRF solver, augmented (KKT)
  constrained Newton, Takahashi selected inverse, CCD integration nodes.
* **`INLAModels`** — likelihoods (Gaussian, Bernoulli, Poisson), latent
  models (`IIDModel`, `RW1Model`, `BesagModel`, `BivariateIIDModel`,
  `SPDEModel`, `ICARModel`, `NonStationarySPDEModel`), the
  `n_hyper / assemble_Q / log_prior / constraint_matrix` interface, and
  log-Gamma / Gaussian priors matching R-INLA's conventions.
* **`INLASpatial`** — 2D Delaunay triangulation via `Meshes.jl`, FEM
  mass/stiffness assembly, SPDE precision construction.

## Installation

`IntegratedNestedLaplace.jl` and its three subpackages (`INLACore`,
`INLAModels`, `INLASpatial`) are published in the
[JuliaRegistry](https://github.com/haavardhvarnes/JuliaRegistry) private
registry. Add it once per Julia installation, then `Pkg.add` works as
usual:

```julia
using Pkg
pkg"registry add https://github.com/haavardhvarnes/JuliaRegistry.git"
Pkg.add("IntegratedNestedLaplace")
```

That's it — the three subpackages are pulled in automatically as
dependencies. The General registry is also required (it ships with
Julia by default).

### Developing from a local clone

If you want to hack on the package or its subpackages directly:

```julia
using Pkg
Pkg.activate(".")
Pkg.develop([
    Pkg.PackageSpec(path = "dev/INLACore"),
    Pkg.PackageSpec(path = "dev/INLAModels"),
    Pkg.PackageSpec(path = "dev/INLASpatial"),
])
Pkg.instantiate()
```

## Performance vs R-INLA 25.10.19

Warm second-run wall time on parity test fixtures, on an Apple M-series CPU.
Reproduce with `julia --project=. bench/parity_bench.jl` after running
`bench/Rrun.sh` to refresh the R-INLA fixtures.

| Example | Julia warm (s) | R-INLA `cpu.used["Total"]` (s) | Ratio |
|---|---:|---:|---:|
| Salamander mating (Bernoulli + 2 IID) | 0.92 | 2.09 | **0.44×** |
| Bivariate meta (Gaussian + 2diid)     | 0.15 | 2.08 | **0.07×** |
| Brunei (Poisson + besag)†             | 0.08 | 1.82 | **0.04×** |
| Stationary SPDE (Gaussian + SPDE)     | 2.38 | n/a  | n/a   |

† Brunei posterior values themselves are still `@test_broken` against
R-INLA — see the *Status* section below. The runtime number is honest.

Cold (TTFX) runs are ~5–10 s due to Julia's compilation. This is a
one-time cost per session, not an algorithmic property.

## What works today

* **`f(covariate, ModelType)` formula syntax** via `StatsModels.jl`.
* **Joint mode finder** with full sparse Hessian
  `H = Q + Aᵀ Diagonal(−h_η) A`, sparse Cholesky factorization, and
  Takahashi selected inverse for marginal variances (in original
  variable order — not the AMD-permuted one).
* **Hard sum-to-zero constraints** on intrinsic GMRFs (Besag/RW1/etc)
  via the augmented (KKT) Newton step. Constraint enforced to machine
  precision.
* **CCD integration over θ** — for `n_hyper ≥ 1` we evaluate the
  Laplace objective at every CCD node, normalize via softmax of the
  log-density gap, and return mixture means and variances.
* **Simplified-Laplace marginal-mean correction** (`Δx ∝ ½ H⁻¹ Aᵀ
  (h⁽³⁾ ⊙ σ²_η)`) projected back onto the constraint set when
  applicable.
* **Edgeworth correction** to `log π̂(y|θ)` (4th-derivative + 3rd-deriv
  cross terms on the constrained η-covariance).
* **R-INLA-style priors** out of the box: log-Gamma(1, 5e-5) on
  log-precisions; configurable per-model
  (e.g. `BivariateIIDModel(; a1=…, b1=…, a2=…, b2=…, rho_precision=…)`).
* **Multi-start BFGS** + PD-fail-safe: BFGS seeds at `theta0`,
  `fill(5, n_h)`, `fill(-2, n_h)` and picks the best feasible
  objective; non-finite seeds are filtered out.

## Status — R-INLA reference examples

| Example | Driver runs | Posterior parity | Test |
|---|:---:|:---:|---|
| **Salamander** (Bernoulli + IID×2) | ✓ | ✓ to 5 dp | [test/parity/salamander_test.jl](test/parity/salamander_test.jl) — 13/13 at 1 % × R-INLA SD |
| **Bivariate meta-analysis** (2diid) | ✓ | ✓ at 30 % rtol on precisions, 0.10 atol on ρ | [test/parity/bivariate_test.jl](test/parity/bivariate_test.jl) — 10/10 |
| **Brunei** (Besag with sum-to-zero) | ✓ | mechanics ✓; per-area means need full per-`x_i` Laplace | [test/parity/brunei_test.jl](test/parity/brunei_test.jl) — 3/3 + 1 broken |
| **Stationary SPDE** | ✓ | smoke (RMSE recovery from synthetic truth) | [test/parity/spde_test.jl](test/parity/spde_test.jl) — 5/5 |
| Dengue (besag → fbesag) | partial | not started | — |
| Joint longitudinal/spatial (Weibull + Gaussian) | not started | — | — |

## Usage

### Salamander mating (Bernoulli + IID random effects)

```julia
using IntegratedNestedLaplace, DataFrames, RDatasets
df = dataset("survey", "salamander")
df.Cross  = string.(df.Cross)
df.Female = string.(df.Female)
df.Male   = string.(df.Male)

res = inla(@formula(Mate ~ 1 + Cross + f(Female, IID) + f(Male, IID)),
           df, family = BernoulliLikelihood(), theta0 = [1.0, 1.0])

println(res)                          # mode → mean ± sd table
res.mean_latent[1:4]                  # posterior means of the 4 fixed effects
hyper_precision_mean(res, 1)          # E[exp(θ_F)] = posterior mean of τ_F
```

### Brunei-style areal Poisson with Besag random effect

```julia
using IntegratedNestedLaplace, SparseArrays, CSV, DataFrames
df  = CSV.read("examples/06_brunei_school_disparities/data/areas.csv", DataFrame)
adj = CSV.read("examples/06_brunei_school_disparities/data/adjacency.csv", DataFrame)
W = sparse(adj.i, adj.j, Float64.(adj.w), nrow(df), nrow(df))

besag = BesagModel(W; scale = true)   # scale.model = TRUE in R-INLA
res = inla(@formula(y ~ 1 + f(area, Besag)), df,
           family = PoissonLikelihood(),
           latent = besag,
           offset = log.(df.E),
           theta0 = [1.0])
```

### Bivariate IID with R-INLA-style prior

```julia
using IntegratedNestedLaplace, DataFrames, CSV
df = CSV.read("examples/05_meta_analysis/data/bivariate_synthetic.csv", DataFrame)

# Match R-INLA's f(., model="2diid", param=c(0.25,0.025,0.25,0.025,0,0.4))
biv = BivariateIIDModel(; a1 = 0.25, b1 = 0.025,
                         a2 = 0.25, b2 = 0.025,
                         rho_precision = 0.4)

res = inla(@formula(y ~ f(study, BivariateIID)), df,
           family = GaussianLikelihood(),
           latent = biv, theta0 = [4.0, 1.0, 0.5, 0.0])

# theta layout: [log τ_y, log τ₁, log τ₂, atanh ρ]
tanh(res.mean_hyper[4])               # posterior mean of ρ
```

## Reproducing the parity benchmark

The R-side fixtures are committed at `test/fixtures/<id>/rinla_reference.json`.
To regenerate (e.g. after upgrading R-INLA), run:

```sh
bench/Rrun.sh                       # all examples
bench/Rrun.sh 04_salamander_mating  # one example
```

Each `examples/<id>/rinla.R` script writes the reference JSON; the matching
Julia parity test loads it and asserts agreement.

## References

1. Rue, H., Martino, S., & Chopin, N. (2009). *Approximate Bayesian
   inference for latent Gaussian models by using integrated nested Laplace
   approximations*. JRSSB **71(2)** 319–392.
2. Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link
   between Gaussian fields and Gaussian Markov random fields: the SPDE
   approach*. JRSSB **73(4)** 423–498.
3. Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). *An
   intuitive Bayesian spatial model for disease mapping that accounts
   for scaling*. Statistical Methods in Medical Research **25(4)**
   1145–1165 (BYM2 reparameterisation, used by `scale.model = TRUE` in
   `BesagModel`).
