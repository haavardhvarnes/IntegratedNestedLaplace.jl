using IntegratedNestedLaplace
using DataFrames
using CSV
using SparseArrays
using Test

include(joinpath(@__DIR__, "parity_helpers.jl"))

const EXAMPLE = "06_brunei_school_disparities"

function load_brunei_data()
    here = joinpath(@__DIR__, "..", "..", "examples", EXAMPLE, "data")
    df  = CSV.read(joinpath(here, "areas.csv"), DataFrame)
    adj = CSV.read(joinpath(here, "adjacency.csv"), DataFrame)
    n = nrow(df)
    W = sparse(adj.i, adj.j, Float64.(adj.w), n, n)
    return df, W
end

@testset "Brunei (besag scale.model=TRUE)" begin
    ref = load_reference(EXAMPLE)
    df, W = load_brunei_data()
    @test nrow(df) == ref["n"]

    besag = BesagModel(W; scale = true)
    # `fixed_precision = 0` matches R-INLA's `prec.intercept = 0` and triggers
    # the improper-prior augmentation path (Phase 6c.2.b in BRUNEI_FIX.md):
    # the unidentifiable direction `v = (1, -1, …, -1)/√(n+1)` reduces, after
    # Gram-Schmidt against the besag sum-to-zero row, to `e_intercept`. The
    # driver appends that row to A_c so `H + A_full' A_full` is full rank.
    res = inla(@formula(y ~ 1 + f(area, Besag)), df,
               family = PoissonLikelihood(),
               latent = besag,
               offset = log.(df.E),
               theta0 = [1.0],
               fixed_precision = 0)

    n_obs = nrow(df)
    @test length(res.mean_latent) == 1 + n_obs

    # The hard sum-to-zero constraint *is* working: the area effect sums to
    # zero numerically, modulo machine precision.
    u_block = res.mean_latent[(2):end]
    @test abs(sum(u_block)) < 1e-8

    # Per-area linear-predictor parity vs R-INLA. With Phase 6g (β-pin
    # removed) BFGS finds the true posterior mode at θ ≈ 1.46 — close to
    # R-INLA's 1.87 in the same local basin but not identical. The mode
    # offset of 0.4 in θ translates to LP differences up to ~35 % of
    # max R-INLA SD on the most-uncertain areas. Bound:
    # `max(0.05, 0.40 × max(R-INLA SD))`.
    rinla_lp_mean = ref["linear_predictor"]["mean"]
    rinla_lp_sd   = ref["linear_predictor"]["sd"]
    julia_lp_mean = res.mean_latent[1] .+ res.mean_latent[1 .+ df.area]
    @test maximum(abs.(julia_lp_mean .- rinla_lp_mean)) <
          max(0.05, 0.40 * maximum(rinla_lp_sd))

    # τ posterior MODE comparison. Critical reframe (Phase 6g+ Phase A):
    # R-INLA's reported posterior MODE for τ is 4.40 (= exp(1.48)), NOT
    # 7.97 (the median, which we previously erroneously compared against).
    # Our BFGS lands at θ ≈ 1.46 (τ ≈ 4.31), within 0.02 nats of R-INLA's
    # 1.48 — both are local minima in the same JP basin. R-INLA's
    # `summary.hyperpar$mode` is the marginal-posterior MAP estimate of τ;
    # our `mode_hyper[1]` is the BFGS-found θ at the joint posterior mode.
    julia_tau_mode = exp(res.mode_hyper[1])
    rinla_tau_mode = ref["hyper"]["Precision for area"]["mode"]
    @test isapprox(julia_tau_mode, rinla_tau_mode; rtol = 0.10)

    # τ posterior MEAN comparison. R-INLA's posterior on τ is heavily
    # right-skewed (mean=19.17, median=7.97, 0.975q=82.32). The factor-of-
    # 4.4 spread between mode and mean reflects the heavy right tail
    # extending past τ=82. Our 11-point CCD grid centered at θ=1.46 with
    # asymmetric skewness corrections (Phase 6d) doesn't extend far
    # enough into the right tail to capture the full mean. We currently
    # land at τ_mean ≈ 6–7, ~65 % below R-INLA's 19.17. The gap is a
    # CCD-coverage issue (R-INLA uses different grid extension and/or
    # tail extrapolation), not a mode-finding error. Tightening this
    # bound past 50 % rtol requires either (a) widening the CCD grid
    # in the heavy-tail direction, (b) implementing R-INLA's tail-
    # extrapolation for marginal hyperposteriors, or (c) full-Laplace
    # strategy.
    julia_tau_mean = hyper_precision_mean(res, 1)
    rinla_tau_mean = ref["hyper"]["Precision for area"]["mean"]
    @test_broken isapprox(julia_tau_mean, rinla_tau_mean; rtol = 0.50)
end
