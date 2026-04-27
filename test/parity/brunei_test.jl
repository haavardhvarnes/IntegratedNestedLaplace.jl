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
    res = inla(@formula(y ~ 1 + f(area, Besag)), df,
               family = PoissonLikelihood(),
               latent = besag,
               offset = log.(df.E),
               theta0 = [1.0])

    n_obs = nrow(df)
    @test length(res.mean_latent) == 1 + n_obs

    # The hard sum-to-zero constraint *is* working: the area effect sums to
    # zero numerically, modulo machine precision. This is the Phase 3
    # correctness milestone for intrinsic GMRFs.
    u_block = res.mean_latent[(2):end]
    @test abs(sum(u_block)) < 1e-8

    # Per-area linear-predictor parity vs R-INLA is still blocked on the
    # simplified-Laplace skewness correction (Phase 4). The Gaussian Laplace
    # at the mode underestimates the spatial-effect posterior for sharply
    # log-concave likelihoods like Poisson on a rank-deficient prior; R-INLA
    # gets ≈ R-INLA τ = 19, Julia's mode (without skew correction) drifts to
    # τ → ∞ where the latent collapses. The constraint mechanics are correct;
    # the approximation quality is not yet good enough for parity.
    rinla_lp_mean = ref["linear_predictor"]["mean"]
    julia_lp_mean = res.mean_latent[1] .+ res.mean_latent[1 .+ df.area]
    @test_broken maximum(abs.(julia_lp_mean .- rinla_lp_mean)) < 0.1
end
