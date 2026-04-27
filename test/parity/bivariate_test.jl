using IntegratedNestedLaplace
using DataFrames
using CSV
using Test

include(joinpath(@__DIR__, "parity_helpers.jl"))

const EXAMPLE = "05_meta_analysis"

@testset "Bivariate IID parity vs R-INLA (2diid)" begin
    ref = load_reference(EXAMPLE)
    data_path = joinpath(@__DIR__, "..", "..", "examples", EXAMPLE, "data", "bivariate_synthetic.csv")
    df = CSV.read(data_path, DataFrame)
    @test nrow(df) == ref["n"]
    n_studies = ref["n_studies"]

    # Match R-INLA's prior: param = c(0.25, 0.025, 0.25, 0.025, 0, 0.4)
    biv = BivariateIIDModel(; a1 = 0.25, b1 = 0.025,
                              a2 = 0.25, b2 = 0.025,
                              rho_precision = 0.4)

    res = inla(@formula(y ~ f(study, BivariateIID)), df,
               family = GaussianLikelihood(),
               latent = biv,
               theta0 = [4.0, 1.0, 0.5, 0.0])
    t = @elapsed (res = inla(@formula(y ~ f(study, BivariateIID)), df,
                              family = GaussianLikelihood(),
                              latent = biv,
                              theta0 = [4.0, 1.0, 0.5, 0.0]))

    # n_latent = 1 (intercept) + 2 * n_studies (per-pair (u, v))
    @test length(res.mean_latent) == 1 + 2 * n_studies
    @test length(res.mode_hyper) == 4   # τ_y, τ₁, τ₂, atanh ρ

    # Hyperparameter parity. Order in `mean_hyper`:
    #   [log τ_y, log τ₁, log τ₂, atanh ρ]
    julia_tau_y = exp(res.mean_hyper[1])
    julia_tau1  = exp(res.mean_hyper[2])
    julia_tau2  = exp(res.mean_hyper[3])
    julia_rho   = tanh(res.mean_hyper[4])

    rinla_tau_y = ref["hyper"]["Precision for the Gaussian observations"]["mean"]
    rinla_tau1  = ref["hyper"]["Precision for diid (first component)"]["mean"]
    rinla_tau2  = ref["hyper"]["Precision for diid (second component)"]["mean"]
    rinla_rho   = ref["hyper"]["Rho for diid"]["mean"]

    # Heavy-tailed posteriors on precisions (especially τ_y on a tiny n=60
    # sample); use loose 30 % rtol. ρ posterior is concentrated near zero,
    # so use absolute tolerance.
    @test isapprox(julia_tau1, rinla_tau1; rtol = 0.30)
    @test isapprox(julia_tau2, rinla_tau2; rtol = 0.30)
    @test isapprox(julia_rho,  rinla_rho;  atol = 0.10)
    # τ_y goes to 22 000 in both — the posterior is nearly flat on the
    # log-precision scale at high values. Same-order-of-magnitude check:
    @test 0.1 ≤ julia_tau_y / rinla_tau_y ≤ 10

    # Latent posterior parity. Both sides use the *interleaved* layout
    # (u₁, v₁, u₂, v₂, …) — the fixture builds R-INLA's `diid = 2(i-1)+type`,
    # which matches Julia's per-pair indexing exactly. R-INLA's
    # `random.diid` has 2·(2n) rows, the latter half being internal
    # placeholder ID=1 zeros; take the first 2n.
    n_latent_random = 2 * n_studies
    rinla_means = Float64.(ref["random_diid"]["mean"][1:n_latent_random])
    julia_random = res.mean_latent[2:end]
    @test length(julia_random) == n_latent_random
    rmse_means = sqrt(sum((julia_random .- rinla_means).^2) / n_latent_random)
    @test rmse_means < 0.10

    assert_runtime(ref, t; ratio = 5.0)
end
