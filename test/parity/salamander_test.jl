using IntegratedNestedLaplace
using DataFrames
using CSV
using Test

include(joinpath(@__DIR__, "parity_helpers.jl"))

const EXAMPLE = "04_salamander_mating"

@testset "Salamander parity vs R-INLA" begin
    ref = load_reference(EXAMPLE)
    data_path = joinpath(@__DIR__, "..", "..", "examples", EXAMPLE, "data", "salamander.csv")
    df = CSV.read(data_path, DataFrame)
    df.Female = string.(df.Female)
    df.Male   = string.(df.Male)
    df.Cross  = string.(df.Cross)
    @test nrow(df) == ref["n"]

    # Warm-up + timed run (warm).
    res = inla(@formula(Mate ~ 1 + Cross + f(Female, IID) + f(Male, IID)),
               df, family = BernoulliLikelihood(), theta0 = [1.0, 1.0])
    t = @elapsed (res = inla(@formula(Mate ~ 1 + Cross + f(Female, IID) + f(Male, IID)),
                              df, family = BernoulliLikelihood(), theta0 = [1.0, 1.0]))

    # n_latent = 1 (intercept) + 3 (Cross dummies) + 60 (Female) + 60 (Male) = 124.
    n_fixed = 4
    @test length(res.mode_latent) == n_fixed + 60 + 60

    julia_means = res.mode_latent[1:n_fixed]
    julia_sds   = sqrt.(max.(0.0, res.marginals_latent[1:n_fixed]))

    fixed_terms = ["(Intercept)", "CrossR/W", "CrossW/R", "CrossW/W"]
    for (i, term) in pairs(fixed_terms)
        assert_fixed(ref, term, julia_means[i], julia_sds[i])
    end

    assert_log_precision(ref, "Precision for Female", res.mode_hyper[1])
    assert_log_precision(ref, "Precision for Male",   res.mode_hyper[2])
    assert_runtime(ref, t)
end
