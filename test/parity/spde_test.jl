using IntegratedNestedLaplace
using INLASpatial
using DataFrames
using CSV
using Random
using Statistics
using Test

# Stationary SPDE regression test.
#
# Strict parity against R-INLA's `inla.spde2.matern` requires aligning the
# τ/κ parameterization (Lindgren-Rue 2011 α=2 vs Julia's
# `K · diag(C)⁻¹ · K` form). That alignment is queued as separate work.
#
# What this test guarantees:
#   1. The SPDE example runs end-to-end (PD-fail-safe in laplace_eval works).
#   2. The CCD / SLA pipeline with a 3-hyper Gaussian likelihood (τ_y, log κ,
#      log τ_spde) returns a finite posterior.
#   3. The posterior mean of the latent field at observed locations
#      reconstructs the underlying smooth signal within reasonable RMSE.

@testset "Stationary SPDE smoke" begin
    Random.seed!(20260427)
    data_path = joinpath(@__DIR__, "..", "..", "examples", "03_swiss_rainfall", "data", "locations.csv")
    df = CSV.read(data_path, DataFrame)
    coords = collect(zip(df.x, df.y_coord))

    mesh = build_mesh(coords)
    C, G = spde_matrices(mesh)
    spde = SPDEModel(C, G)

    res = inla(@formula(y ~ f(loc_id, SPDE)), df,
               family = GaussianLikelihood(),
               latent = spde,
               theta0 = [3.0, 1.0, 1.0])

    n_v = size(C, 1)
    @test length(res.mean_latent) == 1 + n_v   # intercept + spatial vertices
    @test length(res.mode_hyper) == 3
    @test all(isfinite, res.mean_latent)
    @test all(isfinite, res.mode_hyper)

    # Recover the smooth signal at observed locations.
    truth = [sin(c[1] * 3) + cos(c[2] * 3) for c in coords]
    pred  = res.mean_latent[1] .+ res.mean_latent[1 .+ df.loc_id]
    rmse  = sqrt(mean((pred .- truth).^2))
    @test rmse < 0.2   # synthetic noise SD = 0.1; this is generous
end
