using IntegratedNestedLaplace
using Test
using DataFrames
using SparseArrays

@testset "IntegratedNestedLaplace.jl" begin
    @testset "Intercept-only Gaussian (y ~ 1)" begin
        data = DataFrame(y = [1.1, 0.9, 1.2, 0.8, 1.0], x = 1:5)
        res  = inla(@formula(y ~ 1), data)
        # Latent field = intercept only.
        @test length(res.mode_latent) == 1
        @test length(res.marginals_latent) == 1
        @test all(res.marginals_latent .> 0)
        # Gaussian has one likelihood hyper (log τ_y).
        @test length(res.mode_hyper) == 1
    end

    @testset "Gaussian + IID random effect" begin
        data = DataFrame(y = [1.1, 0.9, 1.2, 0.8, 1.0], grp = [1, 1, 2, 2, 3])
        res  = inla(@formula(y ~ 1 + f(grp, IID)), data)
        # 1 (intercept) + 3 (IID levels) = 4 latent.
        @test length(res.mode_latent) == 4
        @test length(res.marginals_latent) == 4
        # τ_y, τ_grp.
        @test length(res.mode_hyper) == 2
    end

    @testset "Gaussian + RW1" begin
        data = DataFrame(y = [1.1, 0.9, 1.2, 0.8, 1.0], t = 1:5)
        res  = inla(@formula(y ~ 1 + f(t, RW1)), data)
        @test length(res.mode_latent) == 6  # intercept + 5 RW1 levels
        @test length(res.mode_hyper) == 2   # τ_y, τ_RW1
        @test all(res.marginals_latent .> 0)
    end

    @testset "SPDE basic API" begin
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
        data = DataFrame(y = [1.0, 0.5, 0.5, 0.0, 0.8], loc_id = 1:5)
        mesh = build_mesh(coords)
        C, G = spde_matrices(mesh)
        spde = SPDEModel(C, G)
        res  = inla(@formula(y ~ 1 + f(loc_id, SPDE)), data, latent = spde)
        n_v = size(C, 1)
        @test length(res.mode_latent) == 1 + n_v
        @test length(res.mode_hyper)  == 1 + 2  # τ_y, log κ, log τ_spde
        # Smoke-test the show method.
        io = IOBuffer()
        show(io, MIME("text/plain"), res)
        @test occursin("INLA Result", String(take!(io)))
    end

    @testset "Bernoulli + multiple IID effects" begin
        data = DataFrame(
            y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            a = [1, 1, 2, 2, 1, 1, 2, 2, 1, 2],
            b = [1, 2, 1, 2, 1, 2, 1, 2, 2, 1],
        )
        res = inla(@formula(y ~ 1 + f(a, IID) + f(b, IID)), data,
                   family = BernoulliLikelihood(), theta0 = [1.0, 1.0])
        @test length(res.mode_latent) == 1 + 2 + 2
        @test length(res.mode_hyper)  == 2  # τ_a, τ_b (Bernoulli has no hyper)
    end
end
