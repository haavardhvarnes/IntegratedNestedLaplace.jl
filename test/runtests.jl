using IntegratedNestedLaplace
using Test
using DataFrames

@testset "IntegratedNestedLaplace.jl" begin
    @testset "Simple INLA API" begin
        # Create some data
        data = DataFrame(y = [1.1, 0.9, 1.2, 0.8, 1.0], x = 1:5)
        
        # Run a simple INLA
        # y ~ 1
        res = inla(@formula(y ~ 1), data)
        
        @test length(res.mode_latent) == 5
        @test length(res.marginals_latent) == 5
        @test all(res.marginals_latent .> 0)
    end

    @testset "SPDE INLA API" begin
        # Create some spatial data
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        data = DataFrame(y = [1.0, 0.5, 0.5, 0.0], x1 = 1:4)
        
        # 1. Build Mesh and SPDE matrices
        mesh = build_mesh(coords)
        C, G = spde_matrices(mesh)
        
        # 2. Define SPDE latent model
        spde = SPDEModel(C, G)
        
        # 3. Run INLA
        res = inla(@formula(y ~ 1 + f(x1, SPDE)), data, latent=spde)
        
        @test length(res.mode_latent) == 4
        @test length(res.marginals_latent) == 4
        
        # Test summary printing
        show(stdout, MIME("text/plain"), res)
    end

    @testset "Full INLA with Hyper-integration" begin
        data = DataFrame(y = [1.1, 0.9, 1.2, 0.8, 1.0], x = 1:5)
        # Standard INLA: integrate over theta = log(tau)
        # Using the formula to pass RW1
        res = inla(@formula(y ~ 1 + f(x, RW1)), data)
        
        @test length(res.mode_hyper) == 1
        @test length(res.nodes_hyper) > 1
        @test isapprox(sum(res.weights_hyper), 1.0)
    end
end
