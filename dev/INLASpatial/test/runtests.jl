using INLASpatial
using Test
using Meshes
using SparseArrays
using LinearAlgebra

@testset "INLASpatial.jl" begin
    @testset "Mesh Building" begin
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.2)]
        mesh = build_mesh(coords)
        @test nvertices(mesh) >= 4
    end

    @testset "SPDE Matrices" begin
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] # Single triangle
        mesh = build_mesh(coords)
        C, G = spde_matrices(mesh)
        @test size(C) == (3, 3)
        @test size(G) == (3, 3)
        @test sum(C) ≈ 0.5 # Total area
        @test all(diag(G) .> 0)
        
        Q = spde_precision(C, G, 1.0, 1.0)
        @test size(Q) == (3, 3)
        @test isapprox(Q, Q', atol=1e-10) # Symmetry
    end
end
