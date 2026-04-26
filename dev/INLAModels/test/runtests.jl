using INLAModels
using Test
using SparseArrays
using LinearAlgebra

@testset "INLAModels.jl" begin
    @testset "Likelihoods" begin
        g = GaussianLikelihood()
        @test log_pdf(g, 1.0, 1.0, 1.0) ≈ -0.9189385332046727 # log(1/sqrt(2pi))
        
        p = PoissonLikelihood()
        @test log_pdf(p, 0, 0.0) ≈ -1.0 # lambda=1, y=0 -> e^-1 * 1^0 / 0! = e^-1
    end

    @testset "Precision Matrices" begin
        iid = IIDModel()
        Q_iid = precision_matrix(iid, 3, 2.0)
        @test Q_iid == 2.0 * I(3)
        
        rw1 = RW1Model()
        Q_rw1 = precision_matrix(rw1, 3, 1.0)
        @test Array(Q_rw1) == [1.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 1.0]
    end

    @testset "Integrated Mode Finding" begin
        using INLACore
        using OptimizationOptimJL
        
        # Simple Model: y_i ~ N(eta_i, 1), eta ~ N(0, Q^-1)
        # We want to find the mode of x = eta
        n = 5
        y = [1.1, 0.9, 1.2, 0.8, 1.0]
        Q = Array(precision_matrix(RW1Model(), n, 10.0))
        
        # Negative log-posterior (objective to minimize)
        function objective(x, p)
            # Log-likelihood: sum log_pdf(Gaussian, y_i, x_i, 1.0)
            ll = sum(log_pdf(GaussianLikelihood(), y[i], x[i], 1.0) for i in 1:n)
            # Log-prior: -0.5 * x' * Q * x
            lp = -0.5 * dot(x, Q * x)
            return -(ll + lp)
        end
        
        x0 = zeros(n)
        res = find_mode(objective, x0)
        
        @test length(res.u) == n
        # The mode should be somewhere near y because precision is high
        @test isapprox(res.u, y, atol=0.5)
    end
end
