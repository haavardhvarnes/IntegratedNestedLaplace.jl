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
        # Canonical RW1 Q with a tiny numerical jitter on the diagonal.
        expected = [1.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 1.0]
        @test isapprox(Array(Q_rw1), expected; atol = 1e-4)
    end

    @testset "Hyperparameter interface" begin
        @test n_hyper(IIDModel()) == 1
        @test n_hyper(RW1Model()) == 1
        @test n_hyper(BivariateIIDModel()) == 3
        @test n_hyper(GaussianLikelihood()) == 1
        @test n_hyper(BernoulliLikelihood()) == 0
        @test n_hyper(PoissonLikelihood()) == 0

        # assemble_Q on log-precision parametrization.
        Q = assemble_Q(IIDModel(), [log(2.0)], 4)
        @test isapprox(Array(Q), 2.0 * Matrix{Float64}(I, 4, 4))
    end

    @testset "Mode finding (Gaussian + flat prior)" begin
        using INLACore
        using OptimizationOptimJL

        # Simple model: y_i ~ N(x_i, 1), with a near-flat prior on x.
        # The mode in this case is exactly y.
        n = 5
        y = [1.1, 0.9, 1.2, 0.8, 1.0]
        Q = Matrix(precision_matrix(IIDModel(), n, 1e-8))  # near-flat prior
        function objective(x, _p)
            ll = sum(log_pdf(GaussianLikelihood(), y[i], x[i], 1.0) for i in 1:n)
            lp = -0.5 * dot(x, Q * x)
            return -(ll + lp)
        end
        res = find_mode(objective, zeros(n))
        @test length(res.u) == n
        @test isapprox(res.u, y; atol = 1e-4)
    end
end
