using INLACore
using Test
using OptimizationOptimJL
using LinearAlgebra
using SparseArrays
using ADTypes
using DifferentiationInterface
using SparseConnectivityTracer

@testset "INLACore.jl" begin
    @testset "find_mode" begin
        f(u, p) = (u[1] - 3.0)^2 + (u[2] + 2.0)^2
        u0 = [0.0, 0.0]
        res = find_mode(f, u0)
        @test isapprox(res.u[1], 3.0, atol=1e-4)
        @test isapprox(res.u[2], -2.0, atol=1e-4)
    end

    @testset "Sparse Hessian (DI)" begin
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]
        detector = TracerSparsityDetector()
        backend = AutoSparse(AutoForwardDiff(); sparsity_detector=detector)
        cache = SparseHessianCache(f, backend, x)
        H = sparse(I, 3, 3) * 1.0
        sparse_hessian!(H, f, x, cache)
        @test all(H .== 2.0 * sparse(I, 3, 3))
    end

    @testset "Takahashi Equations" begin
        n = 10
        Q = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.5, n), 1 => fill(-1.0, n-1))
        true_vars = diag(inv(Matrix(Q)))
        computed_vars = takahashi_marginals(Q)
        @test isapprox(computed_vars, true_vars, atol=1e-10)
    end

    @testset "Sparse Trace Inverse" begin
        n = 5
        Q = spdiagm(-1 => fill(-0.5, n-1), 0 => fill(2.0, n), 1 => fill(-0.5, n-1))
        A = spdiagm(0 => ones(n))
        F = cholesky(Symmetric(Q))
        L = sparse(F.L)
        S = copy(L)
        takahashi_factor!(S, L)
        tr_computed = sparse_trace_inverse(S, A)
        tr_true = tr(inv(Matrix(Q)) * Matrix(A))
        @test isapprox(tr_computed, tr_true, atol=1e-10)
    end

    @testset "KernelAbstractions Likelihood" begin
        using KernelAbstractions
        # Use CPU backend for tests
        backend = CPU()
        n = 100
        y = randn(n)
        eta = zeros(n)
        tau = 1.0
        out = zeros(n)
        
        kernel! = gaussian_ll_kernel(backend)
        kernel!(out, y, eta, tau, ndrange=n)
        KernelAbstractions.synchronize(backend)
        
        # Verify first element
        expected = -0.5*log(2π) + 0.5*log(tau) - 0.5*tau*y[1]^2
        @test isapprox(out[1], expected)
    end
end
