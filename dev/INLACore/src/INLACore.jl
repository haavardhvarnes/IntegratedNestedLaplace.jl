module INLACore

using LinearAlgebra
using SparseArrays
using Optimization
using OptimizationOptimJL
using ADTypes
using DifferentiationInterface
using ExperimentalDesign
using Enzyme
import ForwardDiff
using KernelAbstractions

export find_mode, SparseHessianCache, sparse_hessian!, gmrf_newton
export takahashi_factor!, takahashi_marginals, integration_nodes
export sparse_trace_inverse, gaussian_ll_kernel, gaussian_ll_grad_kernel, sparse_logdet

# --- Sparse Hessian (DI) ---

struct SparseHessianCache{B, E}
    backend::B
    extras::E
end

function SparseHessianCache(f, backend::AbstractADType, x)
    extras = prepare_hessian(f, backend, x)
    return SparseHessianCache(backend, extras)
end

function sparse_hessian!(H, f, x, cache::SparseHessianCache)
    hessian!(f, H, cache.extras, cache.backend, x)
    return H
end

# --- GMRF Specialized Solver ---

"""
    gmrf_newton(grad_lik, hess_lik, Q, x0; max_iter=20, tol=1e-8)

Specialized Sparse Newton solver. Finds mode of p(x|y).
grad_lik and hess_lik should return vectors of likelihood gradient and Hessian diagonal.
"""
function gmrf_newton(grad_lik, hess_lik, Q::SparseMatrixCSC{T}, x0::Vector{T}; max_iter=20, tol=1e-8) where T
    x = copy(x0)
    for i in 1:max_iter
        g_l = grad_lik(x)
        h_l = hess_lik(x)
        g = Q * x - g_l
        H = Q + spdiagm(0 => -h_l)
        F = cholesky(Symmetric(H))
        dx = F \ g
        x .-= dx
        if norm(dx, Inf) < tol
            return x
        end
    end
    return x
end

# --- Takahashi ---

function takahashi_factor!(S::SparseMatrixCSC{T}, L::SparseMatrixCSC{T}) where T
    n = size(L, 1)
    fill!(S.nzval, zero(T))
    diag_indices = zeros(Int, n)
    for j in 1:n
        for ptr in L.colptr[j]:(L.colptr[j+1]-1)
            L.rowval[ptr] == j && (diag_indices[j] = ptr; break)
        end
    end
    for i in n:-1:1
        r1_i, r2_i = L.colptr[i], L.colptr[i+1] - 1
        idx_ii = diag_indices[i]
        L_ii = L.nzval[idx_ii]
        inv_L_ii = one(T) / L_ii
        for ptr_j in r2_i:-1:(idx_ii+1)
            j = L.rowval[ptr_j]
            sum_val = zero(T)
            for ptr_k in (idx_ii+1):r2_i
                k = L.rowval[ptr_k]
                sum_val += L.nzval[ptr_k] * get_sparse_val(S, k, j)
            end
            S.nzval[ptr_j] = -sum_val * inv_L_ii
        end
        sum_val_diag = zero(T)
        for ptr_k in (idx_ii+1):r2_i
            sum_val_diag += L.nzval[ptr_k] * S.nzval[ptr_k]
        end
        S.nzval[idx_ii] = inv_L_ii * (inv_L_ii - sum_val_diag)
    end
    return S
end

@inline function get_sparse_val(S::SparseMatrixCSC{T}, i::Int, j::Int) where T
    if i < j; i, j = j, i; end
    r1, r2 = S.colptr[j], S.colptr[j+1] - 1
    ptr = searchsortedfirst(view(S.rowval, r1:r2), i)
    if ptr <= r2 - r1 + 1 && S.rowval[r1 + ptr - 1] == i
        return S.nzval[r1 + ptr - 1]
    end
    return zero(T)
end

function takahashi_marginals(Q::SparseMatrixCSC{T}) where T
    F = cholesky(Symmetric(Q))
    L = sparse(F.L)
    S = copy(L)
    takahashi_factor!(S, L)
    n = size(Q, 1)
    vars = zeros(T, n)
    p = F.p
    for j in 1:n
        for ptr in S.colptr[j]:(S.colptr[j+1]-1)
            if S.rowval[ptr] == j
                vars[p[j]] = S.nzval[ptr]
                break
            end
        end
    end
    return vars
end

function sparse_trace_inverse(S::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}) where T
    n = size(S, 1)
    tr_val = zero(T)
    for j in 1:n
        for ptr_A in A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[ptr_A]
            tr_val += get_sparse_val(S, i, j) * A.nzval[ptr_A]
        end
    end
    return tr_val
end

function sparse_logdet(Q::SparseMatrixCSC{T}) where T
    Q_f64 = SparseMatrixCSC{Float64, Int}(Q)
    F = cholesky(Symmetric(Q_f64))
    return T(2.0 * logdet(F))
end

# --- Kernels ---

@kernel function gaussian_ll_kernel(out, @Const(y), @Const(eta), tau)
    I = @index(Global)
    T = typeof(tau)
    @inbounds out[I] = T(-0.9189385) + T(0.5) * log(tau) - T(0.5) * tau * (y[I] - eta[I])^2
end

@kernel function gaussian_ll_grad_kernel(out_grad, @Const(y), @Const(eta), tau)
    I = @index(Global)
    T = typeof(tau)
    @inbounds out_grad[I] = tau * (y[I] - eta[I])
end

# --- Integration ---

function integration_nodes(mode::AbstractVector{T}, hessian::AbstractMatrix{T}; method=:ccd) where T
    n_dims = length(mode)
    if n_dims == 1
        sd = one(T) / sqrt(hessian[1,1])
        return [mode, mode .+ T(1.5)*sd, mode .- T(1.5)*sd]
    end
    z_points = T.(ccdesign(n_dims))
    evals, evecs = eigen(Symmetric(hessian))
    evals = max.(evals, T(1e-9))
    trans = evecs * diagm(one(T) ./ sqrt.(evals))
    n_nodes = size(z_points, 1)
    nodes = [mode + trans * z_points[i, :] for i in 1:n_nodes]
    return nodes
end

function find_mode(f, x0, p=nothing; solver=Newton(), adtype=Optimization.AutoForwardDiff(), kwargs...)
    optf = OptimizationFunction(f, adtype)
    prob = OptimizationProblem(optf, x0, p)
    return solve(prob, solver; kwargs...)
end

end # module
