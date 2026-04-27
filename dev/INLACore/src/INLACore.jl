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

export find_mode, SparseHessianCache, sparse_hessian!, gmrf_newton, gmrf_newton_full
export takahashi_factor!, takahashi_marginals, integration_nodes
export sparse_trace_inverse, gaussian_ll_kernel, gaussian_ll_grad_kernel, sparse_logdet
export inverse_permutation, takahashi_diag

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

Specialized Sparse Newton solver. Finds the mode of `p(x|y) ∝ exp(ℓ(x) − ½ x' Q x)`
when `ℓ` has a *diagonal* Hessian in `x` itself (the per-coordinate setup, e.g.
when `x = η`). For models with a wide design matrix `A` so that `η = A x`, use
`gmrf_newton_full` instead.

`grad_lik` returns `∇ℓ(x)`; `hess_lik` returns the diagonal of `∇²ℓ(x)`.
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

"""
    gmrf_newton_full(grad_eta, hess_eta_diag, A, Q, x0;
                     constraint_A = nothing, constraint_e = nothing,
                     max_iter = 50, tol = 1e-8)

Sparse Newton for latent x with linear predictor η = A·x and a likelihood whose
Hessian in η is diagonal (the standard generalized-linear case). The full
sparse Hessian in x is `H = Q + A' Diagonal(−h_η(η)) A`.

Inputs:
* `grad_eta(eta)`  — `∂ℓ/∂η_i` per observation, length `n_obs`.
* `hess_eta_diag(eta)` — `∂²ℓ/∂η_i²` per observation (negative for log-concave
  likelihoods), length `n_obs`.
* `A`  — `n_obs × n_latent` sparse design.
* `Q`  — `n_latent × n_latent` sparse latent precision.
* `constraint_A` — optional `k × n_latent` sparse matrix; if supplied the
  Newton step solves the augmented KKT system to enforce `A_c · x = e_c` at
  every iteration.
* `constraint_e` — `k`-vector of constraint targets (defaults to zeros).

Returns the latent mode `x*` (vector). The Cholesky factor is *not* returned;
the caller can reconstruct it cheaply at `x*`.
"""
function gmrf_newton_full(grad_eta, hess_eta_diag,
                          A::SparseMatrixCSC{T}, Q::SparseMatrixCSC{T},
                          x0::Vector{T};
                          constraint_A::Union{Nothing,SparseMatrixCSC{T,Int}} = nothing,
                          constraint_e::Union{Nothing,AbstractVector{T}} = nothing,
                          max_iter::Int = 50, tol = T(1e-8)) where {T}
    x = copy(x0)
    AT = sparse(A')
    have_constraint = constraint_A !== nothing && size(constraint_A, 1) > 0
    if have_constraint
        e = constraint_e === nothing ? zeros(T, size(constraint_A, 1)) : collect(constraint_e)
        AcT = sparse(constraint_A')
        # Project initial x onto the constraint set so subsequent Newton steps
        # only need to update toward the constrained mode.
        _ = e   # used below
    end
    for _ in 1:max_iter
        eta = A * x
        g_eta = grad_eta(eta)
        h_eta = hess_eta_diag(eta)
        g = Q * x - AT * g_eta
        D = spdiagm(0 => -h_eta)
        H = Q + AT * D * A
        F = cholesky(Symmetric(H))
        if have_constraint
            # Solve the augmented system
            #   [H  A_c'] [dx ] = [-g          ]
            #   [A_c  0 ] [dλ ]   [e − A_c x   ]
            # via Schur complement on the dual variable.
            z = F \ g                      # H^{-1} g
            W = F \ Matrix(constraint_A')  # H^{-1} A_c'  (n×k)
            S = constraint_A * W           # k×k Schur
            r = constraint_e === nothing ? -constraint_A * x :
                                            (constraint_e .- constraint_A * x)
            # Augmented step: dx = -z + W * (S^{-1} (A_c z + r))
            rhs = constraint_A * z + r
            dλ  = S \ rhs
            dx  = -z + W * dλ
            x .+= dx
            if norm(dx, Inf) < tol
                return x
            end
        else
            dx = F \ g
            x .-= dx
            if norm(dx, Inf) < tol
                return x
            end
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
    # `logdet(F)` for a CHOLMOD.Factor already returns log|A| where F = chol(A),
    # not log|L|. Earlier code had a stray ×2 factor here that cancelled with a
    # similar bug in the driver; now both call sites use the canonical scaling.
    return T(logdet(F))
end

"""
    inverse_permutation(p)

Return the inverse permutation of `p`, i.e. `q[p[i]] = i`. Useful for mapping
back from CHOLMOD's AMD ordering to the original variable order.
"""
function inverse_permutation(p::AbstractVector{<:Integer})
    q = similar(p)
    @inbounds for i in eachindex(p)
        q[p[i]] = i
    end
    return q
end

"""
    takahashi_diag(F)

Return per-coordinate variances `Σ_ii` of the inverse of a CHOLMOD
sparse-Cholesky `F` (i.e. `F.factors == F.L F.L'` after permutation `F.p`).
The output is in the *original* variable order — the AMD permutation is
inverted internally.

This is the right thing to use to extract `marginals_latent` after factoring a
GMRF Hessian.
"""
function takahashi_diag(F::SparseArrays.CHOLMOD.Factor{T}) where {T}
    L = sparse(F.L)
    S = copy(L)
    takahashi_factor!(S, L)
    n = size(L, 1)
    p = F.p
    vars = zeros(T, n)
    @inbounds for j in 1:n
        # diagonal entry of S in permuted column j → original index p[j]
        for ptr in S.colptr[j]:(S.colptr[j+1]-1)
            if S.rowval[ptr] == j
                vars[p[j]] = S.nzval[ptr]
                break
            end
        end
    end
    return vars
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
