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
* `factor_augmented` — when `true`, the Cholesky used in the Schur step
  factors `H + A_c' A_c` instead of `H`. Required when `H` is rank-deficient
  along `null(H) ⊆ span(constraint_A')` (e.g. improper-prior augmentation
  for an intrinsic GMRF + improper intercept). On `ker(constraint_A)`,
  `H + A_c' A_c = H`, so the resulting Newton direction is unchanged.

Returns the latent mode `x*` (vector). The Cholesky factor is *not* returned;
the caller can reconstruct it cheaply at `x*`.
"""
function gmrf_newton_full(grad_eta, hess_eta_diag,
                          A::SparseMatrixCSC{T}, Q::SparseMatrixCSC{T},
                          x0::Vector{T};
                          constraint_A::Union{Nothing,SparseMatrixCSC{T,Int}} = nothing,
                          constraint_e::Union{Nothing,AbstractVector{T}} = nothing,
                          factor_augmented::Bool = false,
                          max_iter::Int = 50, tol = T(1e-8)) where {T}
    x = copy(x0)
    AT = sparse(A')
    have_constraint = constraint_A !== nothing && size(constraint_A, 1) > 0
    AcT_AcT_Ac = nothing
    if have_constraint
        e = constraint_e === nothing ? zeros(T, size(constraint_A, 1)) : collect(constraint_e)
        AcT = sparse(constraint_A')
        if factor_augmented
            AcT_AcT_Ac = AcT * constraint_A
        end
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
        H_factored = (have_constraint && factor_augmented) ? (H + AcT_AcT_Ac) : H
        F = cholesky(Symmetric(H_factored))
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

# R-INLA's `int.strategy = "grid"` design for `nhyper == 1`. Eleven points
# in *standardized* coordinates plus quadrature weights that compensate for
# the non-uniform spacing. Verbatim from
# https://github.com/hrue/r-inla/blob/devel/gmrflib/design.c lines 39–65.
const _RINLA_GRID_X1 = [-3.5, -2.5, -1.75, -1.0, -0.5, 0.0,
                         0.5,  1.0,  1.75,  2.5,  3.5]
const _RINLA_GRID_W1 = [3.187537795, 1.811358205, 1.937929918,
                        1.431919577, 1.288639321, 1.0,
                        1.288639321, 1.431919577, 1.937929918,
                        1.811358205, 3.187537795]

# R-INLA's `int.strategy = "grid"` design for `nhyper == 2`. Forty-five
# (x, y) standardized pairs and per-node quadrature weights. Verbatim from
# the `x2[]` / `w2[]` tables in gmrflib/design.c lines 67–161.
const _RINLA_GRID_X2 = [
    -2.25 -1.25;  -2.25 -0.5;   -2.25  0.0;  -2.25  0.5;  -2.25  1.25;
    -1.25 -2.25;  -1.25 -1.25;  -1.25 -0.5;  -1.25  0.0;  -1.25  0.5;
    -1.25  1.25;  -1.25  2.25;
    -0.5  -2.25;  -0.5  -1.25;  -0.5  -0.5;  -0.5   0.0;  -0.5   0.5;
    -0.5   1.25;  -0.5   2.25;
     0.0  -2.25;   0.0  -1.25;   0.0  -0.5;   0.0   0.0;   0.0   0.5;
     0.0   1.25;   0.0   2.25;
     0.5  -2.25;   0.5  -1.25;   0.5  -0.5;   0.5   0.0;   0.5   0.5;
     0.5   1.25;   0.5   2.25;
     1.25 -2.25;   1.25 -1.25;   1.25 -0.5;   1.25  0.0;   1.25  0.5;
     1.25  1.25;   1.25  2.25;
     2.25 -1.25;   2.25 -0.5;   2.25  0.0;   2.25  0.5;   2.25  1.25
]
const _RINLA_GRID_W2 = [
    2.277250821, 1.248862019, 1.93160554,  1.248862019, 2.277250821,
    2.277250821, 1.389904145, 0.762234217, 1.17894196,  0.762234217,
    1.389904145, 2.277250821,
    1.248862019, 0.762234217, 0.4180151587, 0.646540918, 0.4180151587,
    0.762234217, 1.248862019,
    1.93160554,  1.17894196,  0.646540918, 1.0,         0.646540918,
    1.17894196,  1.93160554,
    1.248862019, 0.762234217, 0.4180151587, 0.646540918, 0.4180151587,
    0.762234217, 1.248862019,
    2.277250821, 1.389904145, 0.762234217, 1.17894196,  0.762234217,
    1.389904145, 2.277250821,
    2.277250821, 1.248862019, 1.93160554,  1.248862019, 2.277250821
]

"""
    integration_nodes(mode, hessian; method = :auto,
                      stdev_corr_pos = nothing, stdev_corr_neg = nothing)

Build a quadrature design for integrating a posterior on θ.

`method = :auto` (default) reproduces R-INLA's selection
(`inlaprog/src/inla.c:1365–1369`): use the fixed `int.strategy = "grid"`
design for `n_dims ∈ {1, 2}` and CCD for `n_dims ≥ 3`. `method = :ccd`
forces the CCD design at any dimension; `method = :grid` forces the
fixed grid (errors for `n_dims ≥ 3`, since R-INLA only ships tables for
1 and 2).

`stdev_corr_pos` / `stdev_corr_neg` are R-INLA's per-axis asymmetric
skewness corrections (`approx-inference.c:1736–1834`). They scale each
standardized coordinate `z[k]` by `stdev_corr_pos[k]` when `z[k] ≥ 0`
and by `stdev_corr_neg[k]` when `z[k] < 0`, before mapping to θ-space
via the inverse-Hessian eigendecomp. R-INLA derives them by probing the
log-posterior at `z = ±√2` along each principal axis and equating to
the Gaussian drop of 1 nat. Pass `nothing` (default) for the symmetric
Gaussian case where both corrections are 1.

Returns `(nodes, quad_weights)`:

* `nodes` — `Vector{Vector{T}}` of θ-space node locations, each of length
  `n_dims`. Standardized points `z` are mapped through the inverse-Hessian
  eigendecomp `evec · sqrt(1/eval) · z` (with optional asymmetric
  scaling).
* `quad_weights` — non-negative `Vector{T}` of quadrature weights, one per
  node. For the R-INLA grid these compensate for the non-uniform node
  spacing (so the downstream Bayesian quadrature
  `posterior ∝ exp(-obj_at_node) × quad_weight` doesn't over-weight the
  outer points). For the CCD path all weights are 1.0.
"""
function integration_nodes(mode::AbstractVector{T}, hessian::AbstractMatrix{T};
                           method::Symbol = :auto,
                           stdev_corr_pos::Union{Nothing,AbstractVector{<:Real}} = nothing,
                           stdev_corr_neg::Union{Nothing,AbstractVector{<:Real}} = nothing) where T
    n_dims = length(mode)
    method_used = method === :auto ? (n_dims <= 2 ? :grid : :ccd) :
                  method === :grid && n_dims >= 3 ?
                      error("`method = :grid` only available for n_dims ∈ {1, 2}; got n_dims = $n_dims") :
                  method

    # Build the (eigvec · diag(1/√eigval)) basis once. For n_dims == 1 this
    # collapses to a scalar `sd`.
    if n_dims == 1
        sd = one(T) / sqrt(hessian[1, 1])
        z_pts = method_used === :grid ? _RINLA_GRID_X1 : nothing
        z_pts === nothing && (z_pts = vec(T.(ccdesign(n_dims))))
    end

    if method_used === :grid && n_dims == 1
        nodes = [mode .+ _scale_z(T(x), 1, stdev_corr_pos, stdev_corr_neg) * sd
                 for x in _RINLA_GRID_X1]
        weights = T.(_RINLA_GRID_W1)
        return nodes, weights
    elseif method_used === :grid && n_dims == 2
        evals, evecs = eigen(Symmetric(hessian))
        evals = max.(evals, T(1e-9))
        trans = evecs * diagm(one(T) ./ sqrt.(evals))
        nodes = [mode .+ trans * _scale_z_vec(T.(_RINLA_GRID_X2[k, :]),
                                              stdev_corr_pos, stdev_corr_neg)
                 for k in axes(_RINLA_GRID_X2, 1)]
        weights = T.(_RINLA_GRID_W2)
        return nodes, weights
    else
        z_points = T.(ccdesign(n_dims))
        evals, evecs = eigen(Symmetric(hessian))
        evals = max.(evals, T(1e-9))
        trans = evecs * diagm(one(T) ./ sqrt.(evals))
        n_nodes = size(z_points, 1)
        nodes = [mode + trans * _scale_z_vec(z_points[i, :], stdev_corr_pos, stdev_corr_neg)
                 for i in 1:n_nodes]
        weights = ones(T, n_nodes)
        return nodes, weights
    end
end

# Scale a single standardized coordinate by the asymmetric skewness correction.
@inline function _scale_z(z::T, k::Int,
                          pos::Union{Nothing,AbstractVector{<:Real}},
                          neg::Union{Nothing,AbstractVector{<:Real}}) where {T<:Real}
    pos === nothing && return z
    return z >= zero(T) ? z * T(pos[k]) : z * T(neg[k])
end

# Apply per-axis asymmetric skew correction to a standardized z-vector.
function _scale_z_vec(z::AbstractVector{T},
                      pos::Union{Nothing,AbstractVector{<:Real}},
                      neg::Union{Nothing,AbstractVector{<:Real}}) where {T<:Real}
    pos === nothing && neg === nothing && return z
    out = similar(z)
    @inbounds for k in eachindex(z)
        out[k] = _scale_z(z[k], k, pos, neg)
    end
    return out
end

function find_mode(f, x0, p=nothing; solver=Newton(), adtype=Optimization.AutoForwardDiff(), kwargs...)
    optf = OptimizationFunction(f, adtype)
    prob = OptimizationProblem(optf, x0, p)
    return solve(prob, solver; kwargs...)
end

end # module
