module INLAModels

using LinearAlgebra
using SparseArrays
using Distributions
using INLACore
using INLASpatial

export Likelihood, GaussianLikelihood, BernoulliLikelihood, PoissonLikelihood
export LatentModel, AR1Model, RW1Model, IIDModel, SPDEModel, BivariateIIDModel, ICARModel, NonStationarySPDEModel
export BesagModel
export PCPrior, log_pdf, precision_matrix
export n_hyper, assemble_Q, log_prior, has_likelihood_hyperparameter
export constraint_matrix, has_constraint

# --- Priors ---

struct PCPrior
    U::Float64
    alpha::Float64
    type::Symbol
end

function log_pdf(prior::PCPrior, x::T) where T
    if prior.type == :std
        sigma = one(T)/sqrt(x)
        lambda = T(-log(prior.alpha) / prior.U)
        log_p_sigma = log(lambda) - lambda * sigma
        log_jacobian = T(-1.5) * log(x) - log(T(2.0))
        return log_p_sigma + log_jacobian
    end
    return zero(T)
end

# --- Likelihoods ---

abstract type Likelihood end

struct GaussianLikelihood <: Likelihood end

function log_pdf(::GaussianLikelihood, y, eta, tau::T) where T
    return T(-0.5 * log(2π)) + T(0.5) * log(tau) - T(0.5) * tau * (y - eta)^2
end

struct BernoulliLikelihood <: Likelihood end

function log_pdf(::BernoulliLikelihood, y, eta, params=nothing)
    return y * eta - log(one(eta) + exp(eta))
end

struct PoissonLikelihood <: Likelihood end

function log_pdf(::PoissonLikelihood, y, eta, params=nothing)
    return y * eta - exp(eta)
end

# --- Latent Models ---

abstract type LatentModel end

struct IIDModel <: LatentModel end

function precision_matrix(::IIDModel, n::Int, tau::T) where T
    return tau * sparse(T(1.0) * I, n, n)
end

struct BivariateIIDModel <: LatentModel end

function precision_matrix(::BivariateIIDModel, n_pairs::Int, tau1::T, tau2::T, rho::T) where T
    sigma1 = one(T)/sqrt(tau1)
    sigma2 = one(T)/sqrt(tau2)
    cov12 = rho * sigma1 * sigma2
    Sigma_block = [sigma1^2 cov12; cov12 sigma2^2]
    Q_block = inv(Sigma_block)
    return blockdiag([sparse(Q_block) for _ in 1:n_pairs]...)
end

struct RW1Model <: LatentModel end

# RW1 is intrinsic (constant null space). We add a tiny, *prior-neutral* numeric
# jitter so the result is positive-definite for Cholesky factorization. A real
# fix is the sum-to-zero constraint that lands in Phase 2.
const RW1_NUMERIC_JITTER = 1.0e-6

function precision_matrix(::RW1Model, n::Int, tau::T) where {T}
    Q = spzeros(T, n, n)
    for i in 1:n
        if i == 1 || i == n
            Q[i, i] = one(T)
        else
            Q[i, i] = T(2)
        end
        i > 1 && (Q[i, i-1] = -one(T))
        i < n && (Q[i, i+1] = -one(T))
    end
    for i in 1:n
        Q[i, i] += T(RW1_NUMERIC_JITTER)
    end
    return tau * Q
end

struct SPDEModel <: LatentModel 
    C::SparseMatrixCSC{Float64, Int}
    G::SparseMatrixCSC{Float64, Int}
end

function precision_matrix(model::SPDEModel, kappa::T, tau::T) where T
    return INLASpatial.spde_precision(sparse(T.(model.C)), sparse(T.(model.G)), kappa, tau)
end

"""
    ICARModel(adj)

Intrinsic Conditional Autoregressive (ICAR) model.
Q = tau * (diag(degree) - adj)
"""
struct ICARModel <: LatentModel
    W::SparseMatrixCSC{Float64, Int}
end

"""
    BesagModel(W; scale=true)

Besag/CAR model on the area adjacency graph encoded by symmetric sparse `W`.
Equivalent to R-INLA's `f(., model="besag", graph=g, scale.model=true/false)`.

The sum-to-zero constraint `1' u = 0` is enforced *exactly* via the augmented
KKT system in the driver's inner Newton step and via determinant corrections
in the Laplace marginal likelihood. Without the constraint the area effect
and the intercept are jointly unidentified — a nuisance for `summary.fixed`
and a real problem for the Laplace approximation, which would otherwise
prefer `τ → ∞` (u → 0, intercept absorbs everything).

* `scale=true` (default) rescales the unscaled precision `Q₀ = D − W` so the
  geometric mean of marginal variances under the constrained ICAR prior is 1.
  This is what R-INLA's `scale.model=TRUE` does.
"""
struct BesagModel <: LatentModel
    W::SparseMatrixCSC{Float64, Int}
    scale::Bool
    scale_factor::Float64
    function BesagModel(W::SparseMatrixCSC{Float64, Int}; scale::Bool = true)
        return new(W, scale, scale ? _besag_scale_factor(W) : 1.0)
    end
end

# Returns the *multiplier* `c` such that `c · (D − W)` has geometric mean
# marginal variance = 1 under the sum-to-zero-constrained ICAR prior.
# This is what R-INLA's scale.model=TRUE applies to make τ interpretable as
# an "average" precision across graphs.
#
# Equals `1 / exp(mean(log(diag(Σ_unscaled_proj))))` where Σ_unscaled_proj
# is the generalized inverse of (D−W) on the orthogonal complement of the
# constant null direction.
function _besag_scale_factor(W::SparseMatrixCSC{Float64, Int})
    n = size(W, 1)
    D = Matrix(spdiagm(0 => vec(sum(W, dims=2))))
    Q0 = Matrix(D - Matrix(W))
    e = ones(n) ./ n
    P = I - ones(n) * e'
    # (Q0 + 1·e') is invertible; the outer projection peels the null direction
    # back out, leaving the generalised inverse on the orthogonal complement.
    Σ = P * inv(Q0 + ones(n) * e') * P
    diag_var = real.(diag(Σ))
    geom_mean = exp(mean(log.(max.(diag_var, 1e-12))))
    return 1.0 / geom_mean
end

function precision_matrix(model::BesagModel, tau::T) where {T}
    W = sparse(T.(model.W))
    D = spdiagm(0 => vec(sum(W, dims=2)))
    Q = (D - W) .* T(model.scale_factor)
    # Q is intrinsically rank-deficient along the constant null direction.
    # We do *not* add a numerical jitter here: the driver augments Q with the
    # `A_c' A_c` rank-1 update from the sum-to-zero constraint when computing
    # log-determinants, and adds `Aᵀ D A` (positive) when forming H. Both
    # provide enough mass along the null direction that Cholesky never sees
    # the rank deficiency.
    return tau * Q
end

function constraint_matrix(::BesagModel, n_block::Int)
    # 1×n: the sum-to-zero constraint 1' u = 0, normalized so A_c A_c' = 1.
    return sparse(ones(1, n_block) ./ sqrt(n_block))
end

n_hyper(::BesagModel) = 1

function assemble_Q(model::BesagModel, theta_block::AbstractVector{T}, _n::Int) where {T}
    return precision_matrix(model, exp(theta_block[1]))
end

log_prior(::BesagModel, theta_block::AbstractVector) =
    loggamma_logprior(theta_block[1])

function precision_matrix(model::ICARModel, n::Int, tau::T) where T
    W = sparse(T.(model.W))
    D = spdiagm(0 => vec(sum(W, dims=2)))
    Q = D - W
    for i in 1:n; Q[i, i] += T(1e-6); end
    return tau * Q
end

"""
    NonStationarySPDEModel(C, G, B_kappa, B_tau)

SPDE model with spatially varying log-kappa and log-tau.
"""
struct NonStationarySPDEModel <: LatentModel
    C::SparseMatrixCSC{Float64, Int}
    G::SparseMatrixCSC{Float64, Int}
    B_kappa::Matrix{Float64}
    B_tau::Matrix{Float64}
end

function precision_matrix(model::NonStationarySPDEModel, theta_kappa::AbstractVector{T}, theta_tau::AbstractVector{T}) where T
    kappa = exp.(T.(model.B_kappa) * theta_kappa)
    tau = exp.(T.(model.B_tau) * theta_tau)
    C_diag = vec(sum(model.C, dims=2))
    K = spdiagm(0 => kappa.^2) * sparse(T.(model.C)) + sparse(T.(model.G))
    inv_C = spdiagm(0 => one(T) ./ T.(C_diag))
    Q = K' * inv_C * K
    Q = spdiagm(0 => sqrt.(tau)) * Q * spdiagm(0 => sqrt.(tau))
    return Q
end

# --- Hyperparameter interface (Phase 1) ---
#
# Every latent model declares:
#   n_hyper(model)                 -> Int            # number of hypers it owns
#   assemble_Q(model, theta_block, n)                # build sparse Q given log-scale hypers
#   log_prior(model, theta_block)  -> Real           # log prior on those hypers
#
# Likewise, every likelihood declares:
#   has_likelihood_hyperparameter(::Likelihood)      # Bool
#   n_hyper(::Likelihood)                            # 0 or 1 by default
#   log_prior(::Likelihood, theta_y)                 # log prior on θ_y
#
# `theta_block` is the slice of the global hyperparameter vector that the
# model owns, on the unconstrained scale (log-precision, atanh-correlation, …).

# Default: zero hypers.
n_hyper(::Any) = 0
log_prior(::Any, theta_block::AbstractVector) = zero(eltype(theta_block))

# Default: no linear constraints. Intrinsic GMRFs (RW1/RW2/Besag/ICAR/BYM/BYM2)
# override `constraint_matrix(model, n_block)` to return a `k × n_block` sparse
# matrix `A_c` so that `A_c · u = 0` is enforced exactly inside the Laplace
# approximation (via the augmented KKT system in the inner Newton, plus
# determinant corrections −½ log det(A_c Σ A_c') in the marginal likelihood).
constraint_matrix(::Any, n_block::Int) = spzeros(0, n_block)

# --- Default log-Gamma(1, 5e-5) prior on log-precision (R-INLA's default). ---
const _LGAMMA_A_DEFAULT = 1.0
const _LGAMMA_B_DEFAULT = 5e-5

"""
    loggamma_logprior(theta_log_tau; a=1.0, b=5e-5)

Log density of the log-Gamma prior on θ = log(τ) used as R-INLA's default for
IID/RW/etc precisions. Formula:

    p(θ) = b^a / Γ(a) · exp(a θ − b exp(θ))

so log p(θ) = a θ − b exp(θ) + a log(b) − log Γ(a).
Constants are kept (drop them only if profiling shows it matters).
"""
function loggamma_logprior(theta::T; a = _LGAMMA_A_DEFAULT, b = _LGAMMA_B_DEFAULT) where {T}
    a_T = T(a); b_T = T(b)
    return a_T * theta - b_T * exp(theta) + a_T * log(b_T)  # − log Γ(a) drops out for a=1
end

# --- IIDModel ---
n_hyper(::IIDModel) = 1

function assemble_Q(model::IIDModel, theta_block::AbstractVector{T}, n::Int) where {T}
    return precision_matrix(model, n, exp(theta_block[1]))
end

log_prior(::IIDModel, theta_block::AbstractVector) =
    loggamma_logprior(theta_block[1])

# --- RW1Model ---
n_hyper(::RW1Model) = 1

function assemble_Q(model::RW1Model, theta_block::AbstractVector{T}, n::Int) where {T}
    return precision_matrix(model, n, exp(theta_block[1]))
end

log_prior(::RW1Model, theta_block::AbstractVector) =
    loggamma_logprior(theta_block[1])

# --- ICARModel ---
n_hyper(::ICARModel) = 1

function assemble_Q(model::ICARModel, theta_block::AbstractVector{T}, n::Int) where {T}
    return precision_matrix(model, n, exp(theta_block[1]))
end

log_prior(::ICARModel, theta_block::AbstractVector) =
    loggamma_logprior(theta_block[1])

# --- BivariateIIDModel ---
# 3 hypers: log τ1, log τ2, atanh(ρ). n is the latent-block length and must be even.
n_hyper(::BivariateIIDModel) = 3

function assemble_Q(model::BivariateIIDModel, theta_block::AbstractVector{T}, n::Int) where {T}
    n_pairs = div(n, 2)
    n_pairs * 2 == n || error("BivariateIIDModel needs an even latent length, got $n")
    tau1 = exp(theta_block[1])
    tau2 = exp(theta_block[2])
    rho  = tanh(theta_block[3])
    return precision_matrix(model, n_pairs, tau1, tau2, rho)
end

# Loose default: independent loggamma on the two precisions, weakly informative N(0,1) on z = atanh ρ.
function log_prior(::BivariateIIDModel, theta_block::AbstractVector{T}) where {T}
    return loggamma_logprior(theta_block[1]) +
           loggamma_logprior(theta_block[2]) +
           T(-0.5) * theta_block[3]^2
end

# --- SPDEModel ---
n_hyper(::SPDEModel) = 2  # log κ, log τ

function assemble_Q(model::SPDEModel, theta_block::AbstractVector{T}, n::Int) where {T}
    kappa = exp(theta_block[1])
    tau   = exp(theta_block[2])
    return precision_matrix(model, kappa, tau)
end

# Weakly informative default; PC priors come in Phase 2.
function log_prior(::SPDEModel, theta_block::AbstractVector{T}) where {T}
    return T(-0.5) * (theta_block[1]^2 + theta_block[2]^2)
end

# --- NonStationarySPDEModel ---
n_hyper(model::NonStationarySPDEModel) = size(model.B_kappa, 2) + size(model.B_tau, 2)

function assemble_Q(model::NonStationarySPDEModel, theta_block::AbstractVector{T}, n::Int) where {T}
    p_k = size(model.B_kappa, 2)
    p_t = size(model.B_tau,   2)
    @assert length(theta_block) == p_k + p_t
    theta_kappa = theta_block[1:p_k]
    theta_tau   = theta_block[(p_k+1):(p_k+p_t)]
    return precision_matrix(model, theta_kappa, theta_tau)
end

function log_prior(model::NonStationarySPDEModel, theta_block::AbstractVector{T}) where {T}
    return T(-0.5) * sum(abs2, theta_block)
end

# --- Likelihoods ---
has_likelihood_hyperparameter(::Likelihood) = false
has_likelihood_hyperparameter(::GaussianLikelihood) = true

n_hyper(::Likelihood) = 0
n_hyper(::GaussianLikelihood) = 1

log_prior(::Likelihood, theta::AbstractVector) = zero(eltype(theta))
log_prior(::GaussianLikelihood, theta::AbstractVector) = loggamma_logprior(theta[1])

end # module
