module INLAModels

using LinearAlgebra
using SparseArrays
using Distributions
using INLACore
using INLASpatial

export Likelihood, GaussianLikelihood, BernoulliLikelihood, PoissonLikelihood
export LatentModel, AR1Model, RW1Model, IIDModel, SPDEModel
export PCPrior, log_pdf, precision_matrix

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

struct RW1Model <: LatentModel end

function precision_matrix(::RW1Model, n::Int, tau::T) where T
    Q = spzeros(T, n, n)
    for i in 1:n
        if i == 1 || i == n; Q[i, i] = one(T)
        else Q[i, i] = T(2.0) * one(T)
        end
        i > 1 && (Q[i, i-1] = -one(T))
        i < n && (Q[i, i+1] = -one(T))
    end
    for i in 1:n; Q[i, i] += T(0.1); end
    return tau * Q
end

struct SPDEModel <: LatentModel 
    C::SparseMatrixCSC{Float64, Int}
    G::SparseMatrixCSC{Float64, Int}
end

function precision_matrix(model::SPDEModel, kappa::T, tau::T) where T
    # Convert mesh matrices to type T
    return INLASpatial.spde_precision(sparse(T.(model.C)), sparse(T.(model.G)), kappa, tau)
end

end # module
