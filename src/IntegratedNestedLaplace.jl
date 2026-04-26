module IntegratedNestedLaplace

using Reexport
@reexport using INLACore
@reexport using INLAModels
@reexport using INLASpatial
using StatsModels
using LinearAlgebra
using SparseArrays
import ForwardDiff
using ADTypes
using DifferentiationInterface
using OptimizationOptimJL
using Enzyme
import Meshes
using Printf
using RecipesBase
using KernelAbstractions
using ImplicitDifferentiation

export inla, @formula, f
export GaussianLikelihood, BernoulliLikelihood, PoissonLikelihood
export IIDModel, RW1Model, SPDEModel, BivariateIIDModel

f(args...) = nothing

struct INLAResult{T}
    mode_latent::Vector{T}
    mode_hyper::Vector{T}
    marginals_latent::Vector{T}
    nodes_hyper::Vector{Vector{T}}
    weights_hyper::Vector{T}
    formula::FormulaTerm
end

# --- Result Summary ---

function Base.show(io::IO, mime::MIME"text/plain", res::INLAResult)
    println(io, "INLA Result (Multi-Effect Optimized)")
    println(io, "----------------------------------")
    println(io, "Formula: ", res.formula)
    println(io, "")
    println(io, "Hyperparameters (Mode):")
    for i in 1:length(res.mode_hyper)
        @printf(io, "  theta[%d]: %10.4f\n", i, res.mode_hyper[i])
    end
    println(io, "")
    n = length(res.mode_latent)
    println(io, "Latent Field Summary (n=$n):")
    limit = min(n, 5)
    for i in 1:limit
        var_val = res.marginals_latent[i]
        sd = sqrt(max(0.0, var_val))
        @printf(io, "  [%3d] %10.4f  %10.4f\n", i, res.mode_latent[i], sd)
    end
    n > 5 && println(io, "  ...")
end

# --- Newton Likelihood Helpers ---

function get_lik_grad_hess(family::GaussianLikelihood, y, n, T)
    gl(x) = y .- x
    hl(x) = fill(-one(T), n)
    return gl, hl
end

function get_lik_grad_hess(family::PoissonLikelihood, y, n, T)
    gl(x) = y .- exp.(x)
    hl(x) = .-exp.(x)
    return gl, hl
end

function get_lik_grad_hess(family::BernoulliLikelihood, y, n, T)
    function gl(x)
        p = 1.0 ./ (1.0 .+ exp.(.-x))
        return y .- p
    end
    function hl(x)
        p = 1.0 ./ (1.0 .+ exp.(.-x))
        return .-p .* (1.0 .- p)
    end
    return gl, hl
end

# Internal helper for latent term parsing
struct LatentEffect
    name::Symbol
    model_type::Symbol
    A::SparseMatrixCSC{Float64, Int}
    n_params::Int
    model::Any
end

# Multi-Effect Cache
mutable struct MultiHyperCache
    theta::Vector{Float64}
    obj::Float64
    grad::Vector{Float64}
    x_star::Vector{Float64}
    S::SparseMatrixCSC{Float64, Int}
end

"""
    inla(formula, data; family=GaussianLikelihood(), latent=nothing, theta0=nothing, backend=CPU())
"""
function inla(form::FormulaTerm, data; family=GaussianLikelihood(), latent=nothing, theta0=nothing, backend=CPU())
    # 1. Parse Latent Components
    rhs = form.rhs
    latent_terms = []
    if rhs isa Tuple
        for t in rhs
            t isa FunctionTerm{typeof(f)} && push!(latent_terms, t)
        end
        regular_terms = Tuple(t for t in rhs if !(t isa FunctionTerm{typeof(f)}))
        clean_rhs = isempty(regular_terms) ? ConstantTerm(1) : (length(regular_terms) == 1 ? regular_terms[1] : regular_terms)
    elseif rhs isa FunctionTerm{typeof(f)}
        push!(latent_terms, rhs)
        clean_rhs = ConstantTerm(1)
    else
        clean_rhs = rhs
    end
    
    # 2. Extract Data
    clean_form = FormulaTerm(form.lhs, clean_rhs)
    sch = schema(clean_form, data)
    f_applied = apply_schema(clean_form, sch)
    y_raw, X = modelcols(f_applied, data)
    n_obs = length(y_raw)
    n_fixed = size(X, 2)

    # 3. Process Latent Terms
    effects = LatentEffect[]
    for (i, t) in enumerate(latent_terms)
        cov_name = t.args_parsed[1].sym
        model_sym = length(t.args_parsed) > 1 ? t.args_parsed[2].sym : :IID
        
        cov_data = data[!, cov_name]
        unique_vals = unique(cov_data)
        val_map = Dict(v => i for (i, v) in enumerate(unique_vals))
        n_unique = length(unique_vals)
        
        n_params = model_sym === :BivariateIID ? 2 * n_unique : n_unique
        if model_sym === :SPDE && latent isa NonStationarySPDEModel; n_params = size(latent.C, 1); end
        
        row_idx = Int[]; col_idx = Int[]; val_A = Float64[]
        for j in 1:n_obs
            push!(row_idx, j); push!(col_idx, val_map[cov_data[j]]); push!(val_A, 1.0)
        end
        A = sparse(row_idx, col_idx, val_A, n_obs, n_params)
        
        model = if model_sym === :SPDE; latent
        elseif model_sym === :RW1; RW1Model()
        else IIDModel()
        end
        push!(effects, LatentEffect(cov_name, model_sym, A, n_params, model))
    end

    n_latent = n_fixed + sum(e.n_params for e in effects)
    
    # 4. Multi-Effect Cache Initialization
    n_h = isempty(effects) ? 1 : length(effects)
    cache = MultiHyperCache(zeros(n_h), 0.0, zeros(n_h), zeros(n_latent), spzeros(n_latent, n_latent))

    function update_cache!(theta)
        if theta == cache.theta; return end
        
        TT = eltype(theta)
        Blocks = Vector{SparseMatrixCSC{Float64, Int}}()
        push!(Blocks, sparse(1e-6 * I, n_fixed, n_fixed))
        for (i, e) in enumerate(effects)
            tau = exp(theta[i])
            push!(Blocks, precision_matrix(e.model, e.n_params, Float64(tau)))
        end
        Q = blockdiag(Blocks...)
        A_total = hcat(sparse(X), [sparse(e.A) for e in effects]...)

        # Newton Mode
        gl_obs, hl_obs = get_lik_grad_hess(family, y_raw, n_obs, Float64)
        function gl(x)
            eta = A_total * x
            g_eta = gl_obs(eta)
            return A_total' * g_eta
        end
        function hl(x)
            eta = A_total * x
            h_eta = hl_obs(eta)
            # Diagonal Hessian approximation: A' * diag(h) * A
            return diag(A_total' * spdiagm(0 => h_eta) * A_total)
        end
        
        x_star = gmrf_newton(gl, hl, Q, cache.x_star)
        
        # log p(x*, theta | y)
        eta_star = A_total * x_star
        ll = sum(INLAModels.log_pdf(family, y_raw[i], eta_star[i], 1.0) for i in 1:n_obs)
        lp = -0.5 * dot(x_star, Q * x_star)
        
        # log det(H)
        # H = Q - A' * diag(h_eta) * A
        H = Q + Symmetric(A_total' * spdiagm(0 => -hl_obs(eta_star)) * A_total)
        F = cholesky(Symmetric(H))
        log_det_H = 2.0 * logdet(F)
        
        obj = -(ll + lp - 0.5 * log_det_H - 0.5 * sum(theta.^2))
        
        # Gradient
        S = copy(sparse(F.L))
        takahashi_factor!(S, sparse(F.L))
        
        grad = zeros(n_h)
        for i in 1:n_h
            # dQ/dtheta_i = blockdiag(0, ..., Q_i, ..., 0)
            # tr(H^-1 * dQ) = tr(S_ii * Q_i)
            # dlp = -0.5 * x_star' * dQ * x_star
            # dprior = -theta[i]
            # This is a simplification for the multi-effect case
            grad[i] = 0.0 # Placeholder for exact multi-grad
        end

        cache.theta = copy(theta)
        cache.obj = obj
        cache.grad = grad
        cache.x_star = x_star
        cache.S = S
    end

    # Use Nelder-Mead for robust multi-hyper search
    res_theta = find_mode((t, p) -> (update_cache!(t); cache.obj), fill(1.0, n_h); solver=NelderMead())
    theta_star = res_theta.u

    update_cache!(theta_star)
    return INLAResult(cache.x_star, theta_star, Vector(diag(cache.S)), [[theta_star[1]]], [1.0], form)
end

end # module
