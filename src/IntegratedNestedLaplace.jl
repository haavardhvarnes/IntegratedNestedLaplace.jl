module IntegratedNestedLaplace

using Reexport
@reexport using INLACore
@reexport using INLAModels
@reexport using INLASpatial
using StatsModels
using LinearAlgebra
using SparseArrays
using OptimizationOptimJL
import Optimization
import ADTypes
using Printf
using KernelAbstractions

export inla, @formula, f
export GaussianLikelihood, BernoulliLikelihood, PoissonLikelihood
export IIDModel, RW1Model, SPDEModel, BivariateIIDModel, ICARModel, NonStationarySPDEModel

# `f(...)` is purely a marker for the formula DSL; StatsModels parses the call
# expression for us via `args_parsed`. Calling it directly is meaningless.
f(args...) = nothing

# R-INLA's default fixed-effect prior: N(0, 1000²) ⇒ precision 0.001.
const DEFAULT_FIXED_PRECISION = 1.0e-3

struct INLAResult{T}
    mode_latent::Vector{T}
    mode_hyper::Vector{T}
    marginals_latent::Vector{T}
    marginals_hyper::Matrix{T}            # one row per hyperparameter, columns = (mean, sd)
    nodes_hyper::Vector{Vector{T}}
    weights_hyper::Vector{T}
    formula::FormulaTerm
end

# --- Result Summary ---

function Base.show(io::IO, ::MIME"text/plain", res::INLAResult)
    println(io, "INLA Result")
    println(io, "-----------")
    println(io, "Formula: ", res.formula)
    println(io)
    println(io, "Hyperparameters (Mode):")
    for i in 1:length(res.mode_hyper)
        @printf(io, "  theta[%d]: %10.4f\n", i, res.mode_hyper[i])
    end
    println(io)
    n = length(res.mode_latent)
    println(io, "Latent Field Summary (n=$n):")
    limit = min(n, 5)
    @printf(io, "  %-4s  %10s  %10s\n", "i", "mean", "sd")
    for i in 1:limit
        var_val = res.marginals_latent[i]
        sd = sqrt(max(zero(var_val), var_val))
        @printf(io, "  %-4d  %10.4f  %10.4f\n", i, res.mode_latent[i], sd)
    end
    n > 5 && println(io, "  ...")
end

# --- Per-likelihood derivatives in η. ---
#
# Each method returns two functions of η = A·x:
#   grad_eta(η)        - ∂ℓ/∂η_i, length n_obs
#   hess_eta_diag(η)   - ∂²ℓ/∂η_i² (negative for log-concave), length n_obs
#
# `theta_y` is the (possibly empty) likelihood hyperparameter vector, on the
# unconstrained scale (e.g. log τ_y for Gaussian).

function eta_derivatives(::GaussianLikelihood, y, theta_y)
    tau_y = exp(theta_y[1])
    grad_eta(eta) = tau_y .* (y .- eta)
    hess_eta_diag(_eta) = fill(-tau_y, length(y))
    return grad_eta, hess_eta_diag
end

function eta_derivatives(::BernoulliLikelihood, y, _theta_y)
    function grad_eta(eta)
        p = inv.(one.(eta) .+ exp.(.-eta))
        return y .- p
    end
    function hess_eta_diag(eta)
        p = inv.(one.(eta) .+ exp.(.-eta))
        return .-p .* (one.(p) .- p)
    end
    return grad_eta, hess_eta_diag
end

function eta_derivatives(::PoissonLikelihood, y, _theta_y)
    grad_eta(eta) = y .- exp.(eta)
    hess_eta_diag(eta) = .-exp.(eta)
    return grad_eta, hess_eta_diag
end

# --- log p(y | η, θ_y) summed over observations. ---

function log_likelihood_total(::GaussianLikelihood, y, eta, theta_y)
    tau_y = exp(theta_y[1])
    n = length(y)
    return n * (-0.5 * log(2π) + 0.5 * log(tau_y)) - 0.5 * tau_y * sum(abs2, y .- eta)
end

function log_likelihood_total(::BernoulliLikelihood, y, eta, _theta_y)
    return sum(y .* eta .- log1pexp.(eta))
end

# Numerically stable log(1 + exp(x)).
@inline log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

function log_likelihood_total(::PoissonLikelihood, y, eta, _theta_y)
    return sum(y .* eta .- exp.(eta))
end

# --- Latent term parsing helpers ---

struct LatentEffect{M}
    name::Symbol
    A::SparseMatrixCSC{Float64, Int}
    n_block::Int
    model::M
end

# Build the per-effect projection matrix and the model instance from a parsed
# `f(covariate, model_symbol)` term plus user-provided overrides.
function _build_latent_effect(t, data, n_obs, latent_override)
    cov_name = t.args_parsed[1].sym
    model_sym = length(t.args_parsed) > 1 ? t.args_parsed[2].sym : :IID

    cov_data = data[!, cov_name]
    unique_vals = unique(cov_data)
    val_map = Dict(v => i for (i, v) in enumerate(unique_vals))
    n_unique = length(unique_vals)

    # Choose model instance (user `latent=` argument wins for SPDE-family).
    model = if model_sym === :SPDE
        latent_override isa SPDEModel || latent_override isa NonStationarySPDEModel ?
            latent_override : error("formula uses f(., SPDE) but no SPDE/NonStationarySPDE was passed via latent=")
    elseif model_sym === :NonStationarySPDE
        latent_override isa NonStationarySPDEModel ?
            latent_override : error("formula uses f(., NonStationarySPDE) but latent= is not a NonStationarySPDEModel")
    elseif model_sym === :ICAR
        latent_override isa ICARModel ?
            latent_override : error("formula uses f(., ICAR) but latent= is not an ICARModel")
    elseif model_sym === :BivariateIID
        BivariateIIDModel()
    elseif model_sym === :RW1
        RW1Model()
    else
        IIDModel()
    end

    # The latent block size depends on the model.
    n_block = if model isa SPDEModel
        size(model.C, 1)
    elseif model isa NonStationarySPDEModel
        size(model.C, 1)
    elseif model isa BivariateIIDModel
        2 * n_unique
    else
        n_unique
    end

    # Build sparse projection A: row j observation, col k latent index = val_map[cov_data[j]].
    # For BivariateIID we need to handle the type=1/type=2 stacking; for the simple
    # single-block case (IID/RW1/ICAR) it's a 1-of-K mapping.
    row_idx = Int[]; col_idx = Int[]; vals = Float64[]
    if model isa BivariateIIDModel
        # Need a `type` covariate too — convention: column named `type` with values in {1,2}
        haskey_type = hasproperty(data, :type)
        haskey_type || error("BivariateIID requires the data to have a `type` column with values 1 or 2")
        types = data.type
        for j in 1:n_obs
            base = 2 * (val_map[cov_data[j]] - 1) + 1
            push!(row_idx, j); push!(col_idx, base + (types[j] - 1)); push!(vals, 1.0)
        end
    else
        for j in 1:n_obs
            push!(row_idx, j); push!(col_idx, val_map[cov_data[j]]); push!(vals, 1.0)
        end
    end
    A = sparse(row_idx, col_idx, vals, n_obs, n_block)
    return LatentEffect(cov_name, A, n_block, model)
end

"""
    inla(formula, data;
         family = GaussianLikelihood(),
         latent = nothing,
         theta0 = nothing,
         fixed_precision = $DEFAULT_FIXED_PRECISION,
         solver = :bfgs,
         max_outer_iter = 50,
         backend = CPU())

Fit an approximate Bayesian latent Gaussian model (LGM) by INLA.

Returns an [`INLAResult`](@ref) carrying:
* `mode_latent` — joint posterior mode of the latent field (in original order).
* `mode_hyper` — posterior mode of the (unconstrained) hyperparameter vector
  `θ = [θ_lik..., θ_block_1..., θ_block_2..., ...]`.
* `marginals_latent` — per-coordinate posterior variances of the latent field.
* `marginals_hyper` — `n_hyper × 2` matrix `[θ_mean θ_sd]` from a Gaussian
  approximation at the mode (CCD integration arrives in Phase 2).
"""
function inla(form::FormulaTerm, data;
              family            = GaussianLikelihood(),
              latent            = nothing,
              theta0::Union{Nothing,AbstractVector{<:Real}} = nothing,
              fixed_precision   = DEFAULT_FIXED_PRECISION,
              solver::Symbol    = :bfgs,
              max_outer_iter::Int = 50,
              backend           = CPU())

    # 1. Split the formula into "regular" RHS terms (fixed effects) and `f(...)`
    # markers. `f` is detected as `FunctionTerm{typeof(f)}`.
    rhs = form.rhs
    latent_terms = Any[]
    if rhs isa Tuple
        for t in rhs
            t isa FunctionTerm{typeof(f)} && push!(latent_terms, t)
        end
        regular_terms = Tuple(t for t in rhs if !(t isa FunctionTerm{typeof(f)}))
        clean_rhs = isempty(regular_terms) ? ConstantTerm(1) :
                    (length(regular_terms) == 1 ? regular_terms[1] : regular_terms)
    elseif rhs isa FunctionTerm{typeof(f)}
        push!(latent_terms, rhs)
        clean_rhs = ConstantTerm(1)
    else
        clean_rhs = rhs
    end

    # 2. Build the fixed-effects design matrix via StatsModels.
    clean_form = FormulaTerm(form.lhs, clean_rhs)
    sch = schema(clean_form, data)
    f_applied = apply_schema(clean_form, sch)
    y_raw, X = modelcols(f_applied, data)
    n_obs   = length(y_raw)
    n_fixed = size(X, 2)

    # 3. Build each latent effect.
    effects = LatentEffect[]
    for t in latent_terms
        push!(effects, _build_latent_effect(t, data, n_obs, latent))
    end

    n_random = isempty(effects) ? 0 : sum(e.n_block for e in effects)
    n_latent = n_fixed + n_random

    # 4. Stack the design: A_total = [X | A_1 | A_2 | …]
    A_blocks = SparseMatrixCSC{Float64,Int}[ sparse(X) ]
    for e in effects
        push!(A_blocks, e.A)
    end
    A_total = hcat(A_blocks...)

    # 5. Hyperparameter layout. Convention:
    #     θ = [ θ_lik …, θ_eff_1 …, θ_eff_2 …, … ]
    n_h_lik  = n_hyper(family)
    n_h_eff  = [n_hyper(e.model) for e in effects]
    n_h      = n_h_lik + sum(n_h_eff; init = 0)

    # Hyper slices.
    function _slices(theta::AbstractVector)
        i = 1
        theta_y = view(theta, i:(i + n_h_lik - 1)); i += n_h_lik
        eff_slices = Vector{SubArray}(undef, length(effects))
        for k in eachindex(effects)
            len = n_h_eff[k]
            eff_slices[k] = view(theta, i:(i + len - 1))
            i += len
        end
        return theta_y, eff_slices
    end

    # 6. Initial hyperparameter vector.
    theta_init = if theta0 === nothing
        zeros(n_h)
    else
        length(theta0) == n_h ||
            error("theta0 has length $(length(theta0)) but the model needs $n_h hyperparameters")
        collect(float.(theta0))
    end

    # 7. Build Q for the joint latent at given θ. The fixed-effect block has a
    # weak Gaussian prior that doesn't depend on θ. Phase 1 keeps everything in
    # Float64; AD-friendly element types come in Phase 2.
    Q_fixed = sparse(fixed_precision * I, n_fixed, n_fixed)

    function build_Q(theta::AbstractVector)
        _, eff_slices = _slices(theta)
        blocks = SparseMatrixCSC{Float64,Int}[copy(Q_fixed)]
        for k in eachindex(effects)
            Q_block = assemble_Q(effects[k].model, eff_slices[k], effects[k].n_block)
            push!(blocks, SparseMatrixCSC{Float64,Int}(Q_block))
        end
        return blockdiag(blocks...)
    end

    # 8. Inner Newton + Laplace at given θ. Reuses last x* as warm start.
    x_warm = zeros(n_latent)

    function laplace_obj(theta::AbstractVector)
        theta_y, eff_slices = _slices(theta)
        Q = build_Q(theta)

        grad_eta_fn, hess_eta_diag_fn = eta_derivatives(family, y_raw, theta_y)

        # Newton on x given θ. Phase 1 runs the inner loop in Float64;
        # SuiteSparse Cholesky is Float64-only and AD-wrapping the outer call
        # is reserved for Phase 2.
        x_star = gmrf_newton_full(grad_eta_fn, hess_eta_diag_fn, A_total, Q, x_warm)
        copyto!(x_warm, x_star)

        # Laplace approximation:
        #   log π̂(θ|y) = log p(y|η*, θ_y) + log p(x*|θ) − ½ log|H| + log π(θ) + const
        eta_star = A_total * x_star
        ll  = log_likelihood_total(family, y_raw, eta_star, theta_y)
        lp  = -0.5 * dot(x_star, Q * x_star) + 0.5 * sparse_logdet(Q)

        h_eta = hess_eta_diag_fn(eta_star)
        H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)
        F = cholesky(Symmetric(H))
        log_det_H = 2.0 * logdet(F)

        # Hyperparameter priors.
        lprior = log_prior(family, theta_y)
        for k in eachindex(effects)
            lprior += log_prior(effects[k].model, eff_slices[k])
        end

        # Minimise the negative log marginal posterior of θ.
        return -(ll + lp - 0.5 * log_det_H + lprior)
    end

    # 9. Optimize. For Phase 1 we use BFGS with finite-diff gradients. The
    # closed-form analytic gradient is a Phase 2/3 milestone; finite diff with
    # n_h ≤ 10 is acceptable runtime overhead.
    optf = Optimization.OptimizationFunction((th, _p) -> laplace_obj(th),
                                             ADTypes.AutoFiniteDiff())
    prob = Optimization.OptimizationProblem(optf, theta_init)

    inner_solver = solver === :bfgs        ? BFGS()        :
                   solver === :neldermead  ? NelderMead()  :
                   solver === :newton      ? Newton()      :
                   error("unknown solver $(solver). Use :bfgs, :neldermead, or :newton.")

    sol = Optimization.solve(prob, inner_solver;
                             maxiters = max_outer_iter,
                             abstol   = 1e-7,
                             reltol   = 1e-7)
    theta_star = collect(sol.u)

    # 10. One final pass at θ* to extract the mode and marginals.
    Q_star = SparseMatrixCSC{Float64,Int}(build_Q(theta_star))
    theta_y_star, _ = _slices(theta_star)
    grad_eta_fn, hess_eta_diag_fn = eta_derivatives(family, y_raw, theta_y_star)

    x_star = gmrf_newton_full(grad_eta_fn, hess_eta_diag_fn,
                              A_total, Q_star, x_warm; max_iter = 100, tol = 1e-10)
    eta_star = A_total * x_star
    h_eta = hess_eta_diag_fn(eta_star)
    H_star = Q_star + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)
    F_star = cholesky(Symmetric(H_star))
    marginals = takahashi_diag(F_star)

    # Hyper marginals: Gaussian Laplace at θ* (covariance = H_θ⁻¹ via finite diff).
    marg_hyper = _hyper_marginals(laplace_obj, theta_star)

    # CCD nodes/weights are populated for Phase 2 — for now we report the mode only.
    nodes_hyper   = [collect(theta_star)]
    weights_hyper = [1.0]

    return INLAResult(x_star, theta_star, marginals, marg_hyper,
                      nodes_hyper, weights_hyper, form)
end

"""
    _hyper_marginals(obj, theta_star; eps=1e-3)

Gaussian approximation of the hyperparameter posterior at the mode: returns an
`n_hyper × 2` matrix of (mean, sd). Diagonal of (∇²obj)⁻¹ via central finite
differences. CCD integration in Phase 2 will replace this.
"""
function _hyper_marginals(obj, theta_star::AbstractVector{T}; eps_::T = T(1e-3)) where {T}
    n = length(theta_star)
    n == 0 && return Matrix{T}(undef, 0, 2)
    H = zeros(T, n, n)
    f0 = obj(theta_star)
    for i in 1:n, j in 1:i
        if i == j
            tp = copy(theta_star); tp[i] += eps_
            tm = copy(theta_star); tm[i] -= eps_
            H[i,i] = (obj(tp) - 2*f0 + obj(tm)) / eps_^2
        else
            tpp = copy(theta_star); tpp[i] += eps_; tpp[j] += eps_
            tpm = copy(theta_star); tpm[i] += eps_; tpm[j] -= eps_
            tmp = copy(theta_star); tmp[i] -= eps_; tmp[j] += eps_
            tmm = copy(theta_star); tmm[i] -= eps_; tmm[j] -= eps_
            H[i,j] = (obj(tpp) - obj(tpm) - obj(tmp) + obj(tmm)) / (4 * eps_^2)
            H[j,i] = H[i,j]
        end
    end
    Σ = inv(Symmetric(H))
    out = Matrix{T}(undef, n, 2)
    for i in 1:n
        out[i,1] = theta_star[i]
        out[i,2] = sqrt(max(zero(T), Σ[i,i]))
    end
    return out
end

end # module
