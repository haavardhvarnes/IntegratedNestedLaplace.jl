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
using Random
using Statistics
using KernelAbstractions

export inla, @formula, f
export INLAResult, hyper_precision_mean
export GaussianLikelihood, BernoulliLikelihood, PoissonLikelihood
export IIDModel, RW1Model, SPDEModel, BivariateIIDModel, ICARModel, NonStationarySPDEModel
export BesagModel

# `f(...)` is purely a marker for the formula DSL; StatsModels parses the call
# expression for us via `args_parsed`. Calling it directly is meaningless.
f(args...) = nothing

# R-INLA's default fixed-effect prior: N(0, 1000²) ⇒ precision 0.001.
const DEFAULT_FIXED_PRECISION = 1.0e-3

"""
    INLAResult

The output of [`inla`](@ref).

Fields:
* `mode_latent`      — joint posterior mode of the latent field at θ*.
* `mean_latent`      — posterior mean of the latent field, integrated over θ
                       via CCD when there are ≥ 1 hyperparameters; otherwise
                       equal to `mode_latent`.
* `mode_hyper`       — θ* on the unconstrained scale.
* `mean_hyper`       — posterior mean of θ from the CCD mixture (or `mode_hyper`
                       if no integration was done).
* `marginals_latent` — per-coordinate posterior variance of the latent field
                       (mixture variance after CCD).
* `marginals_hyper`  — `n_hyper × 2` matrix `[θ_mean θ_sd]` from the CCD
                       mixture (or empty if `n_hyper == 0`).
* `nodes_hyper`      — vector of θ values used for CCD integration; each entry
                       is a length-`n_hyper` vector.
* `weights_hyper`    — normalised mixture weights matching `nodes_hyper`.
* `formula`          — the original `FormulaTerm` passed to `inla`.

Comparison to R-INLA: `summary.fixed\$mean` corresponds to `mean_latent[1:n_fixed]`,
not `mode_latent`. `summary.hyperpar\$mean` (on the precision scale) corresponds
to a mixture average of `exp(θ)` rather than `exp(mean_hyper)` — see the
`hyper_precision_mean` helper.
"""
struct INLAResult{T}
    mode_latent::Vector{T}
    mean_latent::Vector{T}
    mode_hyper::Vector{T}
    mean_hyper::Vector{T}
    marginals_latent::Vector{T}
    marginals_hyper::Matrix{T}            # one row per hyperparameter, columns = (mean, sd)
    nodes_hyper::Vector{Vector{T}}
    weights_hyper::Vector{T}
    formula::FormulaTerm
end

"""
    hyper_precision_mean(res, i)

Posterior mean of `exp(θ_i)` (the "precision" scale) computed from the CCD
mixture. This is what R-INLA prints as `summary.hyperpar\$mean` for a
log-precision hyperparameter.
"""
function hyper_precision_mean(res::INLAResult, i::Int)
    isempty(res.weights_hyper) && return exp(res.mode_hyper[i])
    s = 0.0
    for k in eachindex(res.nodes_hyper)
        s += res.weights_hyper[k] * exp(res.nodes_hyper[k][i])
    end
    return s
end

# --- Result Summary ---

function Base.show(io::IO, ::MIME"text/plain", res::INLAResult)
    println(io, "INLA Result")
    println(io, "-----------")
    println(io, "Formula: ", res.formula)
    println(io)
    println(io, "Hyperparameters (mode → mean ± sd):")
    for i in 1:length(res.mode_hyper)
        @printf(io, "  theta[%d]: %10.4f → %10.4f ± %.4f\n",
                i, res.mode_hyper[i], res.mean_hyper[i], res.marginals_hyper[i, 2])
    end
    println(io)
    n = length(res.mean_latent)
    println(io, "Latent Field (n=$n, posterior mean ± sd):")
    limit = min(n, 5)
    @printf(io, "  %-4s  %10s  %10s\n", "i", "mean", "sd")
    for i in 1:limit
        var_val = res.marginals_latent[i]
        sd = sqrt(max(zero(var_val), var_val))
        @printf(io, "  %-4d  %10.4f  %10.4f\n", i, res.mean_latent[i], sd)
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

# --- Higher derivatives of log p(y_k|η_k) wrt η_k, evaluated at η. ---
#
# Used by both the simplified-Laplace marginal-mean correction (3rd derivative)
# and the Edgeworth-style correction to log π̂(y|θ) (3rd + 4th derivatives).
# For Gaussian both are identically zero — the Laplace approximation is exact.

third_deriv_eta(::GaussianLikelihood, eta, _theta_y) = zeros(eltype(eta), length(eta))
fourth_deriv_eta(::GaussianLikelihood, eta, _theta_y) = zeros(eltype(eta), length(eta))

function third_deriv_eta(::BernoulliLikelihood, eta, _theta_y)
    p = inv.(one.(eta) .+ exp.(.-eta))
    return .-p .* (one.(p) .- p) .* (one.(p) .- 2 .* p)
end

function fourth_deriv_eta(::BernoulliLikelihood, eta, _theta_y)
    p = inv.(one.(eta) .+ exp.(.-eta))
    pq = p .* (one.(p) .- p)            # p(1-p)
    return .-pq .* (one.(p) .- 6 .* pq) # = -p(1-p)(1 - 6 p(1-p))
end

function third_deriv_eta(::PoissonLikelihood, eta, _theta_y)
    return .-exp.(eta)
end

function fourth_deriv_eta(::PoissonLikelihood, eta, _theta_y)
    return .-exp.(eta)
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

# --- Per-observation log p(y_i | η_i, θ_y), needed for the Taylor-at-zero
# marginal-likelihood formula (Phase 6g). The existing
# `log_likelihood_total` returns the *sum* over observations and drops
# y-only constants (e.g. the Poisson `log y_i!`). The per-i version below
# uses the same convention (no `log y_i!` for Poisson, no `log binom(n,y)`
# for binomial Bernoulli) so absolute values match `log_likelihood_total`
# but per-coordinate; only relative behaviour matters for BFGS.
log_pdf_per_obs(::GaussianLikelihood, y, eta, theta_y) = let τ = exp(theta_y[1])
    @. -0.5 * log(2π) + 0.5 * log(τ) - 0.5 * τ * (y - eta)^2
end
log_pdf_per_obs(::BernoulliLikelihood, y, eta, _theta_y) = @. y * eta - log1pexp(eta)
log_pdf_per_obs(::PoissonLikelihood, y, eta, _theta_y)   = @. y * eta - exp(eta)

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

    # Choose model instance (user `latent=` argument wins for graph/spatial models).
    model = if model_sym === :SPDE
        latent_override isa SPDEModel || latent_override isa NonStationarySPDEModel ?
            latent_override : error("formula uses f(., SPDE) but no SPDE/NonStationarySPDE was passed via latent=")
    elseif model_sym === :NonStationarySPDE
        latent_override isa NonStationarySPDEModel ?
            latent_override : error("formula uses f(., NonStationarySPDE) but latent= is not a NonStationarySPDEModel")
    elseif model_sym === :ICAR
        latent_override isa ICARModel ?
            latent_override : error("formula uses f(., ICAR) but latent= is not an ICARModel")
    elseif model_sym === :Besag
        latent_override isa BesagModel ?
            latent_override : error("formula uses f(., Besag) but latent= is not a BesagModel")
    elseif model_sym === :BivariateIID
        latent_override isa BivariateIIDModel ? latent_override : BivariateIIDModel()
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
    elseif model isa BesagModel
        size(model.W, 1)
    elseif model isa ICARModel
        size(model.W, 1)
    elseif model isa BivariateIIDModel
        2 * n_unique
    else
        n_unique
    end

    # Build sparse projection A. Layout depends on model:
    # - Graph models (Besag/ICAR/SPDE/NonStationarySPDE): the covariate value is
    #   already a 1-based index into the latent vector. We use it directly so the
    #   row/column ordering matches the precision matrix W or mesh layout.
    # - BivariateIID: per-pair stacked latent (u_i, v_i). A `type ∈ {1,2}` column
    #   on the data picks which of the two slots applies.
    # - All others (IID, RW1): 1-of-K projection from `cov_data` distinct levels.
    use_direct_index = model isa BesagModel ||
                       model isa ICARModel ||
                       model isa SPDEModel ||
                       model isa NonStationarySPDEModel
    row_idx = Int[]; col_idx = Int[]; vals = Float64[]
    if model isa BivariateIIDModel
        hasproperty(data, :type) ||
            error("BivariateIID requires a `type` column with values 1 or 2")
        types = data.type
        for j in 1:n_obs
            base = 2 * (val_map[cov_data[j]] - 1) + 1
            push!(row_idx, j); push!(col_idx, base + (types[j] - 1)); push!(vals, 1.0)
        end
    elseif use_direct_index
        for j in 1:n_obs
            idx = Int(cov_data[j])
            (1 ≤ idx ≤ n_block) ||
                error("covariate $cov_name has value $idx out of bounds for n_block=$n_block")
            push!(row_idx, j); push!(col_idx, idx); push!(vals, 1.0)
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
              offset::Union{Nothing,AbstractVector{<:Real}} = nothing,
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

    # 4a. Assemble the global constraint matrix A_c (k_total × n_latent) by
    # stacking each effect's per-block constraint, padded with zero columns
    # for the fixed-effect prefix and the other effects.
    constraint_blocks = SparseMatrixCSC{Float64,Int}[]
    col_offset = n_fixed
    for e in effects
        Ac_local = constraint_matrix(e.model, e.n_block)
        if size(Ac_local, 1) > 0
            row = spzeros(size(Ac_local, 1), n_latent)
            row[:, (col_offset + 1):(col_offset + e.n_block)] = Ac_local
            push!(constraint_blocks, row)
        end
        col_offset += e.n_block
    end
    A_constraint = isempty(constraint_blocks) ?
        spzeros(Float64, 0, n_latent) :
        vcat(constraint_blocks...)
    has_intrinsic_constraint = size(A_constraint, 1) > 0

    # 4a-bis. Improper-prior detection. With `fixed_precision == 0` the
    # intercept prior is improper. Combined with an intrinsic random effect,
    # the prior `Q` has nullity 2: the intrinsic null (covered by the user
    # constraint A_c) plus `e_β` (improper intercept).
    #
    # Phase 6c.2.b approach (now superseded): augment A_c with `e_intercept`
    # to make Newton's H factor full rank, and use Rue-Held augmented log-
    # determinants on `A_full = [A_c; e_intercept']`. This pinned `β = 0`,
    # which gave H factorability but also drove BFGS to the τ → ∞ corner
    # via an unintended τ-shape effect (Phase 6f).
    #
    # Phase 6g approach (current): Newton uses A_user only (no e_intercept).
    # `gmrf_newton_full` factors `H + A_user' A_user` via `factor_augmented =
    # true`; this is full rank because the unidentifiable direction
    # `v = e_β − 1_u` has `A_user · v = -√n ≠ 0`, so the augmentation
    # contributes positive curvature in the v direction. The marginal-
    # likelihood formula (in `laplace_eval` below) switches to R-INLA's
    # "evaluate at sample = 0" expression: `Σ aᵢ + ½ log|Q_c|_pseudo
    # − ½ log|H_c|_user + ½ μ' H μ`. The e_intercept augmentation is
    # used *only* internally for `½ log_pseudo|Q|_c` (Q has 2 null
    # directions, so we need both augmentations to make Q PD before
    # Cholesky).
    has_intercept_col = n_fixed >= 1 && all(==(1.0), @view X[:, 1])
    improper_augmented = fixed_precision == 0 &&
                         has_intercept_col &&
                         has_intrinsic_constraint
    has_constraints = size(A_constraint, 1) > 0

    # 4b. Validate / normalize the offset.
    o_vec = if offset === nothing
        zeros(Float64, n_obs)
    else
        length(offset) == n_obs ||
            error("offset has length $(length(offset)) but data has $n_obs observations")
        collect(float.(offset))
    end

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
    # `fixed_precision == 0` ⇒ improper fixed-effect prior; combined with an
    # intrinsic random effect, the augmentation block above pins the
    # unidentifiable direction.
    Q_fixed = fixed_precision == 0 ?
        spzeros(Float64, n_fixed, n_fixed) :
        sparse(fixed_precision * I, n_fixed, n_fixed)

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

    """
    Returns the negative-log-posterior `obj`, the latent mode `x_star`, and the
    sparse Cholesky factor `F` of the Hessian `H`. The optimizer only needs
    `obj`; the CCD pass downstream needs the full triple.
    """
    function laplace_eval(theta::AbstractVector)
        theta_y, eff_slices = _slices(theta)
        Q = build_Q(theta)

        # `η_total = A x + offset`. The likelihood derivatives are with respect
        # to η_total, but `H = Q + A' D(-h_η) A` since the offset is a constant
        # shift in η that doesn't appear in `dη/dx`.
        grad_eta_raw, hess_eta_diag_raw = eta_derivatives(family, y_raw, theta_y)
        grad_eta_offset(eta) = grad_eta_raw(eta .+ o_vec)
        hess_eta_offset(eta) = hess_eta_diag_raw(eta .+ o_vec)

        # When the user's θ pushes Q into a numerically-singular region we
        # return a *smooth, finite* penalty rather than Inf. Inf would NaN the
        # finite-difference gradient and kill BFGS's line search; a large
        # penalty + ‖θ‖² gradient instead nudges the optimizer back toward
        # the feasible region.
        bad_theta_penalty = 1e8 + 1e3 * sum(abs2, theta)
        # Defend against poisoned warm-start (a previous failed eval can leave
        # NaN/Inf in x_warm; reset to zero in that case).
        if any(!isfinite, x_warm)
            fill!(x_warm, 0.0)
        end
        x_star = try
            gmrf_newton_full(grad_eta_offset, hess_eta_offset, A_total, Q, x_warm;
                             constraint_A = has_constraints ? A_constraint : nothing,
                             factor_augmented = improper_augmented)
        catch err
            err isa PosDefException || rethrow()
            return (bad_theta_penalty, copy(x_warm),
                    cholesky(Symmetric(sparse(I * 1.0, n_latent, n_latent))))
        end
        copyto!(x_warm, x_star)

        eta_star = A_total * x_star .+ o_vec
        ll = log_likelihood_total(family, y_raw, eta_star, theta_y)

        h_eta = hess_eta_diag_raw(eta_star)
        H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)
        # In the improper-augmented case `H` is rank-deficient along
        # `v = e_β − 1_u` (the unidentifiable direction; v ∉ ker(A_user)
        # since `A_user · v = -√n ≠ 0`). Factoring `H + A_user' A_user`
        # is positive-definite — the augmentation contributes positive
        # curvature exactly in the v direction. The same factor is reused
        # below for `log|H_c|_user` (textbook PLUS) and for sampling in
        # the IS correction.
        H_factor_target = improper_augmented ?
            (H + sparse(A_constraint') * A_constraint) : H
        F_H = try
            cholesky(Symmetric(H_factor_target))
        catch err
            err isa PosDefException || rethrow()
            return (bad_theta_penalty, x_star,
                    cholesky(Symmetric(sparse(I * 1.0, n_latent, n_latent))))
        end
        # `log_det_H_factor` always corresponds to the matrix the Cholesky
        # factor `F_H` was built on. For the proper branch it equals `log|H|`;
        # for the improper-augmented branch it equals `log|H + A_user' A_user|`.
        log_det_H_factor = logdet(F_H)

        # Constrained-determinant corrections. Three branches:
        #
        # * proper-prior + intrinsic constraint (`has_constraints` w/o
        #   improper_augmented): use textbook PLUS on `A_user`.
        #   `log|H_c| = log|H| + log(A_user H⁻¹ A_user')`. R-INLA's
        #   `problem-setup.c::1048–1049` confirms PLUS via the
        #   `[x|Ax] = [x][Ax|x]/[Ax]` decomposition.
        # * improper-augmented (Phase 6g): use R-INLA's "evaluate at
        #   sample = 0" formulation — `Σ aᵢ + ½ log_pseudo|Q|_c
        #   − ½ log|H_c|_user + ½ μ' H μ`. Here `log|H_c|_user` is via
        #   the textbook PLUS form on `A_user` using the augmented
        #   Cholesky `H + A_user' A_user`, and `½ log_pseudo|Q|_c` is
        #   via the Rue-Held augmented form `log|Q + A_full' A_full|`
        #   (Q has 2 null directions: besag/intrinsic + improper β;
        #   `A_full = [A_user; e_intercept']` covers both).
        # * no constraints: standard `log|Q| − log|H| + ll − ½ x'Qx`.
        obj_main = try
            if improper_augmented
                # Phase 6g formula. `A_constraint = A_user` here (the
                # e_intercept augmentation lives only inside this branch
                # for the `½ log_pseudo|Q|_c` computation).
                AuT = sparse(A_constraint')
                Wc  = F_H \ Matrix(AuT)
                S_user = Symmetric(Matrix(A_constraint * Wc))
                log_det_H_c_user = log_det_H_factor + logdet(S_user)

                # `½ log_pseudo|Q|_c` via Rue-Held augmented on A_full =
                # [A_user; e_intercept']. The two augmentation rows cover
                # both null directions of Q. For our orthonormal A_full
                # the augmented Cholesky's logdet equals
                # `log_pseudo|Q|_ker(A_full)`, which on ker(A_user)
                # excludes the e_β null inside the constraint set.
                e_int_row = sparse([1], [1], [1.0], 1, n_latent)
                A_full_for_Q = vcat(A_constraint, e_int_row)
                AfT_for_Q = sparse(A_full_for_Q')
                log_det_Q_c_pseudo = sparse_logdet(Q + AfT_for_Q * A_full_for_Q)

                # The R-INLA "evaluate at zero" terms.
                r_m = A_total * x_star                              # predictor without offset
                sum_a = _taylor_at_zero_loglik(family, y_raw, r_m, theta_y, o_vec)
                quad_xHx = 0.5 * dot(x_star, H * x_star)            # ½ μ' H μ

                # NOTE on cubic: R-INLA's `aa[i]` from `GMRFLib_2order_approx`
                # actually truncates the Taylor at 2nd-order (the cubic term
                # in their formula is gated on `dd != NULL` which is NULL in
                # `ai_marginal_hyperparam`'s call path). Empirically (Phase 6g+
                # Phase A) our 3rd-order Σ a_i differs from R-INLA's 2nd-order
                # one by +cubic_correction. We *do not* subtract cubic here —
                # subtracting it shifts the obj curve enough to expose the
                # global min at θ → ∞, dominated by the formula's slow right-
                # tail decay. Keeping the 3rd-order Taylor gives BFGS a
                # well-conditioned local mode in the basin near R-INLA's mode
                # θ ≈ 1.87 (R-INLA also converges via local BFGS, not
                # global optimization).
                sum_a + 0.5 * log_det_Q_c_pseudo - 0.5 * log_det_H_c_user +
                    quad_xHx
            elseif has_constraints
                AcT = sparse(A_constraint')
                Q_aug = Q + AcT * A_constraint
                log_det_Q_c = sparse_logdet(Q_aug)
                # log|H_c| = log|H| + log(A_c H⁻¹ A_c')   (full-rank H form)
                Wc = F_H \ Matrix(AcT)
                S_h = Symmetric(Matrix(A_constraint * Wc))
                log_det_H_c = log_det_H_factor + logdet(S_h)
                lp_correct = -0.5 * dot(x_star, Q * x_star) + 0.5 * log_det_Q_c
                ll + lp_correct - 0.5 * log_det_H_c
            else
                lp = -0.5 * dot(x_star, Q * x_star) + 0.5 * sparse_logdet(Q)
                ll + lp - 0.5 * log_det_H_factor
            end
        catch err
            err isa PosDefException || rethrow()
            return (bad_theta_penalty, x_star, F_H)
        end

        lprior = log_prior(family, theta_y)
        for k in eachindex(effects)
            lprior += log_prior(effects[k].model, eff_slices[k])
        end

        # Importance-sampled correction to log π̂(y|θ). The Gaussian Laplace
        # approximates the posterior `π(x|θ, y)` by `N(x*, H⁻¹)`; the true
        # density carries the cubic and higher remainder
        #   R(δ) = log p(y|x* + δ) − log p(y|x*) − grad·δ − ½ δ' (-A'D A) δ
        # of the likelihood Taylor expansion at η* (the latent prior `p(x|θ)`
        # is exactly Gaussian, contributing nothing to R for δ in ker(A_c)).
        # We estimate
        #   log p(y|θ) − log p̂_LA(y|θ) = log E_{N_c(0, H_c⁻¹)}[exp(R(δ))]
        # by Monte Carlo with `N` samples drawn from N(0, H⁻¹) via
        # `δ = F.UP \ z` and projected onto the constraint set.  Bernoulli
        # has `R = O(δ³)`; Poisson has `R = O(δ³)` too. For Gaussian
        # likelihoods R ≡ 0 — Laplace is exact and we skip the correction.
        #
        # In the improper-augmented branch the IS correction is *not*
        # added: the new "evaluate at sample = 0" formulation uses a
        # 3rd-order Taylor of the log-likelihood and the IS correction
        # would double-count the same Taylor remainder. The cubic
        # correction (Phase 6g.3) is the leading term; higher-order
        # corrections are deferred until parity tightens past 0.01 nats.
        if !improper_augmented
            is_correction = _importance_correction(family, A_total, F_H,
                                                   x_star, eta_star, theta_y,
                                                   y_raw, o_vec,
                                                   has_constraints ? A_constraint : nothing)
            obj_main += is_correction
        end

        obj = -(obj_main + lprior)
        return obj, x_star, F_H
    end

    """
    Leading Edgeworth correction to `log π̂(y|θ)` for non-Gaussian likelihoods.
    Returns a Float64; for Gaussian likelihoods returns zero. Returns 0 if any
    intermediate quantity is non-finite (numerically degenerate near constraint
    boundaries for very large τ).

    Kept around as a cheaper fallback / sanity check; the importance-sampled
    correction in `_importance_correction` is preferred (captures all
    higher-order terms, not just leading 4th-derivative + 3rd-derivative
    cross-terms).
    """
    function _edgeworth_correction(family, A_total, F_H, eta_star, theta_y, A_c)
        family isa GaussianLikelihood && return 0.0
        try
            AT = sparse(A_total')
            Z  = F_H \ Matrix(AT)
            Σ_eta = A_total * Z
            if A_c !== nothing && size(A_c, 1) > 0
                Wc = F_H \ Matrix(sparse(A_c'))
                AHcAcT = A_total * Wc
                S = Symmetric(Matrix(A_c * Wc))
                AcHcAcT_inv = inv(S)
                Σ_eta -= AHcAcT * AcHcAcT_inv * AHcAcT'
            end
            σ2_eta = diag(Σ_eta)
            h3 = third_deriv_eta(family, eta_star, theta_y)
            h4 = fourth_deriv_eta(family, eta_star, theta_y)
            c_h4   = -0.125 * sum(h4 .* σ2_eta.^2)
            h3h3   = h3 * h3'
            σ2σ2   = σ2_eta * σ2_eta'
            c_h3a  =  0.125 * sum(h3h3 .* σ2σ2 .* Σ_eta)
            c_h3b  = (1/12) * sum(h3h3 .* (Σ_eta .^ 3))
            c = c_h4 + c_h3a + c_h3b
            return isfinite(c) ? c : 0.0
        catch
            return 0.0
        end
    end

    laplace_obj(theta) = first(laplace_eval(theta))

    # 9. Optimize. BFGS with finite-diff gradients. The Laplace objective is
    # generally non-convex in θ — it has multiple local minima for IID
    # precisions whose log-posterior is flat in tail regions (Salamander is
    # the canonical example). To avoid getting trapped, we run BFGS from
    # several seed θ values: the user-provided `theta0`, a high-precision
    # corner (`theta0 .+ 5`, near the log-Gamma prior mode at log τ ≈ 9.9),
    # and a low-precision corner (`theta0 .- 5`). Whichever attains the
    # smallest objective wins.
    optf = Optimization.OptimizationFunction((th, _p) -> laplace_obj(th),
                                             ADTypes.AutoFiniteDiff())
    inner_solver = solver === :bfgs        ? BFGS()        :
                   solver === :neldermead  ? NelderMead()  :
                   solver === :newton      ? Newton()      :
                   error("unknown solver $(solver). Use :bfgs, :neldermead, or :newton.")

    seeds = [theta_init]
    if n_h > 0
        # Bracket the typical log-precision posterior. One near the log-Gamma
        # prior mode (~log 20000), one in the weak-shrinkage corner. The
        # user's `theta0` remains the primary seed.
        push!(seeds, fill(5.0, n_h))
        push!(seeds, fill(-2.0, n_h))
    end
    # Drop seeds where the objective is non-finite (e.g. degenerate Q for
    # extreme SPDE parameters). Without this filter, BFGS launched from a
    # bad seed produces NaN gradients that propagate to the CCD pass.
    feasible_seeds = filter(s -> isfinite(laplace_obj(s)), seeds)
    isempty(feasible_seeds) && error("no feasible θ seeds; try a smaller theta0")

    best_theta = first(feasible_seeds)
    best_obj   = Inf
    for seed in feasible_seeds
        prob = Optimization.OptimizationProblem(optf, seed)
        try
            sol = Optimization.solve(prob, inner_solver;
                                     maxiters = max_outer_iter,
                                     abstol = 1e-7, reltol = 1e-7)
            cand = collect(sol.u)
            v = laplace_obj(cand)
            if isfinite(v) && v < best_obj
                best_obj = v
                best_theta = cand
            end
        catch
            # one bad seed shouldn't kill the whole call
        end
    end
    theta_star = best_theta

    # 10. CCD integration over θ.
    # The CCD pass is what turns Julia's joint-mode estimator into a posterior
    # *mean* estimator (matching what R-INLA reports under summary.fixed\$mean
    # and summary.hyperpar\$mean). For n_h == 0 we skip integration entirely.
    obj_star, x_star, F_star = laplace_eval(theta_star)
    marginals_at_mode = takahashi_diag(F_star)

    if n_h == 0
        # No hyperparameters: posterior is a single Laplace at θ = (). Return mode.
        return INLAResult(x_star, x_star,
                          theta_star, theta_star,
                          marginals_at_mode,
                          Matrix{Float64}(undef, 0, 2),
                          [collect(theta_star)], [1.0], form)
    end

    # If the converged mode itself has non-finite latent (the inner Newton
    # bailed during the BFGS line search and best_theta is from a degenerate
    # seed), skip the CCD pass altogether.
    if any(!isfinite, x_star) || any(!isfinite, marginals_at_mode)
        return INLAResult(x_star, x_star,
                          theta_star, theta_star,
                          marginals_at_mode,
                          hcat(theta_star, fill(NaN, n_h)),
                          [collect(theta_star)], [1.0], form)
    end

    # Compute the Hessian H_θ at θ* via central finite differences. If the
    # finite-diff probe lands in a PD-fail region (penalty values fill the
    # Hessian entries), eigendecomposition will choke — fall back to a
    # mode-only result with a single-node "CCD".
    H_theta = _finitediff_hessian(laplace_obj, theta_star)
    if !all(isfinite, H_theta)
        return INLAResult(x_star, x_star,
                          theta_star, theta_star,
                          marginals_at_mode,
                          hcat(theta_star, fill(NaN, n_h)),
                          [collect(theta_star)], [1.0], form)
    end
    H_theta = Symmetric(H_theta + 1e-9 * I)

    # Asymmetric skewness corrections per principal axis (R-INLA's
    # `stdev_corr_pos` / `stdev_corr_neg`, gmrflib/approx-inference.c:1736–1834).
    # Probe `laplace_obj` at z = ±√2 along each eigvec; if the log-posterior
    # drops by `f0` from the mode, the local Gaussian would predict
    # `f0 = step² / 2 = 1`. The correction is `sqrt(step²/(2·f0))`, which
    # widens the grid where the posterior is fatter than Gaussian and
    # narrows it where it's tighter. Without this, heavy-tailed precision
    # posteriors (e.g. Brunei's right tail of τ) under-sample the right side.
    stdev_corr_pos, stdev_corr_neg = _compute_skew_corrections(laplace_obj,
                                                               theta_star,
                                                               Matrix(H_theta);
                                                               step = sqrt(2.0))

    # Generate integration nodes around θ* in θ-space. For `n_h ∈ {1, 2}`
    # this returns R-INLA's `int.strategy = "grid"` design (11 points spanning
    # ±3.5σ for 1D, 45 points for 2D) along with quadrature weights that
    # compensate for the non-uniform spacing; for `n_h ≥ 3` it falls back to
    # CCD with equal weights. The downstream Bayesian-quadrature combination
    # is `posterior ∝ exp(-obj_node) × quad_weight`, so the weights enter the
    # softmax additively in log-space (see below).
    nodes, quad_w = integration_nodes(theta_star, Matrix(H_theta);
                                      stdev_corr_pos = stdev_corr_pos,
                                      stdev_corr_neg = stdev_corr_neg)

    # Evaluate the posterior at each node, store (x_k, var_k, obj_k).
    # Also apply the simplified-Laplace marginal-mean correction for x:
    #   E_SLA[x_i | y, θ] = x*_i + ½ (H⁻¹ A')_(i,k) · h⁽³⁾(η*_k) · σ²_(η_k)
    # vector form: Δx = ½ H⁻¹ Aᵀ · diag(h⁽³⁾) · diag(σ²_η) · 1_{n_obs}
    #            = ½ H⁻¹ Aᵀ (h⁽³⁾ ⊙ σ²_η)
    # σ²_(η_k) = (A H⁻¹ Aᵀ)_(k,k) is computed from the joint solve `H⁻¹ Aᵀ`.
    # For Gaussian likelihoods h⁽³⁾ ≡ 0, so the SLA correction vanishes.
    n_nodes  = length(nodes)
    obj_at   = Vector{Float64}(undef, n_nodes)
    x_at     = Vector{Vector{Float64}}(undef, n_nodes)
    var_at   = Vector{Vector{Float64}}(undef, n_nodes)
    has_sla  = has_likelihood_hyperparameter(family) === false &&
               !(family isa GaussianLikelihood)
    for k in 1:n_nodes
        copyto!(x_warm, x_star)
        obj_k, x_k, F_k = laplace_eval(nodes[k])
        obj_at[k] = obj_k

        # If the inner Newton at this CCD node failed (returned the bad-θ
        # penalty), exclude this node from the mixture by giving it an
        # effectively-zero weight downstream.
        if !isfinite(obj_k) || any(!isfinite, x_k)
            obj_at[k] = +Inf
            x_at[k]   = copy(x_star)
            var_at[k] = copy(marginals_at_mode)
            continue
        end

        if has_sla
            theta_y_k, _ = _slices(nodes[k])
            eta_k = A_total * x_k .+ o_vec
            h3 = third_deriv_eta(family, eta_k, theta_y_k)
            AT = sparse(A_total')
            Z  = F_k \ Matrix(AT)
            sigma2_eta = vec(sum(A_total .* Z', dims = 2))
            v_obs = h3 .* sigma2_eta
            Av = A_total' * v_obs
            delta_x = 0.5 .* (F_k \ Av)
            if has_constraints
                violation = A_constraint * delta_x
                delta_x .-= A_constraint' * violation
            end
            x_at[k] = all(isfinite, delta_x) ? x_k .+ delta_x : x_k
        else
            x_at[k] = x_k
        end
        var_k = takahashi_diag(F_k)
        var_at[k] = all(isfinite, var_k) ? var_k : copy(marginals_at_mode)
    end

    # Convert −log posterior values into normalized weights. Nodes with Inf
    # objective (excluded) get weight 0 in the softmax.
    finite_objs = filter(isfinite, obj_at)
    if isempty(finite_objs)
        return INLAResult(x_star, x_star,
                          theta_star, theta_star,
                          marginals_at_mode,
                          hcat(theta_star, fill(NaN, n_h)),
                          [collect(theta_star)], [1.0], form)
    end
    log_w = -(obj_at .- minimum(finite_objs)) .+ log.(quad_w)
    log_w[.!isfinite.(log_w)] .= -Inf
    log_w_max = maximum(filter(isfinite, log_w))
    w = exp.(log_w .- log_w_max)
    w[.!isfinite.(w)] .= 0.0
    w ./= sum(w)

    # Mixture posterior moments for the latent field.
    x_mean = zeros(Float64, n_latent)
    for k in 1:n_nodes
        x_mean .+= w[k] .* x_at[k]
    end
    marginals_mixed = zeros(Float64, n_latent)
    for k in 1:n_nodes
        marginals_mixed .+= w[k] .* (var_at[k] .+ (x_at[k] .- x_mean).^2)
    end

    # Hyperparameter posterior moments via the same node mixture.
    theta_mean = zeros(Float64, n_h)
    for k in 1:n_nodes
        theta_mean .+= w[k] .* nodes[k]
    end
    theta_var = zeros(Float64, n_h)
    for k in 1:n_nodes
        theta_var .+= w[k] .* (nodes[k] .- theta_mean).^2
    end
    marg_hyper = hcat(theta_mean, sqrt.(max.(0.0, theta_var)))

    return INLAResult(x_star, x_mean,
                      theta_star, theta_mean,
                      marginals_mixed, marg_hyper,
                      [collect(n) for n in nodes], collect(w), form)
end

"""
    _compute_skew_corrections(obj, theta_star, hessian; step = √2)

R-INLA-style asymmetric skewness corrections per principal axis
(`gmrflib/approx-inference.c:1736–1834`). Probe `obj` at
`theta_star + step · evec_k · sqrt(1/eval_k) · sign` for each eigenvector
`k` and each `sign ∈ {+1, -1}`. Compute the drop in log-posterior
`f0 = obj(probe) − obj(mode)`. If the posterior were Gaussian at the
mode, `f0 = step² / 2`. The correction is
`sqrt(step² / (2·f0))` — > 1 when the posterior is fatter than Gaussian
on that side, < 1 when tighter. Returns `(pos, neg)` vectors of length
`n_h` (each defaulting to 1.0 if the probe lands in a non-finite
region).
"""
function _compute_skew_corrections(obj, theta_star::AbstractVector{T},
                                   hessian::AbstractMatrix{T};
                                   step::T = T(sqrt(2.0))) where {T}
    n = length(theta_star)
    n == 0 && return (T[], T[])

    evals, evecs = eigen(Symmetric(hessian))
    evals_safe = max.(evals, T(1e-9))
    sd_inv = sqrt.(one(T) ./ evals_safe)

    obj_mode = obj(theta_star)
    isfinite(obj_mode) || return (ones(T, n), ones(T, n))

    pos = ones(T, n)
    neg = ones(T, n)
    z = zeros(T, n)
    for k in 1:n
        # Probe along +z[k]: θ = θ_star + evecs · diag(sd_inv) · (step·e_k)
        fill!(z, zero(T)); z[k] = step
        delta = evecs * (sd_inv .* z)
        f0 = obj(theta_star .+ delta) - obj_mode
        pos[k] = (isfinite(f0) && f0 > 0) ? sqrt(step^2 / (2 * f0)) : one(T)

        # Probe along -z[k]
        fill!(z, zero(T)); z[k] = -step
        delta = evecs * (sd_inv .* z)
        f0 = obj(theta_star .+ delta) - obj_mode
        neg[k] = (isfinite(f0) && f0 > 0) ? sqrt(step^2 / (2 * f0)) : one(T)
    end
    return pos, neg
end

"""
    _finitediff_hessian(f, x; eps=1e-3)

Central-difference Hessian of a scalar function `f` at `x`. Used at θ* to
produce the local quadratic shape that drives CCD node placement and the
hyperparameter Gaussian approximation.
"""
function _finitediff_hessian(f, x::AbstractVector{T}; eps_::T = T(1e-3)) where {T}
    n = length(x)
    n == 0 && return Matrix{T}(undef, 0, 0)
    H = zeros(T, n, n)
    f0 = f(x)
    for i in 1:n, j in 1:i
        if i == j
            tp = copy(x); tp[i] += eps_
            tm = copy(x); tm[i] -= eps_
            H[i,i] = (f(tp) - 2*f0 + f(tm)) / eps_^2
        else
            tpp = copy(x); tpp[i] += eps_; tpp[j] += eps_
            tpm = copy(x); tpm[i] += eps_; tpm[j] -= eps_
            tmp = copy(x); tmp[i] -= eps_; tmp[j] += eps_
            tmm = copy(x); tmm[i] -= eps_; tmm[j] -= eps_
            H[i,j] = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) / (4 * eps_^2)
            H[j,i] = H[i,j]
        end
    end
    return H
end

"""
    _marginal_likelihood_cubic_correction(family, A_total, x_star, eta_star, theta_y)

Cubic correction to `log p̂(y|θ)` to match R-INLA's marginal-likelihood
formulation.

R-INLA evaluates the marginal likelihood at the origin (`x = 0`) using a
3rd-order Taylor expansion of `log p(y|η)` centered at the joint mode `η_m`
(see `gmrflib/blockupdate.c::GMRFLib_2order_approx`). We evaluate the same
quantity at the joint mode using the *exact* log-likelihood. Algebraically
the two approaches differ by

    R-INLA  -  Julia  =  -1/6 · Σ_i f'''_i · (η_m_i)³

where `f'''_i = d³ log p(y_i|η_i)/dη³` evaluated at the linear predictor
mode `η_m_i` (without offset — R-INLA's loglFunc API takes the predictor
and adds the offset internally). The derivation uses the joint-mode
condition `Q·x* = Aᵀ·grad` (which lets the linear `Σ b_i η_m_i` term cancel
between the two formulations) and the identity `½ x*' H x* = ½ x*' Q x* +
½ Σ c_i η_m_i²` that links R-INLA's "evaluate at x=0" trick to our
"evaluate at x=x*" formula.

For Gaussian likelihoods `f''' ≡ 0` so the correction vanishes. For Poisson
`f''' = -λ` (= -exp(η+offset)). For Bernoulli `f''' = -p(1-p)(1-2p)`.

Returns 0.0 on numerical failure.
"""
function _marginal_likelihood_cubic_correction(family, A_total,
                                               x_star, eta_star, theta_y)
    family isa GaussianLikelihood && return 0.0
    try
        # `eta_star = A·x* + offset`. R-INLA's `f'''` is a function of
        # `η_full = predictor + offset`, but its Taylor is in the predictor
        # `r = predictor` (without offset). At the mode, `r_m = A·x*` and
        # `η_full_m = r_m + offset = eta_star`. The third-derivative value
        # depends on `η_full_m` (e.g. Poisson `f''' = -exp(eta_star)`),
        # but the cubic argument `η_m` in the correction formula is the
        # predictor `r_m`, not the full η.
        r_m = A_total * x_star
        h3 = third_deriv_eta(family, eta_star, theta_y)
        c = -(1 / 6) * sum(h3 .* r_m .^ 3)
        return isfinite(c) ? c : 0.0
    catch
        return 0.0
    end
end

"""
    _taylor_at_zero_loglik(family, y, r_m, theta_y, offset)

R-INLA's "evaluate at sample = 0" log-likelihood (Phase 6g formulation).

For each i, compute the 3rd-order Taylor expansion of
`ℓ_i(r) = log p(y_i | r + offset_i, θ_y)` centered at the predictor mode
`r_m_i`, evaluated at `r = 0`:

    T_i(0) = ℓ_i(r_m_i) − ℓ_i'(r_m_i) r_m_i + ½ ℓ_i''(r_m_i) r_m_i²
             − 1/6 ℓ_i'''(r_m_i) r_m_i³

Returns `Σ_i T_i(0)`.

This replaces `log p(y|x*, θ)` in the marginal-likelihood formula
(R-INLA's `gmrflib/blockupdate.c::GMRFLib_2order_approx`). For Gaussian
likelihoods f''' ≡ 0 so the Taylor is exact and reduces to
`log p(y|η = offset, θ_y)` (the likelihood at predictor = 0).
"""
function _taylor_at_zero_loglik(family, y, r_m, theta_y, offset)
    eta_m = r_m .+ offset
    grad_eta, hess_eta_diag = eta_derivatives(family, y, theta_y)
    fp_at  = grad_eta(eta_m)
    fpp_at = hess_eta_diag(eta_m)
    fppp_at = third_deriv_eta(family, eta_m, theta_y)
    f0_at  = log_pdf_per_obs(family, y, eta_m, theta_y)
    return sum(@. f0_at - fp_at * r_m + 0.5 * fpp_at * r_m^2 -
                  (1/6) * fppp_at * r_m^3)
end

"""
    _importance_correction(family, A_total, F_H, x_star, eta_star, theta_y,
                           y_raw, o_vec, A_c; N=100, seed=0x42_42_42)

Monte-Carlo estimate of `log E_{N_c(0, H_c⁻¹)}[exp(R(δ))]`, the correction
the Gaussian Laplace approximation needs to recover the true `log p(y|θ)`.
Returns a Float64; for Gaussian likelihoods returns 0 (Laplace is exact).
On numerical failure returns 0 so BFGS doesn't see a non-finite obj.

Sampling: `δ = F_H.UP \\ z` with `z ~ N(0, I)` gives `δ ~ N(0, H⁻¹)`. If a
constraint `A_c` is supplied the sample is projected to `ker(A_c)` via
`δ ← δ − H⁻¹ A_c' (A_c H⁻¹ A_c')⁻¹ A_c δ`. The resulting δ is a sample
from `N(0, H_c⁻¹)` exactly.

R(δ) = log p(y|η*+Aδ) − log p(y|η*) − grad'·(Aδ) − ½ (Aδ)' D (Aδ)
     = the cubic-and-higher Taylor remainder of log p(y|·) at η*.

The RNG is seeded *deterministically* so BFGS sees a reproducible
objective (per-θ noise stable across calls).
"""
function _importance_correction(family, A_total, F_H, x_star, eta_star,
                                theta_y, y_raw, o_vec, A_c;
                                N::Int = 100, seed::UInt32 = UInt32(0x42_42_42))
    family isa GaussianLikelihood && return 0.0
    try
        rng = MersenneTwister(seed)
        n_latent = length(x_star)

        grad_eta_raw, hess_eta_diag_raw = eta_derivatives(family, y_raw, theta_y)
        grad_at = grad_eta_raw(eta_star)
        hess_at = hess_eta_diag_raw(eta_star)
        log_p_at_star = log_likelihood_total(family, y_raw, eta_star, theta_y)

        have_c = A_c !== nothing && size(A_c, 1) > 0
        local Wc, Sc_inv
        if have_c
            AcT = sparse(A_c')
            Wc  = F_H \ Matrix(AcT)
            Sc_inv = inv(Symmetric(Matrix(A_c * Wc)))
        end

        log_R = Vector{Float64}(undef, N)
        for s in 1:N
            z = randn(rng, n_latent)
            δ = vec(F_H.UP \ z)
            if have_c
                δ -= vec(Wc * (Sc_inv * (A_c * δ)))
            end
            eta_dev = A_total * δ
            eta_new = eta_star .+ eta_dev
            log_p_new = log_likelihood_total(family, y_raw, eta_new, theta_y)
            log_R[s] = log_p_new - log_p_at_star -
                       dot(grad_at, eta_dev) -
                       0.5 * sum(hess_at .* eta_dev .^ 2)
            isfinite(log_R[s]) || (log_R[s] = -Inf)
        end

        log_R_max = maximum(log_R)
        isfinite(log_R_max) || return 0.0
        mean_exp = mean(exp.(log_R .- log_R_max))
        mean_exp > 0 || return 0.0
        return log_R_max + log(mean_exp)
    catch
        return 0.0
    end
end

end # module
