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
                             constraint_A = has_constraints ? A_constraint : nothing)
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
        F_H = try
            cholesky(Symmetric(H))
        catch err
            err isa PosDefException || rethrow()
            return (bad_theta_penalty, x_star,
                    cholesky(Symmetric(sparse(I * 1.0, n_latent, n_latent))))
        end
        log_det_H = logdet(F_H)   # = log|H|, see sparse_logdet docstring

        # Constrained-determinant corrections. The Laplace approximation lives
        # on the constrained subspace `{x : A_c x = 0}`. The right log-det
        # formula depends on whether the matrix is rank-deficient along
        # row(A_c):
        #   * For `Q` (intrinsic GMRF — Besag/RW1/etc — has row(A_c) in its
        #     null space): use the Rue & Held 2005 eq. 2.30 augmented form
        #     `log|Q_c| = log|Q + A_c' A_c|`. The standard
        #     `log|Q| − log(A_c Q⁻¹ A_c')` form is undefined (Q is singular).
        #   * For `H = Q + Aᵀ D A` (full rank thanks to the data term): use
        #     the textbook form `log|H_c| = log|H| − log(A_c H⁻¹ A_c')`. The
        #     augmented form `log|H + A_c' A_c|` gives a wrong (off by
        #     `2 log s` where `s = A_c H⁻¹ A_c'`) result for full-rank H.
        # Empirically these two formulas differed by ≈ −21 nats on Brunei,
        # which is a constant shift (doesn't move the θ optimum) but is the
        # mathematically correct answer.
        obj_main = try
            if has_constraints
                AcT = sparse(A_constraint')
                Q_aug = Q + AcT * A_constraint
                log_det_Q_c = sparse_logdet(Q_aug)
                # log|H_c| = log|H| − log(A_c H⁻¹ A_c')   (full-rank H form)
                Wc = F_H \ Matrix(AcT)
                S_h = Symmetric(Matrix(A_constraint * Wc))
                log_det_H_c = log_det_H - logdet(S_h)
                lp_correct = -0.5 * dot(x_star, Q * x_star) + 0.5 * log_det_Q_c
                ll + lp_correct - 0.5 * log_det_H_c
            else
                lp = -0.5 * dot(x_star, Q * x_star) + 0.5 * sparse_logdet(Q)
                ll + lp - 0.5 * log_det_H
            end
        catch err
            err isa PosDefException || rethrow()
            return (bad_theta_penalty, x_star, F_H)
        end

        lprior = log_prior(family, theta_y)
        for k in eachindex(effects)
            lprior += log_prior(effects[k].model, eff_slices[k])
        end

        # Edgeworth-style correction to log π̂(y|θ). The Gaussian Laplace
        # approximation rests on the second-order Taylor of `log p(y|x)` at x*;
        # the next-order remainder picks up cubic and quartic terms in the
        # latent deviation. To leading order:
        #     correction = -⅛ ∑_k h⁽⁴⁾_k σ²²_k
        #                  + ⅛ ∑_{k,l} h⁽³⁾_k h⁽³⁾_l σ²_k σ²_l Σ_(k,l)
        #                  + ¹⁄₁₂ ∑_{k,l} h⁽³⁾_k h⁽³⁾_l Σ³_(k,l)
        # Where Σ_(k,l) = (A H⁻¹ A')_(k,l) is the marginal η-covariance under
        # the Gaussian Laplace, computed on the constrained subspace if
        # `has_constraints`. Adds to `obj_main` (raises log-density).
        # For Gaussian likelihoods the third and fourth derivatives are zero,
        # so the correction vanishes — Gaussian Laplace is exact.
        edgeworth_correction = _edgeworth_correction(family, A_total, F_H,
                                                    eta_star, theta_y,
                                                    has_constraints ? A_constraint : nothing)
        obj_main += edgeworth_correction

        obj = -(obj_main + lprior)
        return obj, x_star, F_H
    end

    """
    Leading Edgeworth correction to `log π̂(y|θ)` for non-Gaussian likelihoods.
    Returns a Float64; for Gaussian likelihoods returns zero. Returns 0 if any
    intermediate quantity is non-finite (numerically degenerate near constraint
    boundaries for very large τ).
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

    # Generate CCD nodes around θ* in θ-space.
    nodes = integration_nodes(theta_star, Matrix(H_theta))

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
    log_w = -(obj_at .- minimum(finite_objs))
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

end # module
