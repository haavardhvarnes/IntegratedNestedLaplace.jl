#!/usr/bin/env julia --project=.
#
# Phase 6c.1 diagnostic: at a single fixed θ, print every scalar component
# of the constrained-Laplace marginal-log-density formula. Pair with
# `bench/brunei_rinla_dump.R` to identify which term disagrees with R-INLA.

using IntegratedNestedLaplace
using INLAModels
using INLACore
using DataFrames
using CSV
using SparseArrays
using LinearAlgebra
using Printf
using JSON

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(ROOT, "examples", "06_brunei_school_disparities", "data")

const θ_FIXED = parse(Float64, get(ENV, "THETA", "2.0"))

function load_brunei()
    df  = CSV.read(joinpath(DATA_DIR, "areas.csv"), DataFrame)
    adj = CSV.read(joinpath(DATA_DIR, "adjacency.csv"), DataFrame)
    n = nrow(df)
    W = sparse(adj.i, adj.j, Float64.(adj.w), n, n)
    return df, W
end

function dump_components(theta::Real; mode::Symbol = :improper)
    df, W = load_brunei()
    n = nrow(df)
    besag = BesagModel(W; scale = true)

    # Driver-equivalent setup
    n_fixed = 1
    n_random = n
    n_latent = n_fixed + n_random

    # Choose intercept prior + constraint set.
    # `:improper` matches R-INLA `prec.intercept = 0`: zero precision on the
    # intercept and an augmented A_full = [A_c; e_intercept'] so that
    # `H + A_full' A_full` is full rank (Phase 6c.2.b in BRUNEI_FIX.md).
    # `:proper` is the historical 1e-3 + besag-only constraint path.
    fixed_prec = mode === :improper ? 0.0 : 1.0e-3
    Q_fixed_block = fixed_prec == 0 ? spzeros(Float64, n_fixed, n_fixed) :
                                      sparse(fixed_prec * I, n_fixed, n_fixed)
    Q_random_block = INLAModels.precision_matrix(besag, exp(theta))
    Q = blockdiag(Q_fixed_block, sparse(Q_random_block))

    A_random = sparse(1:n, 1:n, ones(n), n, n)
    A_total  = hcat(sparse(ones(n, 1)), A_random)

    # Sum-to-zero constraint on area block, normalized so A_c A_c' = 1.
    A_c_base = sparse(hcat(zeros(1, n_fixed), ones(1, n) ./ sqrt(n)))
    A_c = if mode === :improper
        e_int = sparse([1], [1], [1.0], 1, n_latent)   # intercept column
        vcat(A_c_base, e_int)
    else
        A_c_base
    end
    AcT = sparse(A_c')

    # Constrained Newton at θ. Newton takes the offset-less linear predictor
    # `η = A·x` (its closures add the offset internally); outside Newton we
    # already have `eta_star = A·x* + o` so the H build uses a *raw*
    # (no-offset) hessian to avoid double-applying the offset.
    o = log.(df.E)
    grad_eta_offset(eta) = df.y .- exp.(eta .+ o)
    hess_eta_offset(eta) = .-exp.(eta .+ o)
    hess_eta_raw(eta_total) = .-exp.(eta_total)
    x_warm = zeros(n_latent)
    x_star = gmrf_newton_full(grad_eta_offset, hess_eta_offset, A_total, Q, x_warm;
                              constraint_A = A_c,
                              factor_augmented = mode === :improper,
                              max_iter = 200, tol = 1.0e-10)

    # All the scalar components.
    eta_star = A_total * x_star .+ o
    ll = sum(df.y .* eta_star .- exp.(eta_star))   # log p(y | x*, θ) up to factorial constant

    quad_term = 0.5 * dot(x_star, Q * x_star)       # (1/2) x*' Q x*

    # Q-side log-determinant (Rue-Held augmented form).
    Q_aug = Q + AcT * A_c
    log_det_Q_aug = sparse_logdet(Q_aug)
    log_det_AcAct = logdet(Symmetric(Matrix(A_c * AcT)))   # ≈ 0 when A_c orthonormal

    # H = Q + A^T D A.
    h_eta = hess_eta_raw(eta_star)
    H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)

    # In :improper mode H is rank-deficient — factor H_aug = H + A_c' A_c.
    H_factor_target = mode === :improper ? (H + AcT * A_c) : H
    F_H = cholesky(Symmetric(H_factor_target))
    log_det_H = mode === :improper ? NaN : logdet(F_H)
    log_det_H_aug = sparse_logdet(H + AcT * A_c)   # always finite when A_c spans null(H)

    # Textbook "log(A_c H^{-1} A_c')" (only valid for :proper, where H is full rank).
    log_AcHinv = if mode === :improper
        NaN
    else
        Wc = F_H \ Matrix(AcT)
        Sh = Symmetric(Matrix(A_c * Wc))
        logdet(Sh)
    end

    # Importance-sampled correction (uses F_H and A_c — augmented for :improper).
    is_corr = IntegratedNestedLaplace._importance_correction(
        PoissonLikelihood(), A_total, F_H, x_star, eta_star,
        Float64[], df.y, o, A_c; N = 200)

    lprior = INLAModels.loggamma_logprior(theta)

    # Driver-equivalent objective.
    # :improper — log|H_c| = log|H + A_c' A_c| (Rue-Held augmented).
    # :proper   — log|H_c| = log|H| + log(A_c H^{-1} A_c') (textbook PLUS).
    log_det_H_c = mode === :improper ? log_det_H_aug : (log_det_H + log_AcHinv)
    obj_main_julia = ll - quad_term + 0.5 * log_det_Q_aug - 0.5 * log_det_H_c + is_corr
    obj_julia = -(obj_main_julia + lprior)

    @info "Components at θ" theta=theta tau=exp(theta) n_obs=n n_latent=n_latent mode=mode
    @printf("  intercept x*[1]                = %.6f\n", x_star[1])
    @printf("  u*[1:5]                        = [%s]\n",
            join([@sprintf("%.6f", v) for v in x_star[2:6]], ", "))
    @printf("  predictor (β + u_i)[1:5]       = [%s]\n",
            join([@sprintf("%.6f", x_star[1] + x_star[1+i]) for i in 1:5], ", "))
    @printf("  sum(u*) = sum(x*[2:end])       = %.6e\n", sum(x_star[2:end]))
    @printf("  ||u*||_2                       = %.6f\n", norm(x_star[2:end]))
    @printf("  max|u*|                        = %.6f\n", maximum(abs, x_star[2:end]))
    @printf("\n")
    @printf("  log p(y | x*, θ)               = %.6f\n", ll)
    @printf("  (1/2) x*' Q x*                 = %.6f\n", quad_term)
    @printf("  log|Q + A_c'A_c|               = %.6f   (Rue-Held |Q*|)\n", log_det_Q_aug)
    @printf("  log|A_c A_c'|                  = %.6f   (≈ 0 for orthonormal A_c)\n", log_det_AcAct)
    @printf("\n")
    @printf("  log|H| (full rank only)        = %.6f\n", log_det_H)
    @printf("  log|H + A_c' A_c|              = %.6f   (Rue-Held augmented)\n", log_det_H_aug)
    @printf("  log(A_c H^{-1} A_c') (textbook)= %.6f\n", log_AcHinv)
    @printf("  log|H_c| (driver path)         = %.6f\n", log_det_H_c)
    @printf("\n")
    @printf("  IS correction                  = %.6f\n", is_corr)
    @printf("  log p(θ)                       = %.6f\n", lprior)
    @printf("\n")
    @printf("  obj_main (Julia)               = %.6f\n", obj_main_julia)
    @printf("  obj      (Julia, = -log p̂(y|θ) up to const) = %.6f\n", obj_julia)

    # Also export some internals as JSON so the R script can compare easily.
    out = Dict(
        "theta"          => theta,
        "tau"            => exp(theta),
        "mode"           => string(mode),
        "n_obs"          => n,
        "n_latent"       => n_latent,
        "intercept"      => x_star[1],
        "u_first5"       => x_star[2:6],
        "sum_u"          => sum(x_star[2:end]),
        "norm_u"         => norm(x_star[2:end]),
        "ll"             => ll,
        "quad_xQx"       => quad_term,
        "log_det_Q_aug"  => log_det_Q_aug,
        "log_det_H"      => log_det_H,
        "log_det_H_aug"  => log_det_H_aug,
        "log_AcHinvAct"  => log_AcHinv,
        "log_det_H_c"    => log_det_H_c,
        "is_corr"        => is_corr,
        "lprior"         => lprior,
        "obj_main_julia" => obj_main_julia,
        "obj_julia"      => obj_julia,
    )
    # NaN values (e.g. log|H| in the :improper branch where H is rank-deficient)
    # don't survive standard JSON serialization; map them to `nothing` so the
    # downstream R-comparison script gets explicit nulls instead of failures.
    out_json = Dict{String,Any}(k => (v isa Float64 && isnan(v)) ? nothing : v
                                for (k, v) in out)
    suffix = mode === :improper ? "improper" : "proper"
    open(joinpath(HERE, "brunei_dump_julia_$(suffix).json"), "w") do io
        JSON.print(io, out_json, 2)
    end
    @info "Wrote brunei_dump_julia_$(suffix).json"
end

dump_components(θ_FIXED; mode = Symbol(get(ENV, "MODE", "improper")))
