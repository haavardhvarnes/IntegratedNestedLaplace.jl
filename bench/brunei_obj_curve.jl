#!/usr/bin/env julia --project=.
#
# Diagnostic: print `laplace_eval` components on a fine θ grid for the
# Brunei (Besag + Poisson + sum-to-zero) parity setup.
#
# Used to track Phase 6a progress on BRUNEI_FIX.md — checks whether the
# obj curve has its mode at θ ≈ 1.88 (R-INLA's posterior mode for log τ)
# or at θ → ∞ (the current pathology).
#
# Usage:  julia --project=. bench/brunei_obj_curve.jl

using IntegratedNestedLaplace
using INLAModels
using INLACore
using DataFrames
using CSV
using SparseArrays
using LinearAlgebra
using Printf

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, ".."))

const DATA_DIR = joinpath(ROOT, "examples", "06_brunei_school_disparities", "data")

function load_brunei()
    df  = CSV.read(joinpath(DATA_DIR, "areas.csv"), DataFrame)
    adj = CSV.read(joinpath(DATA_DIR, "adjacency.csv"), DataFrame)
    n = nrow(df)
    W = sparse(adj.i, adj.j, Float64.(adj.w), n, n)
    return df, W
end

"""
    eval_at(theta_log_tau, df, W; mode=:improper)

Reproduce the driver internals at a single θ value. `mode` selects the prior
on the intercept and the constraint set used in the Laplace formula:

* `:proper` — historical 1e-3 prior on the intercept, A_c is just the
  besag sum-to-zero row. Matches the driver's behavior when the user passes
  `fixed_precision = 1e-3` (the default).

* `:improper` — 0 prior on the intercept (matches R-INLA `prec.intercept = 0`)
  + augmented A_full = [A_c; e_intercept']. This is what `fixed_precision = 0`
  triggers in the driver (Phase 6c.2.b in BRUNEI_FIX.md). Q and H are both
  rank-deficient along `null ⊆ span(A_full')`, but A_full A_full' = I so
  the Rue-Held augmented form `log|M + A_full' A_full|` directly gives
  `log|M|_{ker(A_full)}` for both M = Q and M = H.
"""
function eval_at(theta_log_tau::Real, df, W; mode::Symbol = :improper)
    n = nrow(df)
    besag = BesagModel(W; scale = true)

    # Fixed effects: intercept + Besag block.
    n_fixed = 1
    n_random = n
    n_latent = n_fixed + n_random
    fixed_prec = mode === :improper ? 0.0 : 1.0e-3
    Q_fixed = fixed_prec == 0 ? spzeros(Float64, n_fixed, n_fixed) :
                                sparse(fixed_prec * I, n_fixed, n_fixed)
    Q_random = INLAModels.precision_matrix(besag, exp(theta_log_tau))
    Q = blockdiag(Q_fixed, sparse(Q_random))

    # A: intercept column + identity for the area block.
    A_random = sparse(1:n, 1:n, ones(n), n, n)
    A_total  = hcat(sparse(ones(n, 1)), A_random)

    # Constraint: sum-to-zero on area block (always present).
    A_c_base = sparse(hcat(zeros(1, n_fixed), ones(1, n) ./ sqrt(n)))
    # Augmented constraint adds e_intercept' for the improper case so that
    # H + A_full' A_full is full rank.
    A_c = if mode === :improper
        e_int = sparse([1], [1], [1.0], 1, n_latent)
        vcat(A_c_base, e_int)
    else
        A_c_base
    end
    AcT = sparse(A_c')
    AcTAc = AcT * A_c

    # Likelihood (Poisson with offset log E). The Newton step takes the linear
    # predictor *without* the offset (η = A·x), so its grad/hess closures add
    # the offset internally. Outside Newton we have `eta_star = A·x* + o`
    # already including the offset, so we use a *raw* hessian (no offset)
    # there to avoid double-applying it. The driver does the same — it keeps
    # `hess_eta_diag_raw` and `hess_eta_offset` as separate closures.
    o = log.(df.E)
    grad_eta_offset(eta) = df.y .- exp.(eta .+ o)
    hess_eta_offset(eta) = .-exp.(eta .+ o)
    hess_eta_raw(eta_total) = .-exp.(eta_total)        # eta_total = A·x + o

    x_warm = zeros(n_latent)
    x_star = gmrf_newton_full(grad_eta_offset, hess_eta_offset, A_total, Q, x_warm;
                              constraint_A = A_c,
                              factor_augmented = mode === :improper,
                              max_iter = 200, tol = 1.0e-10)

    eta_star = A_total * x_star .+ o
    ll = sum(df.y .* eta_star .- exp.(eta_star))

    h_eta = hess_eta_raw(eta_star)
    H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)

    # Rue & Held 2005 eq. 2.30 augmented form (always valid when null(M) ⊆
    # span(A_c') and A_c A_c' = I): log|M_c| = log|M + A_c' A_c|.
    Q_aug = Q + AcTAc
    H_aug = H + AcTAc
    log_det_Q_c_aug = sparse_logdet(Q_aug)
    log_det_H_c_aug = sparse_logdet(H_aug)

    # Textbook form for full-rank H (only valid when fixed_prec > 0):
    # log|H_c| = log|H| + log(A_c H^{-1} A_c'). For the improper-augmented
    # case H is rank-deficient and we factor H_aug instead.
    H_factor_target = mode === :improper ? H_aug : H
    F_H = cholesky(Symmetric(H_factor_target))
    log_det_H_full = mode === :improper ? NaN : logdet(F_H)
    log_det_H_c_text = if mode === :improper
        NaN
    else
        Wc = F_H \ Matrix(AcT)
        Sh = Matrix(A_c * Wc)
        log_det_H_full + logdet(Symmetric(Sh))
    end

    # Common pieces.
    quad = -0.5 * dot(x_star, Q * x_star)
    lprior = INLAModels.loggamma_logprior(theta_log_tau)

    # obj_main using the augmented form for H (what the driver now uses for
    # the improper case; also valid for the proper case).
    obj_main_aug  = ll + quad + 0.5 * log_det_Q_c_aug - 0.5 * log_det_H_c_aug
    # obj_main using the textbook form (only meaningful for proper case).
    obj_main_text = mode === :improper ? NaN :
                    ll + quad + 0.5 * log_det_Q_c_aug - 0.5 * log_det_H_c_text

    # IS correction at this θ.
    is_corr = IntegratedNestedLaplace._importance_correction(
        PoissonLikelihood(), A_total, F_H, x_star, eta_star,
        Float64[], df.y, o, A_c; N = 200)

    # Driver's actual objective (what BFGS sees) includes the IS correction.
    obj_main_driver = obj_main_aug + is_corr

    return (theta = theta_log_tau, tau = exp(theta_log_tau),
            mode = mode,
            x_intercept = x_star[1],
            sum_u   = sum(x_star[2:end]),
            max_abs_u = maximum(abs, x_star[2:end]),
            ll = ll, quad = quad, lprior = lprior,
            log_det_Q_c = log_det_Q_c_aug,
            log_det_H_c_aug  = log_det_H_c_aug,
            log_det_H_c_text = log_det_H_c_text,
            is_corr = is_corr,
            obj_aug   = -(obj_main_aug    + lprior),    # driver minus IS
            obj_driver = -(obj_main_driver + lprior),   # what driver/BFGS minimizes
            obj_text  = -(obj_main_text + lprior),
            obj_text_is = -(obj_main_text + is_corr + lprior))
end

function main()
    df, W = load_brunei()
    grid = vcat(-1.0, range(0.0, 12.0; length = 25))

    for use_mode in (:improper, :proper)
        println()
        println("=== mode = $use_mode ===")
        @printf("%6s  %10s  %8s  %9s  %9s  %9s  %9s  %9s  %9s  %9s\n",
                "θ", "τ", "intercept", "sum(u)", "max|u|",
                "ll", "0.5log|Q_c|", "−0.5log|H_c|aug", "is_corr",
                "obj_driver")
        for θ in grid
            r = eval_at(θ, df, W; mode = use_mode)
            @printf("%6.2f  %10.2f  %8.4f  %9.1e  %9.4f  %9.3f  %12.3f  %15.3f  %9.3f  %10.3f\n",
                    r.theta, r.tau, r.x_intercept, r.sum_u, r.max_abs_u,
                    r.ll, 0.5 * r.log_det_Q_c, -0.5 * r.log_det_H_c_aug,
                    r.is_corr, r.obj_driver)
        end
    end

    println()
    println("IS correction (Phase 6b) on the proper-prior path:  log E[exp(R(δ))] at each θ")
    @printf("  %6s  %12s  %12s  %12s\n", "θ", "obj_text", "is_corr", "obj_text_is")
    for θ in grid
        r = eval_at(θ, df, W; mode = :proper)
        @printf("  %6.2f  %12.4f  %12.4f  %12.4f\n", r.theta, r.obj_text, r.is_corr, r.obj_text_is)
    end
end

main()
