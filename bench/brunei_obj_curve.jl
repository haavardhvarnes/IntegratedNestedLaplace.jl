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
    eval_at(theta_log_tau, df, W; verbose=false)

Reproduce the Phase 4 / Phase 5 driver internals at a single θ value. Returns
a NamedTuple of components so we can see exactly where the τ → ∞ drift comes
from.
"""
function eval_at(theta_log_tau::Real, df, W)
    n = nrow(df)
    besag = BesagModel(W; scale = true)

    # Fixed effects: intercept + Besag block.
    n_fixed = 1
    n_random = n
    n_latent = n_fixed + n_random
    Q_fixed = sparse(1.0e-3 * I, n_fixed, n_fixed)
    Q_random = INLAModels.precision_matrix(besag, exp(theta_log_tau))
    Q = blockdiag(Q_fixed, sparse(Q_random))

    # A: intercept column + identity for the area block.
    A_random = sparse(1:n, 1:n, ones(n), n, n)
    A_total  = hcat(sparse(ones(n, 1)), A_random)

    # Constraint: sum-to-zero on area block, normalized.
    A_c = sparse(hcat(zeros(1, n_fixed), ones(1, n) ./ sqrt(n)))
    AcT = sparse(A_c')
    AcTAc = AcT * A_c

    # Likelihood (Poisson with offset log E).
    o = log.(df.E)
    grad_eta(eta) = df.y .- exp.(eta .+ o)
    hess_eta(eta) = .-exp.(eta .+ o)

    x_warm = zeros(n_latent)
    x_star = gmrf_newton_full(grad_eta, hess_eta, A_total, Q, x_warm;
                              constraint_A = A_c, max_iter = 200, tol = 1.0e-10)

    eta_star = A_total * x_star .+ o
    ll = sum(df.y .* eta_star .- exp.(eta_star))

    h_eta = hess_eta(eta_star)
    H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)

    # Augmented (Rue & Held 2005 eq. 2.30) form, what's currently in the driver.
    Q_aug = Q + AcTAc
    H_aug = H + AcTAc
    log_det_Q_c_aug = sparse_logdet(Q_aug)
    log_det_H_c_aug = sparse_logdet(H_aug)

    # Textbook form for full-rank H: log|H_c| = log|H| - log(A_c H^{-1} A_c').
    F_H = cholesky(Symmetric(H))
    log_det_H = logdet(F_H)
    Wc = F_H \ Matrix(AcT)
    Sh = Matrix(A_c * Wc)
    log_AcHinv = logdet(Symmetric(Sh))
    log_det_H_c_text = log_det_H - log_AcHinv

    # Common pieces.
    quad = -0.5 * dot(x_star, Q * x_star)
    lprior = INLAModels.loggamma_logprior(theta_log_tau)

    # obj_main using the augmented (current) form for H.
    obj_main_aug  = ll + quad + 0.5 * log_det_Q_c_aug - 0.5 * log_det_H_c_aug
    # obj_main using the textbook form for H (what 6a is testing).
    obj_main_text = ll + quad + 0.5 * log_det_Q_c_aug - 0.5 * log_det_H_c_text

    return (theta = theta_log_tau, tau = exp(theta_log_tau),
            x_intercept = x_star[1],
            sum_u   = sum(x_star[2:end]),
            max_abs_u = maximum(abs, x_star[2:end]),
            ll = ll, quad = quad, lprior = lprior,
            log_det_Q_c = log_det_Q_c_aug,
            log_det_H_c_aug  = log_det_H_c_aug,
            log_det_H_c_text = log_det_H_c_text,
            obj_aug  = -(obj_main_aug  + lprior),
            obj_text = -(obj_main_text + lprior))
end

function main()
    df, W = load_brunei()
    grid = vcat(-1.0, range(0.0, 12.0; length = 25))

    println()
    @printf("%6s  %10s  %8s  %9s  %9s  %9s  %9s  %9s  %9s  %9s\n",
            "θ", "τ", "intercept", "sum(u)", "max|u|",
            "ll", "0.5log|Q_c|", "−0.5log|H_c|aug", "−0.5log|H_c|text",
            "obj_aug")
    for θ in grid
        r = eval_at(θ, df, W)
        @printf("%6.2f  %10.2f  %8.4f  %9.1e  %9.4f  %9.3f  %12.3f  %15.3f  %16.3f  %9.3f\n",
                r.theta, r.tau, r.x_intercept, r.sum_u, r.max_abs_u,
                r.ll, 0.5 * r.log_det_Q_c, -0.5 * r.log_det_H_c_aug,
                -0.5 * r.log_det_H_c_text, r.obj_aug)
    end

    println()
    println("Compare obj_aug (current) vs obj_text (Phase 6a candidate):")
    @printf("  %6s  %12s  %12s  %12s\n", "θ", "obj_aug", "obj_text", "Δ")
    for θ in grid
        r = eval_at(θ, df, W)
        @printf("  %6.2f  %12.4f  %12.4f  %12.4f\n", r.theta, r.obj_aug, r.obj_text, r.obj_text - r.obj_aug)
    end
end

main()
