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

function dump_components(theta::Real)
    df, W = load_brunei()
    n = nrow(df)
    besag = BesagModel(W; scale = true)

    # Driver-equivalent setup
    n_fixed = 1
    n_random = n
    n_latent = n_fixed + n_random
    Q_fixed_block  = sparse(1.0e-3 * I, n_fixed, n_fixed)
    Q_random_block = INLAModels.precision_matrix(besag, exp(theta))
    Q = blockdiag(Q_fixed_block, sparse(Q_random_block))

    A_random = sparse(1:n, 1:n, ones(n), n, n)
    A_total  = hcat(sparse(ones(n, 1)), A_random)

    # Sum-to-zero constraint on area block, normalized so A_c A_c' = 1.
    A_c = sparse(hcat(zeros(1, n_fixed), ones(1, n) ./ sqrt(n)))
    AcT = sparse(A_c')

    # Constrained Newton at θ.
    o = log.(df.E)
    grad_eta(eta) = df.y .- exp.(eta .+ o)
    hess_eta(eta) = .-exp.(eta .+ o)
    x_warm = zeros(n_latent)
    x_star = gmrf_newton_full(grad_eta, hess_eta, A_total, Q, x_warm;
                              constraint_A = A_c, max_iter = 200, tol = 1.0e-10)

    # All the scalar components.
    eta_star = A_total * x_star .+ o
    ll = sum(df.y .* eta_star .- exp.(eta_star))   # log p(y | x*, θ) up to factorial constant

    quad_term = 0.5 * dot(x_star, Q * x_star)       # (1/2) x*' Q x*

    # Q-side log-determinants.
    Q_aug = Q + AcT * A_c
    log_det_Q_aug = sparse_logdet(Q_aug)            # = log|Q + A_c'A_c| = log|Q*| (the Rue-Held form)
    # log|A_c A_c'| = log(1) = 0 for normalized A_c.
    log_det_AcAct = log(only(Matrix(A_c * AcT)))    # check empirically; should be ~0

    # H = Q + A^T D A.
    h_eta = hess_eta(eta_star)
    H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)
    F_H = cholesky(Symmetric(H))
    log_det_H = logdet(F_H)
    Wc = F_H \ Matrix(AcT)
    Sh = Symmetric(Matrix(A_c * Wc))
    log_AcHinv = logdet(Sh)                          # log|A_c H^{-1} A_c'|

    # R-INLA-style "log|A_c Q^{-1} A_c'|". Q is rank-deficient along A_c so the
    # raw solve fails; use the augmented-Cholesky workaround:
    # Q^{+} restricted to A_c gives  A_c (Q + A_c'A_c)^{-1} A_c' / (1 - A_c (Q+A_c'A_c)^{-1} A_c')
    # which simplifies for normalized A_c. Easier: use the identity
    #     A_c Q^{-1} A_c' = "marginal precision of A_c x in unconstrained" = ∞ for intrinsic.
    # So `log|A_c Q^{-1} A_c'|` is +∞ if Q is exactly intrinsic. R-INLA must be
    # solving against a regularized Q. We approximate by solving (Q + ε I) y = A_c'
    # for several ε and extrapolating.
    Q_aug_full = Matrix(Q + AcT * A_c)
    z_aug = Q_aug_full \ Vector(A_c[1, :])
    s_aug = dot(A_c[1, :], z_aug)                    # A_c (Q + A_c'A_c)^{-1} A_c'
    # On the orthogonal complement, A_c (Q+A_c'A_c)^{-1} A_c' = (1)/(1+1) ... not what we want.
    # For diagnostic, also report this raw value.

    # Importance-sampled correction (for completeness).
    is_corr = IntegratedNestedLaplace._importance_correction(
        PoissonLikelihood(), A_total, F_H, x_star, eta_star,
        Float64[], df.y, o, A_c; N = 200)

    lprior = INLAModels.loggamma_logprior(theta)

    # Our objective: obj_main = ll + lp_correct - 0.5 * log_det_H_c
    # with lp_correct = -0.5 x'Qx + 0.5 log|Q + A_c'A_c|
    # and  log_det_H_c (textbook) = log|H| - log(A_c H^{-1} A_c')
    log_det_H_c_text = log_det_H - log_AcHinv
    obj_main_julia = ll - quad_term + 0.5 * log_det_Q_aug - 0.5 * log_det_H_c_text + is_corr
    obj_julia = -(obj_main_julia + lprior)

    @info "Components at θ" theta=theta tau=exp(theta) n_obs=n n_latent=n_latent
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
    @printf("  log|Q + A_c'A_c|               = %.6f   (Rue-Held |Q*|, our log_det_Q_c)\n", log_det_Q_aug)
    @printf("  log|A_c A_c'|                  = %.6f   (≈ 0 for normalized A_c)\n", log_det_AcAct)
    @printf("  A_c (Q+A_c'A_c)^{-1} A_c'      = %.6f   (raw, NOT what R-INLA wants)\n", s_aug)
    @printf("\n")
    @printf("  log|H|                         = %.6f\n", log_det_H)
    @printf("  log(A_c H^{-1} A_c')           = %.6f\n", log_AcHinv)
    @printf("  log|H_c| (textbook)            = %.6f\n", log_det_H_c_text)
    @printf("\n")
    @printf("  IS correction                  = %.6f\n", is_corr)
    @printf("  log p(θ)                       = %.6f\n", lprior)
    @printf("\n")
    @printf("  obj_main (Julia)               = %.6f\n", obj_main_julia)
    @printf("  obj      (Julia, = -log p̂(y|θ) up to const) = %.6f\n", obj_julia)

    # Also export some internals as JSON so the R script can compare easily.
    out = Dict(
        "theta"        => theta,
        "tau"          => exp(theta),
        "n_obs"        => n,
        "n_latent"     => n_latent,
        "intercept"    => x_star[1],
        "u_first5"     => x_star[2:6],
        "sum_u"        => sum(x_star[2:end]),
        "norm_u"       => norm(x_star[2:end]),
        "ll"           => ll,
        "quad_xQx"     => quad_term,
        "log_det_Q_aug"   => log_det_Q_aug,
        "log_det_H"       => log_det_H,
        "log_AcHinvAct"   => log_AcHinv,
        "log_det_H_c"     => log_det_H_c_text,
        "is_corr"         => is_corr,
        "lprior"          => lprior,
        "obj_main_julia"  => obj_main_julia,
        "obj_julia"       => obj_julia,
    )
    open(joinpath(HERE, "brunei_dump_julia.json"), "w") do io
        JSON.print(io, out, 2)
    end
    @info "Wrote brunei_dump_julia.json"
end

dump_components(θ_FIXED)
