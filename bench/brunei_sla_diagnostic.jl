#!/usr/bin/env julia --project=.
#
# Phase B (simplified-laplace) diagnostic. Pair with
# `bench/brunei_sla_diagnostic.R`. At each θ on a fixed grid, dump the
# *components* of our `log p̂(y|θ)` and compare to R-INLA's `mlik_int`
# (which the R-side script just produced). Goal: pin down which scalar
# component disagrees — `ll`, `½ log|Q_c|`, `½ log|H_c|`, `½ x*' Q x*`,
# IS correction, or the prior on θ.

using IntegratedNestedLaplace
using INLAModels, INLACore
using DataFrames, CSV, SparseArrays, LinearAlgebra
using Printf, JSON
using SpecialFunctions: loggamma

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(ROOT, "examples", "06_brunei_school_disparities", "data")

const θ_GRID = [-1.0, 0.0, 0.5, 1.0, 1.5, 1.88, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

function load_brunei()
    df  = CSV.read(joinpath(DATA_DIR, "areas.csv"), DataFrame)
    adj = CSV.read(joinpath(DATA_DIR, "adjacency.csv"), DataFrame)
    n = nrow(df)
    W = sparse(adj.i, adj.j, Float64.(adj.w), n, n)
    return df, W
end

function eval_components(theta::Real, df, W)
    n = nrow(df)
    besag = BesagModel(W; scale = true)

    n_fixed = 1
    n_random = n
    n_latent = n_fixed + n_random
    Q_fixed = spzeros(Float64, n_fixed, n_fixed)        # improper intercept
    Q_random = INLAModels.precision_matrix(besag, exp(theta))
    Q = blockdiag(Q_fixed, sparse(Q_random))

    A_random = sparse(1:n, 1:n, ones(n), n, n)
    A_total  = hcat(sparse(ones(n, 1)), A_random)

    A_c_base = sparse(hcat(zeros(1, n_fixed), ones(1, n) ./ sqrt(n)))
    e_int = sparse([1], [1], [1.0], 1, n_latent)
    A_c = vcat(A_c_base, e_int)
    AcT = sparse(A_c')
    AcTAc = AcT * A_c

    o = log.(df.E)
    grad_eta_offset(eta) = df.y .- exp.(eta .+ o)
    hess_eta_offset(eta) = .-exp.(eta .+ o)
    hess_eta_raw(eta_total) = .-exp.(eta_total)

    x_warm = zeros(n_latent)
    x_star = gmrf_newton_full(grad_eta_offset, hess_eta_offset, A_total, Q, x_warm;
                              constraint_A = A_c,
                              factor_augmented = true,
                              max_iter = 200, tol = 1.0e-10)

    eta_star = A_total * x_star .+ o
    ll = sum(df.y .* eta_star .- exp.(eta_star))
    quad = -0.5 * dot(x_star, Q * x_star)

    h_eta = hess_eta_raw(eta_star)
    H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)
    H_aug = H + AcTAc
    F_H = cholesky(Symmetric(H_aug))
    log_det_Q_c = sparse_logdet(Q + AcTAc)
    log_det_H_c = sparse_logdet(H_aug)

    is_corr = IntegratedNestedLaplace._importance_correction(
        PoissonLikelihood(), A_total, F_H, x_star, eta_star,
        Float64[], df.y, o, A_c; N = 200)

    lprior = INLAModels.loggamma_logprior(theta)

    # Driver objective (negative log posterior up to constants).
    obj_main = ll + quad + 0.5 * log_det_Q_c - 0.5 * log_det_H_c + is_corr
    log_post = obj_main + lprior   # = -obj from the driver

    # Poisson factorial constant (R-INLA includes it; we don't).
    log_y_factorial = sum(loggamma.(df.y .+ 1))

    # Predicted "R - J" correction term derived analytically:
    #   log p̂_R-INLA(y|θ) - log p̂_J(y|θ) = -1/6 Σ f'''_i · η_m_i³
    # where f'''_i = d³ log p(y_i|η_i)/dη³ at η_m_i, and η_m_i is the linear
    # predictor at the joint mode (without offset, since R-INLA's loglFunc
    # parameterizes by η = predictor and adds offset internally).
    # For Poisson: f'''_i = -exp(η_m_i + o_i) = -λ_i.
    # ⇒ R - J = (1/6) Σ λ_i (η_m_i)³
    # where η_m_i = β_m + u_m_{area_i} (no offset). At our improper-augmented
    # constraint β_m ≡ 0 so η_m_i = u_m_{area_i}.
    eta_no_offset = A_total * x_star    # = β + u_{area} for each obs
    lambda_i = exp.(eta_star)            # rate at mode (with offset)
    cubic_correction = (1/6) * sum(lambda_i .* eta_no_offset .^ 3)

    return (theta = theta,
            tau = exp(theta),
            x_intercept = x_star[1],
            max_abs_u = maximum(abs, x_star[2:end]),
            ll = ll,
            quad_xQx = -2 * quad,                       # x*' Q x*
            log_det_Q_c = log_det_Q_c,
            log_det_H_c = log_det_H_c,
            is_corr = is_corr,
            lprior = lprior,
            log_post = log_post,
            log_post_with_corr = log_post + cubic_correction,
            cubic_correction = cubic_correction,
            log_y_factorial = log_y_factorial)
end

function main()
    df, W = load_brunei()
    rinla_path = joinpath(HERE, "brunei_sla_diagnostic_rinla.json")
    have_rinla = isfile(rinla_path)
    rinla_rows = if have_rinla
        d = JSON.parsefile(rinla_path)
        Dict(round(r["theta"]; digits = 4) => r for r in d["rows"])
    else
        Dict()
    end

    println()
    @printf("%6s  %10s  %10s  %10s  %12s  %12s  %12s  %10s  %12s  %12s  %12s\n",
            "θ", "τ", "β*", "max|u|",
            "ll", "0.5log|Q_c|", "-0.5log|H_c|",
            "is_corr", "lprior",
            "log_post", "R-INLA mlik")
    rows_julia = []
    for θ in θ_GRID
        r = eval_components(θ, df, W)
        rinla_mlik = if haskey(rinla_rows, round(θ; digits = 4))
            rinla_rows[round(θ; digits = 4)]["mlik_int"]
        else
            NaN
        end
        @printf("%6.2f  %10.2f  %10.4f  %10.4f  %12.4f  %12.4f  %12.4f  %10.4f  %12.4f  %12.4f  %12.4f\n",
                r.theta, r.tau, r.x_intercept, r.max_abs_u,
                r.ll, 0.5 * r.log_det_Q_c, -0.5 * r.log_det_H_c,
                r.is_corr, r.lprior, r.log_post, rinla_mlik)
        push!(rows_julia, (r..., rinla_mlik = rinla_mlik))
    end

    # Differential analysis: align Julia and R-INLA at the mode (θ ≈ 1.88) and
    # compare the SHAPE across θ. Same shape = same Gaussian-Laplace, just
    # different additive constants.
    println()
    println("=== Differential analysis (anchored at θ = 1.88) ===")
    @printf("%6s  %12s  %12s  %12s  %12s  %14s\n",
            "θ", "Δlog_post_J", "Δlog_post_R", "diff (R-J)",
            "Δcubic_corr", "Δ(J+corr)")
    j_anchor = filter(r -> isapprox(r.theta, 1.88; atol = 0.01), rows_julia)
    have_anchor = !isempty(j_anchor)
    j_anchor_lp = have_anchor ? j_anchor[1].log_post : NaN
    j_anchor_corr = have_anchor ? j_anchor[1].cubic_correction : NaN
    r_anchor_lp = if have_anchor && isfinite(j_anchor[1].rinla_mlik)
        j_anchor[1].rinla_mlik
    else
        NaN
    end
    for r in rows_julia
        Δj = r.log_post - j_anchor_lp
        Δr = r.rinla_mlik - r_anchor_lp
        Δcorr = r.cubic_correction - j_anchor_corr
        Δjp = (r.log_post + r.cubic_correction) - (j_anchor_lp + j_anchor_corr)
        @printf("%6.2f  %12.4f  %12.4f  %12.4f  %12.4f  %14.4f\n",
                r.theta, Δj, Δr, Δr - Δj, Δcorr, Δjp)
    end

    # Also dump JSON for downstream tooling.
    open(joinpath(HERE, "brunei_sla_diagnostic_julia.json"), "w") do io
        JSON.print(io, [Dict(string(k) => v for (k, v) in pairs(r)) for r in rows_julia], 2)
    end
    @info "Wrote brunei_sla_diagnostic_julia.json"
end

main()
