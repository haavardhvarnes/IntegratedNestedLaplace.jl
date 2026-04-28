#!/usr/bin/env julia --project=.
#
# Phase 6f.1 component-level diagnostic (Julia side). Mirror
# `bench/brunei_sla_components.R`: dump every scalar component of our
# `log p̂(y|θ)` plus alternative-form log-determinants (`log|H|_{ker(A_user)}`
# vs `log|H|_{ker(A_full)}`) so we can pinpoint which component drifts with
# τ to explain the residual ~5-nat right-tail gap (Phase 6f).
#
# Phase 6g.1 extension: also evaluate every component at *R-INLA's* mode
# `(β_R, u_R)` from `cfg$mean` (loaded from `brunei_sla_components_rinla.json`).
# Compute the Taylor `Σ a_i` (R-INLA's "evaluate at sample = 0" log-lik),
# the exact `ll` at R's mode, the cubic correction `−1/6 Σ f'''_i r_m_i³`,
# the `½ x_R' Q x_R` and `½ x_R' H_at_R x_R` quadratics, and the
# `log|H_c|_user` textbook PLUS form computed against H rebuilt at R's
# mode. The diagnostic comparator in `brunei_sla_compare.jl` reconstructs
#   mlik_R_path = (Σ aᵢ) + ½ log|Q_c|_pseudo − ½ log|H_c|_user
#                 + ½ x_R' H x_R + lprior
#   mlik_J_path = ll_at_R + ½ log|Q_c|_pseudo − ½ log|H_c|_user
#                 − ½ x_R' Q x_R + lprior
# and verifies (a) `mlik_R_path − mlik_J_path == −cubic` (algebraic
# identity, ≤ 1e-6 nats) and (b) `mlik_R_path == res$mlik[1,1]` (R-INLA
# reconstruction, ≤ 0.01 nats).

using IntegratedNestedLaplace
using INLAModels, INLACore
using DataFrames, CSV, SparseArrays, LinearAlgebra
using Printf, JSON

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(ROOT, "examples", "06_brunei_school_disparities", "data")

const θ_GRID = [0.0, 1.0, 1.88, 3.0, 5.0, 7.0, 10.0]

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
    Q_fixed = spzeros(Float64, n_fixed, n_fixed)
    Q_random = INLAModels.precision_matrix(besag, exp(theta))
    Q = blockdiag(Q_fixed, sparse(Q_random))

    A_random = sparse(1:n, 1:n, ones(n), n, n)
    A_total  = hcat(sparse(ones(n, 1)), A_random)

    # User constraint = sum-to-zero on u (1 row, padded with zero on β).
    A_user = sparse(hcat(zeros(1, n_fixed), ones(1, n) ./ sqrt(n)))
    # Augmented = user constraint + e_intercept.
    e_int = sparse([1], [1], [1.0], 1, n_latent)
    A_full = vcat(A_user, e_int)

    o = log.(df.E)
    grad_eta_offset(eta) = df.y .- exp.(eta .+ o)
    hess_eta_offset(eta) = .-exp.(eta .+ o)
    hess_eta_raw(eta_total) = .-exp.(eta_total)

    x_warm = zeros(n_latent)
    x_star = gmrf_newton_full(grad_eta_offset, hess_eta_offset, A_total, Q, x_warm;
                              constraint_A = A_full,
                              factor_augmented = true,
                              max_iter = 200, tol = 1.0e-10)

    eta_star = A_total * x_star .+ o
    r_m = A_total * x_star                              # predictor without offset
    ll_exact = sum(df.y .* eta_star .- exp.(eta_star))

    h_eta = hess_eta_raw(eta_star)
    H = Q + sparse(A_total' * spdiagm(0 => -h_eta) * A_total)

    # All forms of constraint corrections we might use.
    AuT = sparse(A_user');  AuTAu = AuT * A_user
    AfT = sparse(A_full');  AfTAf = AfT * A_full

    # Full-rank augmented forms (Rue-Held augmented, both subspaces).
    H_aug_user  = H + AuTAu
    H_aug_full  = H + AfTAf
    Q_aug_user  = Q + AuTAu
    Q_aug_full  = Q + AfTAf

    safe_logdet(M) = try sparse_logdet(M) catch; -Inf end
    log_det_H_aug_user  = safe_logdet(H_aug_user)
    log_det_H_aug_full  = safe_logdet(H_aug_full)
    log_det_Q_aug_user  = safe_logdet(Q_aug_user)
    log_det_Q_aug_full  = safe_logdet(Q_aug_full)

    # `log|H|` directly (will be -Inf if H is rank-deficient).
    log_det_H_direct = try
        F = cholesky(Symmetric(H))
        logdet(F)
    catch
        -Inf
    end

    # Textbook PLUS form using the augmented H factor (for both user + full).
    # We compute `log|H|_{ker(A_c)}` for each constraint set via the identity
    #     log|H|_{ker(A_c)} = log|H + A_c' A_c| + log(A_c (H + A_c' A_c)⁻¹ A_c')
    # (valid for orthonormal A_c; both pieces use the *same* augmented H).
    F_H_aug_full = cholesky(Symmetric(H_aug_full))
    F_H_aug_user = cholesky(Symmetric(H_aug_user))   # full rank since v ∉ ker(A_user)

    Wc_user = F_H_aug_user \ Matrix(AuT)
    S_user = Symmetric(Matrix(A_user * Wc_user))
    log_det_AuHinv_proper = log(only(S_user.data))   # 1×1, must be > 0

    log_det_H_c_user_proper = log_det_H_aug_user + log_det_AuHinv_proper

    Wc_full_via_aug = F_H_aug_full \ Matrix(AfT)
    S_full_via_aug = Symmetric(Matrix(A_full * Wc_full_via_aug))
    log_det_AfHinv_via_aug = logdet(S_full_via_aug)
    log_det_H_c_full_proper = log_det_H_aug_full + log_det_AfHinv_via_aug

    # log|A_c · A_c'| Jacobian
    log_det_AuAt = log(only(Matrix(A_user * AuT)))      # ≈ 0
    log_det_AfAt = logdet(Symmetric(Matrix(A_full * AfT)))

    # Quadratic forms at the joint mode
    quad_xQpx = 0.5 * dot(x_star, Q * x_star)
    quad_xHx  = 0.5 * dot(x_star, H * x_star)
    quad_xHaugx = 0.5 * dot(x_star, H_aug_full * x_star)

    # Per-coordinate η-marginal variance σ²_η = diag(A H_aug⁻¹ Aᵀ)
    Z = F_H_aug_full \ Matrix(A_total')
    sigma2_eta = vec(sum(A_total .* Z', dims = 2))

    # Higher derivatives at the mode (Poisson)
    lambda_i = exp.(eta_star)
    d3 = -lambda_i
    d4 = -lambda_i

    # Importance-sampled correction (already in driver)
    is_corr = IntegratedNestedLaplace._importance_correction(
        PoissonLikelihood(), A_total, F_H_aug_full, x_star, eta_star,
        Float64[], df.y, o, A_full; N = 200)

    # Loggamma prior
    lprior = INLAModels.loggamma_logprior(theta)

    return (theta = theta,
            tau = exp(theta),
            n_obs = n,
            n_latent = n_latent,
            beta_mode = x_star[1],
            u_mode_first5 = x_star[2:6],
            sum_u = sum(x_star[2:end]),
            norm_u = norm(x_star[2:end]),
            ll_exact = ll_exact,
            quad_xQpx = quad_xQpx,
            quad_xHx = quad_xHx,
            quad_xHaugx = quad_xHaugx,
            log_det_H_direct = log_det_H_direct,
            log_det_H_aug_user = log_det_H_aug_user,
            log_det_H_aug_full = log_det_H_aug_full,
            log_det_Q_aug_user = log_det_Q_aug_user,
            log_det_Q_aug_full = log_det_Q_aug_full,
            log_det_AuHinv_proper = log_det_AuHinv_proper,
            log_det_H_c_user_proper = log_det_H_c_user_proper,
            log_det_AfHinv_via_aug = log_det_AfHinv_via_aug,
            log_det_H_c_full_proper = log_det_H_c_full_proper,
            log_det_AuAt = log_det_AuAt,
            log_det_AfAt = log_det_AfAt,
            sigma2_eta = sigma2_eta,
            lambda_i = lambda_i,
            d3 = d3,
            d4 = d4,
            eta_star = eta_star,
            r_m = r_m,
            is_corr = is_corr,
            lprior = lprior)
end

# Phase 6g.1: evaluate components at R-INLA's mode `(β_R, u_R)`. This is
# *not* our constrained Newton mode — it's an exogenous point loaded from
# `brunei_sla_components_rinla.json`. The point of the diagnostic is to
# decouple "do our component formulas match" from "do our modes match".
function eval_at_R_mode(theta::Real, beta_R::Real, u_R::AbstractVector,
                        df, W)
    n = nrow(df)
    n_fixed = 1
    n_random = n
    n_latent = n_fixed + n_random
    @assert length(u_R) == n_random "u_R must have length n_random=$n_random; got $(length(u_R))"

    besag = BesagModel(W; scale = true)
    Q_fixed = spzeros(Float64, n_fixed, n_fixed)
    Q_random = INLAModels.precision_matrix(besag, exp(theta))
    Q = blockdiag(Q_fixed, sparse(Q_random))

    A_random = sparse(1:n, 1:n, ones(n), n, n)
    A_total  = hcat(sparse(ones(n, 1)), A_random)

    # Constraint matrices (same as eval_components — Phase 6g operates in
    # the same constraint subspaces, only the evaluation point differs).
    A_user = sparse(hcat(zeros(1, n_fixed), ones(1, n) ./ sqrt(n)))
    e_int  = sparse([1], [1], [1.0], 1, n_latent)
    A_full = vcat(A_user, e_int)

    o = log.(df.E)

    # Plug R-INLA's mode into our layout: x_R = [β_R, u_R...].
    x_R = vcat(beta_R, collect(float.(u_R)))
    eta_R = A_total * x_R .+ o
    r_R   = A_total * x_R                                  # predictor without offset
    lambda_R = exp.(eta_R)

    # Per-i Poisson log-likelihood with the lfactorial constant — included
    # so the absolute reconstruction `mlik_R_path` matches R-INLA's mlik
    # without an unknown constant offset.
    log_y_fact_i = [y == 0 ? 0.0 : sum(log, 1:y) for y in df.y]
    ll_at_R = sum(df.y .* eta_R .- lambda_R .- log_y_fact_i)

    # Taylor at η = 0 of the per-i log-lik centered at η_m_i = r_R[i] + offset[i]
    #     T_i(0) = ℓ(r_m_i) − ℓ'(r_m_i) r_m_i + ½ ℓ''(r_m_i) r_m_i²
    #              − 1/6 ℓ'''(r_m_i) r_m_i³
    # where ℓ(r) = y_i(r + offset_i) − exp(r + offset_i) − log(y_i!) for Poisson.
    fp_i  = df.y .- lambda_R                  # ℓ'(r_m)
    fpp_i = .-lambda_R                        # ℓ''(r_m)
    fppp_i = .-lambda_R                       # ℓ'''(r_m)
    f0_i  = df.y .* eta_R .- lambda_R .- log_y_fact_i      # ℓ(r_m)
    a_i   = f0_i .- fp_i .* r_R .+ 0.5 .* fpp_i .* r_R .^ 2 .-
            (1/6) .* fppp_i .* r_R .^ 3
    sum_a = sum(a_i)
    cubic_correction = -(1/6) * sum(fppp_i .* r_R .^ 3)    # = (mlik_R - mlik_J) algebraically

    # Quadratic forms at R-INLA's mode. The H here is *built* at R-INLA's
    # mode (the data Hessian piece -h_eta(η_R) varies with η).
    H_at_R = Q + sparse(A_total' * spdiagm(0 => lambda_R) * A_total)

    quad_xQpx_at_R   = 0.5 * dot(x_R, Q * x_R)
    quad_xHx_at_R    = 0.5 * dot(x_R, H_at_R * x_R)

    # Constraint-corrected log-determinants at R-INLA's mode.
    AuT = sparse(A_user'); AuTAu = AuT * A_user
    AfT = sparse(A_full'); AfTAf = AfT * A_full

    H_aug_user_at_R = H_at_R + AuTAu
    log_det_H_aug_user_at_R = sparse_logdet(H_aug_user_at_R)
    F_H_aug_user = cholesky(Symmetric(H_aug_user_at_R))
    Wc_user = F_H_aug_user \ Matrix(AuT)
    S_user  = Symmetric(Matrix(A_user * Wc_user))
    log_det_AuHinv_at_R = log(only(S_user.data))
    # Textbook PLUS form on A_user (Phase 6g formula uses this).
    log_det_H_c_user_at_R = log_det_H_aug_user_at_R + log_det_AuHinv_at_R

    # `½ log_pseudo|Q|_c` via augmented A_full (Q has 2 null directions:
    # besag intrinsic null and the improper intercept).
    Q_aug_full = Q + AfTAf
    log_det_Q_c_pseudo = sparse_logdet(Q_aug_full)

    # Loggamma prior — same on both reconstruction paths.
    lprior = INLAModels.loggamma_logprior(theta)

    return (theta = theta,
            beta_R = beta_R,
            sum_u_R = sum(u_R),
            r_R_first5 = collect(r_R[1:5]),
            ll_at_R = ll_at_R,
            sum_a = sum_a,
            cubic_correction = cubic_correction,
            quad_xQpx_at_R = quad_xQpx_at_R,
            quad_xHx_at_R = quad_xHx_at_R,
            log_det_Q_c_pseudo = log_det_Q_c_pseudo,
            log_det_H_c_user_at_R = log_det_H_c_user_at_R,
            log_det_H_aug_user_at_R = log_det_H_aug_user_at_R,
            log_det_AuHinv_at_R = log_det_AuHinv_at_R,
            lprior = lprior)
end

function _load_rinla_modes_if_available()
    path = joinpath(HERE, "brunei_sla_components_rinla.json")
    isfile(path) || return nothing
    try
        data = JSON.parsefile(path)
        return data["rows"]
    catch err
        @warn "could not parse $path; skipping at-R-INLA's-mode pass" exception=err
        return nothing
    end
end

function main()
    df, W = load_brunei()
    rows = [eval_components(θ, df, W) for θ in θ_GRID]

    # Phase 6g.1: optionally also evaluate at R-INLA's mode.
    rinla_rows = _load_rinla_modes_if_available()
    at_R_results = Vector{Union{Nothing,NamedTuple}}(nothing, length(θ_GRID))
    if rinla_rows !== nothing
        @info "Loaded R-INLA modes from brunei_sla_components_rinla.json; running Phase 6g.1 at-R-mode eval"
        for (k, θ) in pairs(θ_GRID)
            # Find the R-INLA row whose θ matches.
            match = findfirst(r -> haskey(r, "theta") &&
                                   abs(Float64(r["theta"]) - θ) < 1.0e-6 &&
                                   haskey(r, "beta_mode") &&
                                   haskey(r, "u_mode_full"),
                              rinla_rows)
            if match === nothing
                @warn "no R-INLA row for θ=$θ (or `u_mode_full` missing); rerun bench/brunei_sla_components.R"
                continue
            end
            beta_R = Float64(rinla_rows[match]["beta_mode"])
            u_R    = Float64.(rinla_rows[match]["u_mode_full"])
            at_R_results[k] = eval_at_R_mode(θ, beta_R, u_R, df, W)
        end
    else
        @info "no brunei_sla_components_rinla.json found; skipping Phase 6g.1 at-R-INLA's-mode eval"
    end

    println()
    @printf("%6s  %10s  %10s  %14s  %14s  %14s  %14s\n",
            "θ", "τ", "ll_exact",
            "log|H_aug|_user", "log|H_c|_user_PLUS",
            "log|H_aug|_full", "log|Q_aug|_full")
    for r in rows
        @printf("%6.2f  %10.2f  %10.3f  %14.3f  %18.3f  %14.3f  %14.3f\n",
                r.theta, r.tau, r.ll_exact,
                r.log_det_H_aug_user, r.log_det_H_c_user_proper,
                r.log_det_H_aug_full, r.log_det_Q_aug_full)
    end

    # Strip arrays before serialising; map non-finite scalars to nothing
    # (R-INLA's R script uses `null` for the same reason).
    sanitize(v) = v isa AbstractArray ? sanitize.(collect(v)) :
                  (v isa Real && !isfinite(v)) ? nothing : v
    out = [Dict{String,Any}(string(k) => sanitize(v)
                            for (k, v) in pairs(r)) for r in rows]
    # Phase 6g.1: attach at-R-INLA's-mode quantities under `at_R_mode`.
    for (k, ar) in pairs(at_R_results)
        ar === nothing && continue
        out[k]["at_R_mode"] = Dict{String,Any}(
            string(kk) => sanitize(vv) for (kk, vv) in pairs(ar))
    end
    open(joinpath(HERE, "brunei_sla_components_julia.json"), "w") do io
        JSON.print(io, out, 2)
    end
    @info "Wrote brunei_sla_components_julia.json"

    # Print the Phase 6g.1 reconstruction summary if at-R-mode quantities
    # were computed. Algebraic identity check requires matching θ rows.
    if any(!isnothing, at_R_results)
        println()
        println("=== Phase 6g.1 at-R-INLA's-mode reconstruction (θ grid) ===")
        @printf("%6s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                "θ", "ll_at_R", "Σ a_i", "cubic", "½ x_R'Qx_R", "½ x_R'Hx_R",
                "log|H_c|_u")
        for (k, ar) in pairs(at_R_results)
            ar === nothing && continue
            @printf("%6.2f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f\n",
                    ar.theta, ar.ll_at_R, ar.sum_a, ar.cubic_correction,
                    ar.quad_xQpx_at_R, ar.quad_xHx_at_R,
                    ar.log_det_H_c_user_at_R)
        end
    end
end

main()
