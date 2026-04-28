#!/usr/bin/env julia --project=.
#
# Cross-compare R-INLA and Julia component dumps from
# `bench/brunei_sla_components.{R,jl}`. Print a per-θ table of how each
# scalar component differs, then flag the components whose τ-slope
# matches the observed ~5-nat right-tail gap on Brunei.

using JSON
using Printf

const HERE = @__DIR__

function load_dumps()
    rinla_path = joinpath(HERE, "brunei_sla_components_rinla.json")
    julia_path = joinpath(HERE, "brunei_sla_components_julia.json")
    rinla = JSON.parsefile(rinla_path)
    julia = JSON.parsefile(julia_path)
    rinla_rows = rinla["rows"]
    julia_rows = julia
    return rinla_rows, julia_rows
end

function diff_per_theta(rinla_rows, julia_rows)
    comps = [
        ("ll_exact",            "ll_exact",                  "ll_exact"),
        ("quad_xQpx",           "quad_xQpx",                 "½ x'Qpx"),
        ("quad_xHx",            "quad_xHx",                  "½ x'Hx"),
        ("log_det_H_post",      "log_det_H_aug_user",        "log|H| (R) vs log|H+A_u'A_u| (J)"),
        ("log_det_H_c_user",    "log_det_H_c_user_proper",   "log|H_c|_user (textbook PLUS)"),
        ("log_det_H_c_full",    "log_det_H_c_full_proper",   "log|H_c|_full (textbook PLUS)"),
        ("log_det_H_aug_user",  "log_det_H_aug_user",        "log|H + A_u'A_u| (Rue-Held aug)"),
        ("log_det_H_aug_full",  "log_det_H_aug_full",        "log|H + A_f'A_f| (Rue-Held aug)"),
        ("log_det_AcHinv_user", "log_det_AuHinv_proper",     "log(A_u H⁻¹ A_u')"),
        ("mlik_int",            "ll_exact",                  "mlik(R) vs ll(J) [drift-only]"),
    ]
    println()
    @printf("%6s  ", "θ")
    for (_, _, label) in comps
        @printf("%24s  ", label)
    end
    println()
    for (rr, jr) in zip(rinla_rows, julia_rows)
        @printf("%6.2f  ", rr["theta"])
        for (rkey, jkey, _label) in comps
            r_val = get(rr, rkey, nothing)
            j_val = get(jr, jkey, nothing)
            if r_val !== nothing && j_val !== nothing
                r = r_val isa Real ? Float64(r_val) : NaN
                j = j_val isa Real ? Float64(j_val) : NaN
                d = r - j
                @printf("%8.3f  %8.3f  %6.3f  ", r, j, d)
            else
                @printf("%24s  ", "—")
            end
        end
        println()
    end
end

function tau_slope_table(rinla_rows, julia_rows)
    # Anchor at θ = 1.88 (or closest), then per-θ deltas to identify which
    # component carries the missing τ-slope.
    function find_anchor(rows)
        for r in rows
            if abs(r["theta"] - 1.88) < 1e-3
                return r
            end
        end
        return rows[1]
    end
    r_anchor = find_anchor(rinla_rows)
    j_anchor = find_anchor(julia_rows)

    comps = [
        ("ll_exact", "ll_exact", "Δll"),
        ("log_det_H_c_user", "log_det_H_c_user_proper", "Δ½log|H_c|_user"),
        ("log_det_H_aug_full", "log_det_H_aug_full", "Δ½log|H_aug_full|"),
        ("log_det_Q_aug_full", "log_det_Q_aug_full", "Δ½log|Q_aug_full|"),
        ("mlik_int", nothing, "Δmlik(R)"),
    ]
    println()
    println("=== τ-slope analysis (anchored at θ=1.88; Δ from anchor) ===")
    @printf("%6s  ", "θ")
    for (_, _, label) in comps
        @printf("%14s  ", label * "_R")
        @printf("%14s  ", label * "_J")
        @printf("%10s  ", "ΔR-ΔJ")
    end
    println()
    for (rr, jr) in zip(rinla_rows, julia_rows)
        @printf("%6.2f  ", rr["theta"])
        for (rkey, jkey, _label) in comps
            rv = get(rr, rkey, nothing)
            jv = jkey === nothing ? nothing : get(jr, jkey, nothing)
            r_a = get(r_anchor, rkey, nothing)
            j_a = jkey === nothing ? nothing : get(j_anchor, jkey, nothing)

            ΔR = (rv !== nothing && r_a !== nothing) ? Float64(rv) - Float64(r_a) : NaN
            ΔJ = (jv !== nothing && j_a !== nothing) ? Float64(jv) - Float64(j_a) : NaN

            r_str = isnan(ΔR) ? "—" : @sprintf("%14.3f", ΔR)
            j_str = isnan(ΔJ) ? "—" : @sprintf("%14.3f", ΔJ)
            diff_str = (isnan(ΔR) || isnan(ΔJ)) ? "—" : @sprintf("%10.3f", ΔR - ΔJ)
            @printf("%14s  %14s  %10s  ", r_str, j_str, diff_str)
        end
        println()
    end
end

"""
    reconstructions(rinla_rows, julia_rows)

Phase 6g.1 verification table.

For each θ on the grid where Julia's `at_R_mode` block is present, we
plug R-INLA's joint mode `(β_R, u_R)` into both formulas and compare
against R-INLA's `res\$mlik[1,1]`:

    mlik_J_path = ll_at_R + ½ log|Q_c|_pseudo − ½ log|H_c|_user
                  − ½ x_R' Q x_R + lprior        # our "evaluate at the mode" formula

    mlik_R_path = (Σ a_i) + ½ log|Q_c|_pseudo − ½ log|H_c|_user
                  + ½ x_R' H_R x_R + lprior     # R-INLA's "evaluate at sample = 0" formula

Algebraically, `R_path − J_path = cubic` (= `−1/6 Σ f'''_i r_m_i³`).
So `(R_path − J_path) − cubic` should be ≈ 0 to ~machine precision.

Two checks:
1. Algebraic identity: `(mlik_R_path − mlik_J_path) − cubic ≈ 0`
   to ~machine precision. This must hold by construction; if not,
   there's a sign/term error in this reconstruction.
2. R-INLA reconstruction: `mlik_R_path − res\$mlik[1,1]` should be
   ≤ 0.01 nats. If larger and θ-dependent, there's a hidden τ-shape
   contribution from R-INLA's `extra(θ)` we haven't accounted for.
"""
function reconstructions(rinla_rows, julia_rows)
    println()
    println("=== Phase 6g.1 reconstruction (at R-INLA's mode) ===")
    @printf("%6s  %12s  %12s  %12s  %12s  %12s  %12s\n",
            "θ", "mlik(R)", "mlik_R_path", "mlik_J_path",
            "ΔR_path−mlik", "Δ(R−J)−cub", "cubic")
    any_present = false
    for (rr, jr) in zip(rinla_rows, julia_rows)
        ar = get(jr, "at_R_mode", nothing)
        ar isa AbstractDict || continue
        any_present = true

        mlik_int = Float64(rr["mlik_int"])

        ll_at_R          = Float64(ar["ll_at_R"])
        sum_a            = Float64(ar["sum_a"])
        cubic            = Float64(ar["cubic_correction"])
        quad_xQpx_at_R   = Float64(ar["quad_xQpx_at_R"])
        quad_xHx_at_R    = Float64(ar["quad_xHx_at_R"])
        log_det_Q_c_pseudo = Float64(ar["log_det_Q_c_pseudo"])
        log_det_H_c_user_at_R = Float64(ar["log_det_H_c_user_at_R"])
        lprior = Float64(ar["lprior"])

        mlik_J_path = ll_at_R + 0.5 * log_det_Q_c_pseudo - 0.5 * log_det_H_c_user_at_R -
                      quad_xQpx_at_R + lprior
        mlik_R_path = sum_a + 0.5 * log_det_Q_c_pseudo - 0.5 * log_det_H_c_user_at_R +
                      quad_xHx_at_R + lprior

        # Acceptance gates.
        identity_residual    = (mlik_R_path - mlik_J_path) - cubic   # ≈ 0
        reconstruct_residual = mlik_R_path - mlik_int                # ≤ 0.01 nats

        @printf("%6.2f  %12.4f  %12.4f  %12.4f  %12.4f  %12.4e  %12.4f\n",
                Float64(rr["theta"]), mlik_int, mlik_R_path, mlik_J_path,
                reconstruct_residual, identity_residual, cubic)
    end
    if !any_present
        println("  (no Julia row contains an `at_R_mode` block — rerun the R")
        println("   diagnostic first, then `julia bench/brunei_sla_components.jl`)")
    end
end

function main()
    rinla_rows, julia_rows = load_dumps()
    diff_per_theta(rinla_rows, julia_rows)
    tau_slope_table(rinla_rows, julia_rows)
    reconstructions(rinla_rows, julia_rows)
end

main()
