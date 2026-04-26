using JSON
using Test

const FIXTURES_DIR = joinpath(@__DIR__, "..", "fixtures")

"""
    load_reference(name)

Load `test/fixtures/<name>/rinla_reference.json` produced by the matching
`examples/<name>/rinla.R` script. Returns a `Dict{String, Any}`.
"""
function load_reference(name::AbstractString)
    path = joinpath(FIXTURES_DIR, name, "rinla_reference.json")
    isfile(path) || error("missing reference fixture at $path. Run bench/Rrun.sh $name to regenerate.")
    return JSON.parsefile(path)
end

"""
    fixed_mean(ref, term) / fixed_sd(ref, term)

Look up R-INLA's posterior mean/sd for a fixed-effect term by name.
"""
fixed_mean(ref, term) = ref["fixed"][term]["mean"]
fixed_sd(ref, term)   = ref["fixed"][term]["sd"]

"""
    hyper_mean(ref, name) / hyper_sd(ref, name)

Look up the posterior mean/sd of a hyperparameter by R-INLA's row name
(e.g. "Precision for Female", "Rho for diid", etc.).
"""
hyper_mean(ref, name) = ref["hyper"][name]["mean"]
hyper_sd(ref, name)   = ref["hyper"][name]["sd"]

"""
    assert_fixed(ref, term, julia_mean, julia_sd; mean_atol_factor=0.10, sd_rtol=0.10, mean_floor=1e-3)

Assert agreement of a fixed-effect summary against R-INLA. The mean tolerance is
`max(mean_floor, mean_atol_factor × R-INLA SD)`. SD tolerance is relative.

Phase 1 default is 10 % of R-INLA SD for the mean: Julia returns the joint mode
at θ*, while R-INLA reports the marginal posterior mean integrated over θ. They
differ by O(0.1 × SD) on typical Gaussian-like posteriors. Phase 2 (CCD
integration) will tighten this back to ~1 % of SD.
"""
function assert_fixed(ref, term, julia_mean, julia_sd;
                      mean_atol_factor = 0.10,
                      sd_rtol          = 0.10,
                      mean_floor       = 1e-3)
    rmean = fixed_mean(ref, term)
    rsd   = fixed_sd(ref, term)
    mean_tol = max(mean_floor, mean_atol_factor * rsd)
    @test isapprox(julia_mean, rmean; atol = mean_tol)
    @test isapprox(julia_sd,   rsd;   rtol = sd_rtol)
end

"""
    assert_log_precision(ref, name, julia_log_tau; atol=0.15)

Compare R-INLA's posterior mean precision (on the log scale) against a Julia
log-precision *mode*. Default Phase 1 tolerance is 0.15 on the log-precision
scale (≈15 % on the precision scale) to account for the systematic
mode-vs-mean offset of the log-Gamma posterior. Tightens to ~0.05 once CCD
integration lands.
"""
function assert_log_precision(ref, name, julia_log_tau; atol = 0.15)
    rmean = hyper_mean(ref, name)
    @test isapprox(julia_log_tau, log(rmean); atol = atol)
end

"""
    assert_runtime(ref, julia_seconds; ratio=2.0)

Check that Julia warm wall-time is ≤ `ratio` × R-INLA's `cpu.used["Total"]`.
"""
function assert_runtime(ref, julia_seconds; ratio = 2.0)
    rinla_total = ref["cpu"]["Total"]
    @test julia_seconds ≤ ratio * rinla_total
end
