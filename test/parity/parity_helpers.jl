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

Phase 2 default (after CCD integration over θ) is 10 % of R-INLA SD. CCD covers
the integration over θ, but the Gaussian Laplace approximation of `π(x|y,θ)`
still underestimates the marginal posterior mean for skewed likelihoods like
Bernoulli (typically by O(0.1 × SD) on a single coordinate). The simplified-
Laplace skewness correction lands in Phase 3 and tightens this to ~1 % of SD.
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
    assert_precision_mean(ref, name, julia_tau_mean; rtol=0.10)

Compare R-INLA's posterior precision mean against the Julia CCD-mixture mean
of `exp(θ_i)`. Default tolerance is 10 % on the precision scale (the prior
on log-precision is heavy-tailed, so even with CCD the agreement is looser
than for fixed effects).
"""
function assert_precision_mean(ref, name, julia_tau_mean; rtol = 0.10)
    rmean = hyper_mean(ref, name)
    @test isapprox(julia_tau_mean, rmean; rtol = rtol)
end

"""
    assert_runtime(ref, julia_seconds; ratio=2.0)

Check that Julia warm wall-time is ≤ `ratio` × R-INLA's `cpu.used["Total"]`.
"""
function assert_runtime(ref, julia_seconds; ratio = 2.0)
    rinla_total = ref["cpu"]["Total"]
    @test julia_seconds ≤ ratio * rinla_total
end
