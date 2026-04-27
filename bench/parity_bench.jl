#!/usr/bin/env julia --project=.
#
# Run every passing parity fit twice; record warm wall time and compare to
# the R-INLA `cpu.used["Total"]` recorded in the JSON fixture.
#
#   julia --project=. bench/parity_bench.jl
#
# Output is a markdown table on stdout, plus `bench/results.json`.

using IntegratedNestedLaplace
using DataFrames
using CSV
using JSON
using SparseArrays
using Statistics
using Printf

const FIX = joinpath(@__DIR__, "..", "test", "fixtures")
const EX  = joinpath(@__DIR__, "..", "examples")

# Each entry: (id, label, fit-fn) where fit-fn returns the warm wall time.
function run_fit(name, fit_fn)
    fit_fn()                    # cold (compile + warm-up)
    return @elapsed fit_fn()    # warm
end

function bench_salamander()
    df = CSV.read(joinpath(EX, "04_salamander_mating", "data", "salamander.csv"), DataFrame)
    df.Female = string.(df.Female)
    df.Male   = string.(df.Male)
    df.Cross  = string.(df.Cross)
    fit() = inla(@formula(Mate ~ 1 + Cross + f(Female, IID) + f(Male, IID)),
                 df, family = BernoulliLikelihood(), theta0 = [1.0, 1.0])
    return run_fit("salamander", fit)
end

function bench_bivariate()
    df = CSV.read(joinpath(EX, "05_meta_analysis", "data", "bivariate_synthetic.csv"), DataFrame)
    biv = BivariateIIDModel(; a1 = 0.25, b1 = 0.025, a2 = 0.25, b2 = 0.025, rho_precision = 0.4)
    fit() = inla(@formula(y ~ f(study, BivariateIID)), df,
                 family = GaussianLikelihood(),
                 latent = biv, theta0 = [4.0, 1.0, 0.5, 0.0])
    return run_fit("bivariate", fit)
end

function bench_brunei()
    df  = CSV.read(joinpath(EX, "06_brunei_school_disparities", "data", "areas.csv"), DataFrame)
    adj = CSV.read(joinpath(EX, "06_brunei_school_disparities", "data", "adjacency.csv"), DataFrame)
    n = nrow(df)
    W = sparse(adj.i, adj.j, Float64.(adj.w), n, n)
    besag = BesagModel(W; scale = true)
    fit() = inla(@formula(y ~ 1 + f(area, Besag)), df,
                 family = PoissonLikelihood(),
                 latent = besag, offset = log.(df.E), theta0 = [1.0])
    return run_fit("brunei", fit)
end

function bench_spde()
    df = CSV.read(joinpath(EX, "03_swiss_rainfall", "data", "locations.csv"), DataFrame)
    coords = collect(zip(df.x, df.y_coord))
    mesh = build_mesh(coords)
    C, G = spde_matrices(mesh)
    spde = SPDEModel(C, G)
    fit() = inla(@formula(y ~ f(loc_id, SPDE)), df,
                 family = GaussianLikelihood(),
                 latent = spde, theta0 = [3.0, 1.0, 1.0])
    return run_fit("spde", fit)
end

const TARGETS = [
    ("Salamander mating",      "04_salamander_mating",        bench_salamander),
    ("Bivariate meta (2diid)", "05_meta_analysis",            bench_bivariate),
    ("Brunei (besag)",         "06_brunei_school_disparities", bench_brunei),
    ("Stationary SPDE",        nothing,                       bench_spde),  # no R fixture for SPDE yet
]

results = []
for (label, fixture_id, fn) in TARGETS
    rinla_total = if fixture_id === nothing
        nothing
    else
        ref_path = joinpath(FIX, fixture_id, "rinla_reference.json")
        ref = JSON.parsefile(ref_path)
        Float64(ref["cpu"]["Total"])
    end
    julia_warm = fn()
    push!(results, (label = label, julia = julia_warm, rinla = rinla_total))
end

println()
println("# IntegratedNestedLaplace.jl — warm-run perf vs R-INLA 25.10.19")
println()
println("| Example | Julia warm (s) | R-INLA cpu.used Total (s) | Ratio (Julia / R) |")
println("|---|---:|---:|---:|")
for r in results
    rinla_str = r.rinla === nothing ? "n/a" : @sprintf("%.3f", r.rinla)
    ratio_str = r.rinla === nothing ? "n/a" : @sprintf("%.2f×", r.julia / r.rinla)
    @printf("| %s | %.3f | %s | %s |\n", r.label, r.julia, rinla_str, ratio_str)
end
println()

open(joinpath(@__DIR__, "results.json"), "w") do io
    JSON.print(io, [Dict(string(k) => v for (k, v) in pairs(r)) for r in results], 2)
end
