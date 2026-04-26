using IntegratedNestedLaplace
using RDatasets
using DataFrames
using Random
using Statistics

function get_tokyo_data(rng = MersenneTwister(20260426))
    n = 366
    y = rand(rng, 0:3, n)
    df = DataFrame(y = y, day = 1:n)
    return df
end

println("--- Tutorial 1: Tokyo Rainfall (RW1) ---")
data = get_tokyo_data()

# 2. Run Julia INLA
# Use Poisson for better numerical stability in Newton-Raphson during dev
println("Running IntegratedNestedLaplace.jl...")
start_julia = time()
res = inla(@formula(y ~ f(day, RW1)), data, family=PoissonLikelihood())
end_julia = time()

println(res)
println("Julia Execution Time: ", round(end_julia - start_julia, digits=4), "s")
