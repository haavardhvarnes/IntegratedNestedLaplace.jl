using IntegratedNestedLaplace
using DataFrames
using Random
using SparseArrays
using LinearAlgebra
using Distributions

function get_cancer_data(n=100; rng = MersenneTwister(20260426))
    E = rand(rng, 0.5:0.1:2.0, n)
    y = [rand(rng, Poisson(ei * 1.5)) for ei in E]
    df = DataFrame(y = Float64.(y), district = 1:n)
    return df
end

println("--- Tutorial 2: German Oral Cancer (IID + Covariates) ---")
data = get_cancer_data(50)

# 2. Run Model: y ~ 1 + f(district, IID)
println("Running IntegratedNestedLaplace.jl...")
start_time = time()
res = inla(@formula(y ~ 1 + f(district, IID)), data)
end_time = time()

println(res)
println("Execution Time: ", round(end_time - start_time, digits=4), "s")
