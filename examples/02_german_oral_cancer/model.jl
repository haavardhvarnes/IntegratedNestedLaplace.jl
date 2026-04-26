using IntegratedNestedLaplace
using DataFrames
using SparseArrays
using LinearAlgebra
using Distributions

# 1. Mock Oral Cancer Data (544 Districts)
# For the prototype, we use a 100-district subset for speed
function get_cancer_data(n=100)
    # y ~ Poisson(E * exp(eta))
    # For now, simplify to Gaussian for demonstration of the DSL
    E = rand(0.5:0.1:2.0, n)
    y = [rand(Poisson(ei * 1.5)) for ei in E]
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
