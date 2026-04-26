using IntegratedNestedLaplace
using DataFrames
using SparseArrays
using LinearAlgebra

println("--- Example 06: Brunei School Disparities (ICAR) ---")

# 1. Mock Brunei Data (District Disparities)
# n=10 districts
function get_brunei_data(n=10)
    # y ~ Poisson(exp(eta))
    y = rand(5:15, n)
    df = DataFrame(y = Float64.(y), district = 1:n)
    
    # Simple Ring Adjacency Matrix
    W = spzeros(n, n)
    for i in 1:n
        next_idx = i == n ? 1 : i + 1
        prev_idx = i == 1 ? n : i - 1
        W[i, next_idx] = 1.0
        W[i, prev_idx] = 1.0
    end
    return df, W
end

df, W = get_brunei_data(10)

# 2. Run Model
# Mate ~ Intercept + f(district, ICAR)
println("Running IntegratedNestedLaplace.jl...")
start_time = time()

# Provision the ICARModel with the adjacency matrix
icar = ICARModel(W)

res = inla(
    @formula(y ~ f(district, ICAR)),
    df,
    family=PoissonLikelihood(),
    latent=icar,
    theta0=[1.0]
)

end_time = time()

# 3. Results
println(res)
println("Total Execution Time: ", round(end_time - start_time, digits=4), "s")
