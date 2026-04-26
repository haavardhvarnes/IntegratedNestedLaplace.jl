using IntegratedNestedLaplace
using DataFrames
using SparseArrays
using LinearAlgebra
using Distributions
using Meshes
using Unitful

println("--- Example 07: Dengue in Brazil (Non-Stationary SPDE) ---")

# 1. Generate Non-Stationary Spatial Data
function get_brazil_data(n=50)
    coords = [(rand(), rand()) for _ in 1:n]
    # Spatially varying field
    y = [sin(c[1]*3) * exp(c[2]) for c in coords]
    df = DataFrame(y = y, loc_id = 1:n)
    return df, coords
end

df, coords = get_brazil_data(50)

# 2. Build Mesh and Basis Functions
mesh = build_mesh(coords)
C, G = spde_matrices(mesh)
n_v = nvertices(mesh)

# Simple non-stationary basis: Intercept + x-coordinate
# B_kappa: log(kappa(s)) = theta_k1 + theta_k2 * x(s)
B_kappa = hcat(ones(n_v), [ustrip.(to(vertex(mesh, i)))[1] for i in 1:n_v])
B_tau = hcat(ones(n_v), [ustrip.(to(vertex(mesh, i)))[2] for i in 1:n_v])

# 3. Run Model
println("Running IntegratedNestedLaplace.jl...")
start_time = time()

# Provision NonStationarySPDEModel
ns_spde = NonStationarySPDEModel(C, G, B_kappa, B_tau)

# We have 2 params for kappa and 2 for tau = 4 hypers
res = inla(
    @formula(y ~ f(loc_id, NonStationarySPDE)),
    df,
    latent=ns_spde,
    theta0=[0.5, 0.1, 0.5, 0.1]
)

end_time = time()

# 4. Results
println(res)
println("Total Execution Time: ", round(end_time - start_time, digits=4), "s")
println("Estimated non-stationary log-kappa basis coeffs: ", res.mode_hyper[1:2])
println("Estimated non-stationary log-tau basis coeffs:   ", res.mode_hyper[3:4])
